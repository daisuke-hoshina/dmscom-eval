"""Utility helpers for the DMSCOM evaluation scripts.

This module centralizes audio loading, label-tree conversion, timing
wrappers, and basic persistence helpers so other scripts can stay lean.
"""
from __future__ import annotations

import contextlib
import dataclasses
import os
import time
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

import igraph as ig
import librosa
import numpy as np


@dataclasses.dataclass
class TreeNode:
    """Simple tree node representing a labeled segment.

    Attributes
    ----------
    label: str
        Label assigned by DMSCOM at this level.
    level: int
        Hierarchical level (0 is the coarsest segmentation level).
    start: int
        Start frame index (inclusive).
    end: int
        End frame index (exclusive).
    children: list[TreeNode]
        Child segments nested under this node.
    """

    label: str
    level: int
    start: int
    end: int
    children: List["TreeNode"] = dataclasses.field(default_factory=list)

    def duration(self) -> int:
        return self.end - self.start

    def is_leaf(self) -> bool:
        return len(self.children) == 0


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def time_block() -> Iterator[Tuple[callable, callable]]:
    start = time.perf_counter()

    def _elapsed() -> float:
        return time.perf_counter() - start

    yield (start, _elapsed)


def timed_call(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


# ---------------------------------------------------------------------------
# Audio + persistence helpers
# ---------------------------------------------------------------------------


def load_audio(path: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Load audio with librosa.

    Parameters
    ----------
    path:
        Path to the audio file.
    sr:
        Target sampling rate. If ``None`` librosa defaults (22050 Hz).
    """

    audio, rate = librosa.load(path, sr=sr, mono=True)
    return audio, rate


def extract_features(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 1024,
    n_mfcc: int = 13,
) -> np.ndarray:
    """Compute chroma + MFCC features aligned per frame.

    The extractor is intentionally lightweight: it combines ``chroma_cqt`` and
    MFCCs with a shared hop length, concatenates them, transposes to
    ``(n_frames, n_features)``, and performs simple per-frame L2 normalization
    for robustness.
    """

    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length
    )

    # Ensure equal frame counts by truncating to the shortest sequence.
    n_frames = min(chroma.shape[1], mfcc.shape[1])
    if chroma.shape[1] != n_frames:
        chroma = chroma[:, :n_frames]
    if mfcc.shape[1] != n_frames:
        mfcc = mfcc[:, :n_frames]

    features = np.concatenate([chroma, mfcc], axis=0).T  # (frames, dims)

    # Per-frame L2 normalization to stabilize scale across frames.
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    features = features / norms

    return features


def extract_features_from_path(
    path: str | Path, sr: int | None = None
) -> tuple[np.ndarray, int, np.ndarray]:
    """Load audio then compute features.

    Returns a tuple ``(audio, sr, features)`` for convenience.
    """

    audio, rate = load_audio(str(path), sr=sr)
    feats = extract_features(audio, rate)
    return audio, rate, feats


def compute_similarity_matrices(
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute recurrence (R) and proximity (Δ) similarity matrices.

    Parameters
    ----------
    features : np.ndarray, shape (n_frames, n_dims)
        Frame-level features from extract_features().

    Returns
    -------
    R : np.ndarray, shape (n_frames, n_frames)
        Recurrence similarity matrix (cosine similarity, diagonal zeroed).
    Delta : np.ndarray, shape (n_frames, n_frames)
        Proximity matrix connecting only temporally adjacent frames.
    """

    X = np.asarray(features)
    if X.ndim != 2:
        raise ValueError("features must be a 2D array with shape (n_frames, n_dims)")

    # L2-normalize per frame, guarding against zero rows.
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    Xn = X / norms

    R = Xn @ Xn.T
    np.fill_diagonal(R, 0.0)

    n_frames = X.shape[0]
    Delta = np.zeros_like(R)
    for i in range(n_frames - 1):
        d = np.linalg.norm(X[i] - X[i + 1])
        w = float(np.exp(-d))
        Delta[i, i + 1] = w
        Delta[i + 1, i] = w

    return R, Delta


def build_graph_from_matrices(
    R: np.ndarray,
    Delta: np.ndarray,
    k: int = 10,
    alpha: float = 0.5,
):
    """
    Build an undirected weighted graph G from recurrence and proximity matrices.

    Parameters
    ----------
    R : np.ndarray, shape (n_frames, n_frames)
        Recurrence similarity matrix.
    Delta : np.ndarray, shape (n_frames, n_frames)
        Proximity matrix.
    k : int
        Number of strongest recurrence neighbors to keep per node.
    alpha : float
        Weighting factor between recurrence and proximity when combining.

    Returns
    -------
    G : igraph.Graph
        Undirected weighted graph with 'weight' edge attributes.
    """

    if R.shape != Delta.shape or R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("R and Delta must be square matrices of the same shape")

    n_frames = R.shape[0]
    g = ig.Graph(n=n_frames, directed=False)

    edge_weights: dict[tuple[int, int], float] = {}

    # Add top-k recurrence edges per node.
    for i in range(n_frames):
        row = R[i].copy()
        row[i] = -np.inf
        if k < row.size:
            top_k_indices = np.argpartition(row, -k)[-k:]
        else:
            top_k_indices = np.arange(n_frames)

        for j in top_k_indices:
            if j == i:
                continue
            w_rec = float(R[i, j])
            key = (min(i, j), max(i, j))
            edge_weights[key] = w_rec

    # Add proximity edges.
    prox_indices = np.argwhere(Delta > 0)
    for i, j in prox_indices:
        if i == j:
            continue
        w_prox = float(Delta[i, j])
        key = (min(i, j), max(i, j))
        if key in edge_weights:
            edge_weights[key] = alpha * edge_weights[key] + (1.0 - alpha) * w_prox
        else:
            edge_weights[key] = w_prox

    edge_list = list(edge_weights.keys())
    weight_list = list(edge_weights.values())

    g.add_edges(edge_list)
    g.es["weight"] = weight_list

    return g


def run_multilevel_community_detection(
    graph: ig.Graph, num_levels: int = 5
) -> np.ndarray:
    """
    Run multi-resolution community detection on a graph and return a label matrix.

    Parameters
    ----------
    graph : ig.Graph
        Undirected weighted graph with 'weight' edge attributes.
    num_levels : int
        Number of resolution levels to compute.

    Returns
    -------
    label_matrix : np.ndarray, shape (n_levels, n_frames)
        Each row corresponds to one resolution level; each column is a node/frame.
        Entries are integer community labels (0..C_l-1 for each level l).
    """

    if graph.ecount() == 0 or graph.vcount() == 0:
        return np.zeros((0, graph.vcount()), dtype=int)

    weights = np.asarray(graph.es["weight"], dtype=float)
    w_min = float(weights.min())
    w_max = float(weights.max())
    thresholds = np.linspace(w_min, w_max, num_levels)

    level_memberships: list[np.ndarray] = []
    for thr in thresholds:
        keep_idx = np.where(weights >= thr)[0]
        if keep_idx.size == 0:
            labels = np.arange(graph.vcount(), dtype=int)
            level_memberships.append(labels)
            continue

        edge_tuples = [graph.es[i].tuple for i in keep_idx]
        g_thr = ig.Graph(n=graph.vcount(), edges=edge_tuples, directed=False)
        g_thr.es["weight"] = [float(weights[i]) for i in keep_idx]

        if g_thr.ecount() == 0:
            labels = np.arange(graph.vcount(), dtype=int)
        else:
            communities = g_thr.community_multilevel(weights=g_thr.es["weight"])
            labels = np.asarray(communities.membership, dtype=int)

        level_memberships.append(labels)

    if not level_memberships:
        return np.zeros((0, graph.vcount()), dtype=int)

    label_matrix = np.stack(level_memberships, axis=0)
    return label_matrix


def DMSCOM(
    features: np.ndarray,
    num_levels: int = 5,
    k: int = 10,
    alpha: float = 0.5,
) -> np.ndarray:
    """Lightweight DMSCOM wrapper: from features to hierarchical label matrix.

    Parameters
    ----------
    features : np.ndarray, shape (n_frames, n_dims)
        Frame-level features from `extract_features()`.
    num_levels : int
        Number of community resolution levels.
    k : int
        Number of strongest recurrence neighbors per node when building the graph.
    alpha : float
        Mixing factor between recurrence and proximity weights in the graph.

    Returns
    -------
    label_matrix : np.ndarray, shape (n_levels, n_frames)
        Each row is a level, each column is a frame index, with integer community labels.
    """

    X = np.asarray(features)
    if X.size == 0:
        return np.zeros((0, 0), dtype=int)

    R, Delta = compute_similarity_matrices(X)
    graph = build_graph_from_matrices(R, Delta, k=k, alpha=alpha)
    label_matrix = run_multilevel_community_detection(graph, num_levels=num_levels)
    return label_matrix


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_numpy(array: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    np.save(path, array)
    return path


def save_csv(rows: Iterable[dict], path: str | Path) -> Path:
    import pandas as pd

    df = pd.DataFrame(rows)
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Label tree conversion
# ---------------------------------------------------------------------------


def _labels_to_segments(labels: np.ndarray) -> List[Tuple[str, int, int]]:
    segments: List[Tuple[str, int, int]] = []
    if labels.size == 0:
        return segments

    current = labels[0]
    start = 0
    for idx in range(1, len(labels)):
        if labels[idx] != current:
            segments.append((str(current), start, idx))
            current = labels[idx]
            start = idx
    segments.append((str(current), start, len(labels)))
    return segments


def convert_labels_to_tree(label_matrix: np.ndarray) -> TreeNode:
    """Convert a level x frame label matrix to a hierarchical tree.

    The output tree has a synthetic root (level ``-1``) with all level-0
    segments as its direct children. Child relationships for deeper
    levels are inferred by containment of time spans.
    """

    if label_matrix.ndim != 2:
        raise ValueError("Expected label matrix with shape (n_levels, n_frames)")

    n_levels, n_frames = label_matrix.shape
    root = TreeNode(label="root", level=-1, start=0, end=n_frames)

    # Build segments for every level
    level_segments: List[List[TreeNode]] = []
    for level in range(n_levels):
        segments = []
        for label, start, end in _labels_to_segments(label_matrix[level]):
            segments.append(TreeNode(label=label, level=level, start=start, end=end))
        level_segments.append(segments)

    # Attach children based on containment
    for level in range(n_levels):
        parents = root.children if level == 0 else level_segments[level - 1]
        children = level_segments[level]
        for child in children:
            # find parent covering child window
            for parent in parents:
                if parent.start <= child.start and parent.end >= child.end:
                    parent.children.append(child)
                    break

    return root


__all__ = [
    "TreeNode",
    "load_audio",
    "extract_features",
    "extract_features_from_path",
    "compute_similarity_matrices",
    "build_graph_from_matrices",
    "run_multilevel_community_detection",
    "DMSCOM",
    "ensure_dir",
    "save_numpy",
    "save_csv",
    "convert_labels_to_tree",
    "timed_call",
]
