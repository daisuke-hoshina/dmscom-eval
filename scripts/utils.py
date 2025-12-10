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
    Run flat multi-resolution community detection on a graph and return a label matrix.

    This variant thresholds the entire graph independently per level without
    enforcing parent/child relationships across levels.

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


def run_hierarchical_community_detection(
    graph: ig.Graph,
    num_levels: int = 5,
    low_percentile: float = 10.0,
    high_percentile: float = 80.0,
    min_community_size: int = 64,
) -> np.ndarray:
    """Hierarchical community detection that splits each parent community.

    Level 0 runs Louvain once on the full graph. Higher levels re-cluster each
    parent community's induced subgraph using a thresholded edge set so that
    children are always contained within their parent community. The
    ``min_community_size`` guard prevents over-fragmentation and
    ``low_percentile``/``high_percentile`` control the threshold sweep range.
    """

    n_vertices = graph.vcount()
    if n_vertices == 0:
        return np.zeros((num_levels, 0), dtype=int)

    # Level 0: single community detection over the full graph.
    if graph.ecount() == 0:
        labels0 = np.arange(n_vertices, dtype=int)
    else:
        communities0 = graph.community_multilevel(weights=graph.es["weight"])
        labels0 = np.asarray(communities0.membership, dtype=int)

    label_matrix = np.empty((num_levels, n_vertices), dtype=int)
    label_matrix[0] = labels0
    current_labels = labels0

    # Prepare thresholds from the global edge weights.
    if graph.ecount() == 0:
        thresholds = np.zeros(max(num_levels - 1, 1), dtype=float)
    else:
        weights = np.asarray(graph.es["weight"], dtype=float)
        lo = float(np.percentile(weights, low_percentile))
        hi = float(np.percentile(weights, high_percentile))
        thresholds = np.linspace(lo, hi, max(num_levels - 1, 1))

    for level_idx in range(1, num_levels):
        thr = thresholds[min(level_idx - 1, thresholds.size - 1)]
        new_labels = np.empty(n_vertices, dtype=int)
        new_labels.fill(-1)
        next_label = 0

        for parent_id in np.unique(current_labels):
            nodes = np.where(current_labels == parent_id)[0]
            if nodes.size == 0:
                continue

            if nodes.size < min_community_size:
                new_labels[nodes] = next_label
                next_label += 1
                continue

            subg = graph.induced_subgraph(nodes)
            weights_sub = np.asarray(subg.es["weight"], dtype=float)

            keep_idx = np.where(weights_sub >= thr)[0]
            if keep_idx.size == 0:
                new_labels[nodes] = next_label
                next_label += 1
                continue

            sub_edges = [subg.es[i].tuple for i in keep_idx]
            subg_thr = ig.Graph(n=subg.vcount(), edges=sub_edges, directed=False)
            subg_thr.es["weight"] = [float(weights_sub[i]) for i in keep_idx]

            if subg_thr.ecount() == 0:
                new_labels[nodes] = next_label
                next_label += 1
                continue

            communities = subg_thr.community_multilevel(weights=subg_thr.es["weight"])
            membership = np.asarray(communities.membership, dtype=int)
            uniq, counts = np.unique(membership, return_counts=True)

            valid_clusters = counts[counts >= min_community_size]
            if uniq.size < 2 or valid_clusters.size < 2:
                new_labels[nodes] = next_label
                next_label += 1
                continue

            for cluster_id in uniq:
                cluster_nodes_local = np.where(membership == cluster_id)[0]
                cluster_nodes_global = nodes[cluster_nodes_local]
                new_labels[cluster_nodes_global] = next_label
                next_label += 1

        assert np.all(new_labels >= 0)
        label_matrix[level_idx] = new_labels
        current_labels = new_labels

    return label_matrix


def DMSCOM(
    features: np.ndarray,
    num_levels: int = 5,
    k: int = 10,
    alpha: float = 0.5,
    min_segment_frames: int | None = None,
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
    min_segment_frames : int or None
        If not None and > 0, merge segments shorter than this many frames in
        each level (postprocessing in the time domain).

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
    label_matrix = run_hierarchical_community_detection(
        graph,
        num_levels=num_levels,
    )
    if min_segment_frames is not None and min_segment_frames > 0:
        label_matrix = postprocess_label_matrix(label_matrix, min_segment_frames)
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


def merge_short_segments(labels: np.ndarray, min_segment_frames: int) -> np.ndarray:
    """Merge contiguous runs shorter than min_segment_frames into their neighbors.

    Parameters
    ----------
    labels : np.ndarray, shape (n_frames,)
        1D array of integer community labels for a single level.
    min_segment_frames : int
        Minimum allowed segment length in frames. Any contiguous run of
        identical labels shorter than this will be merged into its left
        and/or right neighbor.

    Returns
    -------
    np.ndarray
        New 1D label array with short segments merged. The input is not modified.
    """

    labels_clean = np.asarray(labels).copy()
    if labels_clean.size == 0:
        return labels_clean

    n = labels_clean.size
    idx = 0
    while idx < n:
        start = idx
        current_label = labels_clean[start]
        while idx < n and labels_clean[idx] == current_label:
            idx += 1
        end = idx
        run_length = end - start

        if run_length >= min_segment_frames:
            continue

        left_label = labels_clean[start - 1] if start > 0 else None
        right_label = labels_clean[end] if end < n else None

        if left_label is None and right_label is None:
            break
        if left_label is None:
            labels_clean[start:end] = right_label
            continue
        if right_label is None:
            labels_clean[start:end] = left_label
            idx = start
            continue

        left_run_start = start - 1
        while left_run_start >= 0 and labels_clean[left_run_start] == left_label:
            left_run_start -= 1
        left_run_length = start - (left_run_start + 1)

        right_run_end = end
        while right_run_end < n and labels_clean[right_run_end] == right_label:
            right_run_end += 1
        right_run_length = right_run_end - end

        if left_run_length >= right_run_length:
            labels_clean[start:end] = left_label
            idx = start
        else:
            labels_clean[start:end] = right_label
            idx = start

    return labels_clean


def postprocess_label_matrix(label_matrix: np.ndarray, min_segment_frames: int) -> np.ndarray:
    """Apply merge_short_segments row-wise to a (n_levels, n_frames) label matrix."""

    labels_copy = np.asarray(label_matrix).copy()
    if labels_copy.ndim != 2 or labels_copy.shape[1] == 0:
        return labels_copy

    for level_idx in range(labels_copy.shape[0]):
        labels_copy[level_idx] = merge_short_segments(
            labels_copy[level_idx], min_segment_frames
        )

    return labels_copy


__all__ = [
    "TreeNode",
    "load_audio",
    "extract_features",
    "extract_features_from_path",
    "compute_similarity_matrices",
    "build_graph_from_matrices",
    "run_multilevel_community_detection",
    "run_hierarchical_community_detection",
    "DMSCOM",
    "merge_short_segments",
    "postprocess_label_matrix",
    "ensure_dir",
    "save_numpy",
    "save_csv",
    "convert_labels_to_tree",
    "timed_call",
]
