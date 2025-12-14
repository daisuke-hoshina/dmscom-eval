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
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

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


def extract_features_with_meta(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 1024,
    n_mfcc: int = 13,
    target_hop_sec: float | None = 0.5,
    pool_agg: str = "median",
) -> tuple[np.ndarray, dict]:
    """Compute pooled chroma + MFCC features with metadata.

    The extractor computes chroma and MFCCs on a common hop length, concatenates
    them into frame-wise feature vectors, and optionally performs temporal
    pooling to reduce time resolution. Pooling aggregates consecutive frames by
    median (default) or mean and returns the pooled feature matrix along with
    metadata describing the effective hop duration.
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

    base_hop_sec = hop_length / float(sr)
    pool_size = 1
    if target_hop_sec is not None:
        pool_size = max(1, int(round(target_hop_sec / base_hop_sec)))
    if pool_size > 1:
        pooled_frames: list[np.ndarray] = []
        for start in range(0, features.shape[0], pool_size):
            end = min(start + pool_size, features.shape[0])
            slice_ = features[start:end]
            if pool_agg == "mean":
                pooled = slice_.mean(axis=0)
            else:
                pooled = np.median(slice_, axis=0)
            pooled_frames.append(pooled)

        features = np.vstack(pooled_frames)

        # Re-normalize pooled frames for stability.
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        features = features / norms

    effective_hop_sec = base_hop_sec * pool_size
    meta = {
        "sr": sr,
        "hop_length": hop_length,
        "pool_size": pool_size,
        "effective_hop_sec": effective_hop_sec,
        "target_hop_sec": target_hop_sec,
        "pool_agg": pool_agg,
        "n_chroma": 12,
        "n_mfcc": n_mfcc,
        "feature_slices": {
            "chroma": (0, 12),
            "mfcc": (12, 12 + n_mfcc),
        },
    }
    return features, meta


def extract_features(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 1024,
    n_mfcc: int = 13,
    target_hop_sec: float | None = None,
    pool_agg: str = "median",
) -> np.ndarray:
    """Backward compatible wrapper returning features only."""

    feats, _ = extract_features_with_meta(
        audio,
        sr,
        hop_length=hop_length,
        n_mfcc=n_mfcc,
        target_hop_sec=target_hop_sec,
        pool_agg=pool_agg,
    )
    return feats


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
    feature_slices: dict[str, tuple[int, int]] | None = None,
    max_proximity_hop: int = 1,
    max_dense_frames: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute recurrence (R) and proximity (Δ) similarity matrices.

    Recurrence is computed on chroma features via cosine similarity, while
    proximity weights come from MFCC distances with an exponential decay. A
    safety guard prevents accidentally materializing extremely large dense
    matrices.

    Parameters
    ----------
    features : np.ndarray, shape (n_frames, n_dims)
        Frame-level features from extract_features().
    feature_slices : dict[str, tuple[int, int]] | None
        Optional mapping defining column ranges for feature subsets. If None,
        chroma is assumed to occupy [0, 12) and MFCC the remainder.
    max_proximity_hop : int
        Maximum temporal hop to connect when constructing the proximity band.
    max_dense_frames : int
        Maximum number of frames allowed before raising to avoid dense blow-up.

    Returns
    -------
    R : np.ndarray, shape (n_frames, n_frames)
        Recurrence similarity matrix (cosine similarity, diagonal zeroed).
    Delta : np.ndarray, shape (n_frames, n_frames)
        Proximity matrix connecting temporally nearby frames.
    """

    X = np.asarray(features)
    if X.ndim != 2:
        raise ValueError("features must be a 2D array with shape (n_frames, n_dims)")

    n_frames, n_dims = X.shape
    if n_frames > max_dense_frames:
        raise ValueError(
            "Too many frames for dense similarity computation; "
            "enable pooling with --target-hop-sec (e.g., 0.5) or increase it to reduce frames."
        )

    slices = feature_slices or {"chroma": (0, 12), "mfcc": (12, n_dims)}
    chroma_slice = slices.get("chroma", (0, min(12, n_dims)))
    mfcc_slice = slices.get("mfcc", (chroma_slice[1], n_dims))

    chroma_feats = X[:, chroma_slice[0] : chroma_slice[1]]
    mfcc_feats = X[:, mfcc_slice[0] : mfcc_slice[1]]

    # Normalize to guard cosine similarity stability.
    chroma_norms = np.linalg.norm(chroma_feats, axis=1, keepdims=True)
    chroma_norms = np.where(chroma_norms == 0, 1.0, chroma_norms)
    chroma_normed = chroma_feats / chroma_norms

    R = chroma_normed @ chroma_normed.T
    np.fill_diagonal(R, 0.0)

    mfcc_norms = np.linalg.norm(mfcc_feats, axis=1, keepdims=True)
    mfcc_norms = np.where(mfcc_norms == 0, 1.0, mfcc_norms)
    mfcc_normed = mfcc_feats / mfcc_norms

    if n_frames < 2:
        return R, np.zeros_like(R)

    neighbor_dists = np.linalg.norm(mfcc_normed[1:] - mfcc_normed[:-1], axis=1)
    sigma = float(np.median(neighbor_dists)) + 1e-6

    Delta = np.zeros_like(R)
    for hop in range(1, max_proximity_hop + 1):
        if hop >= n_frames:
            break
        offsets = mfcc_normed[hop:] - mfcc_normed[:-hop]
        dists = np.linalg.norm(offsets, axis=1)
        weights = np.exp(-dists / sigma)
        i_idx = np.arange(n_frames - hop)
        j_idx = i_idx + hop
        Delta[i_idx, j_idx] = weights
        Delta[j_idx, i_idx] = weights

    return R, Delta


def build_graph_from_matrices(
    R: np.ndarray,
    Delta: np.ndarray,
    k: int = 10,
    alpha: float = 0.5,
    exclude_radius: int = 0,
    mutual_knn: bool = True,
    min_rec_sim: float = 0.0,
    chain_weight: float = 0.8,
    chain_hops: int = 1,
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
    exclude_radius : int
        Number of frames to exclude around the diagonal when selecting recurrence edges.
    mutual_knn : bool
        If True, keep only mutual recurrence neighbors.
    min_rec_sim : float
        Minimum recurrence similarity for an edge to be considered.
    chain_weight : float
        Weight for temporal chain edges.
    chain_hops : int
        Number of forward hops to connect in the temporal chain.

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

    # Precompute neighbor candidates for recurrence edges with exclusion radius.
    neighbor_sets: list[set[int]] = []
    for i in range(n_frames):
        row = R[i].copy()
        mask = np.ones_like(row, dtype=bool)
        idxs = np.arange(n_frames)
        mask[np.abs(idxs - i) <= exclude_radius] = False
        row[~mask] = -np.inf
        row[row <= min_rec_sim] = -np.inf

        if k < row.size:
            candidate_idx = np.argpartition(row, -k)[-k:]
        else:
            candidate_idx = np.where(row > -np.inf)[0]

        # Filter out any -inf leftovers after partitioning.
        valid_idx = [int(j) for j in candidate_idx if row[j] > -np.inf]
        neighbor_sets.append(set(valid_idx))

    # Add top-k recurrence edges per node, optionally keeping only mutual ones.
    for i, neighbors in enumerate(neighbor_sets):
        for j in neighbors:
            if mutual_knn and i not in neighbor_sets[j]:
                continue
            if i == j:
                continue
            w_rec = float(R[i, j])
            if w_rec <= 0:
                continue
            key = (min(i, j), max(i, j))
            edge_weights[key] = w_rec

    # Add proximity edges derived from Delta (scaled by chain_weight).
    nz_upper = np.argwhere(np.triu(Delta, k=1) > 0)
    for i, j in nz_upper:
        w_prox = float(chain_weight * Delta[i, j])
        if w_prox <= 0:
            continue
        key = (int(min(i, j)), int(max(i, j)))
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
    min_community_size: int = 8,
    init_single: bool = True,
    min_contiguous_frames_per_level: Sequence[int] | None = None,
) -> np.ndarray:
    """Hierarchical community detection that splits each parent community.

    Level 0 starts from a single community by default. Higher levels re-cluster
    each parent community's induced subgraph using a parent-relative threshold
    on edge weights so that children are always contained within their parent
    community. The ``min_community_size`` guard prevents over-fragmentation and
    ``low_percentile``/``high_percentile`` control the threshold sweep range.
    """

    n_vertices = graph.vcount()
    if n_vertices == 0:
        return np.zeros((num_levels, 0), dtype=int)

    # Level 0: either a single root community or a flat detection.
    if init_single:
        labels0 = np.zeros(n_vertices, dtype=int)
    elif graph.ecount() == 0:
        labels0 = np.arange(n_vertices, dtype=int)
    else:
        communities0 = graph.community_multilevel(weights=graph.es["weight"])
        labels0 = np.asarray(communities0.membership, dtype=int)

    label_matrix = np.empty((num_levels, n_vertices), dtype=int)
    label_matrix[0] = labels0
    current_labels = labels0

    # Prepare percentile schedule for deeper levels.
    thresholds = np.linspace(low_percentile, high_percentile, max(num_levels - 1, 1))
    if min_contiguous_frames_per_level is None:
        contig_frames = [0] * (num_levels - 1)
    else:
        contig_frames = list(min_contiguous_frames_per_level)
        if len(contig_frames) < num_levels - 1:
            contig_frames.extend([contig_frames[-1]] * (num_levels - 1 - len(contig_frames)))

    for level_idx in range(1, num_levels):
        thr_percentile = thresholds[min(level_idx - 1, thresholds.size - 1)]
        min_contig = contig_frames[min(level_idx - 1, len(contig_frames) - 1)]
        new_labels = np.empty(n_vertices, dtype=int)
        new_labels.fill(-1)
        next_label = 0
        any_split = False

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

            if weights_sub.size == 0:
                new_labels[nodes] = next_label
                next_label += 1
                continue

            thr = float(np.percentile(weights_sub, thr_percentile))
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

            cluster_sizes = {int(cid): int(count) for cid, count in zip(uniq, counts)}

            max_run_per_cluster: dict[int, int] = {}
            run_start = 0
            for idx in range(1, membership.size):
                contiguous = (
                    membership[idx] == membership[idx - 1]
                    and nodes[idx] == nodes[idx - 1] + 1
                )
                if not contiguous:
                    cid = int(membership[run_start])
                    run_len = idx - run_start
                    max_run_per_cluster[cid] = max(
                        max_run_per_cluster.get(cid, 0), run_len
                    )
                    run_start = idx
            cid_last = int(membership[run_start])
            max_run_per_cluster[cid_last] = max(
                max_run_per_cluster.get(cid_last, 0), membership.size - run_start
            )

            min_size = max(min_community_size, 2)
            valid_clusters = [
                cid
                for cid, count in cluster_sizes.items()
                if count >= min_size and max_run_per_cluster.get(cid, 0) >= min_contig
            ]
            if len(valid_clusters) < 2:
                new_labels[nodes] = next_label
                next_label += 1
                continue

            any_split = True
            largest_valid = max(valid_clusters, key=lambda cid: cluster_sizes[cid])
            label_map: dict[int, int] = {}
            for cluster_id in valid_clusters:
                cluster_nodes_local = np.where(membership == cluster_id)[0]
                cluster_nodes_global = nodes[cluster_nodes_local]
                new_labels[cluster_nodes_global] = next_label
                label_map[int(cluster_id)] = next_label
                next_label += 1

            for cluster_id in uniq:
                if cluster_id in valid_clusters:
                    continue
                cluster_nodes_local = np.where(membership == cluster_id)[0]
                cluster_nodes_global = nodes[cluster_nodes_local]
                new_labels[cluster_nodes_global] = label_map[largest_valid]

        assert np.all(new_labels >= 0)
        if not any_split:
            label_matrix[level_idx] = current_labels
            for remaining in range(level_idx + 1, num_levels):
                label_matrix[remaining] = current_labels
            break

        label_matrix[level_idx] = new_labels
        current_labels = new_labels

    return label_matrix


def DMSCOM(
    features: np.ndarray,
    num_levels: int = 5,
    k: int = 10,
    alpha: float = 0.5,
    min_segment_frames: int | None = None,
    min_segment_sec_per_level: float | Sequence[float] | None = None,
    hop_sec: float | None = None,
    min_split_sec_per_level: float | Sequence[float] | None = None,
    exclude_radius: int = 0,
    mutual_knn: bool = True,
    min_rec_sim: float = 0.0,
    chain_weight: float = 0.8,
    chain_hops: int = 1,
    init_single: bool = True,
    max_dense_frames: int = 5000,
    feature_slices: dict[str, tuple[int, int]] | None = None,
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
    min_segment_sec_per_level : float | Sequence[float] | None
        Minimum segment duration in seconds per level. Requires ``hop_sec``.
    hop_sec : float | None
        Effective hop duration in seconds for seconds-based postprocessing.
    min_split_sec_per_level : float | Sequence[float] | None
        Minimum contiguous run length required to consider a split valid, in
        seconds. Converted to frames via ``hop_sec``.
    exclude_radius : int
        Number of frames to exclude around the diagonal when selecting recurrence edges.
    mutual_knn : bool
        If True, keep only mutual recurrence neighbors.
    min_rec_sim : float
        Minimum recurrence similarity to include an edge.
    chain_weight : float
        Weight for temporal chain edges.
    chain_hops : int
        Number of forward hops to connect with temporal chains.
    init_single : bool
        Whether to start the hierarchy from a single community.
    max_dense_frames : int
        Maximum frames allowed before dense similarity computation is aborted.
    feature_slices : dict[str, tuple[int, int]] | None
        Optional column ranges describing feature composition (e.g., chroma vs MFCC).

    Returns
    -------
    label_matrix : np.ndarray, shape (n_levels, n_frames)
        Each row is a level, each column is a frame index, with integer community labels.
    """

    X = np.asarray(features)
    if X.size == 0:
        return np.zeros((0, 0), dtype=int)

    contig_frames_per_level: Sequence[int] | None = None
    if min_split_sec_per_level is not None and hop_sec is not None and hop_sec > 0:
        if isinstance(min_split_sec_per_level, (float, int)):
            contig_frames_per_level = [int(np.ceil(float(min_split_sec_per_level) / hop_sec))] * (
                num_levels - 1
            )
        else:
            sec_list = list(min_split_sec_per_level)
            if len(sec_list) < num_levels - 1:
                sec_list.extend([sec_list[-1]] * (num_levels - 1 - len(sec_list)))
            contig_frames_per_level = [int(np.ceil(float(s) / hop_sec)) for s in sec_list[: num_levels - 1]]

    R, Delta = compute_similarity_matrices(
        X,
        feature_slices=feature_slices,
        max_proximity_hop=chain_hops,
        max_dense_frames=max_dense_frames,
    )
    graph = build_graph_from_matrices(
        R,
        Delta,
        k=k,
        alpha=alpha,
        exclude_radius=exclude_radius,
        mutual_knn=mutual_knn,
        min_rec_sim=min_rec_sim,
        chain_weight=chain_weight,
        chain_hops=chain_hops,
    )
    label_matrix = run_hierarchical_community_detection(
        graph,
        num_levels=num_levels,
        init_single=init_single,
        min_contiguous_frames_per_level=contig_frames_per_level,
    )
    if min_segment_sec_per_level is not None and hop_sec is not None:
        label_matrix = postprocess_label_matrix_seconds(
            label_matrix, hop_sec, min_segment_sec_per_level
        )
    elif min_segment_frames is not None and min_segment_frames > 0:
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


def postprocess_label_matrix_seconds(
    label_matrix: np.ndarray,
    hop_sec: float,
    min_segment_sec_per_level: float | Sequence[float],
) -> np.ndarray:
    """Postprocess label matrix enforcing minimum segment durations in seconds."""

    labels_copy = np.asarray(label_matrix).copy()
    if labels_copy.ndim != 2 or labels_copy.shape[1] == 0:
        return labels_copy

    if isinstance(min_segment_sec_per_level, (float, int)):
        min_sec = [float(min_segment_sec_per_level)] * labels_copy.shape[0]
    else:
        min_sec = list(min_segment_sec_per_level)
        if len(min_sec) != labels_copy.shape[0]:
            if len(min_sec) == 0:
                return labels_copy
            # Extend last value if shorter than number of levels.
            last_val = float(min_sec[-1])
            min_sec = min_sec + [last_val] * (labels_copy.shape[0] - len(min_sec))

    for level_idx, min_sec_val in enumerate(min_sec[: labels_copy.shape[0]]):
        min_frames = int(np.ceil(min_sec_val / hop_sec))
        if min_frames > 0:
            labels_copy[level_idx] = merge_short_segments(
                labels_copy[level_idx], min_frames
            )

    return labels_copy


__all__ = [
    "TreeNode",
    "load_audio",
    "extract_features_with_meta",
    "extract_features",
    "extract_features_from_path",
    "compute_similarity_matrices",
    "build_graph_from_matrices",
    "run_multilevel_community_detection",
    "run_hierarchical_community_detection",
    "DMSCOM",
    "merge_short_segments",
    "postprocess_label_matrix",
    "postprocess_label_matrix_seconds",
    "ensure_dir",
    "save_numpy",
    "save_csv",
    "convert_labels_to_tree",
    "timed_call",
]
