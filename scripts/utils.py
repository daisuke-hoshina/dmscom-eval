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
    "ensure_dir",
    "save_numpy",
    "save_csv",
    "convert_labels_to_tree",
    "timed_call",
]
