"""Compute segment length statistics per level from a saved DMSCOM npz file."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _segment_lengths(labels: np.ndarray) -> list[int]:
    if labels.size == 0:
        return []
    lengths: list[int] = []
    start = 0
    for idx in range(1, labels.size):
        if labels[idx] != labels[start]:
            lengths.append(idx - start)
            start = idx
    lengths.append(labels.size - start)
    return lengths


def main() -> None:
    parser = argparse.ArgumentParser(description="Print per-level segment stats from npz output")
    parser.add_argument("npz", type=Path, help="Path to .npz file produced by run_dmscom.py")
    args = parser.parse_args()

    with np.load(args.npz) as data:
        labels = np.asarray(data["labels"])
        if "hop_sec" not in data.files:
            raise ValueError("hop_sec metadata is missing; rerun run_dmscom.py")
        hop_sec = float(data["hop_sec"].item())

    if labels.ndim != 2:
        raise ValueError("Expected labels with shape (levels, frames)")
    if hop_sec <= 0:
        raise ValueError("hop_sec metadata is missing or invalid")

    print(f"Levels: {labels.shape[0]}, Frames: {labels.shape[1]}, Hop: {hop_sec:.4f} sec")
    for level_idx in range(labels.shape[0]):
        lengths = _segment_lengths(labels[level_idx])
        if not lengths:
            print(f"Level {level_idx}: no segments")
            continue
        lengths_sec = np.asarray(lengths, dtype=float) * hop_sec
        print(
            f"Level {level_idx}: segments={len(lengths_sec)}, "
            f"mean={lengths_sec.mean():.3f}s, median={np.median(lengths_sec):.3f}s, "
            f"min={lengths_sec.min():.3f}s, max={lengths_sec.max():.3f}s"
        )
if __name__ == "__main__":
    main()
