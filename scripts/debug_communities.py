"""
Debug script: run multi-level community detection for a single audio file.

Usage:
    python scripts/debug_communities.py path/to/audio.wav
"""
from __future__ import annotations

import argparse
import numpy as np

from utils import (
    build_graph_from_matrices,
    compute_similarity_matrices,
    extract_features,
    load_audio,
    run_multilevel_community_detection,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audio", type=str, help="Path to an audio file (wav, mp3, etc.)"
    )
    parser.add_argument(
        "--num-levels", type=int, default=5, help="Number of community levels"
    )
    args = parser.parse_args()

    audio, sr = load_audio(args.audio, sr=None)
    feats = extract_features(audio, sr)
    R, Delta = compute_similarity_matrices(feats)
    graph = build_graph_from_matrices(R, Delta)
    labels = run_multilevel_community_detection(graph, num_levels=args.num_levels)

    print(f"Loaded: {args.audio}")
    print(f"  audio shape: {audio.shape}, sr={sr}")
    print(f"  features shape: {feats.shape} (frames, dims)")
    print(f"  Graph: {graph.vcount()} nodes, {graph.ecount()} edges")
    print(f"  Label matrix shape: {labels.shape}")

    for level_idx, level_labels in enumerate(labels):
        num_comms = np.unique(level_labels).size
        print(f"  Level {level_idx}: {num_comms} communities")


if __name__ == "__main__":
    main()
