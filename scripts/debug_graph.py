"""
Debug script: compute similarity matrices and graph for a single audio file.

Usage:
    python scripts/debug_graph.py path/to/audio.wav
"""
from __future__ import annotations

import argparse

from utils import (
    build_graph_from_matrices,
    compute_similarity_matrices,
    extract_features,
    load_audio,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=str, help="Path to an audio file (wav, mp3, etc.)")
    args = parser.parse_args()

    audio, sr = load_audio(args.audio, sr=None)
    feats = extract_features(audio, sr)
    R, Delta = compute_similarity_matrices(feats)
    graph = build_graph_from_matrices(R, Delta)

    print(f"Loaded: {args.audio}")
    print(f"  audio shape: {audio.shape}, sr={sr}")
    print(f"  features shape: {feats.shape} (frames, dims)")
    print(f"  R shape: {R.shape}")
    print(f"  Delta shape: {Delta.shape}")
    print(f"  Graph: {graph.vcount()} nodes, {graph.ecount()} edges")

    avg_degree = (2 * graph.ecount()) / graph.vcount() if graph.vcount() > 0 else 0.0
    weights = graph.es["weight"] if graph.ecount() > 0 else []
    min_w = min(weights) if weights else 0.0
    max_w = max(weights) if weights else 0.0
    print(f"  Average degree: {avg_degree:.3f}")
    print(f"  Edge weight range: min={min_w:.6f}, max={max_w:.6f}")


if __name__ == "__main__":
    main()
