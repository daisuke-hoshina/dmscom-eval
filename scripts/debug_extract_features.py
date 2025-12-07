"""
Debug script: compute features for a single audio file and print basic info.

Usage:
    python scripts/debug_extract_features.py path/to/audio.wav
"""
from __future__ import annotations

import argparse
from pathlib import Path

from utils import ensure_dir, extract_features, load_audio, save_numpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=str, help="Path to an audio file (wav, mp3, etc.)")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("outputs/features_debug"),
        help="Where to save the feature matrix as a .npy file",
    )
    args = parser.parse_args()

    audio, sr = load_audio(args.audio, sr=None)
    feats = extract_features(audio, sr)

    print(f"Loaded: {args.audio}")
    print(f"  audio shape: {audio.shape}, sr={sr}")
    print(f"  features shape: {feats.shape} (frames, dims)")

    ensure_dir(args.out_dir)
    out_path = args.out_dir / (Path(args.audio).stem + "_features.npy")
    save_numpy(feats, out_path)
    print(f"Saved features to: {out_path}")


if __name__ == "__main__":
    main()
