"""Run DMSCOM on a single audio file and store labels."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from utils import (
    DMSCOM,
    ensure_dir,
    extract_features,
    load_audio,
    save_numpy,
    timed_call,
)



def run_pipeline(audio_path: Path, out_dir: Path, num_levels: int = 5) -> Path:
    audio, sr = load_audio(str(audio_path), sr=None)

    features, feat_time = timed_call(extract_features, audio, sr)

    label_matrix, dmscom_time = timed_call(DMSCOM, features, num_levels)
    if isinstance(label_matrix, tuple):
        label_matrix = label_matrix[0]

    output_path = out_dir / f"{audio_path.stem}_labels.npy"
    save_numpy(np.asarray(label_matrix), output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight DMSCOM on a single audio file")
    parser.add_argument("audio", type=str, help="Path to input WAV file")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("metrics"),
        help="Directory to store output label matrices",
    )
    parser.add_argument(
        "--num-levels",
        type=int,
        default=5,
        help="Number of hierarchical community levels",
    )
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    output_path = run_pipeline(Path(args.audio), args.out_dir, num_levels=args.num_levels)
    print(f"Saved labels to {output_path}")


if __name__ == "__main__":
    main()
