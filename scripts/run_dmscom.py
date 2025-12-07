"""Run DMSCOM on a single audio file and store labels."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from utils import ensure_dir, load_audio, save_numpy, timed_call

try:
    from mscom.feature_extraction import extract_features
    from mscom.dmscom import DMSCOM
except Exception as exc:  # pragma: no cover - defensive import guard
    raise ImportError(
        "mscom is required. Install via `pip install git+https://github.com/jonnybluesman/mscom`."
    ) from exc



def run_pipeline(audio_path: Path, out_dir: Path) -> Path:
    audio, sr = load_audio(str(audio_path))
    features, feat_time = timed_call(extract_features, audio, sr)

    labels_output = DMSCOM(features)
    if isinstance(labels_output, tuple):
        label_matrix = labels_output[0]
    else:
        label_matrix = labels_output

    output_path = out_dir / f"{audio_path.stem}_labels.npy"
    save_numpy(np.asarray(label_matrix), output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DMSCOM on a single audio file")
    parser.add_argument("audio", type=str, help="Path to input WAV file")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("metrics"),
        help="Directory to store output label matrices",
    )
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    output_path = run_pipeline(Path(args.audio), args.out_dir)
    print(f"Saved labels to {output_path}")


if __name__ == "__main__":
    main()
