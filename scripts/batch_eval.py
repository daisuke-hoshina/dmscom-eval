"""Batch evaluation for DMSCOM across all WAV files in audio/."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from metrics import compute_metrics
from utils import ensure_dir, load_audio, save_csv, timed_call

try:
    from mscom.feature_extraction import extract_features
    from mscom.dmscom import DMSCOM
except Exception as exc:  # pragma: no cover - defensive import guard
    raise ImportError(
        "mscom is required. Install via `pip install git+https://github.com/jonnybluesman/mscom`."
    ) from exc


def _run_dmscom(audio_path: Path) -> Dict[str, object]:
    audio, sr = load_audio(str(audio_path))

    features, feat_time = timed_call(extract_features, audio, sr)
    labels_output, dmscom_time = timed_call(DMSCOM, features)

    if isinstance(labels_output, dict):
        label_matrix = labels_output.get("labels") or labels_output.get("label_matrix")
        graph_time = labels_output.get("graph_time")
        community_time = labels_output.get("community_time")
    elif isinstance(labels_output, tuple):
        label_matrix = labels_output[0]
        graph_time = None
        community_time = None
    else:
        label_matrix = labels_output
        graph_time = None
        community_time = None

    return {
        "file": audio_path.name,
        "feature_time": feat_time,
        "graph_time": graph_time if graph_time is not None else dmscom_time,
        "community_time": community_time if community_time is not None else dmscom_time,
        "metrics_time": 0.0,
        "label_matrix": np.asarray(label_matrix),
    }


def batch_process(audio_dir: Path, output_csv: Path, labels_dir: Path) -> List[Dict[str, object]]:
    ensure_dir(output_csv.parent)
    ensure_dir(labels_dir)

    results: List[Dict[str, object]] = []
    audio_files = sorted(audio_dir.glob("*.wav"))

    for audio_path in tqdm(audio_files, desc="Processing audio"):
        record = _run_dmscom(audio_path)

        metrics_start = time.perf_counter()
        metrics = compute_metrics(record["label_matrix"])
        metrics_elapsed = time.perf_counter() - metrics_start

        record.update(metrics)
        record["metrics_time"] = metrics_elapsed

        label_path = labels_dir / f"{audio_path.stem}_labels.npy"
        np.save(label_path, record["label_matrix"])
        record.pop("label_matrix")

        results.append(record)

    save_csv(results, output_csv)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-evaluate DMSCOM for WAV files")
    parser.add_argument("--audio_dir", type=Path, default=Path("audio"), help="Input WAV directory")
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("metrics/results.csv"),
        help="Path to save the aggregated CSV",
    )
    parser.add_argument(
        "--labels_dir",
        type=Path,
        default=Path("metrics"),
        help="Directory to store per-file label matrices",
    )
    args = parser.parse_args()

    batch_process(args.audio_dir, args.output_csv, args.labels_dir)


if __name__ == "__main__":
    main()
