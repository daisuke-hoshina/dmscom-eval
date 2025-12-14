"""Run DMSCOM on a single audio file and store labels."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from utils import DMSCOM, ensure_dir, extract_features_with_meta, load_audio, timed_call


def _parse_float_list(value: str | None, num_levels: int, name: str) -> float | list[float] | None:
    """Parse a comma-separated float string into a value or per-level list."""

    if value is None:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        return None
    try:
        floats = [float(p) for p in parts]
    except ValueError as exc:  # pragma: no cover - argparse protects this
        raise ValueError(f"Invalid {name} specification: {value}") from exc

    if len(floats) == 1:
        return floats[0]
    if len(floats) < num_levels:
        floats.extend([floats[-1]] * (num_levels - len(floats)))
    return floats



def run_pipeline(
    audio_path: Path,
    out_dir: Path,
    num_levels: int = 5,
    target_hop_sec: float | None = 0.5,
    exclude_sec: float = 2.0,
    k: int = 20,
    alpha: float = 0.8,
    chain_weight: float = 0.8,
    chain_hops: int = 1,
    min_seg_sec: float | list[float] | None = None,
    min_split_sec: float | list[float] | None = None,
    max_dense_frames: int = 5000,
) -> Path:
    audio, sr = load_audio(str(audio_path), sr=None)

    pooled_target_hop = None if target_hop_sec is not None and target_hop_sec <= 0 else target_hop_sec

    (features, meta), _ = timed_call(
        extract_features_with_meta, audio, sr, target_hop_sec=pooled_target_hop
    )

    n_frames = features.shape[0]
    if n_frames > max_dense_frames:
        raise ValueError(
            "Too many frames for dense similarity computation; enable pooling "
            "with --target-hop-sec (e.g., 0.5) or increase it to reduce frames."
        )

    hop_sec = float(meta.get("effective_hop_sec", 0))
    exclude_radius = int(round(exclude_sec / hop_sec)) if hop_sec > 0 else 0

    effective_min_split_sec = min_split_sec if min_split_sec is not None else min_seg_sec

    label_matrix, _ = timed_call(
        DMSCOM,
        features,
        num_levels,
        k,
        alpha,
        min_segment_frames=None,
        min_segment_sec_per_level=min_seg_sec,
        min_split_sec_per_level=effective_min_split_sec,
        hop_sec=hop_sec,
        exclude_radius=exclude_radius,
        chain_weight=chain_weight,
        chain_hops=chain_hops,
        max_dense_frames=max_dense_frames,
        feature_slices=meta.get("feature_slices"),
    )
    if isinstance(label_matrix, tuple):
        label_matrix = label_matrix[0]

    output_path = out_dir / f"{audio_path.stem}_labels.npz"
    ensure_dir(output_path.parent)
    np.savez(
        output_path,
        labels=np.asarray(label_matrix),
        hop_sec=hop_sec,
        sr=meta.get("sr", sr),
        hop_length=meta.get("hop_length"),
        pool_size=meta.get("pool_size"),
        target_hop_sec=meta.get("target_hop_sec"),
        exclude_radius=exclude_radius,
        k=k,
        alpha=alpha,
        chain_weight=chain_weight,
        chain_hops=chain_hops,
        min_seg_sec=min_seg_sec,
        min_split_sec=effective_min_split_sec,
    )
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
    parser.add_argument(
        "--target-hop-sec",
        type=float,
        default=0.5,
        help="Target hop duration in seconds for feature pooling (0 or negative disables)",
    )
    parser.add_argument(
        "--exclude-sec",
        type=float,
        default=2.0,
        help="Exclusion radius in seconds around the diagonal for recurrence graph",
    )
    parser.add_argument("--k", type=int, default=20, help="Number of recurrence neighbors")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="Mixing factor between recurrence and proximity weights",
    )
    parser.add_argument(
        "--chain-weight",
        type=float,
        default=0.8,
        help="Weight assigned to temporal chain edges (0.5-1.0 often stabilizes)",
    )
    parser.add_argument(
        "--chain-hops",
        type=int,
        default=1,
        help="Number of forward hops to connect with temporal chain edges",
    )
    parser.add_argument(
        "--min-seg-sec",
        type=str,
        default="20,10,5,2",
        help="Minimum segment duration in seconds (single value or comma-separated per level)",
    )
    parser.add_argument(
        "--min-split-sec",
        type=str,
        default=None,
        help=(
            "Minimum contiguous duration in seconds to accept a split (single value or "
            "comma-separated per level); defaults to --min-seg-sec when omitted"
        ),
    )
    parser.add_argument(
        "--max-dense-frames",
        type=int,
        default=5000,
        help="Maximum frames allowed before requiring pooling to avoid dense blow-up",
    )
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    min_seg_sec = _parse_float_list(args.min_seg_sec, args.num_levels, "min-seg-sec")
    min_split_sec = _parse_float_list(args.min_split_sec, args.num_levels, "min-split-sec")
    output_path = run_pipeline(
        Path(args.audio),
        args.out_dir,
        num_levels=args.num_levels,
        target_hop_sec=args.target_hop_sec,
        exclude_sec=args.exclude_sec,
        k=args.k,
        alpha=args.alpha,
        chain_weight=args.chain_weight,
        chain_hops=args.chain_hops,
        min_seg_sec=min_seg_sec,
        min_split_sec=min_split_sec,
        max_dense_frames=args.max_dense_frames,
    )
    print(f"Saved labels to {output_path}")


if __name__ == "__main__":
    main()
