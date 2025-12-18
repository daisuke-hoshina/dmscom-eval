# DMSCOM Evaluation Pipeline

A lightweight reproduction pipeline for **DMSCOM** (de Berardinis et al., "Measuring the Structural Complexity of Music"). It loads WAV audio, runs the [mscom](https://github.com/jonnybluesman/mscom) implementation, computes structural metrics, and exports results for downstream analysis.

## Contents
- `scripts/run_dmscom.py` — run DMSCOM on a single file and save label matrices with metadata.
- `scripts/metrics.py` — depth, fragment imbalance, and singleton fragmentation metrics.
- `scripts/batch_eval.py` — batch processing for all WAV files in `audio/` with timing breakdowns.
- `scripts/utils.py` — shared helpers for audio I/O, tree conversion, and timing.
- `notebooks/analysis.ipynb` — starter notebook for PCA/plotting of metrics.
- `scripts/debug_seg_stats.py` — quick segment-length statistics from saved outputs.
- `scripts/debug_extract_features.py` — profile feature extraction hops/pooling.
- `scripts/debug_graph.py` — inspect recurrence graphs and diagonal exclusions.
- `scripts/debug_communities.py` — explore community assignments across levels.

## Setup (RunPod GPU)
1. **Start a RunPod instance** with a Python-ready GPU image.
2. **Clone the repository**:
   ```bash
   git clone https://github.com/daisuke-hoshina/dmscom-eval.git
   cd dmscom-eval
   ```
3. **Create a virtual environment (recommended)**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
4. **Install dependencies** (mscom comes from GitHub):
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install git+https://github.com/jonnybluesman/mscom
   ```
5. **Add audio**: place WAV files into the `audio/` directory.

## Usage
### Single file
`scripts/run_dmscom.py` exposes the full set of knobs used by the current pipeline:

```bash
python scripts/run_dmscom.py audio/example.wav \
  --out_dir metrics \
  --num-levels 5 \
  --target-hop-sec 0.5 \
  --exclude-sec 2.0 \
  --k 20 \
  --alpha 0.8 \
  --chain-weight 0.8 \
  --chain-hops 1 \
  --min-seg-sec 20,10,5,2 \
  --min-split-sec 20,10,5,2 \
  --max-dense-frames 5000
```

Key flags (all optional):
- `--target-hop-sec`: target hop for feature pooling; use `0` or a negative value to disable pooling and keep dense features.
- `--exclude-sec`: diagonal exclusion radius for the recurrence graph.
- `--chain-weight` / `--chain-hops`: strength and reach of temporal chain edges to stabilize communities.
- `--min-seg-sec` / `--min-split-sec`: per-level constraints; accepts a single float or comma-separated list, automatically padded to the number of levels.
- `--max-dense-frames`: guardrail to force pooling when the feature matrix would be too large for dense similarity.

Outputs for a single run are stored as `metrics/<basename>_labels.npz` with the label matrix and metadata (hop seconds, pooling, graph parameters) needed for downstream analysis.

### Batch evaluation
Process every WAV in `audio/`, saving labels and metrics:

```bash
python scripts/batch_eval.py --audio_dir audio --output_csv metrics/results.csv --labels_dir metrics
```

This script runs feature extraction and DMSCOM with mscom defaults, saves each label matrix to `metrics/<basename>_labels.npy`, and aggregates timing plus the three metrics into `metrics/results.csv`.

## Notes
- Timing columns in `results.csv` record feature extraction, graph/community detection (when exposed by mscom; otherwise total DMSCOM time), and metric computation durations.
- The `analysis.ipynb` notebook expects `metrics/results.csv` to exist and can be extended with additional visualizations.
- Use the debug scripts under `scripts/` to profile feature hops, visualize recurrence graphs/communities, or sanity-check segmentation statistics when adjusting parameters.
