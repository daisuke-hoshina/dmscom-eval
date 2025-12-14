# DMSCOM Evaluation Pipeline

A lightweight reproduction pipeline for **DMSCOM** (de Berardinis et al., "Measuring the Structural Complexity of Music"). It loads WAV audio, runs the [mscom](https://github.com/jonnybluesman/mscom) implementation, computes structural metrics, and exports results for downstream analysis.

## Contents
- `scripts/run_dmscom.py` — run DMSCOM on a single file and save label matrices.
- `scripts/metrics.py` — depth, fragment imbalance, and singleton fragmentation metrics.
- `scripts/batch_eval.py` — batch processing for all WAV files in `audio/` with timing breakdowns.
- `scripts/utils.py` — shared helpers for audio I/O, tree conversion, and timing.
- `notebooks/analysis.ipynb` — starter notebook for PCA/plotting of metrics.
- `scripts/debug_seg_stats.py` — quick segment-length statistics from saved outputs.

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
Run DMSCOM for a single file (now with pooled feature hops and diagonal exclusion):
```bash
python scripts/run_dmscom.py audio/example.wav --num-levels 6 --target-hop-sec 0.5 --exclude-sec 3 --k 20 --alpha 0.8 --min-seg-sec 20,10,5,2,1,1
```

Batch-evaluate every WAV in `audio/`, measuring timings and exporting `metrics/results.csv`:
```bash
python scripts/batch_eval.py
```

Outputs:
- Label matrices and metadata saved to `metrics/<basename>_labels.npz` (includes hop seconds for postprocessing)
- Aggregate metrics/timings saved to `metrics/results.csv`

## Notes
- The timing columns in `results.csv` record feature extraction, graph/community detection (when exposed by mscom; otherwise total DMSCOM time), and metric computation durations.
- The `analysis.ipynb` notebook expects `metrics/results.csv` to exist and can be extended with additional visualizations.
