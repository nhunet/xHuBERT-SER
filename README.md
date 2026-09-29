# xHuBERT: A Selective Layer Aggregation Framework for Robust Speaker-Independent Speech Emotion Recognition

Official implementation of the paper **"xHuBERT: A Selective Layer Aggregation Framework for Robust Speaker-Independent Speech Emotion Recognition"** by Tan Nhu Nguyen and Thanh Hung Bui (Industrial University of Ho Chi Minh City).

xHuBERT combines a HuBERT-base backbone with **Selective Layer Aggregation (SLA)** over all 13 hidden states, **Attention Pooling (AP)**, and a 256-dimensional emotion embedding, fine-tuned end-to-end with optional speaker-adversarial regularisation. The framework is evaluated on RAVDESS under both 5-fold Stratified Cross-Validation and a **Leave-One-Speaker-Group-Out (LOSGO)** protocol that eliminates speaker leakage.

Key results on RAVDESS (speech-only, 1,440 utterances, 24 actors, 8 emotions):

| Protocol | Accuracy |
|----------|----------|
| xHuBERT — 5-fold CV | **88.40%** |
| xHuBERT — LOSGO (3-seed mean) | **69.56 ± 3.11%** |
| Best single-seed (augmentation + GRL) | **73.61%** |

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Dataset](#dataset)
5. [Reproducing Paper Results](#reproducing-paper-results)
6. [Pretrained Checkpoints](#pretrained-checkpoints)
7. [Results](#results)
8. [Citation](#citation)
9. [License](#license)
10. [Acknowledgments](#acknowledgments)

## Repository Structure

```
xHuBERT-SER/
├── config.py                    # Central hyperparameters and paths
├── data.py                      # RAVDESS dataset loader (16 kHz / 22.05 kHz)
├── features.py                  # MFCC, LogMel, LPCC, Chroma, Prosody, Statistical
├── protocols.py                 # 5-fold Stratified CV + LOSGO 6-group split
├── stats.py                     # Paired t-test, Cohen's d, TOST, Wilcoxon
├── utils.py                     # pos_conv_embed fix, logging, seeding
├── run_all.py                   # Orchestrator for all experiments
│
├── models/
│   ├── xhubert.py               # xHuBERT model (SLA + AP + emotion head)
│   ├── hubert_vanilla.py        # HuBERT baseline (mean pooling, last layer)
│   ├── fusion.py                # Gated fusion architectures (F1, F2, F3)
│   ├── ml_classifiers.py        # SVM, Random Forest, KNN wrappers
│   ├── dl_1d.py                 # 1D-CNN baseline
│   └── dl_2d.py                 # 2D-CNN baseline
│
├── experiments/
│   ├── exp1_feature_comparison.py    # Table 4: handcrafted features × ML
│   ├── exp2_hubert_frozen.py         # Table 5: frozen HuBERT + SVM/RF
│   ├── exp3_xhubert_finetune.py      # Tables 6, 7; Figure 5: main result
│   ├── exp4_ablation.py              # Table 9: SLA + AP + emotion-head ablation
│   ├── exp5_fusion.py                # Tables 10, 11; Figure 8: fusion redundancy
│   ├── exp6_speaker_adversarial.py   # Tables 13, 14: GRL analysis
│   ├── exp7_layer_probing.py         # Figures 6, 7: per-layer probing
│   └── exp8_sota_comparison.py       # Table 15: SOTA comparison
│
├── notebooks/                   # Colab-ready notebooks (NB1–NB7)
├── visualization/plots.py       # Figure generation
├── results/                     # 9 CSV files with all reported numbers
├── figures/                     # Paper figures (main + EDA)
│   ├── fig_exp1_*.png
│   ├── fig_exp3_comparison.png
│   ├── fig_exp5_gate_weights.png
│   ├── fig_exp6_adversarial.png
│   ├── fig_exp7_layer_curve.png
│   └── eda/                     # RAVDESS exploratory plots (class, duration, MFCC…)
│
├── requirements.txt
├── LICENSE                      # MIT
├── CITATION.cff
└── .zenodo.json
```

## System Requirements

- Python **3.10** or newer
- PyTorch **2.11.0** with CUDA **12.8** (other CUDA versions work; adjust `torch` install accordingly)
- A CUDA-capable GPU with **≥ 12 GB VRAM** (tested on NVIDIA T4 / A100)
- Approximate wall-clock on a single T4 (Google Colab): ~4 h for xHuBERT 6-fold LOSGO, single seed

All exact package versions used to produce the paper numbers are listed in [`requirements.txt`](requirements.txt).

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/nhunet/xHuBERT-SER.git
cd xHuBERT-SER

python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

# Install PyTorch matching your CUDA version first (see https://pytorch.org/)
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

## Dataset

xHuBERT is trained and evaluated on **RAVDESS Speech**, publicly available at Zenodo:

> Livingstone, S. R., & Russo, F. A. (2018). *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)*. Zenodo. https://doi.org/10.5281/zenodo.1188976

Download and extract the speech-only archive:

```bash
mkdir -p RAVDESS
cd RAVDESS
# Download Audio_Speech_Actors_01-24.zip from the DOI above (~200 MB) and unzip
unzip Audio_Speech_Actors_01-24.zip
cd ..

export RAVDESS_ROOT="$(pwd)/RAVDESS"     # Windows PowerShell: $env:RAVDESS_ROOT = "$(Get-Location)\RAVDESS"
```

The resulting directory should contain 24 `Actor_XX/` subfolders with 60 `.wav` files each (1,440 utterances total, 8 emotions).

## Reproducing Paper Results

Every experiment is fully self-contained. Results are written to `${XHUBERT_SAVE_DIR:-./results}/csv/` and can be compared directly with the reference CSVs in `results/`.

```bash
# Reproduce a single experiment
python -m experiments.exp3_xhubert_finetune          # Tables 6, 7 (main result)
python -m experiments.exp4_ablation                  # Table 9

# Or run everything sequentially
python run_all.py
```

Each experiment maps to one or more paper tables/figures:

| Script | Paper artifact | Reference CSV |
|--------|----------------|---------------|
| `exp1_feature_comparison.py` | Table 4 | `results/results_exp1_feature_comparison.csv` |
| `exp2_hubert_frozen.py` | Table 5 | `results/results_exp2_hubert_frozen.csv` |
| `exp3_xhubert_finetune.py` | Tables 6, 7; Fig. 5 | `results/results_exp3_xhubert_full.csv` |
| `exp4_ablation.py` | Table 9 | `results/results_exp4_ablation_full.csv` |
| `exp5_fusion.py` | Tables 10, 11; Fig. 8 | `results/results_exp5_fusion.csv` |
| `exp6_speaker_adversarial.py` | Tables 13, 14 | `results/results_exp6_adversarial.csv` |
| `exp7_layer_probing.py` | Figs. 6, 7 | `results/results_exp7_layer_probing.csv` |
| `exp8_sota_comparison.py` | Table 15 | `results/results_exp8_sota.csv` |

For a lower-friction, cell-by-cell reproduction, use the Colab notebooks under `notebooks/` (NB1–NB7).

### Important reproducibility note

Before instantiating any HuBERT-based model, the code applies a **`pos_conv_embed` weight-norm fix** that is required for PyTorch ≥ 2.1 (see `utils.py`). Without this fix the positional convolution is silently re-initialised, degrading downstream accuracy by roughly 8 percentage points. All experiments in this repository use the fix.

## Pretrained Checkpoints

Trained model weights for every fold / seed / configuration reported in the paper (**90 checkpoints, ~32 GB total**) are hosted on Google Drive:

**https://drive.google.com/drive/folders/1EWpNoZTVCajftoJfp4xCyy6B7SDPmzQe**

Checkpoints are not required to reproduce results — running the scripts above regenerates them — but are provided for direct evaluation, ablation extension, and downstream reuse.

## Results

The `results/` folder contains the exact CSV files backing every table and figure in the paper:

- `results_exp1_feature_comparison.csv` — 7 handcrafted feature configurations × 3 classifiers
- `results_exp2_hubert_frozen.csv` — Frozen HuBERT + SVM / RF (5-fold CV + LOSGO)
- `results_exp3_xhubert_full.csv` — xHuBERT fine-tuned (5-fold CV + LOSGO, 3 seeds)
- `results_exp4_ablation_full.csv` — Component ablation (4 configs × 3 seeds × 6 folds)
- `results_exp5_fusion.csv` — Fusion architectures and gate weights
- `results_exp6_adversarial.csv` — Speaker-adversarial 2×2 grid
- `results_exp7_layer_probing.csv` — Per-layer frozen probing (13 layers × 6 folds)
- `results_exp8_sota.csv` — Comparison with recent literature

## Citation

If you use this code, please cite the paper:

```bibtex
@article{nguyen2026xhubert,
  title   = {{xHuBERT}: A Selective Layer Aggregation Framework for Robust
             Speaker-Independent Speech Emotion Recognition},
  author  = {Nguyen, Tan Nhu and Bui, Thanh Hung},
  journal = {Under review},
  year    = {2026}
}
```

The software artifact itself is archived on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23044713.svg)](https://doi.org/10.5281/zenodo.23044713)

> Nguyen, T. N., & Bui, T. H. (2026). *xHuBERT: A Selective Layer Aggregation Framework for Robust Speaker-Independent Speech Emotion Recognition*. Zenodo. https://doi.org/10.5281/zenodo.23044713

The DOI above is the **concept DOI**, which always resolves to the latest release. To cite the exact version used to reproduce a specific result, use the version-specific DOI (v1.0.0: [10.5281/zenodo.23044714](https://doi.org/10.5281/zenodo.23044714)).

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgments

- The RAVDESS corpus is provided by Livingstone and Russo (2018) under CC BY-NC-SA 4.0.
- HuBERT weights are provided by Meta AI Research via the [Hugging Face model hub](https://huggingface.co/facebook/hubert-base-ls960).
- We thank the reviewers of the initial submission for constructive feedback that shaped the final framework.
