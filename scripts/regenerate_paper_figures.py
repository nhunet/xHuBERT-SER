"""Regenerate paper figures 6 and 9 from saved artefacts.

Rebuilds two figures whose original versions still had Vietnamese
axis / title labels, using only artefacts that Exp3 already produced:

  * Figure 6 (SLA layer weights) — from ``layer_weights_fold{i}_seed{s}.npy``
    files written by Exp3, averaged across LOSGO folds.
  * Figure 9 (HuBERT fine-tuned confusion matrix) — by loading the LOSGO
    xHuBERT checkpoints (``exp3_LOSGO_fold{i}_seed{s}.pt``) and running
    inference on each fold's test split, then aggregating predictions.

Usage
-----
    # Layer-weight figure only (fast, no GPU required)
    python scripts/regenerate_paper_figures.py --only fig6

    # Confusion matrix only (requires GPU + checkpoints)
    python scripts/regenerate_paper_figures.py --only fig9

    # Both, with a specific seed
    python scripts/regenerate_paper_figures.py --seed 42

Outputs are written to ``${XHUBERT_SAVE_DIR}/figures/`` as
``hubert_ft_layer_weights.png`` and ``confusion_hubert_finetune.png`` —
the exact filenames referenced by ``paper/main.tex``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Make the project importable when the script is run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from utils import ensure_dirs, load_checkpoint
from visualization.plots import plot_sla_layer_weights, plot_confusion_matrix


def regenerate_fig6(seed: int) -> bool:
    pattern = os.path.join(config.EMB_DIR, f"layer_weights_fold*_seed{seed}.npy")
    import glob
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[WARN] No layer-weight files found under {pattern}")
        print(f"       Run experiments.exp3_xhubert_finetune first, or copy the "
              f"exp3 embeddings folder into {config.EMB_DIR}.")
        return False

    weights = np.stack([np.load(f) for f in files], axis=0)
    mean_w = weights.mean(axis=0)
    plot_sla_layer_weights(mean_w, filename="hubert_ft_layer_weights.png")
    print(f"[INFO] Averaged {len(files)} folds; peak layer = {int(mean_w.argmax())}")
    return True


def regenerate_fig9(seed: int) -> bool:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from data import RavdessDataset
    from protocols import get_losgo_splits, verify_losgo_splits
    from models.xhubert import xHuBERT
    from utils import compute_metrics

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    print("[INFO] Loading RAVDESS …")
    dataset = RavdessDataset(sr=config.SR_HUBERT)
    splits = get_losgo_splits(dataset.actors)
    verify_losgo_splits(splits, dataset.actors)

    all_true, all_pred = [], []
    for split in splits:
        fold_idx = split["fold_idx"]
        state = load_checkpoint("exp3_LOSGO", fold_idx, seed)
        if state is None:
            print(f"[WARN] Missing checkpoint exp3_LOSGO_fold{fold_idx}_seed{seed}.pt "
                  f"under {config.CKPT_DIR}; skipping fold")
            continue

        model = xHuBERT(use_sla=True, use_attention_pool=True).to(device)
        model.load_state_dict(state)
        model.eval()

        idx = split["test_idx"]
        X = torch.tensor(np.array([dataset.waveforms[i] for i in idx]),
                         dtype=torch.float32)
        y = torch.tensor(dataset.labels[idx], dtype=torch.long)
        loader = DataLoader(TensorDataset(X, y),
                            batch_size=config.BATCH_SIZE, shuffle=False)

        with torch.no_grad():
            for xb, yb in loader:
                logits = model(xb.to(device))
                all_pred.extend(logits.argmax(-1).cpu().numpy())
                all_true.extend(yb.numpy())

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[INFO] Fold {fold_idx} done ({len(idx)} samples)")

    if not all_true:
        print("[ERROR] No predictions collected — no checkpoints found.")
        return False

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    metrics = compute_metrics(y_true, y_pred)
    plot_confusion_matrix(
        y_true, y_pred,
        title="Confusion matrix — xHuBERT fine-tuned (LOSGO, 6-fold aggregate)",
        filename="confusion_hubert_finetune.png",
        normalize=True,
        acc=metrics["accuracy"], f1=metrics["f1_macro"],
    )
    print(f"[INFO] Aggregated {len(y_true)} predictions. "
          f"Acc = {metrics['accuracy']:.2f}%, F1 = {metrics['f1_macro']:.2f}%")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["fig6", "fig9", "both"],
                        default="both", help="Which figure to regenerate")
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED,
                        help=f"Seed to use (default {config.DEFAULT_SEED})")
    args = parser.parse_args()

    ensure_dirs()

    ok = True
    if args.only in {"fig6", "both"}:
        print("\n=== Figure 6: SLA layer weights ===")
        ok &= regenerate_fig6(args.seed)
    if args.only in {"fig9", "both"}:
        print("\n=== Figure 9: Confusion matrix (LOSGO) ===")
        ok &= regenerate_fig9(args.seed)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
