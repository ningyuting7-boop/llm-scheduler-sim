"""Evaluate a trained predictor checkpoint on the held-out TEST split.

Reconstructs the test rows from test_prompts.json (written by
train_predictor.py's grouped_split) + the original labels CSV, runs the
model, and reports MAE/RMSE/R^2 for both mu and sigma (same metric style as
ELIS, so directly comparable to the numbers in docs/Phase2_Predictor_Training_Plan.md
and the earlier simulation-phase report).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from transformers import AutoTokenizer

from src.predictor_model import DEBERTA_MODEL_NAME, LengthDistributionPredictor, NormalizeStats
from scripts.train_predictor import DEFAULT_LABELS_CSV, load_labels, make_collate_fn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a predictor checkpoint on the held-out test split.")
    parser.add_argument("--checkpoint-dir", type=str, default=os.path.join(ROOT, "checkpoints", "predictor_full"))
    parser.add_argument("--labels-csv", type=str, default=DEFAULT_LABELS_CSV)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(os.path.join(args.checkpoint_dir, "test_prompts.json")) as f:
        test_prompts = set(json.load(f))
    stats = NormalizeStats.load(os.path.join(args.checkpoint_dir, "normalize_stats.json"))

    all_rows = load_labels(args.labels_csv)
    test_rows = [r for r in all_rows if r["prompt"] in test_prompts]
    print(f"test set: {len(test_rows)} rows, {len(test_prompts)} unique prompts")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_MODEL_NAME)
    model = LengthDistributionPredictor.from_pretrained(DEBERTA_MODEL_NAME)
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "best_model.pt"), map_location="cpu"))
    model.to(device)
    model.eval()

    collate_fn = make_collate_fn(tokenizer)

    true_mus, true_sigmas, pred_mus, pred_sigmas = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(test_rows), args.batch_size):
            chunk = test_rows[start : start + args.batch_size]
            batch = collate_fn(
                [
                    {
                        "prompt": r["prompt"],
                        "mu_z": torch.tensor(0.0),  # unused placeholder, real labels read below
                        "sigma_z": torch.tensor(0.0),
                    }
                    for r in chunk
                ]
            )
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mu_hat_z, sigma_hat_z = model(input_ids, attention_mask)

            for r, mu_z, sigma_z in zip(chunk, mu_hat_z.cpu().numpy(), sigma_hat_z.cpu().numpy()):
                true_mus.append(float(r["logt_mu"]))
                true_sigmas.append(float(r["logt_sigma"]))
                pred_mus.append(stats.denormalize_mu(mu_z))
                pred_sigmas.append(float(stats.denormalize_sigma(sigma_z)))

            print(f"  {min(start + args.batch_size, len(test_rows))}/{len(test_rows)}", end="\r")

    print()
    mu_metrics = _metrics(np.array(true_mus), np.array(pred_mus))
    sigma_metrics = _metrics(np.array(true_sigmas), np.array(pred_sigmas))

    print(f"mu:    MAE={mu_metrics['mae']:.4f}  RMSE={mu_metrics['rmse']:.4f}  R2={mu_metrics['r2']:.4f}")
    print(f"sigma: MAE={sigma_metrics['mae']:.4f}  RMSE={sigma_metrics['rmse']:.4f}  R2={sigma_metrics['r2']:.4f}")

    with open(os.path.join(args.checkpoint_dir, "test_metrics.json"), "w") as f:
        json.dump({"mu": mu_metrics, "sigma": sigma_metrics, "n_test_rows": len(test_rows)}, f, indent=2)
    print(f"Wrote {os.path.join(args.checkpoint_dir, 'test_metrics.json')}")


if __name__ == "__main__":
    main()
