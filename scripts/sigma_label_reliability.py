"""How much of the (mu, sigma) label variance is real signal vs. sampling
noise? Split-half reliability: fit log-t independently on two disjoint
halves of each prompt's 20 sampled lengths, then correlate the two halves'
estimates across prompts.

This bounds what ANY predictor can achieve. If a 10-sample sigma estimate
barely predicts another 10-sample estimate of the *same prompt*, then no
model can predict the 20-sample label much better -- the label itself is
mostly noise, and a low test R^2 says more about the labeling budget
(Phase 1's 20 samples/prompt) than about the model or its hyperparameters.

CPU-only, no torch, no GPU. See conversation notes: written after three
training runs (sigma_weight 1.0 / 2.0 / 3.0-4.0) all landed at sigma
R^2 ~= 0.01-0.05 on the test set while mu R^2 stayed 0.58-0.77.
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.logt_fit import _DEGENERATE_SIGMA_FLOOR, fit_logt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SAMPLES_CSV = os.path.join(ROOT, "data", "qwen3_8b_length_samples.csv")


def _reliability(a: np.ndarray, b: np.ndarray) -> dict:
    """Agreement between two independent estimates of the same quantity.

    - r2_corr: squared Pearson correlation. The relevant ceiling for a
      trained model, which is free to learn any linear rescaling of the
      target and so is not penalized for a systematic offset/scale.
    - r2_identity: R^2 treating `a` as a literal prediction of `b` (no
      rescaling allowed). Stricter, and the closer analogue of how
      evaluate_predictor.py scores the model.
    - spearman_brown: split-half correlation corrected to estimate the
      reliability of the FULL 20-sample label (each half here only has 10
      samples, so it is noisier than the real label by construction).
    """
    r = float(np.corrcoef(a, b)[0, 1])
    ss_res = float(np.sum((b - a) ** 2))
    ss_tot = float(np.sum((b - b.mean()) ** 2))
    return {
        "r": r,
        "r2_corr": r**2,
        "r2_identity": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "spearman_brown": 2 * r / (1 + r) if r > -1 else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Split-half reliability of the log-t (mu, sigma) labels.")
    parser.add_argument("--samples-csv", type=str, default=DEFAULT_SAMPLES_CSV)
    parser.add_argument("--num-prompts", type=int, default=2000, help="random subsample; 2000 is ample for a correlation")
    parser.add_argument("--num-splits", type=int, default=5, help="random half-splits per prompt, averaged over")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.samples_csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["sample_lengths"].strip()]

    rng = random.Random(args.seed)
    if args.num_prompts < len(rows):
        rows = rng.sample(rows, args.num_prompts)

    sample_lists = [ast.literal_eval(r["sample_lengths"]) for r in rows]
    fully_degenerate = np.array([len(set(s)) == 1 for s in sample_lists])
    print(f"{len(sample_lists)} prompts; {fully_degenerate.sum()} ({100*fully_degenerate.mean():.1f}%) have all 20 samples identical")

    per_split = []
    for split_idx in range(args.num_splits):
        mu_a, mu_b, sig_a, sig_b = [], [], [], []
        for samples in sample_lists:
            shuffled = list(samples)
            rng.shuffle(shuffled)
            half = len(shuffled) // 2
            fit_a = fit_logt(shuffled[:half])
            fit_b = fit_logt(shuffled[half:])
            mu_a.append(fit_a.mu)
            mu_b.append(fit_b.mu)
            sig_a.append(fit_a.sigma)
            sig_b.append(fit_b.sigma)
        per_split.append(
            {
                "mu": _reliability(np.array(mu_a), np.array(mu_b)),
                "sigma": _reliability(np.array(sig_a), np.array(sig_b)),
                "sigma_nondegenerate": _reliability(
                    np.array(sig_a)[~fully_degenerate], np.array(sig_b)[~fully_degenerate]
                ),
            }
        )
        print(f"  split {split_idx + 1}/{args.num_splits} done", end="\r")

    print()
    for target in ("mu", "sigma", "sigma_nondegenerate"):
        for metric in ("r2_corr", "r2_identity", "spearman_brown"):
            values = np.array([s[target][metric] for s in per_split])
            print(f"{target:>20} {metric:>16}: {values.mean():.4f} +/- {values.std():.4f}")
        print()

    floor_note = (
        f"(fully-degenerate prompts are floored to sigma={_DEGENERATE_SIGMA_FLOOR} in BOTH halves, so they "
        f"agree perfectly by construction -- 'sigma_nondegenerate' excludes them)"
    )
    print(floor_note)


if __name__ == "__main__":
    main()
