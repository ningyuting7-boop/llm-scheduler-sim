"""Sweep the log-t degrees-of-freedom parameter (nu) and find which value
best fits our own Qwen3-8B data, rather than assuming the paper's nu=3.5
(tuned on Llama-3-8B-Instruct) transfers unchanged.

Mirrors the paper's own tuning methodology (Appendix D.3): for each
candidate nu, fit (mu, sigma | nu fixed) per prompt and score by the
*mean* KS p-value across all prompts (a more continuous, sensitive
signal than pass/fail at p>0.05 -- matches what the paper actually
optimized). Runs entirely locally on already-generated data, no GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.validate_logt_fit import _logt_ks
from src.logt_fit import fit_logt

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SAMPLES_PATH = os.path.join(_REPO_ROOT, "data", "qwen3_8b_length_samples.csv")


def load_sample_arrays(samples_path: str, min_samples: int = 5):
    arrays = []
    with open(samples_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lengths = json.loads(row["sample_lengths"])
            if len(lengths) < min_samples:
                continue
            arr = np.asarray(lengths, dtype=float)
            if float(np.std(np.log(arr))) == 0.0:
                continue  # degenerate, excluded same as ks_pass_rate.py
            arrays.append(arr)
    return arrays


def score_nu(sample_arrays, nu: float, alpha: float = 0.05):
    p_values = []
    for arr in sample_arrays:
        fit = fit_logt(arr, nu=nu)
        _, p = _logt_ks(arr, fit.mu, fit.sigma, nu)
        if np.isfinite(p):
            p_values.append(p)
    p_values = np.asarray(p_values)
    return {
        "nu": nu,
        "n_valid": len(p_values),
        "mean_p": float(np.mean(p_values)) if len(p_values) else float("nan"),
        "pass_rate": float(np.mean(p_values > alpha)) if len(p_values) else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep log-t nu and score fit quality on real Qwen3-8B data.")
    parser.add_argument("--samples-path", type=str, default=DEFAULT_SAMPLES_PATH)
    parser.add_argument("--nu-values", type=float, nargs="+", default=[1, 2, 3, 3.5, 4, 5, 6, 8, 10])
    args = parser.parse_args()

    print("Loading samples ...")
    sample_arrays = load_sample_arrays(args.samples_path)
    print(f"{len(sample_arrays)} valid (non-degenerate) prompts loaded.\n")

    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for nu in args.nu_values:
            r = score_nu(sample_arrays, nu)
            results.append(r)
            print(f"nu={nu:5.1f}  mean_p={r['mean_p']:.4f}  pass_rate={r['pass_rate']:.1%}  (n_valid={r['n_valid']})")

    best = max(results, key=lambda r: r["mean_p"])
    print(f"\nBest nu by mean p-value: {best['nu']} (mean_p={best['mean_p']:.4f}, pass_rate={best['pass_rate']:.1%})")
    print("Paper's fixed nu=3.5 for comparison is included in the sweep above.")


if __name__ == "__main__":
    main()
