"""Does the log-t vs log-normal KS pass-rate gap shrink as sample size
shrinks? If so, that's direct evidence (on our own data, no new GPU work
needed) that our 20-samples/prompt gap being much smaller than the
paper's 100-samples/prompt gap is mostly a statistical-power artifact,
not a sign that log-t fits Qwen3-8B's outputs worse than it fits
Llama-3-8B-Instruct's.

For each candidate n <= 20, randomly subsample each prompt's 20 real
samples down to n (without replacement), refit + KS-test both
distributions, and report the pass-rate gap at that n.
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

from scripts.validate_logt_fit import _logt_ks, _lognormal_ks
from src.logt_fit import DEFAULT_NU, fit_logt

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SAMPLES_PATH = os.path.join(_REPO_ROOT, "data", "qwen3_8b_length_samples.csv")


def load_sample_arrays(samples_path: str, min_samples: int = 20):
    arrays = []
    with open(samples_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lengths = json.loads(row["sample_lengths"])
            if len(lengths) < min_samples:
                continue
            arr = np.asarray(lengths, dtype=float)
            if float(np.std(np.log(arr))) == 0.0:
                continue
            arrays.append(arr)
    return arrays


def pass_rates_at_n(sample_arrays, n: int, rng: np.random.Generator, alpha: float = 0.05):
    n_logt_pass = n_lognorm_pass = n_valid = 0
    for arr in sample_arrays:
        sub = arr if n >= len(arr) else rng.choice(arr, size=n, replace=False)
        if float(np.std(np.log(sub))) == 0.0:
            continue
        fit = fit_logt(sub, nu=DEFAULT_NU)
        _, logt_p = _logt_ks(sub, fit.mu, fit.sigma, fit.nu)
        _, lognorm_p = _lognormal_ks(sub)
        if not (np.isfinite(logt_p) and np.isfinite(lognorm_p)):
            continue
        n_valid += 1
        n_logt_pass += logt_p > alpha
        n_lognorm_pass += lognorm_p > alpha
    return n_logt_pass / n_valid, n_lognorm_pass / n_valid, n_valid


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether the log-t/log-normal gap grows with sample size.")
    parser.add_argument("--samples-path", type=str, default=DEFAULT_SAMPLES_PATH)
    parser.add_argument("--n-values", type=int, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("Loading samples (prompts with exactly 20 real samples) ...")
    sample_arrays = load_sample_arrays(args.samples_path)
    print(f"{len(sample_arrays)} prompts loaded.\n")

    rng = np.random.default_rng(args.seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n in args.n_values:
            logt_rate, lognorm_rate, n_valid = pass_rates_at_n(sample_arrays, n, rng)
            gap = (logt_rate - lognorm_rate) * 100
            print(f"n={n:2d}  log-t={logt_rate:.1%}  log-normal={lognorm_rate:.1%}  "
                  f"gap={gap:+.1f}pp  (n_valid={n_valid})")

    print("\nPaper's own numbers for reference: n=100 -> log-t 93.1%, log-normal 60.3%, gap=+32.8pp")


if __name__ == "__main__":
    main()
