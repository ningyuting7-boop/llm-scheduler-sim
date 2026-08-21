"""Per-prompt KS goodness-of-fit pass rate for log-t vs log-normal,
computed directly from already-generated Phase 1 data (no GPU needed).

Mirrors the TIE paper's own validation methodology (Section 3.1: fit each
prompt's repeated-sampling length distribution, run a KS test, report the
fraction of prompts with p > 0.05) -- but run over every prompt in
data/qwen3_8b_length_samples.csv (the full labeling-run population, not a
hand-picked subset) rather than a fresh 1000-prompt/100-sample experiment.

Caveat: the paper's own pass rates (93.1% log-t / 60.3% log-normal) used
100 samples/prompt; we only have `--samples-per-prompt` (20 in Phase 1),
so our KS test has less statistical power to detect a bad fit -- expect
higher pass rates than the paper's numbers for that reason alone, not
necessarily because our fit is "better."
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.validate_logt_fit import _logt_ks, _lognormal_ks
from src.logt_fit import fit_logt

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SAMPLES_PATH = os.path.join(_REPO_ROOT, "data", "qwen3_8b_length_samples.csv")


def compute_pass_rates(samples_path: str, alpha: float = 0.05, min_samples: int = 5):
    n_logt_pass = 0
    n_lognorm_pass = 0
    n_logt_valid = 0
    n_lognorm_valid = 0
    n_tested = 0
    n_skipped = 0
    n_degenerate = 0  # zero variance in log-space (e.g. all samples identical)

    with open(samples_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lengths = json.loads(row["sample_lengths"])
            if len(lengths) < min_samples:
                n_skipped += 1
                continue

            arr = np.asarray(lengths, dtype=float)
            if float(np.std(np.log(arr))) == 0.0:
                n_degenerate += 1
                continue

            fit = fit_logt(arr)
            logt_stat, logt_p = _logt_ks(arr, fit.mu, fit.sigma, fit.nu)
            lognorm_stat, lognorm_p = _lognormal_ks(arr)

            n_tested += 1
            if np.isfinite(logt_p):
                n_logt_valid += 1
                if logt_p > alpha:
                    n_logt_pass += 1
            if np.isfinite(lognorm_p):
                n_lognorm_valid += 1
                if lognorm_p > alpha:
                    n_lognorm_pass += 1

    return {
        "n_tested": n_tested,
        "n_skipped": n_skipped,
        "n_degenerate": n_degenerate,
        "n_logt_valid": n_logt_valid,
        "n_lognorm_valid": n_lognorm_valid,
        "logt_pass_rate": n_logt_pass / n_logt_valid if n_logt_valid else float("nan"),
        "lognormal_pass_rate": n_lognorm_pass / n_lognorm_valid if n_lognorm_valid else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-prompt KS pass-rate for log-t vs log-normal.")
    parser.add_argument("--samples-path", type=str, default=DEFAULT_SAMPLES_PATH)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    result = compute_pass_rates(args.samples_path, alpha=args.alpha)

    print(f"Tested {result['n_tested']} prompts (skipped {result['n_skipped']} with too few samples, "
          f"{result['n_degenerate']} degenerate/zero-variance excluded)")
    print(f"log-t      KS pass rate (p>{args.alpha}): {result['logt_pass_rate']:.1%} "
          f"of {result['n_logt_valid']} valid fits  (paper: 93.1%, n=100/prompt)")
    print(f"log-normal KS pass rate (p>{args.alpha}): {result['lognormal_pass_rate']:.1%} "
          f"of {result['n_lognorm_valid']} valid fits  (paper: 60.3%, n=100/prompt)")


if __name__ == "__main__":
    main()
