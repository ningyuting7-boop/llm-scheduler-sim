"""Validation experiment (not part of the main labeling pipeline): for a
small set of hand-picked high-internal-variance prompts (identified from
the Phase 1 pilot run -- see docs/VLLM_Qwen3_Reproduction_Plan.md), sample
many more completions per prompt and check:

1. Reproducibility: split the samples into two independent halves, fit
   log-t to each half separately, and compare (mu, sigma) -- if the two
   halves roughly agree, the high-variance pattern seen in the n=20 pilot
   is a real property of the prompt, not noise from too few samples.
2. Distributional shape: does log-t(nu=3.5) actually fit better than a
   log-normal, via a KS goodness-of-fit test on each candidate
   distribution -- mirroring the comparison the TIE paper itself reports
   (log-t passes KS on 93.1% of real distributions vs 60.3% for
   log-normal, docs/Report.md section 2).

Caveat (documented simplification): the KS test below fits each
distribution's parameters and tests goodness-of-fit on the *same* data,
which technically biases the test statistic optimistically (no Lilliefors
correction applied). Both log-t and log-normal are scored the same way,
so it's still valid as a *relative* comparison ("which fits better"),
just don't over-interpret the absolute p-values.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_predictor_labels import generate_completion_lengths
from src.logt_fit import fit_logt

# Hand-picked from the Phase 1 pilot's high-internal-variance prompts
# (max sample >= 3x that prompt's own median); chosen for being
# interpretable in English/Chinese, avoiding garbled/adversarial-string
# prompts that are harder to reason about.
DEFAULT_PROMPTS = [
    "Write a single word",
    "今天天气：",
    "Wrong answers only: what is a potato?",
]

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(_REPO_ROOT, "data", "logt_validation.csv")


def _lognormal_ks(samples: np.ndarray):
    log_samples = np.log(samples)
    mu_ln, sigma_ln = float(np.mean(log_samples)), float(np.std(log_samples))
    result = stats.kstest(samples, lambda x: stats.norm.cdf((np.log(x) - mu_ln) / sigma_ln))
    return result.statistic, result.pvalue


def _logt_ks(samples: np.ndarray, mu: float, sigma: float, nu: float):
    result = stats.kstest(samples, lambda x: stats.t.cdf((np.log(x) - mu) / sigma, df=nu))
    return result.statistic, result.pvalue


def validate_prompt(prompt: str, lengths: list) -> dict:
    lengths_arr = np.asarray(lengths, dtype=float)
    half = len(lengths_arr) // 2
    batch_a, batch_b = lengths_arr[:half], lengths_arr[half:]

    fit_a = fit_logt(batch_a)
    fit_b = fit_logt(batch_b)
    fit_full = fit_logt(lengths_arr)

    logt_ks_stat, logt_ks_p = _logt_ks(lengths_arr, fit_full.mu, fit_full.sigma, fit_full.nu)
    lognorm_ks_stat, lognorm_ks_p = _lognormal_ks(lengths_arr)

    return {
        "prompt": prompt,
        "n_samples": len(lengths_arr),
        "mu_batch_a": fit_a.mu,
        "sigma_batch_a": fit_a.sigma,
        "mu_batch_b": fit_b.mu,
        "sigma_batch_b": fit_b.sigma,
        "mu_full": fit_full.mu,
        "sigma_full": fit_full.sigma,
        "logt_ks_stat": logt_ks_stat,
        "logt_ks_pvalue": logt_ks_p,
        "lognormal_ks_stat": lognorm_ks_stat,
        "lognormal_ks_pvalue": lognorm_ks_p,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate log-t fit reproducibility/shape for high-variance prompts.")
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=300,
        help="total completions per prompt (split in half for the reproducibility check)",
    )
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    args = parser.parse_args()

    print(f"Generating {args.num_samples} completions each for {len(args.prompts)} prompt(s) ...")
    lengths_per_prompt = generate_completion_lengths(args.prompts, samples_per_prompt=args.num_samples)

    results = [validate_prompt(prompt, lengths) for prompt, lengths in zip(args.prompts, lengths_per_prompt)]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    for r in results:
        print(f"\nprompt: {r['prompt']!r}  (n={r['n_samples']})")
        print(f"  batch A: mu={r['mu_batch_a']:.3f} sigma={r['sigma_batch_a']:.3f}")
        print(f"  batch B: mu={r['mu_batch_b']:.3f} sigma={r['sigma_batch_b']:.3f}")
        print(f"  log-t    KS stat={r['logt_ks_stat']:.4f} p={r['logt_ks_pvalue']:.4f}")
        print(f"  lognorm  KS stat={r['lognormal_ks_stat']:.4f} p={r['lognormal_ks_pvalue']:.4f}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
