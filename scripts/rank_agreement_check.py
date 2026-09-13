"""How much does TIE's CVaR term actually change the scheduling order?

Answers that from the labels alone, before any GPU time is spent. TIE ranks
waiting requests by ``E[X] + beta * CVaR_0.9[X]``; if that ordering is
indistinguishable from ranking by ``E[X]`` on a given workload, then the
paper's contribution -- uncertainty-awareness -- cannot show up in a
benchmark on that workload no matter how good the sigma predictor is, and
any win over FCFS is attributable to shortest-job-first ordering instead.

Run against the oracle labels (phase 1's fits to 20 real Qwen3-8B samples),
so the answer is about the workload rather than about our predictor:

    python scripts/rank_agreement_check.py

Two views, because they answer different questions:

  * Spearman/Kendall over all prompts -- does the global ordering differ?
  * top-k overlap -- does the *head* of the queue differ? Scheduling only
    ever looks at the head, and a high global correlation can coexist with a
    reshuffled head, so the global number alone would be misleading.

See docs/Phase3_Scheduling_Evaluation_Plan.md section 5.1.1.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from vllm_tie import score_calculator as sc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="data/benchmark_oracle_labels.csv")
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5],
        help="beta values to check; TIE clips adaptive beta to [0.1, 0.5]",
    )
    parser.add_argument("--top-k", type=int, nargs="+", default=[8, 32, 64, 128])
    args = parser.parse_args()

    df = pd.read_csv(args.labels)
    mu = df["logt_mu"].to_numpy()
    sigma = df["logt_sigma"].to_numpy()

    means = np.array([sc.mean(m, s) for m, s in zip(mu, sigma)])
    cvars = np.array([sc.cvar(m, s, 0.9) for m, s in zip(mu, sigma)])
    ratio = cvars / means

    print(f"n = {len(df):,}   ({args.labels})")
    print(
        f"CVaR/E[X]:  p10={np.percentile(ratio, 10):.2f}  "
        f"median={np.median(ratio):.2f}  p90={np.percentile(ratio, 90):.2f}  "
        f"max={ratio.max():.2f}"
    )
    print(
        "sigma:      "
        + "  ".join(f"p{p}={np.percentile(sigma, p):.3f}" for p in (10, 50, 90, 99))
    )
    high = ratio > 2.0
    print(f"CVaR/E[X] > 2.0:  {high.sum():,} / {len(df):,} = {high.mean():.1%}")

    # Lower score is scheduled first, so the head of the queue is the front
    # of the argsort.
    base_order = np.argsort(means)

    print()
    header = f"{'beta':>5} | {'Spearman':>9} | {'Kendall':>8} | " + " | ".join(
        f"top{k:<4}" for k in args.top_k
    )
    print(header)
    print("-" * len(header))
    for beta in args.betas:
        order = np.argsort(means + beta * cvars)
        rho = spearmanr(means + beta * cvars, means).correlation
        tau = kendalltau(means + beta * cvars, means).correlation
        overlaps = [
            f"{len(set(order[:k]) & set(base_order[:k])) / k:6.1%}" for k in args.top_k
        ]
        print(
            f"{beta:>5.1f} | {rho:>9.5f} | {tau:>8.5f} | "
            + " | ".join(f"{o:>7}" for o in overlaps)
        )

    print()
    print(
        "A Spearman near 1.0 with high top-k overlap means the CVaR term is "
        "near-inert on this\nworkload: arm 3 (TIE) and arm 5 (predicted-SJF) "
        "would be measuring almost the same policy."
    )


if __name__ == "__main__":
    main()
