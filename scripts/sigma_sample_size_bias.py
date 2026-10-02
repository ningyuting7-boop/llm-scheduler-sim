"""Does fitting sigma from 20 samples per prompt underestimate it?

The reference implementation up-weights prompts with logt_sigma > 1.0 as a
group worth targeting, but that threshold matches 0.36% of these labels
while its mu threshold matches a sensible 51.3%. Their sigma distribution
is therefore wider than ours, and the obvious suspect is sample count:
20 draws from a heavy-tailed distribution rarely include the tail, so the
MLE for sigma should be biased low.

This script tests that directly, without a GPU: refit each prompt's sigma
from random subsets of its 20 samples and see how the estimate moves with
n. If sigma were still climbing at n=20, more samples would be the fix.

It reports two things, because they fail differently. The level says
whether sigma is systematically low; the rank correlation says whether the
ordering across prompts survives, which is what a predictor would have to
learn and what the scheduler would have to act on.

Usage:
    python scripts/sigma_sample_size_bias.py
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import spearmanr

from src.logt_fit import fit_logt

csv.field_size_limit(10**7)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default="data/qwen3_8b_length_samples.csv")
    parser.add_argument("--sizes", type=int, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    full = max(args.sizes)

    with open(args.samples) as f:
        data = [
            np.asarray(ast.literal_eval(r["sample_lengths"]), dtype=float)
            for r in csv.DictReader(f)
        ]
    data = [x for x in data if len(x) == full]
    print(f"prompts with {full} samples: {len(data):,}\n")

    sigma = {n: [] for n in args.sizes}
    for lengths in data:
        for n in args.sizes:
            subset = lengths if n == full else rng.choice(lengths, size=n, replace=False)
            sigma[n].append(fit_logt(subset).sigma)
    sigma = {n: np.asarray(v) for n, v in sigma.items()}

    print(f"{'n':>4}{'mean':>10}{'median':>10}{'p90':>9}{'p99':>9}{'max':>8}{'>1.0':>8}")
    print("-" * 58)
    for n in args.sizes:
        s = sigma[n]
        print(
            f"{n:>4}{s.mean():>10.4f}{np.median(s):>10.4f}{np.percentile(s, 90):>9.4f}"
            f"{np.percentile(s, 99):>9.4f}{s.max():>8.2f}{100 * (s > 1.0).mean():>7.2f}%"
        )

    ref = sigma[full]
    print(f"\nrelative to n={full}:")
    for n in args.sizes[:-1]:
        print(
            f"  n={n:>3}  mean {100 * (sigma[n].mean() - ref.mean()) / ref.mean():>+6.1f}%"
            f"   rank corr {spearmanr(sigma[n], ref).correlation:.4f}"
        )

    # sigma(n) = sigma_inf - c/n, fitted on the two extreme sizes. The
    # asymptote is what more sampling could buy at most.
    lo = min(args.sizes)
    c = (ref.mean() - sigma[lo].mean()) / (1 / lo - 1 / full)
    inf = ref.mean() + c / full
    print(f"\nsigma(n) = sigma_inf - c/n   ->   sigma_inf={inf:.4f}, c={c:.4f}")
    print(f"{'n':>6}{'predicted':>12}{'measured':>11}")
    for n in args.sizes:
        print(f"{n:>6}{inf - c / n:>12.4f}{sigma[n].mean():>11.4f}")
    print(f"{100:>6}{inf - c / 100:>12.4f}{'-':>11}")
    print(
        f"\nn={full} reaches {100 * ref.mean() / inf:.1f}% of the asymptote; "
        f"going to n=100 would add {100 * ((inf - c / 100) - ref.mean()) / ref.mean():+.1f}%"
    )

    # Where the bias does bite: the upper tail, which is the part the CVaR
    # term depends on.
    print("\nby sigma range (how much n=%d underestimates):" % lo)
    cuts = np.percentile(ref, [50, 90, 99])
    bands = [
        (0.0, cuts[0], "lower half"),
        (cuts[1], cuts[2], "p90-p99"),
        (cuts[2], np.inf, "top 1%"),
    ]
    half = args.sizes[len(args.sizes) // 2 - 1] if len(args.sizes) > 2 else lo
    for low, high, name in bands:
        m = (ref >= low) & (ref < high)
        print(
            f"  {name:<12} n={m.sum():>5}   sigma({half}) vs sigma({full}): "
            f"{100 * (sigma[half][m].mean() - ref[m].mean()) / ref[m].mean():>+6.1f}%"
        )
    top = ref >= cuts[1]
    print(
        f"\n  rank corr overall      {spearmanr(sigma[half], ref).correlation:.4f}"
        f"\n  rank corr within top 10% {spearmanr(sigma[half][top], ref[top]).correlation:.4f}"
    )


if __name__ == "__main__":
    main()
