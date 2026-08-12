"""Exp2 (the core experiment): does ARRS degrade more gracefully than
Predicted SJF as predictor quality (k) worsens? See docs/Week2_3_Plan.md
section 10 for how k/REFERENCE_SCALE/QUALITY_TIERS are sourced (ELIS,
arXiv:2505.09142).

Same workload (same seed) reused across every k, so k is the only thing that
changes between runs of a given scheduler. FCFS/Oracle SJF don't depend on k
at all, so they're run once each and replicated across every k as flat
reference lines.

Feeds Figure 4 (avg/p95 response vs k) -- the headline figure.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_experiment import run_once
from src.predictor import QUALITY_TIERS

K_LEVELS = [0.0, QUALITY_TIERS["low"], QUALITY_TIERS["realistic"], QUALITY_TIERS["high"], 5.00, 10.00, 20.00, 50.00, 100.00, 200.00]
PREDICTION_AWARE_SCHEDULERS = ["predicted_sjf", "arrs"]
REFERENCE_SCHEDULERS = ["fcfs", "oracle_sjf"]

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp2_prediction_robustness.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp2: robustness to prediction error, Predicted SJF vs ARRS.")
    parser.add_argument("--num-requests", type=int, default=2000)
    parser.add_argument("--arrival-rate", type=float, default=5.0)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--decode-time-per-step", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    args = parser.parse_args()

    common = dict(
        workload_source="lmsys",
        num_requests=args.num_requests,
        arrival_rate=args.arrival_rate,
        max_batch_size=args.max_batch_size,
        decode_time_per_step=args.decode_time_per_step,
        seed=args.seed,
    )

    rows = []

    for scheduler in REFERENCE_SCHEDULERS:
        _, summary = run_once(scheduler_name=scheduler, **common)
        for k in K_LEVELS:
            rows.append(
                {
                    "scheduler": scheduler,
                    "k": k,
                    "avg_response": summary.avg_response_time,
                    "p95_response": summary.p95_response_time,
                }
            )
        print(f"{scheduler:>15} (reference) avg_response={summary.avg_response_time:.3f} p95={summary.p95_response_time:.3f}")

    for scheduler in PREDICTION_AWARE_SCHEDULERS:
        for k in K_LEVELS:
            _, summary = run_once(
                scheduler_name=scheduler,
                k=k,
                alpha=args.alpha,
                beta=args.beta,
                **common,
            )
            rows.append(
                {
                    "scheduler": scheduler,
                    "k": k,
                    "avg_response": summary.avg_response_time,
                    "p95_response": summary.p95_response_time,
                }
            )
            print(f"{scheduler:>15} k={k:>7.3f} avg_response={summary.avg_response_time:.3f} p95={summary.p95_response_time:.3f}")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
