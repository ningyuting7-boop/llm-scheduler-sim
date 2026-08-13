"""Exp1: overall performance of all three schedulers, on the real LMSYS
workload, across three load levels. See docs/Week2_3_Plan.md section 5.2.

Feeds Figure 2 (avg response vs RPS), Figure 3 (p95 response vs RPS), and
Figure 6 (throughput vs RPS).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_experiment import run_once

SCHEDULERS = ["fcfs", "oracle_sjf", "predicted_sjf"]
LOAD_LEVELS = [2.0, 5.0, 10.0]  # requests/sec (Poisson lambda)

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp1_overall_performance.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp1: overall scheduler performance across load levels.")
    parser.add_argument("--num-requests", type=int, default=2000)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--decode-time-per-step", type=float, default=0.05)
    parser.add_argument("--k", type=float, default=1.0, help="predictor quality tier, see src/predictor.py QUALITY_TIERS (1.0=realistic)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = []
    for scheduler in SCHEDULERS:
        for rps in LOAD_LEVELS:
            _, summary = run_once(
                scheduler_name=scheduler,
                workload_source="lmsys",
                num_requests=args.num_requests,
                arrival_rate=rps,
                k=args.k,
                max_batch_size=args.max_batch_size,
                decode_time_per_step=args.decode_time_per_step,
                seed=args.seed,
            )
            rows.append(
                {
                    "scheduler": scheduler,
                    "rps": rps,
                    "avg_wait": summary.avg_waiting_time,
                    "avg_response": summary.avg_response_time,
                    "p50_response": summary.p50_response_time,
                    "p95_response": summary.p95_response_time,
                    "p99_response": summary.p99_response_time,
                    "throughput": summary.throughput,
                    "fairness_jain": summary.fairness_jain_index,
                }
            )
            print(f"{scheduler:>15} rps={rps:>5} avg_response={summary.avg_response_time:.3f} p95={summary.p95_response_time:.3f}")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
