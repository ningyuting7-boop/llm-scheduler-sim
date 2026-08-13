"""Exp3: does Predicted SJF stay robust as the system gets more congested?
See docs/Week2_3_Plan.md section 5.4.

Fixed prediction error, arrival rate swept from light to heavy load.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_experiment import run_once

SCHEDULERS = ["fcfs", "oracle_sjf", "predicted_sjf"]
LOAD_LEVELS = [2.0, 5.0, 10.0, 15.0, 20.0]  # requests/sec

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp3_congestion.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp3: performance under increasing congestion.")
    parser.add_argument("--num-requests", type=int, default=3000)
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
                    "avg_response": summary.avg_response_time,
                    "p95_response": summary.p95_response_time,
                    "avg_wait": summary.avg_waiting_time,
                    "throughput": summary.throughput,
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
