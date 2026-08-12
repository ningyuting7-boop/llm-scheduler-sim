"""Exp4: does the aging term in ARRS actually prevent starvation? See
docs/Week2_3_Plan.md section 5.5.

Bimodal workload (many short requests + a few long ones drawn from the real
LMSYS tail) with a deliberately small max_batch_size, so requests actually
queue up -- without contention there's nothing for the aging term to do.
Compares Predicted SJF (expected to let long requests wait indefinitely)
against ARRS with a fixed alpha (expected to bound their wait, at a real
cost to short requests -- the "convoy effect"). A pressure-adaptive alpha
was tried and dropped: this workload is deliberately always congested, so
that mechanism collapses to a relabeled fixed alpha here (see
docs/Week2_3_Plan.md section 9.7).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_experiment import run_once
from src.metrics import percentile, write_csv

SCHEDULERS = ["predicted_sjf", "arrs"]
LONG_REQUEST_THRESHOLD_PERCENTILE = 95.0  # matches generate_bimodal_workload's long_percentile default

DEFAULT_SUMMARY_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp4_starvation_summary.csv")
DEFAULT_DETAIL_OUT_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp4_starvation_{scheduler}.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp4: starvation prevention, Predicted SJF vs ARRS.")
    parser.add_argument("--num-requests", type=int, default=3000)
    parser.add_argument("--arrival-rate", type=float, default=4.0)
    parser.add_argument("--long-request-ratio", type=float, default=0.05)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--decode-time-per-step", type=float, default=0.05)
    parser.add_argument("--k", type=float, default=1.0, help="predictor quality tier, see src/predictor.py QUALITY_TIERS (1.0=realistic)")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-out", type=str, default=DEFAULT_SUMMARY_OUT)
    args = parser.parse_args()

    summary_rows = []
    for scheduler in SCHEDULERS:
        requests, summary = run_once(
            scheduler_name=scheduler,
            workload_source="bimodal",
            num_requests=args.num_requests,
            arrival_rate=args.arrival_rate,
            long_request_ratio=args.long_request_ratio,
            k=args.k,
            alpha=args.alpha,
            beta=args.beta,
            max_batch_size=args.max_batch_size,
            decode_time_per_step=args.decode_time_per_step,
            seed=args.seed,
        )
        waiting_times = [r.waiting_time for r in requests]
        p95_wait = percentile(waiting_times, 95)
        summary_rows.append(
            {
                "scheduler": scheduler,
                "max_waiting_time": summary.max_waiting_time,
                "p95_waiting_time": p95_wait,
                "avg_waiting_time": summary.avg_waiting_time,
            }
        )
        print(f"{scheduler:>15} max_wait={summary.max_waiting_time:.3f} p95_wait={p95_wait:.3f} avg_wait={summary.avg_waiting_time:.3f}")

        detail_path = DEFAULT_DETAIL_OUT_TEMPLATE.format(scheduler=scheduler)
        write_csv(requests, detail_path)
        print(f"Per-request details written to {detail_path}")

    with open(args.summary_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {args.summary_out}")


if __name__ == "__main__":
    main()
