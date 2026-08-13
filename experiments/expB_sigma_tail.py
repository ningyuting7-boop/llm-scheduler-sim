"""Experiment B: fixing tail_rate, how sensitive is TIEScheduler's advantage
to sigma_tail (how wide the tail requests' fitted log-normal is)? See
docs/Week2_3_Plan.md section 11.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import compute_summary
from src.schedulers.predicted_sjf import PredictedSJFScheduler
from src.schedulers.tie import TIEScheduler
from src.simulator import Simulator
from src.workload import generate_contaminated_workload, load_length_pool

SIGMA_TAIL_VALUES = [0.2, 0.4, 0.6, 0.8, 1.0]
SEEDS = [42, 43, 44, 45, 46]

DEFAULT_LMSYS_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "lmsys_output_lengths.csv")
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expB_sigma_tail.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp B: sensitivity of TIEScheduler's advantage to sigma_tail.")
    parser.add_argument("--num-requests", type=int, default=2000)
    parser.add_argument("--arrival-rate", type=float, default=5.0)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--decode-time-per-step", type=float, default=0.05)
    parser.add_argument("--tail-rate", type=float, default=0.03)
    parser.add_argument("--sigma-normal", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--lmsys-csv", type=str, default=DEFAULT_LMSYS_CSV)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    args = parser.parse_args()

    pool = load_length_pool(args.lmsys_csv)

    def run(scheduler_factory, sigma_tail: float, seed: int) -> float:
        requests = generate_contaminated_workload(
            num_requests=args.num_requests,
            arrival_rate=args.arrival_rate,
            length_pool=pool,
            tail_rate=args.tail_rate,
            sigma_normal=args.sigma_normal,
            sigma_tail=sigma_tail,
            seed=seed,
        )
        Simulator(requests, scheduler=scheduler_factory(), decode_time_per_step=args.decode_time_per_step).run()
        return compute_summary(requests).avg_response_time

    rows = []
    for sigma_tail in SIGMA_TAIL_VALUES:
        psjf_vals = [run(lambda: PredictedSJFScheduler(args.max_batch_size), sigma_tail, s) for s in SEEDS]
        tie_vals = [run(lambda: TIEScheduler(args.max_batch_size, beta=args.beta), sigma_tail, s) for s in SEEDS]
        psjf_mean = sum(psjf_vals) / len(psjf_vals)
        tie_mean = sum(tie_vals) / len(tie_vals)
        rows.append({"sigma_tail": sigma_tail, "predicted_sjf": psjf_mean, "tie": tie_mean})
        print(f"sigma_tail={sigma_tail:.2f}  predicted_sjf={psjf_mean:>8.2f}  tie={tie_mean:>8.2f}  gap={psjf_mean - tie_mean:>7.2f}")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
