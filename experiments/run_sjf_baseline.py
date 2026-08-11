"""Run the same synthetic workload as the FCFS baseline using Oracle SJF."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import compute_summary, print_summary, write_csv
from src.schedulers.sjf import SJFScheduler
from src.simulator import Simulator
from src.workload import generate_workload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Oracle SJF baseline experiment."
    )

    parser.add_argument("--num-requests", type=int, default=200)

    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=2.0,
        help="requests per unit time (Poisson lambda)",
    )

    parser.add_argument(
        "--mean-output-len",
        type=float,
        default=50.0,
    )

    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--decode-time-per-step",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--csv-out",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "sjf_baseline.csv",
        ),
    )

    args = parser.parse_args()

    # Same workload generator and same seed as FCFS.
    requests = generate_workload(
        num_requests=args.num_requests,
        arrival_rate=args.arrival_rate,
        mean_output_len=args.mean_output_len,
        seed=args.seed,
    )

    scheduler = SJFScheduler(
        max_batch_size=args.max_batch_size
    )

    sim = Simulator(
        requests,
        scheduler=scheduler,
        decode_time_per_step=args.decode_time_per_step,
    )

    sim.run()

    summary = compute_summary(requests)

    print_summary(
        summary,
        label="Oracle SJF",
    )

    write_csv(
        requests,
        args.csv_out,
    )

    print(
        f"Per-request details written to {args.csv_out}"
    )


if __name__ == "__main__":
    main()