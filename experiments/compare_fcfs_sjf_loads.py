"""Compare FCFS, Oracle SJF, and Predicted SJF under the same workloads.

The script:
1. Generates ONE synthetic workload for each arrival rate.
2. Deep-copies that workload so all three schedulers see identical requests.
3. Adds predicted_output_len only for Predicted SJF using a configurable
   multiplicative Gaussian prediction-error model.
4. Runs FCFS, Oracle SJF, and Predicted SJF.
5. Prints all metrics and writes a comparison CSV.

Interpretation of --prediction-error:
    0.0  = perfect prediction
    0.1  = about 10% standard deviation in multiplicative prediction error
    0.3  = about 30% standard deviation
    0.5  = about 50% standard deviation

Example:
    python3 experiments/compare_fcfs_oracle_predicted.py --prediction-error 0.3
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import random
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import compute_summary
from src.schedulers.fcfs import FCFSScheduler
from src.schedulers.sjf import SJFScheduler
from src.schedulers.predicted_sjf import PredictedSJFScheduler
from src.simulator import Simulator
from src.workload import generate_workload


def add_predictions(requests, error_std: float, seed: int) -> None:
    """Populate predicted_output_len for each request.

    Prediction model:
        predicted = true_length * (1 + Gaussian(0, error_std))

    Predictions are clipped to at least 1 token.
    """
    rng = random.Random(seed)

    for request in requests:
        error = rng.gauss(0.0, error_std)
        predicted = request.output_len * (1.0 + error)
        request.predicted_output_len = max(1.0, predicted)


def run_scheduler(requests, scheduler):
    sim = Simulator(
        requests=requests,
        scheduler=scheduler,
        decode_time_per_step=ARGS.decode_time_per_step,
    )
    sim.run()
    return compute_summary(requests)


def percent_change(baseline_value: float, new_value: float) -> float:
    """Percent change relative to baseline. Negative means the metric decreased."""
    if baseline_value == 0:
        return 0.0
    return (new_value - baseline_value) / baseline_value * 100.0


def summary_row(arrival_rate, algorithm, summary, prediction_error=None):
    row = {
        "arrival_rate": arrival_rate,
        "algorithm": algorithm,
        "num_requests": summary.num_requests,
        "avg_waiting_time": summary.avg_waiting_time,
        "avg_response_time": summary.avg_response_time,
        "p99_response_time": summary.p99_response_time,
        "throughput": summary.throughput,
        "fairness_jain_index": summary.fairness_jain_index,
        "waiting_time_stdev": summary.waiting_time_stdev,
        "prediction_error_std": prediction_error if prediction_error is not None else "",
    }
    return row


def print_result(arrival_rate, label, summary):
    print(
        f"[lambda={arrival_rate:g} | {label}] "
        f"requests={summary.num_requests} "
        f"avg_wait={summary.avg_waiting_time:.3f} "
        f"avg_response={summary.avg_response_time:.3f} "
        f"p99_response={summary.p99_response_time:.3f} "
        f"throughput={summary.throughput:.3f} "
        f"fairness(jain)={summary.fairness_jain_index:.3f} "
        f"wait_stdev={summary.waiting_time_stdev:.3f}"
    )


def main():
    rows = []

    for arrival_rate in ARGS.arrival_rates:
        # Generate ONE base workload.
        base_requests = generate_workload(
            num_requests=ARGS.num_requests,
            arrival_rate=arrival_rate,
            mean_output_len=ARGS.mean_output_len,
            seed=ARGS.seed,
        )

        # All algorithms receive the same true workload.
        fcfs_requests = copy.deepcopy(base_requests)
        oracle_requests = copy.deepcopy(base_requests)
        predicted_requests = copy.deepcopy(base_requests)

        # Only Predicted SJF gets noisy predictions.
        # A different but deterministic seed is used for prediction noise.
        prediction_seed = ARGS.seed + int(arrival_rate * 1000) + 12345
        add_predictions(
            predicted_requests,
            error_std=ARGS.prediction_error,
            seed=prediction_seed,
        )

        fcfs_summary = run_scheduler(
            fcfs_requests,
            FCFSScheduler(max_batch_size=ARGS.max_batch_size),
        )

        oracle_summary = run_scheduler(
            oracle_requests,
            SJFScheduler(max_batch_size=ARGS.max_batch_size),
        )

        predicted_summary = run_scheduler(
            predicted_requests,
            PredictedSJFScheduler(max_batch_size=ARGS.max_batch_size),
        )

        print_result(arrival_rate, "FCFS", fcfs_summary)
        print_result(arrival_rate, "Oracle SJF", oracle_summary)
        print_result(arrival_rate, "Predicted SJF", predicted_summary)

        print(
            f"  Oracle SJF vs FCFS: "
            f"avg-wait change={percent_change(fcfs_summary.avg_waiting_time, oracle_summary.avg_waiting_time):+.1f}%, "
            f"avg-response change={percent_change(fcfs_summary.avg_response_time, oracle_summary.avg_response_time):+.1f}%"
        )

        print(
            f"  Predicted SJF vs FCFS: "
            f"avg-wait change={percent_change(fcfs_summary.avg_waiting_time, predicted_summary.avg_waiting_time):+.1f}%, "
            f"avg-response change={percent_change(fcfs_summary.avg_response_time, predicted_summary.avg_response_time):+.1f}%"
        )

        print(
            f"  Predicted SJF vs Oracle SJF: "
            f"avg-wait change={percent_change(oracle_summary.avg_waiting_time, predicted_summary.avg_waiting_time):+.1f}%, "
            f"avg-response change={percent_change(oracle_summary.avg_response_time, predicted_summary.avg_response_time):+.1f}%"
        )
        print()

        rows.append(summary_row(arrival_rate, "FCFS", fcfs_summary))
        rows.append(summary_row(arrival_rate, "Oracle SJF", oracle_summary))
        rows.append(
            summary_row(
                arrival_rate,
                "Predicted SJF",
                predicted_summary,
                prediction_error=ARGS.prediction_error,
            )
        )

    fieldnames = [
        "arrival_rate",
        "algorithm",
        "num_requests",
        "avg_waiting_time",
        "avg_response_time",
        "p99_response_time",
        "throughput",
        "fairness_jain_index",
        "waiting_time_stdev",
        "prediction_error_std",
    ]

    with open(ARGS.csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Comparison written to {ARGS.csv_out}")


parser = argparse.ArgumentParser(
    description="Compare FCFS, Oracle SJF, and Predicted SJF."
)

parser.add_argument(
    "--arrival-rates",
    type=float,
    nargs="+",
    default=[2.0, 4.0, 8.0, 12.0],
)

parser.add_argument("--num-requests", type=int, default=200)
parser.add_argument("--mean-output-len", type=float, default=50.0)
parser.add_argument("--max-batch-size", type=int, default=8)
parser.add_argument("--decode-time-per-step", type=float, default=0.05)
parser.add_argument("--seed", type=int, default=42)

parser.add_argument(
    "--prediction-error",
    type=float,
    default=0.30,
    help="Std. dev. of multiplicative Gaussian prediction error (default: 0.30).",
)

parser.add_argument(
    "--csv-out",
    type=str,
    default=os.path.join(
        os.path.dirname(__file__),
        "fcfs_oracle_predicted_comparison.csv",
    ),
)

ARGS = parser.parse_args()

if __name__ == "__main__":
    main()
