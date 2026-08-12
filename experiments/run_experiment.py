"""Generic experiment runner: build a workload, optionally inject prediction
error, run it through one scheduler, and report metrics.

`run_once` is the reusable core (used by exp1-4 to sweep parameters without
paying subprocess overhead per run); `main` is a thin CLI wrapper around it
for one-off invocations.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import MetricsSummary, compute_summary, print_summary, write_csv
from src.predictor import predict_length
from src.request import Request
from src.scheduler import Scheduler
from src.schedulers.arrs import ARRSScheduler
from src.schedulers.fcfs import FCFSScheduler
from src.schedulers.oracle_sjf import OracleSJFScheduler
from src.schedulers.predicted_sjf import PredictedSJFScheduler
from src.simulator import Simulator
from src.workload import (
    generate_bimodal_workload,
    generate_workload,
    generate_workload_from_lengths,
    load_length_pool,
)

DEFAULT_LMSYS_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "lmsys_output_lengths.csv")

# Schedulers that rank by predicted_output_len need predict_length run over
# the workload first; fcfs/oracle_sjf never look at that field.
SCHEDULERS_NEEDING_PREDICTION = {"predicted_sjf", "arrs"}


def _build_workload(
    workload_source: str,
    num_requests: int,
    arrival_rate: float,
    mean_output_len: float,
    lmsys_csv: str,
    long_request_ratio: float,
    seed: Optional[int],
) -> List[Request]:
    if workload_source == "lognormal":
        return generate_workload(num_requests, arrival_rate, mean_output_len, seed=seed)
    if workload_source == "lmsys":
        pool = load_length_pool(lmsys_csv)
        return generate_workload_from_lengths(num_requests, arrival_rate, pool, seed=seed)
    if workload_source == "bimodal":
        pool = load_length_pool(lmsys_csv)
        return generate_bimodal_workload(num_requests, arrival_rate, pool, long_request_ratio, seed=seed)
    raise ValueError(f"Unknown workload_source: {workload_source!r}")


def _build_scheduler(scheduler_name: str, max_batch_size: int, alpha: float, beta: float, decode_time_per_step: float) -> Scheduler:
    if scheduler_name == "fcfs":
        return FCFSScheduler(max_batch_size)
    if scheduler_name == "oracle_sjf":
        return OracleSJFScheduler(max_batch_size)
    if scheduler_name == "predicted_sjf":
        return PredictedSJFScheduler(max_batch_size)
    if scheduler_name == "arrs":
        return ARRSScheduler(max_batch_size, alpha=alpha, beta=beta, decode_time_per_step=decode_time_per_step)
    raise ValueError(f"Unknown scheduler_name: {scheduler_name!r}")


def run_once(
    scheduler_name: str,
    workload_source: str = "lognormal",
    num_requests: int = 200,
    arrival_rate: float = 2.0,
    mean_output_len: float = 50.0,
    lmsys_csv: str = DEFAULT_LMSYS_CSV,
    long_request_ratio: float = 0.05,
    k: float = 0.0,
    alpha: float = 1.0,
    beta: float = 0.0,
    max_batch_size: int = 8,
    decode_time_per_step: float = 0.05,
    seed: Optional[int] = None,
) -> Tuple[List[Request], MetricsSummary]:
    """`k` is the predictor quality tier passed to predict_length (see
    src/predictor.py QUALITY_TIERS: 0.0 = perfect, ~0.34/1.0/2.22 = the
    ELIS-grounded low/realistic/high tiers, larger = ungrounded stress test).
    """
    requests = _build_workload(workload_source, num_requests, arrival_rate, mean_output_len, lmsys_csv, long_request_ratio, seed)

    if scheduler_name in SCHEDULERS_NEEDING_PREDICTION:
        # A rng independent of (but derived from) the workload's seed, so
        # the same workload can be re-scored at a different k without
        # perturbing which requests exist / when they arrive.
        predictor_rng = random.Random(None if seed is None else seed + 1)
        for request in requests:
            predicted, uncertainty = predict_length(request.output_len, k, predictor_rng)
            request.predicted_output_len = predicted
            request.prediction_uncertainty = uncertainty

    scheduler = _build_scheduler(scheduler_name, max_batch_size, alpha, beta, decode_time_per_step)
    Simulator(requests, scheduler=scheduler, decode_time_per_step=decode_time_per_step).run()

    return requests, compute_summary(requests)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one scheduling experiment and report metrics.")
    parser.add_argument("--scheduler", required=True, choices=["fcfs", "oracle_sjf", "predicted_sjf", "arrs"])
    parser.add_argument("--workload-source", default="lognormal", choices=["lognormal", "lmsys", "bimodal"])
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--arrival-rate", type=float, default=2.0)
    parser.add_argument("--mean-output-len", type=float, default=50.0)
    parser.add_argument("--lmsys-csv", type=str, default=DEFAULT_LMSYS_CSV)
    parser.add_argument("--long-request-ratio", type=float, default=0.05)
    parser.add_argument("--k", type=float, default=0.0, help="predictor quality tier, see src/predictor.py QUALITY_TIERS")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--decode-time-per-step", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv-out", type=str, default=None)
    args = parser.parse_args()

    requests, summary = run_once(
        scheduler_name=args.scheduler,
        workload_source=args.workload_source,
        num_requests=args.num_requests,
        arrival_rate=args.arrival_rate,
        mean_output_len=args.mean_output_len,
        lmsys_csv=args.lmsys_csv,
        long_request_ratio=args.long_request_ratio,
        k=args.k,
        alpha=args.alpha,
        beta=args.beta,
        max_batch_size=args.max_batch_size,
        decode_time_per_step=args.decode_time_per_step,
        seed=args.seed,
    )

    print_summary(summary, label=args.scheduler)
    if args.csv_out:
        write_csv(requests, args.csv_out)
        print(f"Per-request details written to {args.csv_out}")


if __name__ == "__main__":
    main()
