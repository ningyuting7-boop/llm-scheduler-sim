"""Metrics computed over a finished run: waiting/response time, throughput,
and fairness. Also supports grouped breakdowns (e.g. by priority) and CSV
export, since Week 2/3 will need to compare multiple scheduling algorithms
on the same output format.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

from src.request import Request, RequestStatus


@dataclass
class MetricsSummary:
    num_requests: int
    avg_waiting_time: float
    avg_response_time: float
    p99_response_time: float
    throughput: float
    fairness_jain_index: float
    waiting_time_stdev: float


def _finished_requests(requests: Iterable[Request]) -> List[Request]:
    finished = [r for r in requests if r.status is RequestStatus.FINISHED]
    if not finished:
        raise ValueError("No finished requests to compute metrics from")
    return finished


def jains_fairness_index(values: List[float]) -> float:
    """1.0 = perfectly fair (all equal), approaches 1/n = maximally unfair."""
    if not values:
        return 0.0
    n = len(values)
    sum_sq = sum(v * v for v in values)
    if sum_sq == 0:
        return 1.0
    return (sum(values) ** 2) / (n * sum_sq)


def compute_summary(requests: Iterable[Request]) -> MetricsSummary:
    finished = _finished_requests(requests)
    waiting_times = [r.waiting_time for r in finished]
    response_times = sorted(r.response_time for r in finished)

    total_span = max(r.finish_time for r in finished) - min(r.arrival_time for r in finished)
    throughput = len(finished) / total_span if total_span > 0 else float("inf")

    p99_index = min(len(response_times) - 1, int(len(response_times) * 0.99))

    return MetricsSummary(
        num_requests=len(finished),
        avg_waiting_time=statistics.mean(waiting_times),
        avg_response_time=statistics.mean(response_times),
        p99_response_time=response_times[p99_index],
        throughput=throughput,
        fairness_jain_index=jains_fairness_index(waiting_times),
        waiting_time_stdev=statistics.pstdev(waiting_times) if len(waiting_times) > 1 else 0.0,
    )


def group_by(
    requests: Iterable[Request], key_fn: Callable[[Request], object]
) -> Dict[object, MetricsSummary]:
    groups: Dict[object, List[Request]] = {}
    for request in _finished_requests(requests):
        groups.setdefault(key_fn(request), []).append(request)
    return {key: compute_summary(reqs) for key, reqs in groups.items()}


def bucket_by_output_len(request: Request, bucket_size: int = 10) -> str:
    lower = (request.output_len // bucket_size) * bucket_size
    return f"{lower}-{lower + bucket_size - 1}"


def print_summary(summary: MetricsSummary, label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    print(
        f"{prefix}requests={summary.num_requests} "
        f"avg_wait={summary.avg_waiting_time:.3f} "
        f"avg_response={summary.avg_response_time:.3f} "
        f"p99_response={summary.p99_response_time:.3f} "
        f"throughput={summary.throughput:.3f} "
        f"fairness(jain)={summary.fairness_jain_index:.3f} "
        f"wait_stdev={summary.waiting_time_stdev:.3f}"
    )


def write_csv(requests: Iterable[Request], path: str) -> None:
    finished = _finished_requests(requests)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "request_id",
                "arrival_time",
                "output_len",
                "priority",
                "start_time",
                "finish_time",
                "waiting_time",
                "response_time",
            ]
        )
        for r in finished:
            writer.writerow(
                [
                    r.request_id,
                    r.arrival_time,
                    r.output_len,
                    r.priority,
                    r.start_time,
                    r.finish_time,
                    r.waiting_time,
                    r.response_time,
                ]
            )
