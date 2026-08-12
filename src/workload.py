"""Request workload generators: synthetic (lognormal) and empirical (real
LMSYS-Chat-1M output lengths, see scripts/extract_lmsys_lengths.py).

Arrival times always follow a Poisson process (exponential inter-arrival
times) via `_poisson_arrival_times`, shared by every generator below - the
standard, defensible default absent a real arrival trace, and kept identical
across data sources so arrival statistics never become an uncontrolled
variable when comparing schedulers. Only how each request's `output_len` is
produced differs between generators.
"""

from __future__ import annotations

import csv
import math
import random
from typing import List, Optional, Sequence

from src.metrics import percentile
from src.request import Request


def _poisson_arrival_times(num_requests: int, arrival_rate: float, rng: random.Random) -> List[float]:
    """Strictly increasing arrival times, length == num_requests."""
    times: List[float] = []
    t = 0.0
    for _ in range(num_requests):
        t += rng.expovariate(arrival_rate)
        times.append(t)
    return times


def generate_output_len(rng: random.Random, mean_output_len: float, sigma: float = 0.8) -> int:
    mu = math.log(mean_output_len) - (sigma ** 2) / 2
    sampled = rng.lognormvariate(mu, sigma)
    return max(1, round(sampled))


def generate_workload(
    num_requests: int,
    arrival_rate: float,
    mean_output_len: float,
    sigma: float = 0.8,
    priority_levels: Optional[Sequence[int]] = None,
    seed: Optional[int] = None,
) -> List[Request]:
    """Synthetic lognormal workload: the Week 1 baseline, and the fallback
    data source if the LMSYS CSV isn't available.

    arrival_rate: average number of requests per unit time (lambda of the
    Poisson process); mean_output_len: target mean of the output length
    distribution; priority_levels: if given, each request's priority is
    drawn uniformly from this sequence (default: everyone priority 0,
    irrelevant for FCFS/SJF).
    """
    rng = random.Random(seed)
    arrival_times = _poisson_arrival_times(num_requests, arrival_rate, rng)
    requests: List[Request] = []
    for request_id, arrival_time in enumerate(arrival_times):
        output_len = generate_output_len(rng, mean_output_len, sigma)
        priority = rng.choice(priority_levels) if priority_levels else 0
        requests.append(
            Request(
                request_id=request_id,
                arrival_time=arrival_time,
                output_len=output_len,
                priority=priority,
            )
        )
    return requests


def load_length_pool(csv_path: str, column: str = "word_count") -> List[int]:
    """Read scripts/extract_lmsys_lengths.py's output CSV into a list of
    ints, for the two generators below to bootstrap-sample from."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return [int(float(row[column])) for row in reader]


def generate_workload_from_lengths(
    num_requests: int,
    arrival_rate: float,
    length_pool: Sequence[int],
    seed: Optional[int] = None,
) -> List[Request]:
    """Poisson arrivals; output_len bootstrap-sampled (with replacement,
    independent of arrival order) from a real length distribution, e.g. from
    `load_length_pool`."""
    rng = random.Random(seed)
    arrival_times = _poisson_arrival_times(num_requests, arrival_rate, rng)
    requests: List[Request] = []
    for request_id, arrival_time in enumerate(arrival_times):
        output_len = rng.choice(length_pool)
        requests.append(Request(request_id=request_id, arrival_time=arrival_time, output_len=output_len))
    return requests


def generate_bimodal_workload(
    num_requests: int,
    arrival_rate: float,
    length_pool: Sequence[int],
    long_request_ratio: float,
    short_percentile: float = 50.0,
    long_percentile: float = 95.0,
    seed: Optional[int] = None,
) -> List[Request]:
    """Poisson arrivals; each request independently (not in a fixed pattern)
    is drawn from a "short" (<= short_percentile) or "long" (>=
    long_percentile) bucket of a real length distribution, per
    `long_request_ratio`. For the starvation experiment: many short requests
    interleaved with a few long ones, still grounded in real data. See
    docs/Week2_3_Plan.md section 4.7.3 for why the split is per-request
    random rather than a fixed short/long arrival pattern.
    """
    rng = random.Random(seed)
    short_cutoff = percentile(list(length_pool), short_percentile)
    long_cutoff = percentile(list(length_pool), long_percentile)
    short_pool = [length for length in length_pool if length <= short_cutoff]
    long_pool = [length for length in length_pool if length >= long_cutoff]
    if not short_pool or not long_pool:
        raise ValueError("short_pool/long_pool is empty; check short_percentile/long_percentile against length_pool")

    arrival_times = _poisson_arrival_times(num_requests, arrival_rate, rng)
    requests: List[Request] = []
    for request_id, arrival_time in enumerate(arrival_times):
        is_long = rng.random() < long_request_ratio
        output_len = rng.choice(long_pool if is_long else short_pool)
        requests.append(Request(request_id=request_id, arrival_time=arrival_time, output_len=output_len))
    return requests
