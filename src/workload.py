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
from typing import List, Optional, Sequence, Tuple, Union

from src.metrics import percentile
from src.predictor import lognormal_cvar, lognormal_expectation, sample_calibrated_error
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


SigmaSpec = Union[float, Tuple[float, float]]


def _resolve_sigma(spec: SigmaSpec, rng: random.Random) -> float:
    """A SigmaSpec is either a fixed sigma, or a (low, high) range to draw
    independently per request from -- resolved BEFORE true/predicted_length
    are known for that request (see docs/Week2_3_Plan.md section 11 /
    conversation notes on leakage: the range itself is a property of the
    category, chosen blind, never a function of this request's realized
    true/predicted gap).
    """
    if isinstance(spec, tuple):
        low, high = spec
        return rng.uniform(low, high)
    return spec


def generate_contaminated_workload(
    num_requests: int,
    arrival_rate: float,
    length_pool: Sequence[int],
    tail_rate: float,
    sigma_normal: SigmaSpec,
    sigma_tail: SigmaSpec,
    tail_true_percentile: float = 99.0,
    tail_pred_percentile: float = 50.0,
    cvar_alpha: float = 0.9,
    seed: Optional[int] = None,
) -> List[Request]:
    """Poisson arrivals. Each request is, independently, "normal"
    (probability 1-tail_rate) or "tail" (probability tail_rate):

    - normal: true_length sampled from the real pool; predicted_length =
      true_length + eps, eps from the ELIS-calibrated Gaussian mixture (see
      src/predictor.py sample_calibrated_error) -- a genuine, data-grounded
      predictor error, not an assumption.
    - tail: a DELIBERATELY CONSTRUCTED stress-test case, not a claim about
      real predictor behavior -- true_length sampled from the pool's
      >=tail_true_percentile tail (a real long request), predicted_length
      independently sampled from the pool's <=tail_pred_percentile bottom
      half (looks short), i.e. a severe, synthetic underestimation. See
      docs/Week2_3_Plan.md section 11.

    `sigma_normal`/`sigma_tail` are each either a fixed float, or a (low,
    high) tuple to draw sigma from uniformly, independently per request --
    giving within-category spread (not every normal/tail request scored
    identically) without leaking the realized true/predicted gap into sigma
    (see docs/Week2_3_Plan.md section 11.6 and conversation notes).

    Every request also gets `expected_length` (E[X]) and `tail_risk`
    (CVaR_alpha[X]) of a LogNormal(log(predicted_length), sigma) fit for
    TIEScheduler to consume (see src/schedulers/tie.py).
    """
    rng = random.Random(seed)
    tail_true_cutoff = percentile(list(length_pool), tail_true_percentile)
    tail_pred_cutoff = percentile(list(length_pool), tail_pred_percentile)
    tail_true_pool = [length for length in length_pool if length >= tail_true_cutoff]
    tail_pred_pool = [length for length in length_pool if length <= tail_pred_cutoff]
    if not tail_true_pool or not tail_pred_pool:
        raise ValueError("tail_true_pool/tail_pred_pool is empty; check the percentile cutoffs against length_pool")

    arrival_times = _poisson_arrival_times(num_requests, arrival_rate, rng)
    requests: List[Request] = []
    for request_id, arrival_time in enumerate(arrival_times):
        is_tail = rng.random() < tail_rate
        if is_tail:
            true_length = rng.choice(tail_true_pool)
            predicted_length = float(rng.choice(tail_pred_pool))
            sigma = _resolve_sigma(sigma_tail, rng)
        else:
            true_length = rng.choice(length_pool)
            predicted_length = max(1.0, true_length + sample_calibrated_error(rng))
            sigma = _resolve_sigma(sigma_normal, rng)

        mu = math.log(predicted_length)
        request = Request(request_id=request_id, arrival_time=arrival_time, output_len=true_length)
        request.predicted_output_len = predicted_length
        request.expected_length = lognormal_expectation(mu, sigma)
        request.tail_risk = lognormal_cvar(mu, sigma, alpha=cvar_alpha)
        requests.append(request)
    return requests
