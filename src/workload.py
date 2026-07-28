"""Synthetic request workload generator.

Arrival times follow a Poisson process (exponential inter-arrival times) -
the standard, defensible default absent a real trace. Output lengths follow
a log-normal distribution to reflect the long-tail characteristic of real
LLM generation lengths (many short completions, a few very long ones).
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence

from src.request import Request


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
    """Generate `num_requests` requests.

    arrival_rate: average number of requests per unit time (lambda of the
    Poisson process); mean_output_len: target mean of the output length
    distribution; priority_levels: if given, each request's priority is
    drawn uniformly from this sequence (default: everyone priority 0,
    irrelevant for FCFS/SJF).
    """
    rng = random.Random(seed)
    requests: List[Request] = []
    arrival_time = 0.0
    for request_id in range(num_requests):
        arrival_time += rng.expovariate(arrival_rate)
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
