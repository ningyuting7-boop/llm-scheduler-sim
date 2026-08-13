"""TIEScheduler: score = E[X] + beta * CVaR_alpha[X] - alpha * waiting_time,
approximating the scheduling rule in Zheng et al. 2026 ("TIE",
arXiv:2604.00499) Eq. 11, with a log-normal fit in place of their log-t (see
src/predictor.py module docstring for why).

`alpha` defaults to 0.0 (no aging), which is the original design -- this
isolates whether tail-risk-awareness alone (not combined with
starvation-prevention) can beat a naive point-estimate SJF on average
latency; see docs/Week2_3_Plan.md section 11. `alpha > 0.0` adds an aging
term for the separate question of whether TIE's average gain over Predicted
SJF can coexist with better fairness -- TIE's tail-risk term alone makes
fairness WORSE than Predicted SJF, not better, since it deliberately makes
flagged requests wait longer; see conversation notes / docs/Week2_3_Plan.md
section 13.

`alpha` has a direct physical unit: "tokens of priority credit per second
waited" (simulation time is in seconds throughout, see section 12; score is
in the same token/word-count units as expected_length/tail_risk), NOT an
abstract dimensionless multiplier -- e.g. alpha=100 means a request that
has waited 1 second gets treated as if its expected length were 100 tokens
shorter. This makes swept alpha values directly interpretable/reportable
(see docs/Week2_3_Plan.md section 13), unlike a dimensionless alpha whose
meaning would silently drift if decode_time_per_step changed.

Requires Request.expected_length / Request.tail_risk to already be filled
in (see src/workload.py generate_contaminated_workload).
"""

from __future__ import annotations

from typing import Tuple

from src.request import Request
from src.scheduler import Scheduler


class TIEScheduler(Scheduler):
    def __init__(self, max_batch_size: int, beta: float, alpha: float = 0.0) -> None:
        super().__init__(max_batch_size)
        self.beta = beta
        self.alpha = alpha

    def _priority_key(self, request: Request, current_time: float) -> Tuple[float, float, int]:
        if request.expected_length is None or request.tail_risk is None:
            raise ValueError(
                f"Request {request.request_id} has no expected_length/tail_risk; "
                f"use workload.generate_contaminated_workload to build the workload."
            )
        score = request.expected_length + self.beta * request.tail_risk
        if self.alpha:
            waiting_time = current_time - request.arrival_time
            score -= self.alpha * waiting_time
        return (score, request.arrival_time, request.request_id)
