"""ARRS: predicted length + uncertainty penalty - aging bonus.

score(t) = predicted_output_len + beta * uncertainty - alpha * waiting_steps

where waiting_steps = (t - arrival_time) / decode_time_per_step, converting
waiting time into the same units as predicted_output_len (decode steps) so
alpha's meaning doesn't drift with --decode-time-per-step. See
docs/Week2_3_Plan.md section 3.3 for the full rationale.

An earlier version of this class had a `pressure`-adaptive alpha (scaling
alpha by queue length / max_batch_size, borrowed from the TIE paper's Eq. 12
adaptive beta). It was dropped: Exp4's workload is deliberately, persistently
congested (otherwise there's no starvation to measure), so pressure sits at
its ceiling almost the entire run and the "adaptive" scaling became
mathematically identical to a fixed alpha -- min_scale had no effect at all.
The paper's adaptive beta governs a different mechanism (how much to hedge
against a request that might be long) than alpha (how hard to rescue one
that's already waited a long time); the two aren't interchangeable. See
docs/Week2_3_Plan.md section 9.7.
"""

from __future__ import annotations

from src.request import Request
from src.scheduler import Scheduler


class ARRSScheduler(Scheduler):
    def __init__(self, max_batch_size: int, alpha: float, beta: float, decode_time_per_step: float = 1.0) -> None:
        super().__init__(max_batch_size)
        self.alpha = alpha
        self.beta = beta
        self.decode_time_per_step = decode_time_per_step

    def _priority_key(self, request: Request, current_time: float) -> float:
        if request.predicted_output_len is None:
            raise ValueError(
                f"Request {request.request_id} has no predicted_output_len; "
                f"run predictor.predict_length on the workload before scheduling."
            )
        uncertainty = request.prediction_uncertainty or 0.0
        waiting_steps = (current_time - request.arrival_time) / self.decode_time_per_step
        return request.predicted_output_len + self.beta * uncertainty - self.alpha * waiting_steps
