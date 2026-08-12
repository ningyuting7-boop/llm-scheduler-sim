"""Predicted SJF: ranks by predicted_output_len (no knowledge of the true
length). Requires the caller to have already run predictor.predict_length
over the workload and filled in Request.predicted_output_len."""

from __future__ import annotations

from typing import Tuple

from src.request import Request
from src.scheduler import Scheduler


class PredictedSJFScheduler(Scheduler):
    def _priority_key(self, request: Request, current_time: float) -> Tuple[float, float, int]:
        if request.predicted_output_len is None:
            raise ValueError(
                f"Request {request.request_id} has no predicted_output_len; "
                f"run predictor.predict_length on the workload before scheduling."
            )
        return (request.predicted_output_len, request.arrival_time, request.request_id)
