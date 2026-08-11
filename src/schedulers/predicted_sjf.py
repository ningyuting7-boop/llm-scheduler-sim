from typing import Tuple

from src.request import Request
from src.scheduler import Scheduler


class PredictedSJFScheduler(Scheduler):
    """Predicted Shortest Job First scheduler.

    Chooses the waiting request with the smallest predicted output length.
    """

    def _priority_key(self, request: Request) -> Tuple[float, float, int]:
        if request.predicted_output_len is None:
            raise ValueError(
                f"Request {request.request_id} has no predicted_output_len."
            )

        return (
            request.predicted_output_len,
            request.arrival_time,
            request.request_id,
        )