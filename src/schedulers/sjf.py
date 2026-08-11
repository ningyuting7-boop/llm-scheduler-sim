from typing import Tuple

from src.request import Request
from src.scheduler import Scheduler


class SJFScheduler(Scheduler):
    """Oracle Shortest Job First scheduler.

    Chooses the waiting request with the smallest true output length.
    """

    def _priority_key(self, request: Request) -> Tuple[int, float, int]:
        return (
            request.output_len,
            request.arrival_time,
            request.request_id,
        )