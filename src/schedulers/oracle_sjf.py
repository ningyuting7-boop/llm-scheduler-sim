"""Oracle SJF: ranks by the true output_len. Not implementable in a real
system (it requires knowing the future), used only as the theoretical
upper-bound reference the other policies are compared against."""

from __future__ import annotations

from typing import Tuple

from src.request import Request
from src.scheduler import Scheduler


class OracleSJFScheduler(Scheduler):
    def _priority_key(self, request: Request, current_time: float) -> Tuple[int, float, int]:
        return (request.output_len, request.arrival_time, request.request_id)
