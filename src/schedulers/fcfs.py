"""FCFS (first-come, first-served) scheduling policy."""

from __future__ import annotations

from typing import Tuple

from src.request import Request
from src.scheduler import Scheduler


class FCFSScheduler(Scheduler):
    def _priority_key(self, request: Request, current_time: float) -> Tuple[float, int]:
        return (request.arrival_time, request.request_id)
