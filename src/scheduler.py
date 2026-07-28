"""Scheduler abstract base class.

Owns the waiting queue / running set / batch-capacity bookkeeping shared by
every scheduling algorithm. Subclasses only need to define the ordering
policy (which waiting request to admit next) via `_priority_key`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from src.request import Request


class Scheduler(ABC):
    def __init__(self, max_batch_size: int) -> None:
        self.max_batch_size = max_batch_size
        self.waiting: List[Request] = []
        self.running: Dict[int, Request] = {}

    def add_request(self, request: Request) -> None:
        self.waiting.append(request)

    def notify_departure(self, request_id: int) -> None:
        self.running.pop(request_id, None)

    def schedule(self, current_time: float) -> List[Request]:
        """Admit as many waiting requests as there is free batch capacity for.

        Returns the list of requests newly admitted into the running set.
        """
        admitted: List[Request] = []
        while self.waiting and len(self.running) < self.max_batch_size:
            request = min(self.waiting, key=self._priority_key)
            self.waiting.remove(request)
            self.running[request.request_id] = request
            admitted.append(request)
        return admitted

    @abstractmethod
    def _priority_key(self, request: Request) -> Any:
        """Sort key used to pick the next waiting request to admit (smaller = sooner)."""
        raise NotImplementedError
