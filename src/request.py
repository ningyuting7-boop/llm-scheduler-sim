"""Request data class: represents a single LLM inference request in the simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class RequestStatus(Enum):
    WAITING = auto()    # arrived, waiting in queue to be scheduled
    RUNNING = auto()    # currently being processed (occupies a batch slot)
    FINISHED = auto()    # all decode steps completed


@dataclass
class Request:
    request_id: int
    arrival_time: float
    output_len: int  # total number of decode steps to generate

    # Reserved for later scheduling algorithms; unused during the FCFS stage
    priority: int = 0
    predicted_output_len: Optional[float] = None
    prediction_uncertainty: Optional[float] = None  # None treated as 0

    # TIEScheduler only (log-normal + CVaR model, see src/predictor.py):
    # E[X] and CVaR_alpha[X] of the fitted per-request distribution.
    expected_length: Optional[float] = None
    tail_risk: Optional[float] = None

    status: RequestStatus = field(default=RequestStatus.WAITING, compare=False)
    remaining_len: int = field(init=False, compare=False)
    start_time: Optional[float] = field(default=None, compare=False)
    finish_time: Optional[float] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.output_len <= 0:
            raise ValueError(f"output_len must be positive, got {self.output_len}")
        self.remaining_len = self.output_len

    @property
    def waiting_time(self) -> Optional[float]:
        """Time from arrival to first being scheduled to run; None if not yet started."""
        if self.start_time is None:
            return None
        return self.start_time - self.arrival_time

    @property
    def response_time(self) -> Optional[float]:
        """Total time from arrival to completion; None if not yet finished."""
        if self.finish_time is None:
            return None
        return self.finish_time - self.arrival_time

    def mark_running(self, current_time: float) -> None:
        if self.status is RequestStatus.WAITING:
            self.start_time = current_time
        self.status = RequestStatus.RUNNING

    def step(self) -> bool:
        """Execute one decode step; return whether the request just finished."""
        if self.status is not RequestStatus.RUNNING:
            raise RuntimeError(f"Request {self.request_id} is not running, cannot step")
        if self.remaining_len <= 0:
            raise RuntimeError(f"Request {self.request_id} has no remaining work")
        self.remaining_len -= 1
        return self.remaining_len == 0

    def mark_finished(self, current_time: float) -> None:
        self.status = RequestStatus.FINISHED
        self.finish_time = current_time
