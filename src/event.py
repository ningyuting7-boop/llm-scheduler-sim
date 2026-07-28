"""Event data class: the minimal unit driving the event-driven simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import count
from typing import Any, Optional


class EventType(Enum):
    ARRIVAL = auto()      # a request arrives
    DEPARTURE = auto()    # a request completes a decode step (or finishes entirely)


_event_seq = count()


@dataclass(order=True)
class Event:
    """Ordered by (time, seq) for use with heapq.

    seq is only used to tie-break multiple events at the same time (FIFO by
    insertion order); event_type/request_id/payload are excluded from
    comparison to avoid comparing non-orderable objects.
    """

    time: float
    event_type: EventType = field(compare=False)
    request_id: int = field(compare=False)
    payload: Optional[Any] = field(default=None, compare=False)
    seq: int = field(default_factory=lambda: next(_event_seq))

    def __repr__(self) -> str:
        return (
            f"Event(time={self.time}, type={self.event_type.name}, "
            f"request_id={self.request_id})"
        )
