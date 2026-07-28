"""Event-driven main loop of the simulator."""

from __future__ import annotations

import heapq
from typing import Dict, List

from src.event import Event, EventType
from src.request import Request
from src.scheduler import Scheduler

DEFAULT_DECODE_TIME_PER_STEP = 1.0


class Simulator:
    def __init__(
        self,
        requests: List[Request],
        scheduler: Scheduler,
        decode_time_per_step: float = DEFAULT_DECODE_TIME_PER_STEP,
    ) -> None:
        self.scheduler = scheduler
        self.decode_time_per_step = decode_time_per_step
        self.requests: Dict[int, Request] = {r.request_id: r for r in requests}
        self.event_queue: List[Event] = []
        self.current_time: float = 0.0

        for request in self.requests.values():
            heapq.heappush(
                self.event_queue,
                Event(time=request.arrival_time, event_type=EventType.ARRIVAL, request_id=request.request_id),
            )

    def run(self) -> None:
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time
            if event.event_type is EventType.ARRIVAL:
                self._handle_arrival(event)
            elif event.event_type is EventType.DEPARTURE:
                self._handle_departure(event)

    def _handle_arrival(self, event: Event) -> None:
        request = self.requests[event.request_id]
        self.scheduler.add_request(request)
        self._admit_waiting_requests()

    def _handle_departure(self, event: Event) -> None:
        request = self.requests[event.request_id]
        finished = request.step()
        if finished:
            request.mark_finished(current_time=self.current_time)
            self.scheduler.notify_departure(request.request_id)
            self._admit_waiting_requests()
        else:
            self._schedule_next_departure(request)

    def _admit_waiting_requests(self) -> None:
        for request in self.scheduler.schedule(self.current_time):
            request.mark_running(current_time=self.current_time)
            self._schedule_next_departure(request)

    def _schedule_next_departure(self, request: Request) -> None:
        next_time = self.current_time + self.decode_time_per_step
        heapq.heappush(
            self.event_queue,
            Event(time=next_time, event_type=EventType.DEPARTURE, request_id=request.request_id),
        )
