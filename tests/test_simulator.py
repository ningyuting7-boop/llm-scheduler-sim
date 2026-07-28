"""Regression tests for the event mechanism, FCFS ordering, and metrics.

Each test is hand-verifiable: inputs are small enough that expected
waiting/response times were computed by hand before writing the assertions,
so a bug in the underlying framework can't hide behind a complex scenario.
"""

from __future__ import annotations

import heapq
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.event import Event, EventType
from src.metrics import compute_summary, jains_fairness_index
from src.request import Request, RequestStatus
from src.schedulers.fcfs import FCFSScheduler
from src.simulator import Simulator


class TestEventOrdering(unittest.TestCase):
    def test_pops_in_time_order_with_fifo_tiebreak(self) -> None:
        heap: list = []
        heapq.heappush(heap, Event(time=2.0, event_type=EventType.ARRIVAL, request_id=1))
        heapq.heappush(heap, Event(time=1.0, event_type=EventType.ARRIVAL, request_id=2))
        heapq.heappush(heap, Event(time=1.0, event_type=EventType.DEPARTURE, request_id=3))

        popped = [heapq.heappop(heap) for _ in range(3)]
        self.assertEqual([e.request_id for e in popped], [2, 3, 1])
        self.assertEqual([e.time for e in popped], [1.0, 1.0, 2.0])


class TestRequest(unittest.TestCase):
    def test_step_lifecycle(self) -> None:
        r = Request(request_id=1, arrival_time=0.0, output_len=2)
        r.mark_running(current_time=0.5)
        self.assertEqual(r.status, RequestStatus.RUNNING)
        self.assertEqual(r.waiting_time, 0.5)

        self.assertFalse(r.step())
        self.assertTrue(r.step())
        r.mark_finished(current_time=2.5)
        self.assertEqual(r.status, RequestStatus.FINISHED)
        self.assertEqual(r.response_time, 2.5)

    def test_rejects_non_positive_output_len(self) -> None:
        with self.assertRaises(ValueError):
            Request(request_id=1, arrival_time=0.0, output_len=0)


class TestSimulatorUnlimitedBatch(unittest.TestCase):
    def test_no_queueing_when_batch_capacity_is_ample(self) -> None:
        requests = [
            Request(request_id=1, arrival_time=0.0, output_len=3),
            Request(request_id=2, arrival_time=0.5, output_len=1),
            Request(request_id=3, arrival_time=5.0, output_len=2),
        ]
        sim = Simulator(requests, scheduler=FCFSScheduler(max_batch_size=10), decode_time_per_step=1.0)
        sim.run()

        self.assertEqual((requests[0].start_time, requests[0].finish_time), (0.0, 3.0))
        self.assertEqual((requests[1].start_time, requests[1].finish_time), (0.5, 1.5))
        self.assertEqual((requests[2].start_time, requests[2].finish_time), (5.0, 7.0))
        self.assertTrue(all(r.waiting_time == 0.0 for r in requests))


class TestFCFSQueueing(unittest.TestCase):
    def test_serves_strictly_in_arrival_order_when_batch_limited(self) -> None:
        requests = [
            Request(request_id=1, arrival_time=0.0, output_len=5),
            Request(request_id=2, arrival_time=0.1, output_len=1),
            Request(request_id=3, arrival_time=0.2, output_len=1),
        ]
        sim = Simulator(requests, scheduler=FCFSScheduler(max_batch_size=1), decode_time_per_step=1.0)
        sim.run()

        self.assertEqual((requests[0].start_time, requests[0].finish_time), (0.0, 5.0))
        self.assertEqual((requests[1].start_time, requests[1].finish_time), (5.0, 6.0))
        self.assertEqual((requests[2].start_time, requests[2].finish_time), (6.0, 7.0))

    def test_later_arrival_never_jumps_ahead_of_earlier_one(self) -> None:
        # Request 2 arrives after request 1 but would finish "faster" if run
        # first; FCFS must still not reorder by output_len.
        requests = [
            Request(request_id=1, arrival_time=0.0, output_len=3),
            Request(request_id=2, arrival_time=0.5, output_len=1),
        ]
        sim = Simulator(requests, scheduler=FCFSScheduler(max_batch_size=1), decode_time_per_step=1.0)
        sim.run()

        self.assertEqual(requests[0].start_time, 0.0)
        self.assertEqual(requests[1].start_time, 3.0)


class TestMetrics(unittest.TestCase):
    def test_hand_computed_summary(self) -> None:
        requests = [
            Request(request_id=1, arrival_time=0.0, output_len=2),
            Request(request_id=2, arrival_time=0.0, output_len=2),
        ]
        sim = Simulator(requests, scheduler=FCFSScheduler(max_batch_size=2), decode_time_per_step=1.0)
        sim.run()

        summary = compute_summary(requests)
        self.assertAlmostEqual(summary.avg_waiting_time, 0.0)
        self.assertAlmostEqual(summary.avg_response_time, 2.0)
        self.assertAlmostEqual(summary.fairness_jain_index, 1.0)

    def test_jains_fairness_index(self) -> None:
        self.assertAlmostEqual(jains_fairness_index([1, 1, 1, 1]), 1.0)
        idx = jains_fairness_index([1, 0, 0, 0])
        self.assertTrue(0 < idx < 1)


if __name__ == "__main__":
    unittest.main()
