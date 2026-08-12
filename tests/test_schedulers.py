"""Regression tests for Oracle SJF, Predicted SJF, and ARRS.

Numbers here are hand-verified (see docs/Week2_3_Plan.md section 7), not
just "assert it runs" checks, so a bug in the ordering logic can't hide
behind a large/random workload.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.request import Request
from src.schedulers.arrs import ARRSScheduler
from src.schedulers.fcfs import FCFSScheduler
from src.schedulers.oracle_sjf import OracleSJFScheduler
from src.schedulers.predicted_sjf import PredictedSJFScheduler
from src.simulator import Simulator


class TestSanityCheckFCFSVsOracleSJF(unittest.TestCase):
    """Exp0: three requests, identical arrival time, batch size 1.

    Hand-computed (see docs/Week2_3_Plan.md section 7.1):
      FCFS serves in id order (1,2,3):   response times 100, 110, 130 -> avg 113.333
      Oracle SJF serves shortest first (2,3,1): response times 10, 30, 130 -> avg 56.667
    Oracle SJF must beat FCFS here; if it doesn't, either the simulator's
    simultaneous-arrival handling or the scheduler's ordering is broken.
    """

    def _make_requests(self):
        return [
            Request(request_id=1, arrival_time=0.0, output_len=100),
            Request(request_id=2, arrival_time=0.0, output_len=10),
            Request(request_id=3, arrival_time=0.0, output_len=20),
        ]

    def test_fcfs_serves_in_id_order(self):
        requests = self._make_requests()
        Simulator(requests, scheduler=FCFSScheduler(max_batch_size=1), decode_time_per_step=1.0).run()
        response_times = {r.request_id: r.response_time for r in requests}
        self.assertEqual(response_times, {1: 100.0, 2: 110.0, 3: 130.0})

    def test_oracle_sjf_serves_shortest_first(self):
        requests = self._make_requests()
        Simulator(requests, scheduler=OracleSJFScheduler(max_batch_size=1), decode_time_per_step=1.0).run()
        response_times = {r.request_id: r.response_time for r in requests}
        self.assertEqual(response_times, {2: 10.0, 3: 30.0, 1: 130.0})

    def test_oracle_sjf_beats_fcfs_on_average_response_time(self):
        fcfs_requests = self._make_requests()
        Simulator(fcfs_requests, scheduler=FCFSScheduler(max_batch_size=1), decode_time_per_step=1.0).run()
        fcfs_avg = sum(r.response_time for r in fcfs_requests) / len(fcfs_requests)

        oracle_requests = self._make_requests()
        Simulator(oracle_requests, scheduler=OracleSJFScheduler(max_batch_size=1), decode_time_per_step=1.0).run()
        oracle_avg = sum(r.response_time for r in oracle_requests) / len(oracle_requests)

        self.assertAlmostEqual(fcfs_avg, 113.333, places=2)
        self.assertAlmostEqual(oracle_avg, 56.667, places=2)
        self.assertLess(oracle_avg, fcfs_avg)


class TestPredictedSJF(unittest.TestCase):
    def test_orders_by_predicted_not_true_length(self):
        # Request 1 looks short to the predictor but is actually the longest;
        # Predicted SJF must be fooled by the (wrong) prediction and run it first.
        requests = [
            Request(request_id=1, arrival_time=0.0, output_len=100, predicted_output_len=5.0),
            Request(request_id=2, arrival_time=0.0, output_len=10, predicted_output_len=50.0),
        ]
        Simulator(requests, scheduler=PredictedSJFScheduler(max_batch_size=1), decode_time_per_step=1.0).run()
        self.assertEqual(requests[0].start_time, 0.0)  # request 1 ran first...
        self.assertEqual(requests[1].start_time, 100.0)  # ...despite being the truly shorter job

    def test_missing_prediction_raises(self):
        requests = [Request(request_id=1, arrival_time=0.0, output_len=10)]
        with self.assertRaises(ValueError):
            Simulator(requests, scheduler=PredictedSJFScheduler(max_batch_size=1), decode_time_per_step=1.0).run()


class TestARRSStarvationPrevention(unittest.TestCase):
    """A long request competes against a continuous, overloading stream of
    short requests (arrival rate >> service rate, so the queue never
    drains). Under Predicted SJF the long request's score never improves,
    so it only runs once the short stream is fully exhausted. Under ARRS
    the aging term lets it win once it has waited long enough, bounding its
    wait well below that.
    """

    NUM_SHORT_REQUESTS = 400

    def _make_requests(self):
        requests = [
            Request(request_id=0, arrival_time=0.0, output_len=20, predicted_output_len=20.0, prediction_uncertainty=0.0)
        ]
        for i in range(1, self.NUM_SHORT_REQUESTS + 1):
            # request i=1 arrives at the same instant as the long request so
            # the long one is never alone at t=0 with a free batch slot --
            # it must actually compete on priority from the very first
            # scheduling decision, not win by default event-processing order.
            requests.append(
                Request(
                    request_id=i,
                    arrival_time=(i - 1) * 0.1,
                    output_len=1,
                    predicted_output_len=1.0,
                    prediction_uncertainty=0.0,
                )
            )
        return requests

    def test_predicted_sjf_starves_the_long_request(self):
        requests = self._make_requests()
        Simulator(requests, scheduler=PredictedSJFScheduler(max_batch_size=1), decode_time_per_step=1.0).run()
        long_request = requests[0]
        # It only ever gets admitted after every short request has, i.e. its
        # wait equals the full short-request count.
        self.assertEqual(long_request.waiting_time, float(self.NUM_SHORT_REQUESTS))

    def test_arrs_bounds_the_long_requests_wait(self):
        requests = self._make_requests()
        Simulator(
            requests,
            scheduler=ARRSScheduler(max_batch_size=1, alpha=2.0, beta=0.0, decode_time_per_step=1.0),
            decode_time_per_step=1.0,
        ).run()
        long_request = requests[0]
        # Aging lets it win well before the short stream is exhausted.
        self.assertLess(long_request.waiting_time, self.NUM_SHORT_REQUESTS / 2)


if __name__ == "__main__":
    unittest.main()
