"""Waiting queues for the phase 3 benchmark.

``TIERequestQueue``
    A min-heap ordered by ``E[X] + beta * CVaR_0.9[X]``, with scores filled
    in asynchronously and decayed over time to prevent starvation. This is
    the scheduler under test.

``FCFSWithPredictionQueue``
    Plain FCFS order, but it loads the predictor and runs every prediction
    anyway, then discards the result. Its only purpose is to put exactly the
    same load on the GPU as the TIE arm so that the difference between them
    isolates the scheduling decision from the predictor's cost -- see
    docs/Phase3_Scheduling_Evaluation_Plan.md section 4.2.

``TIERequestQueue`` is adapted from the TIE reference implementation
(``UARequestQueue`` in ``vllm/v1/core/sched/request_queue.py`` of the
authors' vLLM fork, Apache 2.0). The heap/lazy-deletion/starvation-decay
algorithm is theirs and is reproduced with its structure intact, since
reproducing the paper's scheduler is the point. Changes made here:

  * ``max_batch_size`` default lowered 128 -> 32. The predictor shares one
    A100 with the served model and only gets the memory vLLM left over (see
    plan doc section 4); a 128-prompt DeBERTa forward would not fit.
  * Scoring calls go to ``vllm_tie.predictor``.
  * The background worker is factored into ``_PredictionWorker`` so the
    FCFS control arm can reuse it unchanged.
  * Added ``stats()``, which counts requests scheduled *before* their
    prediction landed. If that fraction is high the queue is effectively
    running FCFS however good the predictor is -- a failure mode the
    benchmark must be able to detect rather than guess at.
  * Logging is throttled through ``_log``; the reference prints on every
    batch, which floods the server log at benchmark request rates.
"""

from __future__ import annotations

import heapq
import queue as queue_module
import threading
import time
from collections.abc import Iterable, Iterator

from vllm.v1.core.sched.request_queue import FCFSRequestQueue, RequestQueue
from vllm.v1.request import Request

from vllm_tie.predictor import INITIAL_SCORE, load_model, predict_scores

# Starvation prevention (paper section 5): Score' = Score * gamma^(tw/tau)
STARVATION_GAMMA = 0.9
STARVATION_TAU = 30.0
STARVATION_UPDATE_INTERVAL = 5.0

LOG_INTERVAL = 10.0


class _PredictionWorker:
    """Background thread that batches waiting requests and scores them.

    Scoring never blocks the scheduler: `_submit` returns immediately and
    results arrive later through the `_on_scores` hook. Subclasses decide
    what, if anything, to do with them.
    """

    def _init_predictor(
        self,
        tokenizer,
        max_batch_size: int,
        optimal_batch_size: int,
        max_wait_time_ms: float,
        enable_batching: bool,
        predictor_device: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.optimal_batch_size = optimal_batch_size
        self.max_wait_time_ms = max_wait_time_ms
        self.enable_batching = enable_batching

        self._prediction_queue: queue_module.Queue = queue_module.Queue()
        self._running = True

        self._n_scored = 0
        self._n_popped = 0
        self._n_popped_unpredicted = 0
        self._predict_time_total = 0.0
        self._predict_batches = 0
        self._last_log = 0.0

        load_model(device_id=predictor_device)

        self._prediction_thread = threading.Thread(
            target=self._prediction_worker, daemon=True, name="TIE-Predictor"
        )
        self._prediction_thread.start()
        print(
            f"[TIE] {type(self).__name__} ready; prediction thread started "
            f"(max_batch={max_batch_size}, optimal_batch={optimal_batch_size}, "
            f"max_wait={max_wait_time_ms}ms)",
            flush=True,
        )

    # -- hooks for subclasses ------------------------------------------

    def _on_scores(self, requests: list[Request], scores: list[int]) -> None:
        """Called with finished predictions. Default: discard them."""

    def _periodic(self) -> None:
        """Called every STARVATION_UPDATE_INTERVAL seconds on the worker."""

    def _queue_depth(self) -> int:
        raise NotImplementedError

    # -- worker --------------------------------------------------------

    def _submit(self, request: Request) -> None:
        self._prediction_queue.put(request)

    def _prediction_worker(self) -> None:
        batch: list[Request] = []
        last_process = time.time()
        last_periodic = time.time()

        while self._running:
            now = time.time()
            if now - last_periodic >= STARVATION_UPDATE_INTERVAL:
                self._periodic()
                last_periodic = now
                self._log()

            try:
                task = self._prediction_queue.get(timeout=0.005)
                if task is None:
                    break
                batch.append(task)

                elapsed_ms = (time.time() - last_process) * 1000
                should_process = (
                    len(batch) >= self.max_batch_size
                    or (
                        len(batch) >= self.optimal_batch_size
                        and self._prediction_queue.qsize() == 0
                    )
                    or elapsed_ms >= self.max_wait_time_ms
                    or not self.enable_batching
                )
                if should_process:
                    self._process_batch(batch)
                    batch = []
                    last_process = time.time()
            except queue_module.Empty:
                if batch:
                    self._process_batch(batch)
                    batch = []
                    last_process = time.time()
            except Exception as exc:  # noqa: BLE001
                print(f"[TIE] prediction thread error: {exc}", flush=True)
                batch = []

    def _process_batch(self, batch: list[Request]) -> None:
        if not batch or not self.tokenizer:
            return
        try:
            pairs = [(r, r.prompt_token_ids) for r in batch if r.prompt_token_ids]
            if not pairs:
                return
            requests = [r for r, _ in pairs]
            token_ids_list = [t for _, t in pairs]

            start = time.time()
            scores = predict_scores(token_ids_list, self.tokenizer, self._queue_depth())
            self._predict_time_total += time.time() - start
            self._predict_batches += 1
            self._n_scored += len(scores)

            self._on_scores(requests, scores)
        except Exception as exc:  # noqa: BLE001 - must never kill the server
            print(f"[TIE] batch scoring failed ({len(batch)} reqs): {exc}", flush=True)

    # -- reporting -----------------------------------------------------

    def _log(self, force: bool = False) -> None:
        """Throttled one-line status carrying everything needed to tell a
        degenerate run from a real one."""
        now = time.time()
        if not force and now - self._last_log < LOG_INTERVAL:
            return
        self._last_log = now
        s = self.stats()
        print(
            f"[TIE] waiting={self._queue_depth()} scored={s['scored']} "
            f"popped={s['popped']} "
            f"popped_before_prediction={s['popped_before_prediction']} "
            f"avg_batch_predict={s['avg_batch_predict_ms']:.1f}ms",
            flush=True,
        )

    def stats(self) -> dict[str, float]:
        avg_ms = (
            1000.0 * self._predict_time_total / self._predict_batches
            if self._predict_batches
            else 0.0
        )
        return {
            "scored": self._n_scored,
            "popped": self._n_popped,
            "popped_before_prediction": self._n_popped_unpredicted,
            "predict_batches": self._predict_batches,
            "avg_batch_predict_ms": avg_ms,
        }

    def shutdown(self) -> None:
        self._log(force=True)
        print(f"[TIE] final stats: {self.stats()}", flush=True)
        self._running = False
        self._prediction_queue.put(None)
        if self._prediction_thread.is_alive():
            self._prediction_thread.join(timeout=2.0)
            if self._prediction_thread.is_alive():
                print("[TIE] warning: prediction thread did not stop cleanly.", flush=True)


class TIERequestQueue(_PredictionWorker, RequestQueue):
    """Min-heap waiting queue ordered by TIE score.

    A request is enqueued immediately with a pessimistic placeholder score
    and handed to the background thread, which pushes a refined score back.

    Lazy deletion via per-request version counters gives:
        peek   O(1) amortized (stale entries discarded at the top)
        push   O(log n)
        pop    O(log n) amortized
        update O(log n)  (push new entry; old one becomes stale)
        remove O(1)      (invalidate version; cleaned up later)

    Heap entries are ``(effective_score, arrival_time, version, req_id,
    request)``. ``arrival_time`` breaks ties in FCFS order, which matters
    early on when every request still carries the same placeholder score.
    """

    def __init__(
        self,
        tokenizer=None,
        max_batch_size: int = 32,
        optimal_batch_size: int = 8,
        max_wait_time_ms: float = 3.0,
        enable_batching: bool = True,
        predictor_device: int = 0,
    ) -> None:
        self._heap: list[tuple[float, float, int, str, Request]] = []
        self._lock = threading.RLock()
        self._versions: dict[str, int] = {}
        self._base_scores: dict[str, float] = {}
        self._request_info: dict[str, tuple[float, Request]] = {}
        self._predicted: set[str] = set()

        self._init_predictor(
            tokenizer,
            max_batch_size,
            optimal_batch_size,
            max_wait_time_ms,
            enable_batching,
            predictor_device,
        )

    # -- worker hooks --------------------------------------------------

    def _queue_depth(self) -> int:
        with self._lock:
            return len(self._versions)

    def _on_scores(self, requests: list[Request], scores: list[int]) -> None:
        with self._lock:
            for request, score in zip(requests, scores):
                req_id = request.request_id
                if req_id in self._versions:
                    self._push_updated_score(req_id, float(score))
                    self._predicted.add(req_id)

    def _periodic(self) -> None:
        self._apply_starvation_decay()

    # -- internals -----------------------------------------------------

    def _push_updated_score(self, req_id: str, new_base: float) -> None:
        """Push a fresh heap entry for req_id, invalidating the old one.

        Must hold self._lock.
        """
        if req_id not in self._versions:
            return  # already popped or removed
        arrival_time, request = self._request_info[req_id]
        tw = max(0.0, time.time() - arrival_time)
        effective = new_base * (STARVATION_GAMMA ** (tw / STARVATION_TAU))
        version = self._versions[req_id] + 1
        self._versions[req_id] = version
        self._base_scores[req_id] = new_base
        heapq.heappush(self._heap, (effective, arrival_time, version, req_id, request))

    def _apply_starvation_decay(self) -> None:
        """Rebuild the heap with time-decayed scores (paper section 5).

        O(n), but runs on the background thread every
        STARVATION_UPDATE_INTERVAL seconds, off the scheduling path. Bumping
        every version also purges stale entries left by lazy deletion.
        """
        with self._lock:
            if not self._versions:
                return
            now = time.time()
            new_heap = []
            for req_id, (arrival_time, request) in self._request_info.items():
                base = self._base_scores.get(req_id, INITIAL_SCORE)
                tw = max(0.0, now - arrival_time)
                effective = base * (STARVATION_GAMMA ** (tw / STARVATION_TAU))
                version = self._versions[req_id] + 1
                self._versions[req_id] = version
                new_heap.append((effective, arrival_time, version, req_id, request))
            self._heap = new_heap
            heapq.heapify(self._heap)

    # -- RequestQueue interface ----------------------------------------

    def add_request(self, request: Request) -> None:
        """Enqueue at INITIAL_SCORE (O(log n)) and hand off for scoring."""
        req_id = request.request_id
        with self._lock:
            self._versions[req_id] = 0
            self._base_scores[req_id] = INITIAL_SCORE
            self._request_info[req_id] = (request.arrival_time, request)
            heapq.heappush(
                self._heap,
                (INITIAL_SCORE, request.arrival_time, 0, req_id, request),
            )
        self._submit(request)

    def pop_request(self) -> Request:
        """Remove and return the lowest-scoring request (O(log n) amortized)."""
        with self._lock:
            while self._heap:
                _, _, version, req_id, request = heapq.heappop(self._heap)
                if self._versions.get(req_id) != version:
                    continue  # stale entry
                del self._versions[req_id]
                del self._base_scores[req_id]
                del self._request_info[req_id]
                self._n_popped += 1
                if req_id in self._predicted:
                    self._predicted.discard(req_id)
                else:
                    self._n_popped_unpredicted += 1
                return request
            raise IndexError("pop from empty heap")

    def peek_request(self) -> Request:
        """Lowest-scoring request without removing it (O(1) amortized)."""
        with self._lock:
            while self._heap:
                _, _, version, req_id, request = self._heap[0]
                if self._versions.get(req_id) == version:
                    return request
                heapq.heappop(self._heap)
            raise IndexError("peek from empty heap")

    def prepend_request(self, request: Request) -> None:
        """Re-admit a preempted request.

        Note this is NOT the head-insertion FCFSRequestQueue does: the
        request re-enters ranked by score, so a preempted long request can be
        deferred again. The difference is intentional (it is what the
        reference implementation does) but it moves tail latency, so
        preemption counts are reported alongside the percentiles.
        """
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Drop a request (O(1)); its heap entry dies on the next pop/peek."""
        with self._lock:
            req_id = request.request_id
            if req_id in self._versions:
                del self._versions[req_id]
                del self._base_scores[req_id]
                del self._request_info[req_id]
                self._predicted.discard(req_id)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        for request in requests:
            self.remove_request(request)

    def __bool__(self) -> bool:
        with self._lock:
            return bool(self._versions)

    def __len__(self) -> int:
        with self._lock:
            return len(self._versions)

    def __iter__(self) -> Iterator[Request]:
        """Requests in ascending score order, stale entries skipped."""
        with self._lock:
            heap_copy = self._heap[:]
            versions_copy = dict(self._versions)
        while heap_copy:
            _, _, version, req_id, request = heapq.heappop(heap_copy)
            if versions_copy.get(req_id) == version:
                versions_copy.pop(req_id)
                yield request

    def __reversed__(self) -> Iterator[Request]:
        return reversed(list(self))


class FCFSWithPredictionQueue(_PredictionWorker, FCFSRequestQueue):
    """FCFS ordering, but pays the predictor's full GPU cost anyway.

    Everything vLLM's own FCFSRequestQueue does is inherited untouched; the
    only addition is submitting each arriving request to the prediction
    worker and throwing the score away. Comparing this against
    TIERequestQueue isolates the value of the scheduling decision, and
    comparing it against stock FCFS isolates the predictor's overhead.
    """

    def __init__(
        self,
        tokenizer=None,
        max_batch_size: int = 32,
        optimal_batch_size: int = 8,
        max_wait_time_ms: float = 3.0,
        enable_batching: bool = True,
        predictor_device: int = 0,
    ) -> None:
        FCFSRequestQueue.__init__(self)
        self._init_predictor(
            tokenizer,
            max_batch_size,
            optimal_batch_size,
            max_wait_time_ms,
            enable_batching,
            predictor_device,
        )

    def _queue_depth(self) -> int:
        return len(self)

    def add_request(self, request: Request) -> None:
        super().add_request(request)
        self._submit(request)

    def pop_request(self) -> Request:
        self._n_popped += 1
        return super().pop_request()
