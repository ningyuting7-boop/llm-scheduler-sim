"""Pluggable vLLM schedulers for the phase 3 benchmark.

Both classes are loaded by stock vLLM through its ``--scheduler-cls`` flag,
so no part of vLLM is forked or patched::

    vllm serve <model> --scheduler-cls vllm_tie.scheduler.TIEScheduler
    vllm serve <model> --scheduler-cls vllm_tie.scheduler.FCFSWithPredictorScheduler
    vllm serve <model>            # the stock FCFS baseline, unmodified

Everything about scheduling -- KV cache management, chunked prefill,
preemption, continuous batching -- is inherited from vLLM's own
``Scheduler``. The only thing either subclass changes is ``self.waiting``,
the waiting-queue object. That single substitution is also the entire
substance of what the TIE authors' fork changes, which is why forking is
unnecessary; see docs/Phase3_Scheduling_Evaluation_Plan.md section 3.

Caveat: ``SchedulerInterface`` is explicitly not a public API ("compatibility
may not be maintained", per vLLM's own warning when a custom scheduler is
loaded), so the vLLM version is pinned. See requirements.
"""

from __future__ import annotations

import os

from vllm.v1.core.sched.scheduler import Scheduler

from vllm_tie import predictor
from vllm_tie.request_queue import FCFSWithPredictionQueue, TIERequestQueue


class _PredictorSchedulerBase(Scheduler):
    """Runs vLLM's scheduler init unchanged, then swaps in a queue that owns
    a DeBERTa predictor.

    Subclasses differ only in which queue class they build.
    """

    queue_cls: type

    def __init__(self, *args, **kwargs) -> None:
        # Run the stock scheduler's init first: it sets self.vllm_config and
        # builds self.waiting as an ordinary FCFS deque, which is then
        # replaced below.
        super().__init__(*args, **kwargs)

        # Keep the adaptive-beta denominator B tied to the real --max-num-seqs
        # rather than a separately-set environment variable, which is how the
        # reference implementation does it and is easy to let drift out of
        # sync with the server's actual batch size.
        predictor.set_gpu_batch_size(self.max_num_running_reqs)

        tokenizer = self._build_tokenizer()
        device = int(os.environ.get("TIE_PREDICTOR_GPU", "0"))
        max_batch = int(os.environ.get("TIE_MAX_PREDICT_BATCH", "32"))

        self.waiting = self.queue_cls(
            tokenizer=tokenizer,
            max_batch_size=max_batch,
            predictor_device=device,
        )
        print(
            f"[TIE] {type(self).__name__} active: queue={self.queue_cls.__name__}, "
            f"max_num_seqs={self.max_num_running_reqs}, "
            f"mode={os.environ.get('TIE_MODE', 'predict')}",
            flush=True,
        )

    def _build_tokenizer(self):
        """The served model's tokenizer, used to decode prompt_token_ids back
        to text before handing them to the predictor."""
        from vllm.transformers_utils.tokenizer import get_tokenizer

        model_config = self.vllm_config.model_config
        return get_tokenizer(
            tokenizer_name=model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
        )

    def shutdown(self) -> None:
        # Stop the prediction thread and print the run's counters before
        # vLLM tears the rest down.
        waiting = getattr(self, "waiting", None)
        if waiting is not None and hasattr(waiting, "shutdown"):
            waiting.shutdown()
        super().shutdown()


class TIEScheduler(_PredictorSchedulerBase):
    """The scheduler under test: waiting requests are ordered by
    ``E[X] + beta * CVaR_0.9[X]`` over a predicted log-t output-length
    distribution."""

    queue_cls = TIERequestQueue


class FCFSWithPredictorScheduler(_PredictorSchedulerBase):
    """Control arm: identical GPU load to TIEScheduler -- same model, same
    predictions, same memory -- but requests are served in arrival order and
    the predictions are discarded.

    Comparing this against TIEScheduler measures the scheduling decision
    alone; comparing it against stock vLLM measures what the predictor costs
    to deploy. See plan doc section 4.2.
    """

    queue_cls = FCFSWithPredictionQueue
