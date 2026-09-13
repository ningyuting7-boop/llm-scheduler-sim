"""Gate check for phase 3: can our scheduler actually plug into stock vLLM?

``--scheduler-cls`` loads a class by qualified name into a slot that vLLM
documents as **not** a public API ("compatibility may not be maintained"),
so every assumption the phase 3 design rests on is checked here before any
GPU time is spent:

  1. the vLLM version matches the one the design was written against;
  2. ``Scheduler`` and the request-queue types import from the stock package;
  3. ``Scheduler.__init__`` really does build ``self.waiting`` from a
     replaceable queue object -- the one line our subclass overrides;
  4. our queues satisfy the ``RequestQueue`` ABC (an unimplemented abstract
     method would only surface at serve time, mid-benchmark);
  5. our schedulers satisfy ``SchedulerInterface`` and resolve through the
     same ``resolve_obj_by_qualname`` path ``--scheduler-cls`` uses.

Run from the repo root::

    python scripts/verify_vllm_integration.py

No GPU needed, but importing vLLM pulls in torch and is memory-hungry; on a
cluster login node it may be OOM-killed, in which case run it inside a small
interactive allocation instead.
"""

from __future__ import annotations

import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXPECTED_VERSION = "0.11.1"

_failures: list[str] = []
_warnings: list[str] = []


def check(name: str, ok: bool, detail: str = "", fatal: bool = True) -> bool:
    mark = "PASS" if ok else ("FAIL" if fatal else "WARN")
    print(f"[{mark}] {name}" + (f"\n       {detail}" if detail else ""))
    if not ok:
        (_failures if fatal else _warnings).append(name)
    return ok


def main() -> int:
    print("=" * 70)
    print("phase 3 gate check: vllm_tie against stock vLLM")
    print("=" * 70)

    # -- 1. version ----------------------------------------------------
    try:
        import vllm
    except ImportError as exc:
        check("import vllm", False, f"{exc}\n       try: pip install vllm=={EXPECTED_VERSION}")
        return 1

    version = getattr(vllm, "__version__", "unknown")
    check(
        f"vLLM version is {EXPECTED_VERSION}",
        version == EXPECTED_VERSION,
        f"found {version}. SchedulerInterface is not a public API, so a "
        f"different version may have moved or renamed what we subclass.",
        fatal=False,
    )

    # -- 2. stock imports ----------------------------------------------
    try:
        from vllm.v1.core.sched.interface import SchedulerInterface
        from vllm.v1.core.sched.request_queue import FCFSRequestQueue, RequestQueue
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.request import Request  # noqa: F401
    except ImportError as exc:
        check("import vLLM scheduler internals", False, str(exc))
        return 1
    check("import Scheduler / RequestQueue / FCFSRequestQueue / Request", True)

    # -- 3. the replaceable seam ---------------------------------------
    try:
        src = inspect.getsource(Scheduler.__init__)
    except OSError as exc:
        src = ""
        check("read Scheduler.__init__ source", False, str(exc), fatal=False)

    check(
        "Scheduler.__init__ assigns self.waiting",
        "self.waiting" in src,
        "our subclass overrides exactly this attribute; if it is gone, the "
        "whole approach needs rethinking",
    )
    check(
        "self.waiting is built from a queue factory/class",
        "create_request_queue" in src or "RequestQueue(" in src,
        f"found: {[l.strip() for l in src.splitlines() if 'self.waiting' in l]}",
        fatal=False,
    )

    # -- 4. our queues satisfy RequestQueue -----------------------------
    try:
        from vllm_tie.request_queue import FCFSWithPredictionQueue, TIERequestQueue
    except ImportError as exc:
        check("import vllm_tie.request_queue", False, str(exc))
        return 1

    abstract = {
        n for n, v in vars(RequestQueue).items() if getattr(v, "__isabstractmethod__", False)
    }
    for cls in (TIERequestQueue, FCFSWithPredictionQueue):
        missing = sorted(m for m in abstract if getattr(cls, m, None) is getattr(RequestQueue, m, None))
        check(
            f"{cls.__name__} implements all RequestQueue abstract methods",
            not missing,
            f"missing: {missing}",
        )
        check(
            f"{cls.__name__} is a RequestQueue subclass",
            issubclass(cls, RequestQueue),
        )
    check(
        "FCFSWithPredictionQueue reuses vLLM's own FCFS ordering",
        issubclass(FCFSWithPredictionQueue, FCFSRequestQueue),
        "the control arm must order requests exactly as stock vLLM does, or "
        "it is not a control",
    )

    # -- 5. our schedulers resolve and conform --------------------------
    try:
        from vllm_tie.scheduler import FCFSWithPredictorScheduler, TIEScheduler
    except ImportError as exc:
        check("import vllm_tie.scheduler", False, str(exc))
        return 1

    for cls in (TIEScheduler, FCFSWithPredictorScheduler):
        check(f"{cls.__name__} subclasses stock Scheduler", issubclass(cls, Scheduler))
        check(
            f"{cls.__name__} satisfies SchedulerInterface",
            issubclass(cls, SchedulerInterface),
        )
        unimplemented = sorted(getattr(cls, "__abstractmethods__", set()))
        check(
            f"{cls.__name__} has no unimplemented abstract methods",
            not unimplemented,
            f"missing: {unimplemented}",
        )

    # The exact mechanism --scheduler-cls uses.
    try:
        from vllm.utils.import_utils import resolve_obj_by_qualname
    except ImportError:
        try:
            from vllm.utils import resolve_obj_by_qualname
        except ImportError as exc:
            check("import resolve_obj_by_qualname", False, str(exc), fatal=False)
            resolve_obj_by_qualname = None

    if resolve_obj_by_qualname is not None:
        for path, expected in (
            ("vllm_tie.scheduler.TIEScheduler", TIEScheduler),
            ("vllm_tie.scheduler.FCFSWithPredictorScheduler", FCFSWithPredictorScheduler),
        ):
            try:
                resolved = resolve_obj_by_qualname(path)
                check(f"--scheduler-cls {path} resolves", resolved is expected)
            except Exception as exc:  # noqa: BLE001
                check(f"--scheduler-cls {path} resolves", False, str(exc))

    # -- 6. checkpoint is reachable -------------------------------------
    from pathlib import Path

    model_dir = Path(os.environ.get("TIE_MODEL_DIR", "checkpoints/predictor_full"))
    for fname in ("best_model.pt", "normalize_stats.json"):
        path = model_dir / fname
        check(
            f"checkpoint file present: {path}",
            path.exists(),
            "set TIE_MODEL_DIR, or upload the phase 2 checkpoint (it is "
            "gitignored, ~707MB)",
            fatal=False,
        )

    print("=" * 70)
    if _failures:
        print(f"{len(_failures)} FAILED: {_failures}")
        return 1
    if _warnings:
        print(f"all critical checks passed; {len(_warnings)} warning(s): {_warnings}")
    else:
        print("all checks passed -- the external scheduler approach is viable.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
