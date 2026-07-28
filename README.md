# llm-scheduler-sim

An event-driven simulator for comparing LLM inference request scheduling
policies (FCFS, and — starting Week 2 — Oracle SJF / Predicted SJF /
Priority). Week 1 delivers the simulator framework and an FCFS baseline.

## Directory structure

```
src/
  request.py          # Request data class + RequestStatus enum
  event.py             # Event data class + EventType enum (ARRIVAL/DEPARTURE)
  scheduler.py         # Scheduler abstract base class (waiting queue, batch capacity)
  schedulers/
    fcfs.py            # FCFSScheduler
  simulator.py         # Event-driven main loop (heapq-based)
  workload.py          # Synthetic request generator
  metrics.py           # Waiting/response time, throughput, fairness, CSV export
experiments/
  run_fcfs_baseline.py # End-to-end script: generate workload -> run FCFS -> report metrics
tests/
  test_simulator.py    # Regression tests (hand-verifiable small examples)
```

## Running

```bash
# from the llm-scheduler-sim/ directory
python3 -m unittest tests/test_simulator.py -v

python3 experiments/run_fcfs_baseline.py \
  --num-requests 300 --arrival-rate 3 --mean-output-len 40 \
  --max-batch-size 8 --decode-time-per-step 0.05
```

The experiment script prints a metrics summary and writes per-request
details to `experiments/fcfs_baseline.csv`.

## Design overview

**Request lifecycle**: `WAITING -> RUNNING -> FINISHED` (`src/request.py`).
`output_len` is the ground-truth number of decode steps; `remaining_len`
counts down as `step()` is called. `priority` and `predicted_output_len`
are unused by FCFS but reserved for Priority / Predicted SJF.

**Event-driven loop** (`src/simulator.py`): a min-heap of `Event`s ordered
by `(time, seq)` drives everything. There is no fixed time-step — the
simulator jumps straight to the next moment something actually changes.
`ARRIVAL` adds a request to the scheduler's waiting queue; `DEPARTURE`
advances a running request by one decode step, and on completion frees its
batch slot.

**Scheduler / batch model**: `Scheduler` (`src/scheduler.py`) owns the
waiting queue, the running set, and the `max_batch_size` capacity check —
this logic is identical across every algorithm. A concrete scheduler (e.g.
`FCFSScheduler`) only implements `_priority_key(request)`, the sort key
used to decide which waiting request to admit next when a batch slot frees
up. The batch itself is an **unordered set** of currently-running requests;
position within the batch has no effect on anything, since decode-step
cost is modeled as a constant independent of batch composition (see "What
is intentionally not modeled" below).

**Timing model**: only decode is modeled — each running request advances
one step per `decode_time_per_step` (a constant, configurable per run).
Prefill is not modeled; `arrival -> first decode step` has no separate
prefill delay. This keeps `prefill_time` from becoming a load-dependent
quantity, which would break the "compute a request's future event time
once, push it, done" pattern the event loop relies on.

## What is intentionally not modeled (Week 1 scope)

- **Prefill phase**: no `prompt_len`/prefill-time cost. Only decode steps
  count. Add this later only if a specific research question requires it
  (it changes the timing model, not just `Request`'s fields).
- **Static/synchronous batching**: batch slots free up individually as
  each request finishes (continuous batching), not as a whole batch. This
  means requests of very different `output_len` can share a batch with no
  penalty — there is no "bubble" cost to avoid, so schedulers never need
  to group by similar length.
- **Load-dependent step timing**: `decode_time_per_step` does not depend
  on current batch size/composition, so a request's remaining duration is
  always known in closed form the moment it starts running.
- **Aging / starvation prevention**: FCFS has no starvation risk by
  construction; algorithms that could starve (SJF, Priority) are a Week 2
  concern.

## Adding a new scheduling algorithm (Week 2)

Subclass `Scheduler` and implement `_priority_key`:

```python
from src.scheduler import Scheduler

class OracleSJFScheduler(Scheduler):
    def _priority_key(self, request):
        return (request.output_len, request.arrival_time, request.request_id)
```

No changes to `simulator.py` are needed — it only calls
`scheduler.add_request()` / `scheduler.schedule()` / `scheduler.notify_departure()`.

## Metrics (`src/metrics.py`)

`compute_summary(requests)` returns average/p99 waiting and response time,
throughput, and Jain's fairness index over waiting time (1.0 = perfectly
fair). `group_by(requests, key_fn)` produces the same summary per group
(e.g. `group_by(requests, lambda r: bucket_by_output_len(r))`) for
starvation/fairness analysis once Priority/SJF land. `write_csv` produces
one row per finished request for downstream plotting (Week 3).
