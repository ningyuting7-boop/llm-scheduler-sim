# llm-scheduler-sim

An event-driven simulator for comparing LLM inference request scheduling
policies under **prediction uncertainty**. It compares FCFS, Oracle SJF
(perfect-knowledge upper bound), Predicted SJF (point-estimate only), and
**TIEScheduler** — a risk-aware scheduler adapted from Zheng et al. 2026's
*Tail Inflated Expectation* scheduling rule (arXiv:2604.00499), combined
with a classical aging term for starvation control. See
[`docs/Report.md`](docs/Report.md) for the full write-up, design rationale,
and results.

## Setup

```bash
pip install matplotlib scipy numpy datasets
```

`datasets` (HuggingFace) is only needed to re-run the LMSYS extraction
script (`scripts/extract_lmsys_lengths.py`); `data/lmsys_output_lengths.csv`
is already committed, so everything else runs without it. LMSYS-Chat-1M is
a gated dataset — extraction requires `huggingface-cli login` first.

## Directory structure

```
src/
  request.py            # Request data class + RequestStatus enum
  event.py               # Event data class + EventType enum (ARRIVAL/DEPARTURE)
  scheduler.py           # Scheduler abstract base class (waiting queue, batch capacity)
  schedulers/
    fcfs.py              # FCFSScheduler
    oracle_sjf.py         # OracleSJFScheduler (sorts by true output_len)
    predicted_sjf.py       # PredictedSJFScheduler (sorts by predicted_output_len)
    tie.py                # TIEScheduler: E[X] + beta*CVaR_0.9[X] - alpha*waiting_time
  simulator.py            # Event-driven main loop (heapq-based)
  predictor.py            # Synthetic predictor-error models (calibrated to ELIS paper stats)
  workload.py             # Workload generators (Poisson arrivals; real-LMSYS and contaminated variants)
  metrics.py              # Waiting/response time, throughput, fairness (Jain's index), CSV export
scripts/
  extract_lmsys_lengths.py  # LMSYS-Chat-1M -> data/lmsys_output_lengths.csv (word-count lengths)
data/
  lmsys_output_lengths.csv  # 40,395 real response lengths, already extracted
experiments/
  *.py                   # Experiment/plotting scripts -- see "Reproducing the report figures" below
tests/
  test_simulator.py       # Core event loop / metrics regression tests
  test_schedulers.py      # FCFS / Oracle SJF / Predicted SJF ordering + starvation tests
docs/
  Report.md              # Full project report (background, method, experiments, results, limitations)
  figures/                # Figures referenced by Report.md
```

## Design overview

**Request lifecycle**: `WAITING -> RUNNING -> FINISHED` (`src/request.py`).
`output_len` is the ground-truth number of decode steps; `remaining_len`
counts down as `step()` is called. `predicted_output_len` /
`prediction_uncertainty` / `expected_length` / `tail_risk` hold whatever a
given scheduler needs to rank requests (unused fields are simply `None`).

**Event-driven loop** (`src/simulator.py`): a min-heap of `Event`s ordered
by `(time, seq)` drives everything — there is no fixed time-step. `ARRIVAL`
adds a request to the scheduler's waiting queue; `DEPARTURE` advances a
running request by one decode step, and on completion frees its batch slot.
Same-timestamp events are batched before admission decisions are made, so
simultaneous arrivals are scheduled correctly relative to each other.

**Scheduler / batch model** (`src/scheduler.py`): every scheduler shares
the same waiting queue, running set, and `max_batch_size` capacity check; a
concrete scheduler only implements `_priority_key(request, current_time)`,
the sort key used to pick the next request to admit when a slot frees up
(**smallest key wins**). Batching is continuous (slots free up
individually, not as a whole batch); only the decode phase is modeled
(`decode_time_per_step`), not prefill or KV-cache overhead — see
`docs/Report.md` Limitations.

**TIEScheduler's score** (`src/schedulers/tie.py`):

```
score = expected_length + beta * tail_risk - alpha * waiting_time
```

`expected_length`/`tail_risk` (`E[X]`/`CVaR_0.9[X]` of a per-request
log-normal fit) come from `workload.generate_contaminated_workload`, not
from a real trained predictor — see `docs/Report.md` Section 3.2 for how
the error model is calibrated against real predictor accuracy numbers from
the ELIS paper (arXiv:2505.09142). `alpha` defaults to `0.0` (no aging);
`alpha > 0` adds starvation control at the cost of some of `beta`'s benefit
(see Report.md Section 4.4 for the tradeoff).

## Reproducing the report figures

`docs/Report.md` cites 8 figures. Each is produced by one experiment script
(all read `data/lmsys_output_lengths.csv` directly, no other setup needed):

| Figure | Script |
|---|---|
| `lmsys_length_distribution.png` | `experiments/plotting.py` (`figure1_length_distribution`) |
| `fig_four_schedulers_wait_vs_load.png` | `experiments/plot_four_schedulers_wait_vs_load.py` |
| `fig_four_schedulers_response_vs_load.png` | `experiments/plot_four_schedulers_wait_vs_load.py` (same run) |
| `fig_four_schedulers_throughput.png` | `experiments/plot_four_schedulers_wait_vs_load.py` (same run, saved as `fig_four_schedulers_throughput_vs_load.png` -- rename after running) |
| `fig_four_schedulers_p99_wait.png` | `experiments/plot_four_schedulers_p99.py` |
| `fig_four_schedulers_jain_fairness_index.png` | `experiments/plot_four_schedulers_fairness.py` (saved as `fig_four_schedulers_jain.png` -- rename after running) |
| `fig_tie_alpha_curve_rps2.png` | `experiments/plot_tie_alpha_curve_rps2.py` |
| `fig_tie_beta_gap_compare.png` | `experiments/plot_tie_beta_gap_compare.py` |

```bash
python3 experiments/plot_four_schedulers_wait_vs_load.py   # wait/response/throughput vs load
python3 experiments/plot_four_schedulers_p99.py             # P99 tail latency vs load
python3 experiments/plot_four_schedulers_fairness.py        # Jain's fairness index vs load
python3 experiments/plot_tie_alpha_curve_rps2.py             # aging-term (alpha) tradeoff at rps=2
python3 experiments/plot_tie_beta_gap_compare.py             # risk-hedging weight (beta) sensitivity
```

**Note:** `experiments/` also contains a number of exploratory scripts from
earlier stages of this project (e.g. alternate load points, alternate
alpha/beta sweeps, an earlier ARRS-era pipeline) that are *not* cited in the
final report. They were kept rather than deleted so the analysis trail is
reproducible, but `docs/Report.md`'s figure list above is the authoritative
"what actually matters" reference — if a script isn't in that table, its
output isn't part of the reported results.

## Testing

```bash
python3 -m unittest discover -s tests -v
```
