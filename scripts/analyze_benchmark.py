"""Turn a phase 3 benchmark run into the numbers the report needs.

Reads the per-(arm, rate) JSON files written by `vllm bench serve
--save-result` and reports TTFT, TPOT and throughput, then the four planned
contrasts (docs/Phase3_Scheduling_Evaluation_Plan.md section 5.1). Each
contrast changes exactly one thing:

    tie        - sjf          the CVaR term, i.e. the paper's contribution
    sjf        - fcfs_pred    shortest-job-first ordering, GPU load held equal
    fcfs_pred  - fcfs         what deploying the predictor costs
    tie_oracle - tie          headroom from a better sigma predictor

Reading the output:

  * TTFT is where scheduling shows up. It is queueing delay -- how long a
    request waited before the batch had room for it -- so a policy that
    serves short requests first should lower mean TTFT and raise the tail,
    because the long requests it defers are the ones that pay.
  * TPOT should barely move. Once a request is in the running batch it
    decodes at a rate set by batch size and the GPU, not by what put it
    there. A large TPOT gap between arms is more likely a load artefact
    than a scheduling result.
  * Throughput over a whole run is near-constant by construction: every arm
    processes the same 1000 prompts, so total token work is fixed and the
    GPU is the bottleneck. Differences show up in *when* requests finish,
    not how many -- so read throughput as a check that no arm broke, not as
    a result.

Usage:
    python scripts/analyze_benchmark.py results/bench_10330173
    python scripts/analyze_benchmark.py results/bench_10330173 --plot
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Display order and labels. Keys must match the arm names in
# hpc/benchmark_serving.slurm.
ARM_LABELS = {
    "fcfs": "FCFS (stock vLLM)",
    "fcfs_pred": "FCFS + Predictor",
    "sjf": "Predicted-SJF",
    "tie": "TIE",
    "tie_oracle": "TIE (oracle sigma)",
}
ARM_ORDER = ["fcfs", "fcfs_pred", "sjf", "tie", "tie_oracle"]

# (arm, baseline, what the difference isolates)
CONTRASTS = [
    ("tie", "sjf", "CVaR term (the paper's contribution)"),
    ("sjf", "fcfs_pred", "SJF ordering, GPU load held equal"),
    ("fcfs_pred", "fcfs", "cost of deploying the predictor"),
    ("tie_oracle", "tie", "headroom from a better sigma predictor"),
]

# Metrics carried through the summary, the contrasts and the plots.
METRICS = [
    ("mean_ttft_ms", "mean TTFT", "ms"),
    ("p90_ttft_ms", "P90 TTFT", "ms"),
    ("p99_ttft_ms", "P99 TTFT", "ms"),
    ("mean_tpot_ms", "mean TPOT", "ms"),
    ("p99_tpot_ms", "P99 TPOT", "ms"),
    ("mean_e2el_ms", "mean E2E", "ms"),
    ("req_throughput", "req/s", ""),
    ("tok_throughput", "tok/s", ""),
]

FILENAME_RE = re.compile(r"^(?P<arm>[a-z_]+)_rate(?P<rate>[0-9.]+)\.json$")


def load_run(path: Path) -> dict:
    with path.open() as f:
        raw = json.load(f)
    nan = float("nan")
    return {
        "completed": raw.get("completed", 0),
        "failed": raw.get("failed", 0),
        "duration_s": raw.get("duration", nan),
        "req_throughput": raw.get("request_throughput", nan),
        "tok_throughput": raw.get("output_throughput", nan),
        "mean_ttft_ms": raw.get("mean_ttft_ms", nan),
        "p50_ttft_ms": raw.get("p50_ttft_ms", raw.get("median_ttft_ms", nan)),
        "p90_ttft_ms": raw.get("p90_ttft_ms", nan),
        "p99_ttft_ms": raw.get("p99_ttft_ms", nan),
        "mean_tpot_ms": raw.get("mean_tpot_ms", nan),
        "p99_tpot_ms": raw.get("p99_tpot_ms", nan),
        "mean_e2el_ms": raw.get("mean_e2el_ms", nan),
        "p99_e2el_ms": raw.get("p99_e2el_ms", nan),
    }


def load_all(results_dir: Path) -> dict[tuple[str, float], dict]:
    runs: dict[tuple[str, float], dict] = {}
    for path in sorted(results_dir.glob("*.json")):
        m = FILENAME_RE.match(path.name)
        if not m:
            continue
        runs[(m["arm"], float(m["rate"]))] = load_run(path)
    return runs


def fmt(value, width: int = 9, prec: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return " " * (width - 1) + "-"
    return f"{value:{width}.{prec}f}"


def print_summary(runs: dict) -> None:
    arms = [a for a in ARM_ORDER if any(k[0] == a for k in runs)]
    rates = sorted({k[1] for k in runs})

    print("=" * 104)
    print("SUMMARY")
    print("=" * 104)
    header = (
        f"{'arm':<20}{'rate':>6}{'done':>6}{'fail':>5}"
        f"{'req/s':>8}{'tok/s':>9}"
        f"{'meanTTFT':>10}{'p50TTFT':>9}{'p99TTFT':>10}"
        f"{'meanTPOT':>10}{'p99TPOT':>9}{'meanE2E':>10}"
    )
    print(header)
    print("-" * len(header))
    for arm in arms:
        for rate in rates:
            r = runs.get((arm, rate))
            if r is None:
                continue
            print(
                f"{ARM_LABELS.get(arm, arm):<20}{rate:>6.0f}"
                f"{r['completed']:>6}{r['failed']:>5}"
                f"{fmt(r['req_throughput'], 8, 2)}{fmt(r['tok_throughput'], 9, 0)}"
                f"{fmt(r['mean_ttft_ms'], 10)}{fmt(r['p50_ttft_ms'], 9)}"
                f"{fmt(r['p99_ttft_ms'], 10)}"
                f"{fmt(r['mean_tpot_ms'], 10, 2)}{fmt(r['p99_tpot_ms'], 9, 2)}"
                f"{fmt(r['mean_e2el_ms'], 10)}"
            )
        print()


def print_contrasts(runs: dict) -> None:
    rates = sorted({k[1] for k in runs})
    print("=" * 104)
    print("CONTRASTS  (percent change vs the baseline arm)")
    print("  latency: negative is better.  throughput: positive is better.")
    print("=" * 104)

    # p50 comes first because under saturation it is the only TTFT statistic
    # that moves. Total queueing delay is close to conserved once throughput
    # is capped -- the area under the queue-length curve depends on arrivals
    # and service rate, not on the order requests are served in -- so
    # reordering redistributes waiting rather than removing it, leaving the
    # mean flat while the median and the tail pull apart. A table showing
    # only the mean would report a 7x improvement as "no effect".
    keys = [
        ("p50_ttft_ms", "p50TTFT"),
        ("mean_ttft_ms", "meanTTFT"),
        ("p90_ttft_ms", "p90TTFT"),
        ("p99_ttft_ms", "p99TTFT"),
        ("mean_tpot_ms", "meanTPOT"),
        ("req_throughput", "req/s"),
    ]

    for arm, baseline, what in CONTRASTS:
        shared = [r for r in rates if (arm, r) in runs and (baseline, r) in runs]
        if not shared:
            continue
        print(f"\n{ARM_LABELS.get(arm, arm)}  vs  {ARM_LABELS.get(baseline, baseline)}")
        print(f"  isolates: {what}")
        header = "  " + f"{'rate':>6}" + "".join(f"{label:>12}" for _, label in keys)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for rate in shared:
            a, b = runs[(arm, rate)], runs[(baseline, rate)]
            cells = []
            for key, _ in keys:
                x, y = a[key], b[key]
                if not (np.isfinite(x) and np.isfinite(y)) or y == 0:
                    cells.append(f"{'-':>12}")
                else:
                    cells.append(f"{100.0 * (x - y) / y:+11.1f}%")
            print(f"  {rate:>6.0f}" + "".join(cells))


def print_saturation_note(runs: dict) -> None:
    """Which rates actually queued?

    Scheduling order can only matter once the running batch is full; below
    that a request is admitted on arrival and every arm behaves identically.
    A near-zero TTFT is the signature of that regime, so flagging it keeps
    'no difference between arms' at low rates from being read as a finding.
    """
    rates = sorted({k[1] for k in runs})
    ref = "fcfs_pred" if any(k[0] == "fcfs_pred" for k in runs) else ARM_ORDER[0]
    print("\n" + "=" * 104)
    print(f"SATURATION (from the {ARM_LABELS.get(ref, ref)} arm)")
    print("=" * 104)
    queued, idle = [], []
    for rate in rates:
        r = runs.get((ref, rate))
        if r is None or not np.isfinite(r["p99_ttft_ms"]):
            continue
        (queued if r["p99_ttft_ms"] > 500 else idle).append(rate)
        print(
            f"  rate {rate:>4.0f}: p99 TTFT {fmt(r['p99_ttft_ms'], 9)} ms, "
            f"throughput {fmt(r['req_throughput'], 6, 2)} req/s"
            + ("   <- queueing" if r["p99_ttft_ms"] > 500 else "   <- no queueing")
        )
    print(f"\n  rates where scheduling could matter: {queued or 'none'}")
    if idle:
        print(
            f"  rates with an empty waiting queue: {idle}\n"
            f"    arms are expected to be identical here; that is the control, not a null result"
        )


def make_plots(runs: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    rates = sorted({k[1] for k in runs})
    arms = [a for a in ARM_ORDER if any(k[0] == a for k in runs)]
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = [
        ("mean_ttft_ms", "Mean TTFT (ms)", "mean_ttft", True),
        ("p50_ttft_ms", "Median TTFT (ms)", "p50_ttft", True),
        ("p99_ttft_ms", "P99 TTFT (ms)", "p99_ttft", True),
        ("mean_tpot_ms", "Mean TPOT (ms)", "mean_tpot", False),
        ("req_throughput", "Request throughput (req/s)", "req_throughput", False),
        ("tok_throughput", "Output token throughput (tok/s)", "tok_throughput", False),
    ]
    for key, ylabel, stem, log_y in panels:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for arm in arms:
            xs = [r for r in rates if (arm, r) in runs]
            ys = [runs[(arm, r)][key] for r in xs]
            ax.plot(xs, ys, marker="o", linewidth=2, label=ARM_LABELS.get(arm, arm))
        ax.set_xlabel("Request rate (req/s)")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log", base=2)
        ax.set_xticks(rates)
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        if log_y:
            ax.set_yscale("log")
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        path = out_dir / f"phase3_{stem}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", help="e.g. results/bench_10330173")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-dir", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    runs = load_all(results_dir)
    if not runs:
        raise SystemExit(f"no benchmark JSON files found in {results_dir}")

    failed = {k: r["failed"] for k, r in runs.items() if r["failed"]}
    if failed:
        print(f"WARNING: runs with failed requests: {failed}\n")

    print_summary(runs)
    print_contrasts(runs)
    print_saturation_note(runs)

    if args.plot:
        make_plots(runs, Path(args.plot_dir or results_dir / "figures"))


if __name__ == "__main__":
    main()
