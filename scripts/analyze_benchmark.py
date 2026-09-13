"""Turn a phase 3 benchmark run into the numbers the report needs.

Reads the per-(arm, rate) JSON files written by `vllm bench serve
--save-result` and produces three things:

1. A summary table of latency, throughput and fairness per arm and rate.
2. The four planned contrasts (docs/Phase3_Scheduling_Evaluation_Plan.md
   section 5.1), each of which isolates one variable:

       tie  - sjf        the CVaR term, i.e. the paper's contribution
       sjf  - fcfs_pred  shortest-job-first ordering, GPU load held equal
       fcfs_pred - fcfs  what deploying the predictor costs
       tie_oracle - tie  how much a better sigma predictor could buy

3. A mechanism check: the rank correlation between a request's output
   length and its TTFT. Aggregate latency can move for all sorts of
   reasons, but if length-aware scheduling is working at all then short
   requests must be getting served first, and this measures that directly.
   Under FCFS it should sit near zero by construction, which also serves as
   a sanity check that the arms really are running different policies.

Two metrics are computed here rather than taken from the JSON:

  * Jain's fairness index over TTFT, which the benchmark does not report.
    J = (sum x)^2 / (n * sum x^2): 1.0 when every request waits equally,
    1/n when one request absorbs all the waiting.
  * The length/TTFT correlation, which needs the raw per-request arrays.

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

# (better, baseline, what the difference isolates)
CONTRASTS = [
    ("tie", "sjf", "CVaR term (the paper's contribution)"),
    ("sjf", "fcfs_pred", "SJF ordering, GPU load held equal"),
    ("fcfs_pred", "fcfs", "cost of deploying the predictor"),
    ("tie_oracle", "tie", "headroom from a better sigma predictor"),
]

FILENAME_RE = re.compile(r"^(?P<arm>[a-z_]+)_rate(?P<rate>[0-9.]+)\.json$")


def jains_index(values: np.ndarray) -> float:
    """1.0 = every request waited the same; 1/n = one request absorbed it all."""
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.any(values > 0):
        return float("nan")
    return float(values.sum() ** 2 / (values.size * np.square(values).sum()))


def length_ttft_correlation(output_lens, ttfts) -> float:
    """Spearman rho between output length and TTFT.

    Positive means longer requests waited longer, which is what
    length-aware scheduling is supposed to produce. Near zero is the FCFS
    signature: arrival order carries no information about length.
    """
    from scipy.stats import spearmanr

    lens = np.asarray(output_lens, dtype=float)
    ttfts = np.asarray(ttfts, dtype=float)
    ok = np.isfinite(lens) & np.isfinite(ttfts)
    if ok.sum() < 10:
        return float("nan")
    return float(spearmanr(lens[ok], ttfts[ok]).correlation)


def load_run(path: Path) -> dict:
    with path.open() as f:
        raw = json.load(f)

    ttfts_s = np.asarray(raw.get("ttfts", []), dtype=float)
    ttfts_ms = ttfts_s * 1000.0
    out_lens = raw.get("output_lens", [])

    return {
        "completed": raw.get("completed", 0),
        "failed": raw.get("failed", 0),
        "duration_s": raw.get("duration", float("nan")),
        "req_throughput": raw.get("request_throughput", float("nan")),
        "tok_throughput": raw.get("output_throughput", float("nan")),
        "mean_ttft_ms": raw.get("mean_ttft_ms", float("nan")),
        "p50_ttft_ms": raw.get("p50_ttft_ms", raw.get("median_ttft_ms", float("nan"))),
        "p99_ttft_ms": raw.get("p99_ttft_ms", float("nan")),
        "mean_e2el_ms": raw.get("mean_e2el_ms", float("nan")),
        "p99_e2el_ms": raw.get("p99_e2el_ms", float("nan")),
        "jain_ttft": jains_index(ttfts_ms),
        "len_ttft_rho": length_ttft_correlation(out_lens, ttfts_ms),
        "mean_output_len": float(np.mean(out_lens)) if len(out_lens) else float("nan"),
    }


def load_all(results_dir: Path) -> dict[tuple[str, float], dict]:
    runs: dict[tuple[str, float], dict] = {}
    for path in sorted(results_dir.glob("*.json")):
        m = FILENAME_RE.match(path.name)
        if not m:
            continue
        runs[(m["arm"], float(m["rate"]))] = load_run(path)
    return runs


def fmt(value: float, width: int = 9, prec: int = 1) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return " " * (width - 1) + "-"
    return f"{value:{width}.{prec}f}"


def print_summary(runs: dict) -> None:
    arms = [a for a in ARM_ORDER if any(k[0] == a for k in runs)]
    rates = sorted({k[1] for k in runs})

    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    header = (
        f"{'arm':<20}{'rate':>6}{'done':>6}{'fail':>5}"
        f"{'thru/s':>9}{'meanTTFT':>10}{'p50TTFT':>9}{'p99TTFT':>10}"
        f"{'p99E2E':>10}{'Jain':>7}{'len~TTFT':>9}"
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
                f"{fmt(r['req_throughput'], 9, 2)}"
                f"{fmt(r['mean_ttft_ms'], 10)}"
                f"{fmt(r['p50_ttft_ms'], 9)}"
                f"{fmt(r['p99_ttft_ms'], 10)}"
                f"{fmt(r['p99_e2el_ms'], 10)}"
                f"{fmt(r['jain_ttft'], 7, 3)}"
                f"{fmt(r['len_ttft_rho'], 9, 3)}"
            )
        print()


def print_contrasts(runs: dict) -> None:
    rates = sorted({k[1] for k in runs})
    print("=" * 100)
    print("CONTRASTS  (negative = the first arm is faster; % of the baseline)")
    print("=" * 100)

    for better, baseline, what in CONTRASTS:
        pairs = [r for r in rates if (better, r) in runs and (baseline, r) in runs]
        if not pairs:
            continue
        print(f"\n{ARM_LABELS.get(better, better)}  vs  {ARM_LABELS.get(baseline, baseline)}")
        print(f"  isolates: {what}")
        header = f"  {'rate':>6}{'meanTTFT':>12}{'p99TTFT':>12}{'p99E2E':>12}{'Jain':>12}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for rate in pairs:
            a, b = runs[(better, rate)], runs[(baseline, rate)]

            def pct(key: str) -> str:
                x, y = a[key], b[key]
                if not (np.isfinite(x) and np.isfinite(y)) or y == 0:
                    return "         -"
                return f"{100.0 * (x - y) / y:+11.1f}%"

            print(
                f"  {rate:>6.0f}{pct('mean_ttft_ms')}{pct('p99_ttft_ms')}"
                f"{pct('p99_e2el_ms')}{pct('jain_ttft')}"
            )


def print_mechanism_check(runs: dict) -> None:
    """Did the arms actually schedule differently?

    Aggregate latencies can coincide for uninteresting reasons. A
    length-aware policy must show longer requests waiting longer; FCFS
    cannot, because arrival order is independent of length. If the
    length-aware arms do not separate from the FCFS arms here, the run did
    not test what it was meant to -- most likely the queue never had depth.
    """
    rates = sorted({k[1] for k in runs})
    arms = [a for a in ARM_ORDER if any(k[0] == a for k in runs)]

    print("\n" + "=" * 100)
    print("MECHANISM CHECK: Spearman(output length, TTFT)")
    print("  length-aware arms should be clearly positive; FCFS arms near zero")
    print("=" * 100)
    header = f"{'arm':<20}" + "".join(f"{r:>9.0f}" for r in rates)
    print(header)
    print("-" * len(header))
    for arm in arms:
        row = "".join(
            fmt(runs[(arm, r)]["len_ttft_rho"], 9, 3) if (arm, r) in runs else " " * 9
            for r in rates
        )
        print(f"{ARM_LABELS.get(arm, arm):<20}{row}")

    saturated = [
        r
        for r in rates
        if ("fcfs_pred", r) in runs
        and np.isfinite(runs[("fcfs_pred", r)]["p99_ttft_ms"])
        and runs[("fcfs_pred", r)]["p99_ttft_ms"] > 1000
    ]
    print(
        f"\nrates where the queue clearly had depth (baseline p99 TTFT > 1s): "
        f"{saturated if saturated else 'none -- the run never saturated'}"
    )


def make_plots(runs: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rates = sorted({k[1] for k in runs})
    arms = [a for a in ARM_ORDER if any(k[0] == a for k in runs)]
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = [
        ("mean_ttft_ms", "Mean TTFT (ms)", "mean_ttft"),
        ("p99_ttft_ms", "P99 TTFT (ms)", "p99_ttft"),
        ("p99_e2el_ms", "P99 end-to-end latency (ms)", "p99_e2el"),
        ("jain_ttft", "Jain fairness index over TTFT", "jain"),
    ]
    for key, ylabel, stem in panels:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for arm in arms:
            xs = [r for r in rates if (arm, r) in runs]
            ys = [runs[(arm, r)][key] for r in xs]
            ax.plot(xs, ys, marker="o", linewidth=2, label=ARM_LABELS.get(arm, arm))
        ax.set_xlabel("Request rate (req/s)")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log", base=2)
        ax.set_xticks(rates)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        if key != "jain_ttft":
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
    print_mechanism_check(runs)

    if args.plot:
        make_plots(runs, Path(args.plot_dir or results_dir / "figures"))


if __name__ == "__main__":
    main()
