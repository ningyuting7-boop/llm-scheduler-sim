"""Plot: avg waiting time AND avg response time vs. load (arrival rate),
same style as Figure 2 (experiments/plotting.py), for all four schedulers
(FCFS, Oracle SJF, Predicted SJF, TIEScheduler) on the contaminated
workload. 5-seed average per point. See docs/Week2_3_Plan.md section 11.
"""

from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.metrics import compute_summary
from src.schedulers.fcfs import FCFSScheduler
from src.schedulers.oracle_sjf import OracleSJFScheduler
from src.schedulers.predicted_sjf import PredictedSJFScheduler
from src.schedulers.tie import TIEScheduler
from src.simulator import Simulator
from src.workload import generate_contaminated_workload, load_length_pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LMSYS_CSV = os.path.join(ROOT, "data", "lmsys_output_lengths.csv")
FIGURES_DIR = os.path.join(ROOT, "docs", "figures")
CSV_OUT = os.path.join(ROOT, "experiments", "four_schedulers_wait_vs_load.csv")

NUM_REQUESTS = 2000
LOAD_LEVELS = [2.0, 5.0, 10.0, 15.0, 20.0]
MAX_BATCH_SIZE = 8
DECODE_TIME_PER_STEP = 0.05
TAIL_RATE = 0.03
SIGMA_NORMAL = (0.0, 0.1)
SIGMA_TAIL = (0.8, 1.2)
BETA = 2.0
SEEDS = [42, 43, 44, 45, 46]

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

SCHEDULERS = {
    "fcfs": {"label": "FCFS", "color": "#2a78d6", "marker": "o", "factory": lambda: FCFSScheduler(MAX_BATCH_SIZE)},
    "oracle_sjf": {"label": "Oracle SJF", "color": "#eb6834", "marker": "s", "factory": lambda: OracleSJFScheduler(MAX_BATCH_SIZE)},
    "predicted_sjf": {"label": "Predicted SJF", "color": "#1baf7a", "marker": "^", "factory": lambda: PredictedSJFScheduler(MAX_BATCH_SIZE)},
    "tie": {"label": "TIEScheduler", "color": "#eda100", "marker": "D", "factory": lambda: TIEScheduler(MAX_BATCH_SIZE, beta=BETA)},
}
SCHEDULER_ORDER = ["fcfs", "oracle_sjf", "predicted_sjf", "tie"]


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _plot_metric(rows, metric_col: str, ylabel: str, title: str, out_name: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    _style_axes(ax)
    for name in SCHEDULER_ORDER:
        spec = SCHEDULERS[name]
        sub = [r for r in rows if r["scheduler"] == name]
        xs = [r["rps"] for r in sub]
        ys = [r[metric_col] for r in sub]
        ax.plot(xs, ys, color=spec["color"], marker=spec["marker"], markersize=6, linewidth=2, label=spec["label"])

    ax.set_xlabel("Arrival rate (requests/sec)", color=INK_SECONDARY)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.set_title(title, color=INK_PRIMARY)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9)

    fig.patch.set_facecolor(SURFACE)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, out_name)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    pool = load_length_pool(DEFAULT_LMSYS_CSV)

    rows = []
    for name in SCHEDULER_ORDER:
        spec = SCHEDULERS[name]
        for rps in LOAD_LEVELS:
            waits, responses, throughputs = [], [], []
            for seed in SEEDS:
                requests = generate_contaminated_workload(
                    num_requests=NUM_REQUESTS,
                    arrival_rate=rps,
                    length_pool=pool,
                    tail_rate=TAIL_RATE,
                    sigma_normal=SIGMA_NORMAL,
                    sigma_tail=SIGMA_TAIL,
                    seed=seed,
                )
                Simulator(requests, scheduler=spec["factory"](), decode_time_per_step=DECODE_TIME_PER_STEP).run()
                summary = compute_summary(requests)
                waits.append(summary.avg_waiting_time)
                responses.append(summary.avg_response_time)
                throughputs.append(summary.throughput)
            mean_wait = sum(waits) / len(waits)
            mean_response = sum(responses) / len(responses)
            mean_throughput = sum(throughputs) / len(throughputs)
            rows.append({
                "scheduler": name, "rps": rps,
                "avg_wait": mean_wait, "avg_response": mean_response, "throughput": mean_throughput,
            })
            print(f"{spec['label']:>15} rps={rps:>5} avg_wait={mean_wait:.3f} avg_response={mean_response:.3f} throughput={mean_throughput:.3f}")

    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scheduler", "rps", "avg_wait", "avg_response", "throughput"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {CSV_OUT}")

    suffix = f"(tail_rate={TAIL_RATE:.0%}, sigma_tail={SIGMA_TAIL}, {len(SEEDS)}-seed avg)"
    _plot_metric(rows, "avg_wait", "Avg waiting time", f"Avg waiting time vs. load {suffix}", "fig_four_schedulers_wait_vs_load.png")
    _plot_metric(rows, "avg_response", "Avg response time", f"Avg response time vs. load {suffix}", "fig_four_schedulers_response_vs_load.png")
    _plot_metric(rows, "throughput", "Throughput (req/sec)", f"Throughput vs. load {suffix}", "fig_four_schedulers_throughput_vs_load.png")


if __name__ == "__main__":
    main()
