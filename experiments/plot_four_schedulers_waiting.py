"""Plot: average waiting time for all four schedulers (FCFS, Oracle SJF,
Predicted SJF, TIEScheduler) on the same contaminated workload (tail_rate=3%,
sigma_normal=0.2, sigma_tail=0.8 -- the Experiment A/B default). 5-seed
average. See docs/Week2_3_Plan.md section 11.
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
CSV_OUT = os.path.join(ROOT, "experiments", "four_schedulers_waiting.csv")

NUM_REQUESTS = 2000
ARRIVAL_RATE = 5.0
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
    "fcfs": {"label": "FCFS", "color": "#2a78d6", "factory": lambda: FCFSScheduler(MAX_BATCH_SIZE)},
    "oracle_sjf": {"label": "Oracle SJF", "color": "#eb6834", "factory": lambda: OracleSJFScheduler(MAX_BATCH_SIZE)},
    "predicted_sjf": {"label": "Predicted SJF", "color": "#1baf7a", "factory": lambda: PredictedSJFScheduler(MAX_BATCH_SIZE)},
    "tie": {"label": "TIEScheduler", "color": "#eda100", "factory": lambda: TIEScheduler(MAX_BATCH_SIZE, beta=BETA)},
}


def main() -> None:
    pool = load_length_pool(DEFAULT_LMSYS_CSV)

    results = {}
    for name, spec in SCHEDULERS.items():
        waits = []
        for seed in SEEDS:
            requests = generate_contaminated_workload(
                num_requests=NUM_REQUESTS,
                arrival_rate=ARRIVAL_RATE,
                length_pool=pool,
                tail_rate=TAIL_RATE,
                sigma_normal=SIGMA_NORMAL,
                sigma_tail=SIGMA_TAIL,
                seed=seed,
            )
            Simulator(requests, scheduler=spec["factory"](), decode_time_per_step=DECODE_TIME_PER_STEP).run()
            waits.append(compute_summary(requests).avg_waiting_time)
        results[name] = sum(waits) / len(waits)
        print(f"{spec['label']:>15}: avg_waiting_time={results[name]:.3f}")

    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scheduler", "avg_waiting_time"])
        for name in SCHEDULERS:
            writer.writerow([name, results[name]])
    print(f"Wrote {CSV_OUT}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(True, axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    names = list(SCHEDULERS.keys())
    labels = [SCHEDULERS[n]["label"] for n in names]
    colors = [SCHEDULERS[n]["color"] for n in names]
    values = [results[n] for n in names]

    bars = ax.bar(labels, values, color=colors, width=0.55, zorder=3)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
            textcoords="offset points", ha="center", va="bottom", fontsize=10, color=INK_SECONDARY,
        )

    ax.set_ylabel("Avg waiting time", color=INK_SECONDARY)
    ax.set_title(
        f"Avg waiting time by scheduler (tail_rate={TAIL_RATE:.0%}, sigma_tail={SIGMA_TAIL}, {len(SEEDS)}-seed avg)",
        color=INK_PRIMARY,
    )
    ax.tick_params(axis="x", labelsize=10, colors=INK_SECONDARY)

    fig.patch.set_facecolor(SURFACE)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, "fig_four_schedulers_waiting.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
