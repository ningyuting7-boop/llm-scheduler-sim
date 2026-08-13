"""Plot: how many requests are completed within a fixed time window T, for
all four schedulers, at a fixed load (rps=10). Unlike the full-run
throughput (completions / total span until all requests finish, which is
dominated by total capacity and barely differs across schedulers -- see
fig_four_schedulers_throughput_vs_load.png), completions within an early,
fixed window directly reflect scheduling quality: a scheduler that clears
short requests faster completes more of them within the same window,
exactly mirroring the avg_wait/avg_response ranking. See
docs/Week2_3_Plan.md section 11 / conversation notes.
"""

from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.schedulers.fcfs import FCFSScheduler
from src.schedulers.oracle_sjf import OracleSJFScheduler
from src.schedulers.predicted_sjf import PredictedSJFScheduler
from src.schedulers.tie import TIEScheduler
from src.simulator import Simulator
from src.workload import generate_contaminated_workload, load_length_pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LMSYS_CSV = os.path.join(ROOT, "data", "lmsys_output_lengths.csv")
FIGURES_DIR = os.path.join(ROOT, "docs", "figures")
CSV_OUT = os.path.join(ROOT, "experiments", "four_schedulers_goodput.csv")

NUM_REQUESTS = 2000
RPS = 10.0
MAX_BATCH_SIZE = 8
DECODE_TIME_PER_STEP = 0.05
TAIL_RATE = 0.03
SIGMA_NORMAL = (0.0, 0.1)
SIGMA_TAIL = (0.8, 1.2)
BETA = 2.0
SEEDS = [42, 43, 44, 45, 46]
WINDOWS = [50, 100, 150, 200, 300, 400, 500]

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


def main() -> None:
    pool = load_length_pool(DEFAULT_LMSYS_CSV)

    rows = []
    for name in SCHEDULER_ORDER:
        spec = SCHEDULERS[name]
        window_completed = {w: [] for w in WINDOWS}
        for seed in SEEDS:
            requests = generate_contaminated_workload(
                num_requests=NUM_REQUESTS,
                arrival_rate=RPS,
                length_pool=pool,
                tail_rate=TAIL_RATE,
                sigma_normal=SIGMA_NORMAL,
                sigma_tail=SIGMA_TAIL,
                seed=seed,
            )
            Simulator(requests, scheduler=spec["factory"](), decode_time_per_step=DECODE_TIME_PER_STEP).run()
            finishes = sorted(r.finish_time for r in requests)
            for w in WINDOWS:
                completed = sum(1 for f in finishes if f <= w)
                window_completed[w].append(completed)

        for w in WINDOWS:
            vals = window_completed[w]
            mean_completed = sum(vals) / len(vals)
            rows.append({"scheduler": name, "window": w, "completed": mean_completed})
            print(f"{spec['label']:>15} T={w:>4} completed={mean_completed:.1f}")

    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scheduler", "window", "completed"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {CSV_OUT}")

    fig, ax = plt.subplots(figsize=(7, 5))
    _style_axes(ax)
    for name in SCHEDULER_ORDER:
        spec = SCHEDULERS[name]
        sub = [r for r in rows if r["scheduler"] == name]
        xs = [r["window"] for r in sub]
        ys = [r["completed"] for r in sub]
        ax.plot(xs, ys, color=spec["color"], marker=spec["marker"], markersize=6, linewidth=2, label=spec["label"])

    ax.set_xlabel("Window T (seconds since start)", color=INK_SECONDARY)
    ax.set_ylabel("Requests completed within T", color=INK_SECONDARY)
    ax.set_title(f"Completions within a fixed window vs. T (rps={RPS:.0f}, tail_rate={TAIL_RATE:.0%}, {len(SEEDS)}-seed avg)", color=INK_PRIMARY)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9)

    fig.patch.set_facecolor(SURFACE)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, "fig_four_schedulers_goodput.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
