"""Plot: TIE's avg-waiting-time advantage over Predicted SJF (as a % gap,
no aging, alpha=0) vs. beta, comparing rps=4.0 (heavily loaded) against
rps=1.0 (below this workload's ~1.34 req/sec capacity). One chart, two
lines, so the "beta matters a lot under load, barely at all when light"
contrast is directly visible. See docs/Week2_3_Plan.md section 13 /
conversation notes.
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
from src.schedulers.predicted_sjf import PredictedSJFScheduler
from src.schedulers.tie import TIEScheduler
from src.simulator import Simulator
from src.workload import generate_contaminated_workload, load_length_pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LMSYS_CSV = os.path.join(ROOT, "data", "lmsys_output_lengths.csv")
FIGURES_DIR = os.path.join(ROOT, "docs", "figures")
CSV_OUT = os.path.join(ROOT, "experiments", "tie_beta_gap_compare.csv")

NUM_REQUESTS = 2000
MAX_BATCH_SIZE = 8
DECODE_TIME_PER_STEP = 0.05
TAIL_RATE = 0.03
SIGMA_NORMAL = (0.0, 0.1)
SIGMA_TAIL = (0.8, 1.2)
SEEDS = [42, 43, 44, 45, 46]
BETAS = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
RPS_LEVELS = {4.0: "#eb6834", 1.0: "#1baf7a"}

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _run(rps, scheduler_factory, seed):
    pool = load_length_pool(DEFAULT_LMSYS_CSV)
    requests = generate_contaminated_workload(
        num_requests=NUM_REQUESTS,
        arrival_rate=rps,
        length_pool=pool,
        tail_rate=TAIL_RATE,
        sigma_normal=SIGMA_NORMAL,
        sigma_tail=SIGMA_TAIL,
        seed=seed,
    )
    Simulator(requests, scheduler=scheduler_factory(), decode_time_per_step=DECODE_TIME_PER_STEP).run()
    return compute_summary(requests).avg_waiting_time


def main() -> None:
    rows = []
    for rps in RPS_LEVELS:
        psjf_vals = [_run(rps, lambda: PredictedSJFScheduler(MAX_BATCH_SIZE), s) for s in SEEDS]
        psjf_mean = sum(psjf_vals) / len(psjf_vals)
        for beta in BETAS:
            vals = [_run(rps, lambda: TIEScheduler(MAX_BATCH_SIZE, beta=beta, alpha=0.0), s) for s in SEEDS]
            mean = sum(vals) / len(vals)
            gap_pct = 100 * (psjf_mean - mean) / psjf_mean
            rows.append({"rps": rps, "beta": beta, "gap_pct": gap_pct})
            print(f"rps={rps:>4.1f}  beta={beta:>5.1f}  gap%={gap_pct:6.2f}%")

    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rps", "beta", "gap_pct"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {CSV_OUT}")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    _style_axes(ax)
    for rps, color in RPS_LEVELS.items():
        sub = [r for r in rows if r["rps"] == rps]
        xs = [r["beta"] for r in sub]
        ys = [r["gap_pct"] for r in sub]
        ax.plot(xs, ys, color=color, marker="D", markersize=6, linewidth=2, label=f"rps={rps:.0f}")

    ax.axhline(0, color=INK_MUTED, linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("beta (log scale)", color=INK_SECONDARY)
    ax.set_ylabel("Gap vs Predicted SJF (%, + = TIE wins)", color=INK_SECONDARY)
    ax.set_title(f"TIE's avg-wait advantage over Predicted SJF vs. beta (no aging, tail_rate={TAIL_RATE:.0%})", color=INK_PRIMARY)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9)

    fig.patch.set_facecolor(SURFACE)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, "fig_tie_beta_gap_compare.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
