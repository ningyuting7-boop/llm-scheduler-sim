"""Experiment + plot: does adding an aging term to TIEScheduler
(score = E[X] + beta*CVaR90 - alpha*waiting_steps) improve fairness without
giving up too much of TIE's average-latency advantage over Predicted SJF?

Sweeps alpha at fixed beta=2.0, fixed load (rps=10), fixed tail_rate=3%.
For each alpha, reports:
  - Jain's Fairness Index (higher = more equal; TIE at alpha=0 is LESS fair
    than Predicted SJF, see docs/Week2_3_Plan.md section 13 -- this sweep
    asks whether aging can close that gap)
  - the avg_wait gap vs Predicted SJF (how much of TIE's original advantage
    survives)

See docs/Week2_3_Plan.md section 11 / conversation notes.
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
CSV_OUT = os.path.join(ROOT, "experiments", "tie_aging_tradeoff.csv")

NUM_REQUESTS = 2000
RPS = 10.0
MAX_BATCH_SIZE = 8
DECODE_TIME_PER_STEP = 0.05
TAIL_RATE = 0.03
SIGMA_NORMAL = (0.0, 0.1)
SIGMA_TAIL = (0.8, 1.2)
BETA = 2.0
SEEDS = [42, 43, 44, 45, 46]
ALPHAS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

COLOR_FAIRNESS = "#4a3aa7"
COLOR_GAP = "#eda100"


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _run(scheduler_factory, seed):
    pool = load_length_pool(DEFAULT_LMSYS_CSV)
    requests = generate_contaminated_workload(
        num_requests=NUM_REQUESTS,
        arrival_rate=RPS,
        length_pool=pool,
        tail_rate=TAIL_RATE,
        sigma_normal=SIGMA_NORMAL,
        sigma_tail=SIGMA_TAIL,
        seed=seed,
    )
    Simulator(requests, scheduler=scheduler_factory(), decode_time_per_step=DECODE_TIME_PER_STEP).run()
    return compute_summary(requests)


def main() -> None:
    psjf_waits, psjf_jains = [], []
    for seed in SEEDS:
        summary = _run(lambda: PredictedSJFScheduler(MAX_BATCH_SIZE), seed)
        psjf_waits.append(summary.avg_waiting_time)
        psjf_jains.append(summary.fairness_jain_index)
    psjf_wait = sum(psjf_waits) / len(psjf_waits)
    psjf_jain = sum(psjf_jains) / len(psjf_jains)
    print(f"{'Predicted SJF':>15}            avg_wait={psjf_wait:8.3f}  jain={psjf_jain:.4f}")

    rows = []
    for alpha in ALPHAS:
        waits, jains = [], []
        for seed in SEEDS:
            summary = _run(
                lambda: TIEScheduler(MAX_BATCH_SIZE, beta=BETA, alpha=alpha, decode_time_per_step=DECODE_TIME_PER_STEP),
                seed,
            )
            waits.append(summary.avg_waiting_time)
            jains.append(summary.fairness_jain_index)
        mean_wait = sum(waits) / len(waits)
        mean_jain = sum(jains) / len(jains)
        gap = psjf_wait - mean_wait
        rows.append({"alpha": alpha, "avg_wait": mean_wait, "jain": mean_jain, "gap_vs_psjf": gap})
        print(f"alpha={alpha:>5.1f}  avg_wait={mean_wait:8.3f}  jain={mean_jain:.4f}  gap_vs_psjf={gap:7.3f}")

    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "avg_wait", "jain", "gap_vs_psjf"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {CSV_OUT}")

    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    _style_axes(ax1)
    xs = [r["alpha"] for r in rows]
    jain_ys = [r["jain"] for r in rows]
    gap_ys = [r["gap_vs_psjf"] for r in rows]

    ax1.plot(xs, jain_ys, color=COLOR_FAIRNESS, marker="o", markersize=6, linewidth=2, label="Jain's Fairness Index (left)")
    ax1.axhline(psjf_jain, color=COLOR_FAIRNESS, linewidth=1, linestyle="--")
    ax1.text(xs[-1], psjf_jain, " Predicted SJF's Jain", color=COLOR_FAIRNESS, fontsize=8, va="bottom", ha="right")
    ax1.set_xlabel("alpha (aging strength)", color=INK_SECONDARY)
    ax1.set_ylabel("Jain's Fairness Index (higher = more equal)", color=COLOR_FAIRNESS)
    ax1.set_xscale("symlog", linthresh=0.1)

    ax2 = ax1.twinx()
    ax2.set_facecolor(SURFACE)
    for spine in ("top", "left"):
        ax2.spines[spine].set_visible(False)
    ax2.spines["right"].set_color(AXIS)
    ax2.tick_params(colors=INK_MUTED, labelsize=9)
    ax2.axhline(0, color=INK_MUTED, linewidth=1, linestyle=":")
    ax2.plot(xs, gap_ys, color=COLOR_GAP, marker="D", markersize=6, linewidth=2, label="Gap vs Predicted SJF (right, + = TIE still wins)")
    ax2.set_ylabel("Avg_wait gap: Predicted SJF - TIE (+ = TIE still wins)", color=COLOR_GAP)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, labelcolor=INK_SECONDARY, fontsize=8, loc="center right")

    ax1.set_title(
        f"TIE aging tradeoff: fairness vs. average-latency advantage (beta={BETA}, rps={RPS:.0f}, tail_rate={TAIL_RATE:.0%})",
        color=INK_PRIMARY,
    )

    fig.patch.set_facecolor(SURFACE)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, "fig_tie_aging_tradeoff.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
