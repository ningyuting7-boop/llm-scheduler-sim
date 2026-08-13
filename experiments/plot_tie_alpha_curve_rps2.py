"""Plot: at rps=2.0 (just over this workload's ~1.34 req/sec capacity), how
do P99 waiting time and avg response time move as TIEScheduler's aging term
alpha (tokens/sec) increases? P99 WT drops sharply then plateaus; avg
response time climbs (here, not slowly -- it roughly doubles by alpha=20,
unlike the light-load rps=1.0 case). See docs/Week2_3_Plan.md section 13 /
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

from src.metrics import percentile
from src.schedulers.tie import TIEScheduler
from src.simulator import Simulator
from src.workload import generate_contaminated_workload, load_length_pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LMSYS_CSV = os.path.join(ROOT, "data", "lmsys_output_lengths.csv")
FIGURES_DIR = os.path.join(ROOT, "docs", "figures")
CSV_OUT = os.path.join(ROOT, "experiments", "tie_alpha_curve_rps2.csv")

NUM_REQUESTS = 2000
RPS = 2.0
MAX_BATCH_SIZE = 8
DECODE_TIME_PER_STEP = 0.05
TAIL_RATE = 0.03
SIGMA_NORMAL = (0.0, 0.1)
SIGMA_TAIL = (0.8, 1.2)
BETA = 2.0
SEEDS = [42, 43, 44, 45, 46]
ALPHAS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
COLOR_P99 = "#1baf7a"
COLOR_AVG = "#eda100"


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _run(alpha, seed):
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
    Simulator(
        requests,
        scheduler=TIEScheduler(MAX_BATCH_SIZE, beta=BETA, alpha=alpha),
        decode_time_per_step=DECODE_TIME_PER_STEP,
    ).run()
    response_times = [r.response_time for r in requests]
    waiting_times = [r.waiting_time for r in requests]
    return sum(response_times) / len(response_times), percentile(waiting_times, 99)


def main() -> None:
    rows = []
    for alpha in ALPHAS:
        avg_rts, p99_wts = [], []
        for seed in SEEDS:
            avg_rt, p99_wt = _run(alpha, seed)
            avg_rts.append(avg_rt)
            p99_wts.append(p99_wt)
        mean_avg_rt = sum(avg_rts) / len(avg_rts)
        mean_p99_wt = sum(p99_wts) / len(p99_wts)
        rows.append({"alpha": alpha, "avg_rt": mean_avg_rt, "p99_wt": mean_p99_wt})
        print(f"alpha={alpha:>6.1f}  avg_rt={mean_avg_rt:8.2f}  p99_wt={mean_p99_wt:8.2f}")

    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "avg_rt", "p99_wt"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {CSV_OUT}")

    fig, ax1 = plt.subplots(figsize=(7.5, 5.5))
    _style_axes(ax1)
    xs = [r["alpha"] for r in rows]
    p99_ys = [r["p99_wt"] for r in rows]
    avg_ys = [r["avg_rt"] for r in rows]

    ax1.plot(xs, p99_ys, color=COLOR_P99, marker="D", markersize=6, linewidth=2, label="P99 waiting time (left)")
    ax1.set_xlabel("alpha (tokens/sec of aging credit per second waited)", color=INK_SECONDARY)
    ax1.set_ylabel("P99 waiting time", color=COLOR_P99)

    ax2 = ax1.twinx()
    ax2.set_facecolor(SURFACE)
    for spine in ("top", "left"):
        ax2.spines[spine].set_visible(False)
    ax2.spines["right"].set_color(AXIS)
    ax2.tick_params(colors=INK_MUTED, labelsize=9)
    ax2.plot(xs, avg_ys, color=COLOR_AVG, marker="o", markersize=6, linewidth=2, label="Avg response time (right)")
    ax2.set_ylabel("Avg response time", color=COLOR_AVG)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, labelcolor=INK_SECONDARY, fontsize=9, loc="center right")

    ax1.set_title(
        f"TIE aging term at rps={RPS:.0f} (just over ~1.34 req/s capacity): P99 WT drops, avg RT climbs\n"
        f"(beta={BETA}, tail_rate={TAIL_RATE:.0%}, {len(SEEDS)}-seed avg)",
        color=INK_PRIMARY,
        fontsize=10.5,
    )

    fig.patch.set_facecolor(SURFACE)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, "fig_tie_alpha_curve_rps2.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
