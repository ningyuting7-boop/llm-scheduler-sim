"""Plot: how much P99 waiting-time improvement can TIE's aging term (alpha,
tokens/sec -- see src/schedulers/tie.py) buy, under a fixed avg-response-time
budget (avg RT increase <= AVG_RT_MAX_INCREASE), as load increases?

For each rps, sweeps alpha and reports the best P99 WT drop (vs alpha=0)
among alphas that stay within the avg RT budget. See
docs/Week2_3_Plan.md section 13 / conversation notes: at rps=10 (heavily
overloaded relative to this workload's ~1.34 req/sec capacity) the ceiling
was only ~6%; at rps=1.0 (below capacity) alpha=52 alone already reaches
20%. This sweeps the load axis to show that dependency directly.
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
CSV_OUT = os.path.join(ROOT, "experiments", "p99_improvement_vs_load.csv")

NUM_REQUESTS = 2000
RPS_LEVELS = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0]
MAX_BATCH_SIZE = 8
DECODE_TIME_PER_STEP = 0.05
TAIL_RATE = 0.03
SIGMA_NORMAL = (0.0, 0.1)
SIGMA_TAIL = (0.8, 1.2)
BETA = 2.0
SEEDS = [42, 43, 44, 45, 46]
ALPHAS = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1000.0]
AVG_RT_MAX_INCREASE = 0.05

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
COLOR = "#eda100"


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _run(rps, alpha, seed):
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
    for rps in RPS_LEVELS:
        baseline_avg_rt = sum(_run(rps, 0.0, seed)[0] for seed in SEEDS) / len(SEEDS)
        baseline_p99_wt = sum(_run(rps, 0.0, seed)[1] for seed in SEEDS) / len(SEEDS)

        best_drop = 0.0
        best_alpha = 0.0
        for alpha in ALPHAS:
            avg_rts, p99_wts = [], []
            for seed in SEEDS:
                avg_rt, p99_wt = _run(rps, alpha, seed)
                avg_rts.append(avg_rt)
                p99_wts.append(p99_wt)
            mean_avg_rt = sum(avg_rts) / len(avg_rts)
            mean_p99_wt = sum(p99_wts) / len(p99_wts)
            avg_rt_incr = (mean_avg_rt - baseline_avg_rt) / baseline_avg_rt
            p99_wt_drop = (baseline_p99_wt - mean_p99_wt) / baseline_p99_wt
            if avg_rt_incr <= AVG_RT_MAX_INCREASE and p99_wt_drop > best_drop:
                best_drop = p99_wt_drop
                best_alpha = alpha

        rows.append({"rps": rps, "best_alpha": best_alpha, "best_p99_wt_drop_pct": best_drop})
        print(f"rps={rps:>5.1f}  best_alpha={best_alpha:>7.1f}  best_p99_wt_drop={100*best_drop:.1f}%  (within avg_rt budget <= {100*AVG_RT_MAX_INCREASE:.0f}%)")

    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rps", "best_alpha", "best_p99_wt_drop_pct"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {CSV_OUT}")

    fig, ax = plt.subplots(figsize=(7, 5))
    _style_axes(ax)
    xs = [r["rps"] for r in rows]
    ys = [100 * r["best_p99_wt_drop_pct"] for r in rows]
    ax.plot(xs, ys, color=COLOR, marker="D", markersize=7, linewidth=2)
    for r in rows:
        ax.annotate(f"α={r['best_alpha']:.0f}", xy=(r["rps"], 100 * r["best_p99_wt_drop_pct"]), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8, color=INK_SECONDARY)

    ax.set_xlabel("Arrival rate (requests/sec)", color=INK_SECONDARY)
    ax.set_ylabel("Best achievable P99 waiting-time drop (%)", color=INK_SECONDARY)
    ax.set_title(
        f"Best P99 WT improvement within a {100*AVG_RT_MAX_INCREASE:.0f}% avg-RT budget, vs. load\n"
        f"(TIEScheduler aging term, beta={BETA}, tail_rate={TAIL_RATE:.0%}, {len(SEEDS)}-seed avg)",
        color=INK_PRIMARY,
    )

    fig.patch.set_facecolor(SURFACE)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, "fig_p99_improvement_vs_load.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
