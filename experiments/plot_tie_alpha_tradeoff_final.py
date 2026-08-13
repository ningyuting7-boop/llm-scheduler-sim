"""Final tradeoff sweep: for TIEScheduler's aging term alpha (tokens/sec),
find alpha values that give a large P99 waiting-time drop for a small
avg-response-time increase, at rps=1.0 (below this workload's ~1.34 req/sec
capacity -- the only load regime where this tradeoff is favorable at all;
see docs/Week2_3_Plan.md section 13 / conversation notes on why rps=2 and
rps=10 don't work).

Reports, for a few avg-RT budgets, the alpha that maximizes P99 WT drop
without exceeding that budget.
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
CSV_OUT = os.path.join(ROOT, "experiments", "tie_alpha_tradeoff_final.csv")

NUM_REQUESTS = 2000
RPS = 1.0
MAX_BATCH_SIZE = 8
DECODE_TIME_PER_STEP = 0.05
TAIL_RATE = 0.03
SIGMA_NORMAL = (0.0, 0.1)
SIGMA_TAIL = (0.8, 1.2)
BETA = 2.0
SEEDS = [42, 43, 44, 45, 46]
ALPHAS = [0.0, 10.0, 20.0, 30.0, 40.0, 45.0, 50.0, 52.0, 55.0, 60.0, 70.0, 80.0, 90.0, 100.0, 120.0, 150.0, 200.0]
BUDGETS = [0.03, 0.05, 0.10, 0.15]

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
COLOR_DROP = "#1baf7a"
COLOR_INCR = "#898781"
COLOR_MARK = "#eb6834"


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
    baseline_avg_rt = sum(_run(0.0, s)[0] for s in SEEDS) / len(SEEDS)
    baseline_p99_wt = sum(_run(0.0, s)[1] for s in SEEDS) / len(SEEDS)
    print(f"baseline (alpha=0): avg_rt={baseline_avg_rt:.2f}  p99_wt={baseline_p99_wt:.2f}")

    rows = []
    for alpha in ALPHAS:
        avg_rts, p99_wts = [], []
        for seed in SEEDS:
            avg_rt, p99_wt = _run(alpha, seed)
            avg_rts.append(avg_rt)
            p99_wts.append(p99_wt)
        mean_avg_rt = sum(avg_rts) / len(avg_rts)
        mean_p99_wt = sum(p99_wts) / len(p99_wts)
        avg_rt_incr = (mean_avg_rt - baseline_avg_rt) / baseline_avg_rt
        p99_wt_drop = (baseline_p99_wt - mean_p99_wt) / baseline_p99_wt
        rows.append({"alpha": alpha, "avg_rt_incr_pct": avg_rt_incr, "p99_wt_drop_pct": p99_wt_drop})
        print(f"alpha={alpha:>6.1f}  p99_wt_drop={100*p99_wt_drop:6.1f}%  avg_rt_incr={100*avg_rt_incr:6.1f}%")

    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "avg_rt_incr_pct", "p99_wt_drop_pct"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {CSV_OUT}")

    print()
    print("Best alpha per avg-RT budget:")
    best_per_budget = {}
    for budget in BUDGETS:
        candidates = [r for r in rows if r["avg_rt_incr_pct"] <= budget]
        best = max(candidates, key=lambda r: r["p99_wt_drop_pct"]) if candidates else None
        best_per_budget[budget] = best
        if best:
            print(f"  budget<= {100*budget:.0f}%: alpha={best['alpha']:.0f}  p99_wt_drop={100*best['p99_wt_drop_pct']:.1f}%  avg_rt_incr={100*best['avg_rt_incr_pct']:.1f}%")
        else:
            print(f"  budget<= {100*budget:.0f}%: no alpha qualifies")

    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    _style_axes(ax1)
    xs = [r["alpha"] for r in rows]
    drop_ys = [100 * r["p99_wt_drop_pct"] for r in rows]
    incr_ys = [100 * r["avg_rt_incr_pct"] for r in rows]

    ax1.plot(xs, drop_ys, color=COLOR_DROP, marker="D", markersize=6, linewidth=2, label="P99 waiting time drop (%)")
    ax1.plot(xs, incr_ys, color=COLOR_INCR, marker="o", markersize=6, linewidth=2, linestyle="--", label="Avg response time increase (%)")
    ax1.axhline(0, color=INK_MUTED, linewidth=1)

    best5 = best_per_budget.get(0.05)
    if best5:
        ax1.axvline(best5["alpha"], color=COLOR_MARK, linewidth=1.3, linestyle="-", alpha=0.85, zorder=2)
        ax1.annotate(
            f"alpha={best5['alpha']:.0f}\nP99 -{100*best5['p99_wt_drop_pct']:.0f}%, Avg +{100*best5['avg_rt_incr_pct']:.0f}%",
            xy=(best5["alpha"], 100 * best5["p99_wt_drop_pct"]), xytext=(best5["alpha"] + 12, 100 * best5["p99_wt_drop_pct"] - 6),
            fontsize=9, color=COLOR_MARK, arrowprops=dict(arrowstyle="->", color=COLOR_MARK, lw=1),
        )

    ax1.set_xlabel("alpha (tokens/sec of aging credit per second waited)", color=INK_SECONDARY)
    ax1.set_ylabel("% change vs alpha=0 baseline", color=INK_SECONDARY)
    ax1.set_title(f"TIE aging tradeoff at rps={RPS:.0f} (below system capacity): best alpha for max P99 drop per avg-RT budget", color=INK_PRIMARY, fontsize=10.5)
    ax1.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9, loc="center right")

    fig.patch.set_facecolor(SURFACE)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, "fig_tie_alpha_tradeoff_final.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
