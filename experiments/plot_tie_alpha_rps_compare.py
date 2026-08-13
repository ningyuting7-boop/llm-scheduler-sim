"""Compare the TIE aging-term (alpha, tokens/sec) tradeoff at two load
points: rps=1.0 (below this workload's ~1.34 req/sec capacity) vs rps=10.0
(heavily overloaded). At rps=1.0 there exists an alpha satisfying the
constraint (P99 WT drop >= 20%, avg RT increase <= 5%); at rps=10.0 no
alpha in a wide sweep does -- the P99 WT drop plateaus around ~6% no matter
how large alpha gets. See docs/Week2_3_Plan.md section 13 / conversation
notes.
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
CSV_OUT = os.path.join(ROOT, "experiments", "tie_alpha_rps_compare.csv")

NUM_REQUESTS = 2000
MAX_BATCH_SIZE = 8
DECODE_TIME_PER_STEP = 0.05
TAIL_RATE = 0.03
SIGMA_NORMAL = (0.0, 0.1)
SIGMA_TAIL = (0.8, 1.2)
BETA = 2.0
SEEDS = [42, 43, 44, 45, 46]

P99_WT_TARGET_DROP = 0.20
AVG_RT_MAX_INCREASE = 0.05

RPS_ALPHAS = {
    1.0: [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 52.0, 60.0, 100.0],
    10.0: [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 100.0, 500.0],
}

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
COLOR_LIGHT = "#1baf7a"
COLOR_HEAVY = "#eb6834"


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
    all_rows = []
    results_by_rps = {}
    for rps, alphas in RPS_ALPHAS.items():
        baseline_avg_rt = sum(_run(rps, 0.0, s)[0] for s in SEEDS) / len(SEEDS)
        baseline_p99_wt = sum(_run(rps, 0.0, s)[1] for s in SEEDS) / len(SEEDS)

        rows = []
        for alpha in alphas:
            avg_rts, p99_wts = [], []
            for seed in SEEDS:
                avg_rt, p99_wt = _run(rps, alpha, seed)
                avg_rts.append(avg_rt)
                p99_wts.append(p99_wt)
            mean_avg_rt = sum(avg_rts) / len(avg_rts)
            mean_p99_wt = sum(p99_wts) / len(p99_wts)
            avg_rt_incr = 100 * (mean_avg_rt - baseline_avg_rt) / baseline_avg_rt
            p99_wt_drop = 100 * (baseline_p99_wt - mean_p99_wt) / baseline_p99_wt
            rows.append({"rps": rps, "alpha": alpha, "avg_rt_incr_pct": avg_rt_incr, "p99_wt_drop_pct": p99_wt_drop})
            print(f"rps={rps:>5.1f}  alpha={alpha:>6.1f}  p99_wt_drop={p99_wt_drop:6.1f}%  avg_rt_incr={avg_rt_incr:6.1f}%")
        results_by_rps[rps] = rows
        all_rows.extend(rows)

    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rps", "alpha", "avg_rt_incr_pct", "p99_wt_drop_pct"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {CSV_OUT}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, rps, color in [(axes[0], 1.0, COLOR_LIGHT), (axes[1], 10.0, COLOR_HEAVY)]:
        _style_axes(ax)
        rows = results_by_rps[rps]
        xs = [r["alpha"] for r in rows]
        drop_ys = [r["p99_wt_drop_pct"] for r in rows]
        incr_ys = [r["avg_rt_incr_pct"] for r in rows]

        ax.plot(xs, drop_ys, color=color, marker="D", markersize=6, linewidth=2, label="P99 WT drop %")
        ax.plot(xs, incr_ys, color=INK_MUTED, marker="o", markersize=6, linewidth=2, linestyle="--", label="Avg RT increase %")
        ax.axhline(100 * P99_WT_TARGET_DROP, color=color, linewidth=1, linestyle=":", alpha=0.6)
        ax.axhline(100 * AVG_RT_MAX_INCREASE, color=INK_MUTED, linewidth=1, linestyle=":", alpha=0.6)

        ax.set_xlabel("alpha (tokens/sec)", color=INK_SECONDARY)
        ax.set_ylabel("%", color=INK_SECONDARY)
        ax.set_title(f"rps={rps:.0f}", color=INK_PRIMARY)
        ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=8, loc="upper left")

    fig.suptitle(
        "TIE aging tradeoff at light vs. heavy load: rps=1.0 (below ~1.34 req/s capacity) finds a point satisfying\n"
        "P99 WT drop >= 20% & Avg RT increase <= 5%; rps=10.0 (overloaded) never does",
        color=INK_PRIMARY,
    )
    fig.patch.set_facecolor(SURFACE)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, "fig_tie_alpha_rps_compare.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
