"""Plot: TIEScheduler avg waiting time vs. beta (no aging, alpha=0), at
rps=4.0 (heavily loaded) and rps=1.0 (below this workload's ~1.34 req/sec
capacity, lightly loaded), each against the Predicted SJF baseline. At
rps=4 the gap grows monotonically with diminishing returns and saturates;
at rps=1 there's almost no queueing to act on, so the gap is tiny and
noisy/non-monotonic. See docs/Week2_3_Plan.md section 13 / conversation
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

from src.metrics import compute_summary
from src.schedulers.predicted_sjf import PredictedSJFScheduler
from src.schedulers.tie import TIEScheduler
from src.simulator import Simulator
from src.workload import generate_contaminated_workload, load_length_pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LMSYS_CSV = os.path.join(ROOT, "data", "lmsys_output_lengths.csv")
FIGURES_DIR = os.path.join(ROOT, "docs", "figures")

NUM_REQUESTS = 2000
MAX_BATCH_SIZE = 8
DECODE_TIME_PER_STEP = 0.05
TAIL_RATE = 0.03
SIGMA_NORMAL = (0.0, 0.1)
SIGMA_TAIL = (0.8, 1.2)
SEEDS = [42, 43, 44, 45, 46]
BETAS = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
COLOR_TIE = "#eda100"
COLOR_PSJF = "#1baf7a"


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


def _sweep(rps):
    psjf_vals = [_run(rps, lambda: PredictedSJFScheduler(MAX_BATCH_SIZE), s) for s in SEEDS]
    psjf_mean = sum(psjf_vals) / len(psjf_vals)

    rows = []
    for beta in BETAS:
        vals = [_run(rps, lambda: TIEScheduler(MAX_BATCH_SIZE, beta=beta, alpha=0.0), s) for s in SEEDS]
        mean = sum(vals) / len(vals)
        rows.append({"beta": beta, "tie_avg_wait": mean})
        print(f"rps={rps:>4.1f}  beta={beta:>5.1f}  tie_avg_wait={mean:.4f}  psjf={psjf_mean:.4f}")
    return psjf_mean, rows


def _plot(rps, psjf_mean, rows, out_name):
    fig, ax = plt.subplots(figsize=(7, 5))
    _style_axes(ax)
    xs = [r["beta"] for r in rows]
    ys = [r["tie_avg_wait"] for r in rows]

    ax.plot(xs, ys, color=COLOR_TIE, marker="D", markersize=6, linewidth=2, label="TIEScheduler (alpha=0)")
    ax.axhline(psjf_mean, color=COLOR_PSJF, linewidth=2, linestyle="--", label="Predicted SJF (no beta)")
    ax.set_xscale("log")
    ax.set_xlabel("beta (log scale)", color=INK_SECONDARY)
    ax.set_ylabel("Avg waiting time", color=INK_SECONDARY)
    ax.set_title(f"TIE avg waiting time vs. beta at rps={rps:.0f} (tail_rate={TAIL_RATE:.0%}, {len(SEEDS)}-seed avg)", color=INK_PRIMARY)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9)

    fig.patch.set_facecolor(SURFACE)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, out_name)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    for rps, csv_name, fig_name in [
        (4.0, "tie_beta_sweep_rps4.csv", "fig_tie_beta_sweep_rps4.png"),
        (1.0, "tie_beta_sweep_rps1.csv", "fig_tie_beta_sweep_rps1.png"),
    ]:
        psjf_mean, rows = _sweep(rps)
        csv_path = os.path.join(ROOT, "experiments", csv_name)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["beta", "tie_avg_wait"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {csv_path}")
        _plot(rps, psjf_mean, rows, fig_name)


if __name__ == "__main__":
    main()
