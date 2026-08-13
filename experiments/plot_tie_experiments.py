"""Plot Experiment A (tail_rate sweep) and B (sigma_tail sweep): does
TIEScheduler (E[X] + beta*CVaR_90, log-normal fit) beat Predicted SJF on
average latency? See docs/Week2_3_Plan.md section 11.
"""

from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_DIR = os.path.join(ROOT, "experiments")
FIGURES_DIR = os.path.join(ROOT, "docs", "figures")

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

STYLE = {
    "predicted_sjf": {"color": "#1baf7a", "marker": "^", "label": "Predicted SJF"},
    "tie": {"color": "#eda100", "marker": "D", "label": "TIEScheduler (E[X]+beta*CVaR90)"},
}


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _plot_pair(csv_name: str, x_col: str, x_label: str, title: str, out_name: str) -> None:
    rows = list(csv.DictReader(open(os.path.join(EXPERIMENTS_DIR, csv_name))))
    xs = [float(r[x_col]) for r in rows]
    psjf = [float(r["predicted_sjf"]) for r in rows]
    tie = [float(r["tie"]) for r in rows]
    gap = [p - t for p, t in zip(psjf, tie)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    _style_axes(ax)
    for name, vals in [("predicted_sjf", psjf), ("tie", tie)]:
        style = STYLE[name]
        ax.plot(xs, vals, color=style["color"], marker=style["marker"], markersize=6, linewidth=2, label=style["label"])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Avg response time")
    ax.set_title("Avg response time")
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=8, loc="upper left")

    ax2 = axes[1]
    _style_axes(ax2)
    ax2.axhline(0, color=INK_MUTED, linewidth=1, linestyle="--")
    ax2.plot(xs, gap, color="#4a3aa7", marker="o", markersize=6, linewidth=2)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Gap: predicted_sjf − tie (positive = TIE wins)")
    ax2.set_title("TIE advantage")

    fig.suptitle(title, color=INK_PRIMARY)
    fig.patch.set_facecolor(SURFACE)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, out_name)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    _plot_pair(
        "expA_tail_rate.csv", "tail_rate", "tail_rate (fraction of severely-underestimated requests)",
        "Experiment A: TIE vs Predicted SJF as contamination grows (sigma_tail=0.8 fixed)",
        "figA_tail_rate.png",
    )
    _plot_pair(
        "expB_sigma_tail.csv", "sigma_tail", "sigma_tail (log-space spread of tail requests)",
        "Experiment B: sensitivity to sigma_tail (tail_rate=3% fixed)",
        "figB_sigma_tail.png",
    )


if __name__ == "__main__":
    main()
