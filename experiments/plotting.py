"""Render the report figures from the CSVs the exp1-4 scripts produce.

Reads only; run exp1/exp3 first. Figures go to docs/figures/.
Color assignment is fixed (never cycled) so a scheduler's color/marker is
identical across every figure: fcfs=blue/o, oracle_sjf=orange/s,
predicted_sjf=aqua/^.
"""

from __future__ import annotations

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import percentile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_DIR = os.path.join(ROOT, "experiments")
DATA_DIR = os.path.join(ROOT, "data")
FIGURES_DIR = os.path.join(ROOT, "docs", "figures")

# --- validated palette (docs skill: dataviz/references/palette.md), fixed order ---
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

SCHEDULER_STYLE = {
    "fcfs": {"color": "#2a78d6", "marker": "o", "label": "FCFS"},
    "oracle_sjf": {"color": "#eb6834", "marker": "s", "label": "Oracle SJF"},
    "predicted_sjf": {"color": "#1baf7a", "marker": "^", "label": "Predicted SJF"},
}
SCHEDULER_ORDER = ["fcfs", "oracle_sjf", "predicted_sjf"]


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.title.set_color(INK_PRIMARY)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def _save(fig, name: str) -> None:
    fig.patch.set_facecolor(SURFACE)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {path}")


def _read_csv(path: str):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def figure1_length_distribution() -> None:
    rows = _read_csv(os.path.join(DATA_DIR, "lmsys_output_lengths.csv"))
    lengths = [float(r["word_count"]) for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    _style_axes(ax)
    ax.hist(lengths, bins=60, color=SCHEDULER_STYLE["fcfs"]["color"], alpha=0.85, edgecolor=SURFACE, linewidth=0.3)
    ax.set_yscale("log")
    ax.set_xlabel("Output length (words)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Figure 1: LMSYS-Chat-1M output length distribution (n={:,})".format(len(lengths)))

    for q, style in [(50, "--"), (95, ":"), (99, "-.")]:
        v = percentile(lengths, q)
        ax.axvline(v, color=INK_MUTED, linestyle=style, linewidth=1)
        ax.text(v, ax.get_ylim()[1] * 0.7, f" p{q}={v:.0f}", color=INK_SECONDARY, fontsize=8, rotation=90, va="top")

    _save(fig, "fig1_length_distribution.png")


def _plot_metric_vs_rps(csv_name: str, metric_col: str, title: str, ylabel: str, out_name: str) -> None:
    rows = _read_csv(os.path.join(EXPERIMENTS_DIR, csv_name))
    fig, ax = plt.subplots(figsize=(6, 4))
    _style_axes(ax)

    for scheduler in SCHEDULER_ORDER:
        sub = [r for r in rows if r["scheduler"] == scheduler]
        sub.sort(key=lambda r: float(r["rps"]))
        xs = [float(r["rps"]) for r in sub]
        ys = [float(r[metric_col]) for r in sub]
        style = SCHEDULER_STYLE[scheduler]
        ax.plot(xs, ys, color=style["color"], marker=style["marker"], markersize=6, linewidth=2, label=style["label"])

    ax.set_xlabel("Arrival rate (requests/sec)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9)
    _save(fig, out_name)


def figure2_avg_response_vs_rps() -> None:
    _plot_metric_vs_rps(
        "exp3_congestion.csv", "avg_response", "Figure 2: Average response time vs. load", "Avg response time", "fig2_avg_response_vs_rps.png"
    )


def figure3_p95_response_vs_rps() -> None:
    _plot_metric_vs_rps(
        "exp3_congestion.csv", "p95_response", "Figure 3: P95 response time vs. load", "P95 response time", "fig3_p95_response_vs_rps.png"
    )


def figure6_throughput_vs_rps() -> None:
    _plot_metric_vs_rps(
        "exp3_congestion.csv", "throughput", "Figure 6: Throughput vs. load", "Throughput (req/sec)", "fig6_throughput_vs_rps.png"
    )


def main() -> None:
    figure1_length_distribution()
    figure2_avg_response_vs_rps()
    figure3_p95_response_vs_rps()
    figure6_throughput_vs_rps()


if __name__ == "__main__":
    main()
