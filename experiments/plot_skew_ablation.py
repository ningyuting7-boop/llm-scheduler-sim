"""Plot the skewed-uncertainty ablation: does ARRS's beta*uncertainty term
(alpha=0, isolated from aging) improve average latency over Predicted SJF,
and how does the gap move as k grows? See conversation / docs update for
context: skew=1.0 means high uncertainty is deliberately biased toward
underestimation (a constructed scenario, not ELIS-grounded) so that
uncertainty carries directional information Predicted SJF can't see.
"""

from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "experiments", "skew_ablation.csv")
FIGURES_DIR = os.path.join(ROOT, "docs", "figures")

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

STYLE = {
    "predicted_sjf": {"color": "#1baf7a", "marker": "^", "label": "Predicted SJF"},
    "arrs": {"color": "#eda100", "marker": "D", "label": "ARRS (alpha=0, beta=0.2, skew=1.0)"},
}


def main() -> None:
    rows = list(csv.DictReader(open(CSV_PATH)))
    ks = [float(r["k"]) for r in rows]
    psjf = [float(r["predicted_sjf"]) for r in rows]
    arrs = [float(r["arrs"]) for r in rows]
    gap = [p - a for p, a in zip(psjf, arrs)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for name in ("predicted_sjf", "arrs"):
        vals = psjf if name == "predicted_sjf" else arrs
        style = STYLE[name]
        ax.plot(ks, vals, color=style["color"], marker=style["marker"], markersize=6, linewidth=2, label=style["label"])
    ax.set_xlabel("k (predictor error scale)")
    ax.set_ylabel("Avg response time")
    ax.set_title("Avg response time vs k")
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=8, loc="upper left")

    ax2 = axes[1]
    ax2.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax2.spines[spine].set_color(AXIS)
    ax2.tick_params(colors=INK_MUTED, labelsize=9)
    ax2.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)
    ax2.axhline(0, color=INK_MUTED, linewidth=1, linestyle="--")
    ax2.plot(ks, gap, color="#4a3aa7", marker="o", markersize=6, linewidth=2)
    ax2.set_xlabel("k (predictor error scale)")
    ax2.set_ylabel("Gap: predicted_sjf − arrs (positive = ARRS wins)")
    ax2.set_title("ARRS advantage vs k (peaks, then shrinks)")

    fig.suptitle("Skewed-uncertainty ablation (alpha=0, isolating beta*uncertainty alone)", color=INK_PRIMARY)
    fig.patch.set_facecolor(SURFACE)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, "fig_skew_ablation.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
