"""Plot the collateral-damage breakdown: under Predicted SJF, severely
underestimated ("tail") requests finish fast for themselves but drag down
everyone else via HOL blocking; TIEScheduler trades a slower tail for a
faster majority. See docs/Week2_3_Plan.md section 11 / conversation notes.
"""

from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "experiments", "collateral_damage.csv")
FIGURES_DIR = os.path.join(ROOT, "docs", "figures")

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

COLOR_PSJF = "#1baf7a"
COLOR_TIE = "#eda100"


def main() -> None:
    rows = {r["scheduler"]: r for r in csv.DictReader(open(CSV_PATH))}
    psjf = rows["predicted_sjf"]
    tie = rows["tie"]

    categories = ["Tail requests\n(severely underestimated)", "Non-tail requests\n(the other 97%)"]
    psjf_vals = [float(psjf["tail_response"]), float(psjf["normal_response"])]
    tie_vals = [float(tie["tail_response"]), float(tie["normal_response"])]

    x = np.arange(len(categories))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(True, axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    bars1 = ax.bar(x - width / 2, psjf_vals, width, label="Predicted SJF", color=COLOR_PSJF, zorder=3)
    bars2 = ax.bar(x + width / 2, tie_vals, width, label="TIEScheduler", color=COLOR_TIE, zorder=3)

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                textcoords="offset points", ha="center", va="bottom", fontsize=9, color=INK_SECONDARY,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Avg response time", color=INK_SECONDARY)
    ax.set_title("Who pays for the tail? (tail_rate=3%, sigma_tail=0.8, 5-seed avg)", color=INK_PRIMARY)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9)

    fig.patch.set_facecolor(SURFACE)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, "fig_collateral_damage.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
