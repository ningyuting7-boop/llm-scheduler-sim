"""Select a default aging strength (alpha, in tokens/sec of priority credit
per second waited -- see src/schedulers/tie.py) for TIEScheduler using a
constraint-based rule instead of eyeballing a chart or comparing against
Predicted SJF:

    Choose the smallest alpha such that, relative to alpha=0:
      - P99 waiting time drops by at least P99_WT_TARGET_DROP
      - avg response time rises by at most AVG_RT_MAX_INCREASE

All comparisons are TIE-vs-itself across alpha (not vs Predicted SJF) --
the question here is purely "how much aging does TIE need for the tail",
not "does TIE still beat Predicted SJF". See docs/Week2_3_Plan.md section 13
/ conversation notes.
"""

from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import percentile
from src.schedulers.tie import TIEScheduler
from src.simulator import Simulator
from src.workload import generate_contaminated_workload, load_length_pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LMSYS_CSV = os.path.join(ROOT, "data", "lmsys_output_lengths.csv")
CSV_OUT = os.path.join(ROOT, "experiments", "tie_alpha_selection.csv")

NUM_REQUESTS = 2000
RPS = float(os.environ.get("SWEEP_RPS", "10.0"))
MAX_BATCH_SIZE = 8
DECODE_TIME_PER_STEP = 0.05
TAIL_RATE = 0.03
SIGMA_NORMAL = (0.0, 0.1)
SIGMA_TAIL = (0.8, 1.2)
BETA = 2.0
SEEDS = [42, 43, 44, 45, 46]

ALPHAS = [0.0, 50.0, 52.0, 54.0, 56.0, 58.0, 60.0]  # tokens/sec of priority credit per second waited

P99_WT_TARGET_DROP = 0.20  # require at least a 20% reduction in P99 waiting time
AVG_RT_MAX_INCREASE = 0.05  # allow at most a 5% increase in avg response time


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
    n = len(requests)
    sum_sq = sum(w * w for w in waiting_times)
    jain = (sum(waiting_times) ** 2) / (n * sum_sq) if sum_sq > 0 else 1.0
    return {
        "avg_rt": sum(response_times) / n,
        "p99_rt": percentile(response_times, 99),
        "p99_wt": percentile(waiting_times, 99),
        "jain": jain,
    }


def main() -> None:
    rows = []
    for alpha in ALPHAS:
        accum = {"avg_rt": [], "p99_rt": [], "p99_wt": [], "jain": []}
        for seed in SEEDS:
            result = _run(alpha, seed)
            for key in accum:
                accum[key].append(result[key])
        row = {"alpha": alpha, **{key: sum(vals) / len(vals) for key, vals in accum.items()}}
        rows.append(row)

    baseline = rows[0]
    print(f"{'alpha':>8}{'avg_rt':>10}{'p99_rt':>10}{'p99_wt':>10}{'jain':>8}{'p99_wt_drop%':>14}{'avg_rt_incr%':>14}")
    for row in rows:
        p99_wt_drop = (baseline["p99_wt"] - row["p99_wt"]) / baseline["p99_wt"]
        avg_rt_incr = (row["avg_rt"] - baseline["avg_rt"]) / baseline["avg_rt"]
        row["p99_wt_drop_pct"] = p99_wt_drop
        row["avg_rt_incr_pct"] = avg_rt_incr
        print(
            f"{row['alpha']:>8.0f}{row['avg_rt']:>10.2f}{row['p99_rt']:>10.2f}{row['p99_wt']:>10.2f}"
            f"{row['jain']:>8.4f}{100*p99_wt_drop:>13.1f}%{100*avg_rt_incr:>13.1f}%"
        )

    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "avg_rt", "p99_rt", "p99_wt", "jain", "p99_wt_drop_pct", "avg_rt_incr_pct"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {CSV_OUT}")

    chosen = None
    for row in rows[1:]:  # skip alpha=0, which trivially satisfies neither/both at 0%
        if row["p99_wt_drop_pct"] >= P99_WT_TARGET_DROP and row["avg_rt_incr_pct"] <= AVG_RT_MAX_INCREASE:
            chosen = row
            break

    print()
    if chosen:
        print(
            f"Selected alpha={chosen['alpha']:.0f} tokens/sec: "
            f"P99 waiting time -{100*chosen['p99_wt_drop_pct']:.1f}%, "
            f"avg response time +{100*chosen['avg_rt_incr_pct']:.1f}% "
            f"(constraint: P99 WT drop >= {100*P99_WT_TARGET_DROP:.0f}%, avg RT increase <= {100*AVG_RT_MAX_INCREASE:.0f}%)"
        )
    else:
        print(
            f"No alpha in {ALPHAS} satisfies the constraint "
            f"(P99 WT drop >= {100*P99_WT_TARGET_DROP:.0f}%, avg RT increase <= {100*AVG_RT_MAX_INCREASE:.0f}%) -- "
            f"need a finer sweep or looser constraints."
        )


if __name__ == "__main__":
    main()
