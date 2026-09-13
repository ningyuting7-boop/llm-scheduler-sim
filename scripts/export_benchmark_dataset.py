"""Build the phase 3 benchmark workload from phase 2's held-out test split.

Writes two files:

  * a JSONL of ``{"prompt": ...}`` lines, which is what
    ``vllm bench serve --dataset-name custom --dataset-path ...`` reads;
  * a CSV of the same prompts' phase 1 labels, which the oracle arm reads
    through ``TIE_ORACLE_CSV``.

Using the test split (rather than fresh LMSYS prompts) is what makes the
oracle arm possible at all: these prompts were never seen during training
*and* we already have 20 real Qwen3-8B samples per prompt behind their
(mu, sigma). See docs/Phase3_Scheduling_Evaluation_Plan.md section 5.2.

The check that matters here is coverage. A benchmark prompt with no label
would silently fall through to the model in oracle mode, quietly turning the
oracle arm into a mixture of the two conditions and making its result
uninterpretable -- so a miss is a hard error, not a warning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-prompts",
        default="checkpoints/predictor_full/test_prompts.json",
        help="phase 2's held-out prompts, written by scripts/train_predictor.py",
    )
    parser.add_argument(
        "--labels",
        default="data/qwen3_8b_logt_labels.csv",
        help="phase 1 labels (prompt, logt_mu, logt_sigma)",
    )
    parser.add_argument("--out-jsonl", default="data/benchmark_prompts.jsonl")
    parser.add_argument("--out-oracle", default="data/benchmark_oracle_labels.csv")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="keep only the first N prompts (0 = all); useful for smoke tests",
    )
    args = parser.parse_args()

    prompts = json.loads(Path(args.test_prompts).read_text())
    if not isinstance(prompts, list):
        raise ValueError(f"{args.test_prompts} should hold a list of prompt strings")
    if args.limit:
        prompts = prompts[: args.limit]

    labels = pd.read_csv(args.labels)
    by_prompt = {
        str(p).strip(): (mu, sigma)
        for p, mu, sigma in zip(labels["prompt"], labels["logt_mu"], labels["logt_sigma"])
    }

    missing = [p for p in prompts if str(p).strip() not in by_prompt]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(prompts)} benchmark prompts have no phase 1 "
            f"label, which would contaminate the oracle arm. First one:\n"
            f"  {missing[0][:200]!r}"
        )

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w") as f:
        for prompt in prompts:
            f.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")

    rows = [
        {"prompt": p, "logt_mu": by_prompt[str(p).strip()][0],
         "logt_sigma": by_prompt[str(p).strip()][1]}
        for p in prompts
    ]
    oracle = pd.DataFrame(rows)
    oracle.to_csv(args.out_oracle, index=False)

    print(f"wrote {len(prompts):,} prompts -> {out_jsonl}")
    print(f"wrote {len(oracle):,} labels  -> {args.out_oracle}")
    print(
        f"  sigma: min={oracle['logt_sigma'].min():.4f} "
        f"median={oracle['logt_sigma'].median():.4f} "
        f"max={oracle['logt_sigma'].max():.4f}"
    )
    print(
        f"  mu:    min={oracle['logt_mu'].min():.4f} "
        f"median={oracle['logt_mu'].median():.4f} "
        f"max={oracle['logt_mu'].max():.4f}"
    )


if __name__ == "__main__":
    main()
