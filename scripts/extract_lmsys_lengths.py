"""One-time preprocessing: stream LMSYS-Chat-1M and extract the output-length
distribution of individual assistant responses into a small CSV.

Does not download the full ~1.49GB dataset: uses `streaming=True` and reads
only the first `--num-conversations` conversations, discarding the raw text
as soon as its length is measured. The output CSV contains only numbers (no
conversation content), see docs/Week2_3_Plan.md section 4 for the rationale.

Requires a HuggingFace account that has accepted the lmsys-chat-1m usage
terms and is logged in locally (`hf auth login`).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import statistics
from typing import Dict, List

DEFAULT_OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "lmsys_output_lengths.csv"
)

# 4096 tokens is a standard max_generated_tokens setting for LLM inference
# serving. Using OpenAI's rule of thumb (~0.75 words per token), that's
# ~3072 words. Responses longer than this are treated as data artifacts
# (e.g. a model looping on repeated output) rather than genuine generations,
# since no real serving config would produce them; measured on a
# 20k-conversation sample this drops ~0.06% of turns (25/40420, all in the
# 3087-157431 word range with no gray area right below the cutoff) while
# leaving the real distribution (median/p95/p99) essentially unchanged.
MAX_WORD_COUNT = round(4096 * 0.75)


def extract_lengths(num_conversations: int, max_word_count: int = MAX_WORD_COUNT) -> List[Dict[str, float]]:
    """Stream `num_conversations` conversations and return one row per
    assistant turn: {"word_count", "char_count", "char_div4"}. Empty
    responses (word_count == 0) and responses longer than `max_word_count`
    (see MAX_WORD_COUNT) are dropped; everything in between is kept as-is to
    preserve the real heavy tail.
    """
    from datasets import load_dataset

    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
    rows: List[Dict[str, float]] = []
    for conversation in itertools.islice(ds, num_conversations):
        for turn in conversation["conversation"]:
            if turn["role"] != "assistant":
                continue
            content = turn["content"]
            word_count = len(content.split())
            if word_count == 0 or word_count > max_word_count:
                continue
            char_count = len(content)
            rows.append({"word_count": word_count, "char_count": char_count, "char_div4": char_count / 4})
    return rows


def write_lengths_csv(rows: List[Dict[str, float]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["word_count", "char_count", "char_div4"])
        writer.writeheader()
        writer.writerows(rows)


def print_distribution_summary(rows: List[Dict[str, float]]) -> None:
    word_counts = sorted(r["word_count"] for r in rows)
    n = len(word_counts)
    if n == 0:
        print("No samples extracted.")
        return

    def percentile(q: float) -> float:
        idx = min(n - 1, int(n * q / 100))
        return word_counts[idx]

    print(f"samples={n}")
    print(
        f"word_count: mean={statistics.mean(word_counts):.1f} "
        f"median={percentile(50):.1f} p95={percentile(95):.1f} "
        f"p99={percentile(99):.1f} max={word_counts[-1]:.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract LMSYS-Chat-1M assistant-response length distribution.")
    parser.add_argument("--num-conversations", type=int, default=20000)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    print(f"Streaming {args.num_conversations} conversations from lmsys/lmsys-chat-1m ...")
    rows = extract_lengths(args.num_conversations)
    print(f"Extracted {len(rows)} assistant-turn samples (after dropping empty responses).")
    print_distribution_summary(rows)
    write_lengths_csv(rows, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
