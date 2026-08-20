"""Generate (prompt, logt_mu, logt_sigma) training labels for the TIE
predictor (see docs/VLLM_Qwen3_Reproduction_Plan.md, Phase 1).

For each of `--num-prompts` real LMSYS-Chat-1M prompts, sample
`--samples-per-prompt` completions from the served model via vLLM, then
MLE-fit a log-t(mu, sigma | nu=3.5 fixed) distribution to that prompt's
output-token-length samples (src/logt_fit.py, TIE Eq. 5-6).

`vllm`/`datasets` are only imported inside the functions that need them,
so the CSV-writing and prompt-streaming logic can be exercised without a
GPU (see tests/test_logt_fit.py for the fitting code itself; this script
is not unit tested end-to-end since it requires vLLM + a GPU + network
access to download LMSYS-Chat-1M and Qwen3-8B).

Run on an HPC GPU node, not the login node -- see hpc/generate_labels.slurm.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from typing import Iterator, List

from src.logt_fit import DEFAULT_NU, fit_logt

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LABELS_OUT = os.path.join(_REPO_ROOT, "data", "qwen3_8b_logt_labels.csv")
DEFAULT_SAMPLES_OUT = os.path.join(_REPO_ROOT, "data", "qwen3_8b_length_samples.csv")

MODEL_NAME = "Qwen/Qwen3-8B"
SAMPLES_PER_PROMPT = 20  # paper: 20 repeated generations per prompt
# Mirrors ua_predictor.py's MAX_GENERATED_TOKENS / CVAR_MAX_GENERATED_TOKENS.
MAX_GENERATED_TOKENS = 2048
# Assumption (undocumented in the paper/repo): standard stochastic sampling,
# not greedy -- output length varies because *when* EOS gets sampled is
# genuinely random, and a low temperature would collapse that variance and
# make the log-t fit degenerate. See docs/VLLM_Qwen3_Reproduction_Plan.md.
SAMPLING_TEMPERATURE = 1.0


def stream_first_turn_prompts(num_prompts: int) -> Iterator[str]:
    """Yield the first turn's content from each of the first `num_prompts`
    lmsys-chat-1m conversations, as a flat prompt string.

    Matches ua_predictor.py's extract_original_prompt(), which also just
    wants a flat prompt string -- multi-turn context is dropped, a
    documented simplification (see plan doc, Phase 1 step 3).
    """
    from datasets import load_dataset

    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
    for conversation in itertools.islice(ds, num_prompts):
        turns = conversation["conversation"]
        if not turns:
            continue
        content = turns[0]["content"].strip()
        if content:
            yield content


def generate_completion_lengths(
    prompts: List[str], samples_per_prompt: int = SAMPLES_PER_PROMPT
) -> List[List[int]]:
    """Batch-generate `samples_per_prompt` completions per prompt via vLLM.

    Returns one list of output-token-counts per prompt, in the same order
    as `prompts`.
    """
    from vllm import LLM, SamplingParams

    llm = LLM(model=MODEL_NAME, dtype="bfloat16")
    sampling_params = SamplingParams(
        n=samples_per_prompt,
        temperature=SAMPLING_TEMPERATURE,
        max_tokens=MAX_GENERATED_TOKENS,
    )
    outputs = llm.generate(prompts, sampling_params)

    lengths_per_prompt: List[List[int]] = []
    for output in outputs:
        lengths = [max(1, len(completion.token_ids)) for completion in output.outputs]
        lengths_per_prompt.append(lengths)
    return lengths_per_prompt


def write_labels_csv(prompts: List[str], fits, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "logt_mu", "logt_sigma"])
        for prompt, fit in zip(prompts, fits):
            writer.writerow([prompt, fit.mu, fit.sigma])


def write_samples_csv(prompts: List[str], lengths_per_prompt: List[List[int]], path: str) -> None:
    """Persist the raw per-prompt length samples separately, so a bug in
    the MLE fitting step never requires re-running the expensive
    generation step."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_id", "prompt", "sample_lengths"])
        for prompt_id, (prompt, lengths) in enumerate(zip(prompts, lengths_per_prompt)):
            writer.writerow([prompt_id, prompt, json.dumps(lengths)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TIE predictor training labels from Qwen3-8B.")
    parser.add_argument("--num-prompts", type=int, default=3000)
    parser.add_argument("--samples-per-prompt", type=int, default=SAMPLES_PER_PROMPT)
    parser.add_argument("--labels-out", type=str, default=DEFAULT_LABELS_OUT)
    parser.add_argument("--samples-out", type=str, default=DEFAULT_SAMPLES_OUT)
    args = parser.parse_args()

    print(f"Streaming {args.num_prompts} prompts from lmsys/lmsys-chat-1m ...")
    prompts = list(stream_first_turn_prompts(args.num_prompts))
    print(f"Got {len(prompts)} non-empty prompts.")

    print(f"Generating {args.samples_per_prompt} completions/prompt via {MODEL_NAME} ...")
    lengths_per_prompt = generate_completion_lengths(prompts, args.samples_per_prompt)

    print("Fitting log-t(mu, sigma | nu=%.1f) per prompt ..." % DEFAULT_NU)
    fits = [fit_logt(lengths) for lengths in lengths_per_prompt]

    write_labels_csv(prompts, fits, args.labels_out)
    write_samples_csv(prompts, lengths_per_prompt, args.samples_out)
    print(f"Wrote {args.labels_out} and {args.samples_out}")


if __name__ == "__main__":
    main()
