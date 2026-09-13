"""Turns a waiting request's prompt into a TIE scheduling score.

Two modes, selected by the ``TIE_MODE`` environment variable:

``predict`` (default)
    Run phase 2's DeBERTa predictor to get (mu_hat, sigma_hat).

``oracle``
    Look (mu, sigma) up in a table of phase 1 labels -- the parameters fitted
    to 20 real Qwen3-8B samples per prompt. This is the arm that separates
    "the CVaR mechanism does not help" from "our sigma predictor is the
    thing that does not work"; see docs/Phase3_Scheduling_Evaluation_Plan.md
    section 5.2. Falls back to ``predict`` for any prompt not in the table.

Written for this project rather than adapted from the TIE reference
implementation (``vllm/v1/core/sched/ua_predictor.py``), because most of
that file defines a model class we do not use -- phase 2 trained
``src.predictor_model.LengthDistributionPredictor``, whose branches are
single nn.Sequentials rather than the reference's split
feature-extractor/head pair. Carried over from the reference: the adaptive
beta formula and the chat-template stripping in `extract_user_prompt`.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from vllm_tie import score_calculator

if TYPE_CHECKING:
    from src.predictor_model import NormalizeStats

# torch / transformers / src.predictor_model are imported lazily inside
# load_model and _run_model. Keeping them out of module scope lets the
# prompt-extraction and beta logic -- the parts most likely to be silently
# wrong -- be unit-tested without a CUDA-capable environment.

# Max running batch size B in the adaptive beta formula (paper Eq. 12).
# Must be kept equal to vLLM's --max-num-seqs.
GPU_BATCH_SIZE = int(os.environ.get("TIE_GPU_BATCH_SIZE", "32"))

CVAR_ALPHA = 0.9
BETA_MIN, BETA_MAX = 0.1, 0.5

# Pin beta to a constant instead of adapting it to queue depth. Setting
# TIE_BETA=0 drops the CVaR term entirely, leaving score = E[X] -- i.e.
# predicted-SJF. That arm is what isolates the paper's actual contribution:
# on this workload the TIE ranking and the plain E[X] ranking agree to
# Spearman >= 0.996 even with oracle sigma (median fitted sigma is only
# 0.103), so without it a win over FCFS cannot be attributed to
# uncertainty-awareness rather than to shortest-job-first ordering.
# See docs/Phase3_Scheduling_Evaluation_Plan.md section 5.1.
_beta_env = os.environ.get("TIE_BETA")
BETA_OVERRIDE: float | None = float(_beta_env) if _beta_env is not None else None

# Score given to a request whose prediction has not landed yet. Equal to
# vLLM's default max_tokens ceiling, so unpredicted requests sort to the
# bottom of the heap (pessimistic: assume the worst until we know better).
INITIAL_SCORE = 2048.0

_MODEL = None
_TOKENIZER = None
_STATS: "NormalizeStats | None" = None
_DEVICE = None
_MAX_LENGTH = 512  # overwritten from src.predictor_model at load time
_ORACLE: dict[str, tuple[float, float]] = {}
_LOAD_LOCK = threading.Lock()

_ORACLE_HITS = 0
_ORACLE_MISSES = 0


def set_gpu_batch_size(n: int) -> None:
    """Set B in the adaptive beta formula from the server's real
    --max-num-seqs. Called by the scheduler at startup so the two cannot
    drift apart (the reference implementation passes B through a separate
    environment variable, which can silently disagree with the server)."""
    global GPU_BATCH_SIZE
    if n > 0:
        GPU_BATCH_SIZE = n


def adaptive_beta(waiting_count: int) -> float:
    """beta = clip(0.1 * Lq / B, 0.1, 0.5) -- paper Eq. 12.

    Scales the tail-risk term with queue pressure: when few requests are
    waiting there is little to gain from hedging against long outliers, so
    beta stays at its floor and the score is nearly a plain E[X] (i.e. SJF).

    Overridden wholesale by TIE_BETA when that is set.
    """
    if BETA_OVERRIDE is not None:
        return BETA_OVERRIDE
    return min(BETA_MAX, max(BETA_MIN, 0.1 * waiting_count / GPU_BATCH_SIZE))


def compute_score(mu: float, sigma: float, waiting_count: int) -> float:
    """TIE score for one request: E[X] + beta(Lq) * CVaR_0.9[X].

    With TIE_BETA=0 the CVaR term is multiplied by zero, so the Monte Carlo
    for it is skipped. Both arms still draw one 10k-sample batch for the
    censored mean; the saving is a fraction of a millisecond on a background
    thread and does not change what the GPU is doing.
    """
    beta = adaptive_beta(waiting_count)
    if beta == 0.0:
        return score_calculator.mean(mu, sigma)
    return score_calculator.tie_score(mu, sigma, alpha=CVAR_ALPHA, beta=beta)


_USER_MARKER = re.compile(r"(?:\A|\n)user\n")
_ASSISTANT_MARKER = re.compile(r"\nassistant\n?")


def extract_user_prompt(text: str) -> str:
    """Recover the raw user prompt from a decoded, chat-templated string.

    Phase 2 trained on bare LMSYS prompt text, but `vllm bench serve` applies
    the model's chat template before sending, so decoding a request's
    prompt_token_ids yields the templated form. Feeding that to the predictor
    would be a train/serve mismatch. With skip_special_tokens=True Qwen3's
    template decodes to roughly::

        [system\n{system}\n]user\n{prompt}\nassistant\n[<think>\n\n</think>\n\n]

    so the prompt is what sits between the *first* user marker and the *last*
    assistant marker. Both extremes matter: searching for the first assistant
    marker instead would truncate any prompt that itself contains the line
    "assistant", and LMSYS prompts do sometimes discuss chat transcripts.
    """
    start = 0
    first_user = _USER_MARKER.search(text)
    if first_user:
        start = first_user.end()

    end = len(text)
    last_assistant = None
    for match in _ASSISTANT_MARKER.finditer(text, start):
        last_assistant = match
    if last_assistant:
        end = last_assistant.start()

    extracted = text[start:end].strip()
    if extracted:
        return extracted

    # Untemplated (raw completion) requests, or a template shape we do not
    # recognise: fall back to the whole string, which is the closest thing to
    # what phase 2 was trained on.
    return text.strip()


def load_model(device_id: int = 0) -> None:
    """Load the phase 2 checkpoint, tokenizer, and (in oracle mode) the label
    table. Idempotent -- safe to call from more than one place.

    Paths come from the environment so the same code serves every arm:
        TIE_MODEL_DIR    directory with best_model.pt + normalize_stats.json
        TIE_ENCODER      encoder name or local path (default: deberta-v3-base)
        TIE_MODE         "predict" (default) or "oracle"
        TIE_ORACLE_CSV   labels CSV, required when TIE_MODE=oracle
    """
    global _MODEL, _TOKENIZER, _STATS, _DEVICE, _MAX_LENGTH

    import torch
    from transformers import AutoTokenizer

    from src.predictor_model import (
        DEBERTA_MODEL_NAME,
        MAX_LENGTH,
        LengthDistributionPredictor,
        NormalizeStats,
    )

    with _LOAD_LOCK:
        if _MODEL is not None:
            return

        _MAX_LENGTH = MAX_LENGTH
        model_dir = Path(os.environ.get("TIE_MODEL_DIR", "checkpoints/predictor_full"))
        encoder = os.environ.get("TIE_ENCODER", DEBERTA_MODEL_NAME)

        weights = model_dir / "best_model.pt"
        stats_path = model_dir / "normalize_stats.json"
        if not weights.exists():
            raise FileNotFoundError(f"[TIE] checkpoint not found: {weights}")
        if not stats_path.exists():
            raise FileNotFoundError(f"[TIE] normalize stats not found: {stats_path}")

        _DEVICE = (
            torch.device(f"cuda:{device_id}")
            if device_id >= 0 and torch.cuda.is_available()
            else torch.device("cpu")
        )

        _STATS = NormalizeStats.load(str(stats_path))
        model = LengthDistributionPredictor.from_pretrained(encoder)
        # Phase 2 saved a bare state_dict via torch.save(model.state_dict()),
        # not a {"model_state_dict": ...} training checkpoint.
        model.load_state_dict(torch.load(weights, map_location="cpu"))
        model.eval()
        # fp16 halves both the weights and the activation peak. This matters
        # because vLLM has already claimed --gpu-memory-utilization of the
        # card by the time the scheduler is constructed, so the predictor
        # lives in the remainder (see plan doc section 4).
        if _DEVICE.type == "cuda":
            model = model.half()
        _MODEL = model.to(_DEVICE)

        _TOKENIZER = AutoTokenizer.from_pretrained(encoder)

        n_params = sum(p.numel() for p in _MODEL.parameters())
        print(
            f"[TIE] predictor loaded: {n_params:,} params on {_DEVICE}, "
            f"dtype={next(_MODEL.parameters()).dtype}",
            flush=True,
        )

        if os.environ.get("TIE_MODE", "predict") == "oracle":
            _load_oracle_table()


def _load_oracle_table() -> None:
    """Load phase 1's fitted (mu, sigma) labels, keyed by exact prompt text."""
    import pandas as pd

    csv_path = os.environ.get("TIE_ORACLE_CSV")
    if not csv_path:
        raise ValueError("[TIE] TIE_MODE=oracle requires TIE_ORACLE_CSV")

    df = pd.read_csv(csv_path)
    for prompt, mu, sigma in zip(df["prompt"], df["logt_mu"], df["logt_sigma"]):
        _ORACLE[str(prompt).strip()] = (float(mu), float(sigma))
    print(f"[TIE] oracle table loaded: {len(_ORACLE):,} prompts", flush=True)


def predict_scores(
    token_ids_list: list[list[int]],
    llm_tokenizer,
    waiting_count: int = 0,
) -> list[int]:
    """Score a batch of waiting requests from their prompt token IDs.

    `llm_tokenizer` is the *served model's* tokenizer (Qwen3), used only to
    decode prompt_token_ids back to text; the text is then re-tokenized with
    the predictor's own (DeBERTa) tokenizer.

    Returns one integer score per request, in input order. On failure every
    request gets INITIAL_SCORE rather than raising -- a scoring error should
    degrade the scheduler toward FCFS, not take down the server.
    """
    global _ORACLE_HITS, _ORACLE_MISSES

    if _MODEL is None:
        raise RuntimeError("[TIE] load_model() must be called first")
    if not token_ids_list:
        return []

    try:
        decoded = llm_tokenizer.batch_decode(token_ids_list, skip_special_tokens=True)
        prompts = [extract_user_prompt(t) for t in decoded]

        oracle_mode = bool(_ORACLE)
        params: list[tuple[float, float] | None] = [None] * len(prompts)
        if oracle_mode:
            for i, prompt in enumerate(prompts):
                hit = _ORACLE.get(prompt.strip())
                if hit is not None:
                    params[i] = hit
                    _ORACLE_HITS += 1
                else:
                    _ORACLE_MISSES += 1

        # Anything without an oracle hit (all of them in predict mode) goes
        # through the model.
        todo = [i for i, p in enumerate(params) if p is None]
        if todo:
            for i, (mu, sigma) in zip(todo, _run_model([prompts[i] for i in todo])):
                params[i] = (mu, sigma)

        return [
            max(1, int(round(compute_score(mu, sigma, waiting_count))))
            for mu, sigma in params  # type: ignore[misc]
        ]

    except Exception as exc:  # noqa: BLE001 - must never kill the server
        print(f"[TIE] scoring failed for batch of {len(token_ids_list)}: {exc}", flush=True)
        return [int(INITIAL_SCORE)] * len(token_ids_list)


def _run_model(prompts: list[str]) -> list[tuple[float, float]]:
    """Forward `prompts` through the predictor, returning denormalized
    (mu, sigma) pairs."""
    import torch

    encoding = _TOKENIZER(
        prompts,
        max_length=_MAX_LENGTH,
        padding=True,  # dynamic, matching phase 2's collate_fn
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(_DEVICE)
    attention_mask = encoding["attention_mask"].to(_DEVICE)

    with torch.no_grad():
        mu_z, sigma_z = _MODEL(input_ids, attention_mask)

    out = []
    for i in range(len(prompts)):
        mu = float(_STATS.denormalize_mu(mu_z[i].item()))
        sigma = float(_STATS.denormalize_sigma(sigma_z[i].item()))
        # denormalize_sigma is expm1(...), which can go slightly negative for
        # predictions below the training minimum; sigma <= 0 is not a valid
        # scale parameter and would make the log-t draw degenerate.
        out.append((mu, max(sigma, 1e-6)))
    return out


def oracle_stats() -> tuple[int, int]:
    """(hits, misses) for the oracle table -- a miss means a benchmark prompt
    had no phase 1 label and silently fell back to the model, which would
    contaminate the oracle arm. Expected to be 0 when the workload is the
    phase 2 test split."""
    return _ORACLE_HITS, _ORACLE_MISSES
