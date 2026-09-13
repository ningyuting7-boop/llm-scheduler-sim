"""Fine-tune DeBERTa-v3-base to predict (mu, sigma) of each prompt's log-t
output-length distribution. See docs/Phase2_Predictor_Training_Plan.md for
the full design rationale; this implements section 5 (training config) and
the "known limitations" section (grouped split to avoid duplicate-prompt
leakage; the label data's censoring/degenerate-fit caveats live in that doc,
not here -- this script trusts data/qwen3_8b_logt_labels.csv as given).

Intended to run on a GPU machine (HPC), not this repo's dev environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoTokenizer

from src.predictor_model import DEBERTA_MODEL_NAME, MAX_LENGTH, LengthDistributionPredictor, NormalizeStats, predictor_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LABELS_CSV = os.path.join(ROOT, "data", "qwen3_8b_logt_labels.csv")
DEFAULT_OUTPUT_DIR = os.path.join(ROOT, "checkpoints", "predictor")

TRAIN_FRAC, VAL_FRAC = 0.6, 0.2  # remaining 0.2 is test
BATCH_SIZE = 32
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 3  # epochs with no val-loss improvement before stopping
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1

# Fixed weight used ONLY for the val-loss that drives early stopping / best-
# checkpoint selection -- NOT the same as the epoch-varying sigma_weight
# used for the training loss. Using the dynamic weight for both was a bug:
# since sigma_weight grows every epoch, sigma_weight*loss_sigma can inflate
# even while loss_mu and loss_sigma are BOTH still improving, making val_loss
# look worse for reasons that have nothing to do with the model overfitting
# -- confirmed on a real run where train_loss dropped steeply every epoch
# (5.41->2.32) while val_loss "stopped improving" after epoch 1 and
# early-stopped at epoch 4. Monitoring needs a ruler that doesn't change
# across epochs; training itself still uses the dynamic weight below.
MONITORING_SIGMA_WEIGHT = 1.0

# Weighted sampling for the high-sigma/extreme-mu minority (see plan doc
# section 5): weight=2.0 for the most extreme decile/vigintile, 1.5 for the
# next band, 1.0 otherwise. Percentile thresholds are computed on the TRAIN
# split only (never peek at val/test to decide sampling weights).
HIGH_SIGMA_P95, HIGH_SIGMA_P90 = 0.95, 0.90
EXTREME_MU_P95, EXTREME_MU_P90 = 0.95, 0.90


def load_labels(path: str) -> List[Dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def grouped_split(rows: List[Dict], seed: int) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split by UNIQUE prompt text, not by row -- ~11% of prompts in this
    dataset repeat (e.g. "Hi" appears 50 times), and a naive row-level split
    would let the same prompt appear in both train and val/test, letting the
    model "memorize" it rather than generalize. See plan doc known limitations.
    """
    by_prompt: Dict[str, List[Dict]] = {}
    for row in rows:
        by_prompt.setdefault(row["prompt"], []).append(row)

    unique_prompts = list(by_prompt.keys())
    random.Random(seed).shuffle(unique_prompts)

    n = len(unique_prompts)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train_prompts = unique_prompts[:n_train]
    val_prompts = unique_prompts[n_train : n_train + n_val]
    test_prompts = unique_prompts[n_train + n_val :]

    train_rows = [r for p in train_prompts for r in by_prompt[p]]
    val_rows = [r for p in val_prompts for r in by_prompt[p]]
    test_rows = [r for p in test_prompts for r in by_prompt[p]]
    return train_rows, val_rows, test_rows


class PromptLengthDataset(Dataset):
    """Returns raw prompt text (not tokenized) -- tokenization happens once
    per BATCH in collate_fn, not once per example here. A fast (Rust-backed)
    tokenizer processes a batch of raw strings in one parallelized call;
    tokenizing one example at a time in __getitem__ (as this used to do)
    forfeits that and triggers transformers' own "use __call__ on the batch,
    not encode+pad separately" warning. See conversation notes.
    """

    def __init__(self, rows: List[Dict], stats: NormalizeStats) -> None:
        self.rows = rows
        self.stats = stats

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        mu = float(row["logt_mu"])
        sigma = float(row["logt_sigma"])
        return {
            "prompt": row["prompt"],
            "mu_z": torch.tensor(self.stats.normalize_mu(mu), dtype=torch.float32),
            "sigma_z": torch.tensor(self.stats.normalize_sigma(sigma), dtype=torch.float32),
        }


def make_collate_fn(tokenizer):
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # One batched call: the fast tokenizer encodes AND pads the whole
        # batch together (to the longest prompt in THIS batch, not a fixed
        # 512 -- see the OOM/dynamic-padding discussion in conversation
        # notes), using its internal (Rust) parallelism across the batch.
        encoding = tokenizer(
            [b["prompt"] for b in batch],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "mu_z": torch.stack([b["mu_z"] for b in batch]),
            "sigma_z": torch.stack([b["sigma_z"] for b in batch]),
        }

    return collate_fn


def compute_sample_weights(rows: List[Dict]) -> List[float]:
    mus = np.array([float(r["logt_mu"]) for r in rows])
    sigmas = np.array([float(r["logt_sigma"]) for r in rows])

    sigma_p95, sigma_p90 = np.percentile(sigmas, [HIGH_SIGMA_P95 * 100, HIGH_SIGMA_P90 * 100])
    mu_lo95, mu_hi95 = np.percentile(mus, [(1 - EXTREME_MU_P95) * 100, EXTREME_MU_P95 * 100])
    mu_lo90, mu_hi90 = np.percentile(mus, [(1 - EXTREME_MU_P90) * 100, EXTREME_MU_P90 * 100])

    weights = []
    for mu, sigma in zip(mus, sigmas):
        is_extreme_mu = mu <= mu_lo95 or mu >= mu_hi95
        is_high_sigma = sigma >= sigma_p95
        if is_high_sigma or is_extreme_mu:
            weights.append(2.0)
            continue
        is_mild_extreme_mu = mu <= mu_lo90 or mu >= mu_hi90
        is_mild_high_sigma = sigma >= sigma_p90
        weights.append(1.5 if (is_mild_high_sigma or is_mild_extreme_mu) else 1.0)
    return weights


def make_lr_scheduler(optimizer, num_training_steps: int):
    num_warmup_steps = int(num_training_steps * WARMUP_FRACTION)

    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        remaining = num_training_steps - step
        return max(0.0, remaining / max(1, num_training_steps - num_warmup_steps))

    return LambdaLR(optimizer, lr_lambda)


def run_epoch(
    model, loader, device, optimizer=None, scheduler=None, grad_accum_steps: int = 1, sigma_weight: float = 1.0,
    mu_weight: float = 1.0,
) -> float:
    """`grad_accum_steps` > 1 accumulates gradients over that many physical
    batches before each optimizer step, so `--batch-size 8
    --grad-accum-steps 4` trains with the same effective batch size (32,
    same optimizer-step/LR-schedule dynamics) as a single batch_size=32 step
    would, while only ever materializing 8 examples in GPU memory at once
    (deberta-v2's disentangled attention OOM'd at batch_size=32 on a 32GB
    GPU -- see conversation notes). The loss is scaled by 1/grad_accum_steps
    before backward so the accumulated gradient matches a true batch-of-32
    average, not a sum grad_accum_steps times too large.
    """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n_micro_batches = 0.0, 0

    if is_train:
        optimizer.zero_grad()

    with torch.set_grad_enabled(is_train):
        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mu_z = batch["mu_z"].to(device)
            sigma_z = batch["sigma_z"].to(device)

            mu_hat_z, sigma_hat_z = model(input_ids, attention_mask)
            loss = predictor_loss(
                mu_hat_z, sigma_hat_z, mu_z, sigma_z, sigma_weight=sigma_weight, mu_weight=mu_weight
            )

            if is_train:
                (loss / grad_accum_steps).backward()
                is_last_micro_batch = i == len(loader) - 1
                if (i + 1) % grad_accum_steps == 0 or is_last_micro_batch:
                    # Clip AFTER accumulation, right before the optimizer
                    # step -- clipping every micro-batch instead would cap
                    # each 1/grad_accum_steps-scaled slice individually
                    # rather than the accumulated gradient as a whole,
                    # which is not the same thing and under-clips relative
                    # to a true (unaccumulated) batch_size=32 step. Needed
                    # once sigma_weight > 1 amplifies loss_sigma's gradient
                    # contribution and makes spikes more likely (see TIE's
                    # own train/model_train.py, which pairs sigma_weight=3-4
                    # with this same max_norm=1.0 clip -- conversation notes).
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            total_loss += loss.item()
            n_micro_batches += 1
    return total_loss / max(1, n_micro_batches)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the DeBERTa length-distribution predictor.")
    parser.add_argument("--labels-csv", type=str, default=DEFAULT_LABELS_CSV)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="override for smoke tests, e.g. --epochs 2")
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help="physical batch size actually materialized in GPU memory per step; lower this if you hit "
        "CUDA OOM (deberta-v2's disentangled attention is memory-hungry), and raise --grad-accum-steps "
        "by the same factor to keep the effective batch size (batch_size * grad_accum_steps) at 32",
    )
    parser.add_argument(
        "--grad-accum-steps", type=int, default=1,
        help="accumulate gradients over this many physical batches before each optimizer step -- "
        "e.g. --batch-size 8 --grad-accum-steps 4 trains with effective batch size 32 (matching "
        "the TIE-aligned default -- see docs/Phase2_Predictor_Training_Plan.md section 5) while "
        "only holding 8 examples in GPU memory at a time",
    )
    parser.add_argument(
        "--sigma-weight-start", type=float, default=3.0,
        help="sigma_weight at epoch 0 (loss = loss_mu + sigma_weight*loss_sigma). Two rounds of "
        "evaluation showed sigma_weight=1.0 (equal) gives sigma R^2=0.05, and 2.0 made it WORSE "
        "(R^2=0.01) -- TIE's own train/model_train.py uses 3.0-4.0, much higher than either value "
        "we'd tried, which is the actual working range; see conversation notes",
    )
    parser.add_argument(
        "--sigma-weight-end", type=float, default=4.0,
        help="sigma_weight at the final epoch; linearly interpolated from --sigma-weight-start, "
        "matching TIE's `3 + (epoch/NUM_EPOCHS)*1.0` schedule",
    )
    parser.add_argument(
        "--mu-weight", type=float, default=1.0,
        help="weight on the mu MSE term. Set to 0.0 to train sigma ALONE, which isolates whether "
        "sigma's stuck ~0.05 test R^2 is caused by competing with mu for the shared encoder. Note "
        "the mu head then receives no gradient, so that run's mu metrics are meaningless",
    )
    parser.add_argument(
        "--freeze-encoder-epoch", type=int, default=2,
        help="freeze the DeBERTa encoder at the start of this epoch and train only the MLP heads "
        "(pass a value >= --epochs to never freeze). Default 2: our val_loss bottoms at epoch 1 and "
        "degrades after, while train_loss keeps falling -- i.e. past epoch 1 the encoder is "
        "memorizing, not generalizing. A split-half reliability check put the sigma label's own "
        "ceiling at r^2~0.6-0.87, so sigma IS learnable in principle and the 0.05 we measured is a "
        "training problem (overfitting before sigma can be learned), not a label-noise problem",
    )
    parser.add_argument(
        "--lr-frozen", type=float, default=5e-5,
        help="learning rate after the encoder is frozen (matching TIE's LR_FREEZE_ENCODER) -- the "
        "remaining heads are small enough to tolerate larger steps, and there are no pretrained "
        "encoder weights left to damage",
    )
    parser.add_argument(
        "--limit-rows", type=int, default=None,
        help="use only the first N labeled rows (before splitting) -- for the "
        "small-scale HPC sanity check in docs/Phase2_Predictor_Training_Plan.md "
        "(e.g. --limit-rows 200), not for real training runs",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    rows = load_labels(args.labels_csv)
    if args.limit_rows is not None:
        rows = rows[: args.limit_rows]
    train_rows, val_rows, test_rows = grouped_split(rows, seed=args.seed)
    print(f"split sizes (rows): train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    # Fit normalization on TRAIN only, save immediately so it's available
    # even if training is interrupted before finishing.
    stats = NormalizeStats.fit(
        mus=[float(r["logt_mu"]) for r in train_rows],
        sigmas=[float(r["logt_sigma"]) for r in train_rows],
    )
    stats.save(os.path.join(args.output_dir, "normalize_stats.json"))

    with open(os.path.join(args.output_dir, "test_prompts.json"), "w") as f:
        json.dump(sorted({r["prompt"] for r in test_rows}), f)

    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_MODEL_NAME)
    train_dataset = PromptLengthDataset(train_rows, stats)
    val_dataset = PromptLengthDataset(val_rows, stats)

    sample_weights = compute_sample_weights(train_rows)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    collate_fn = make_collate_fn(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device(args.device)
    model = LengthDistributionPredictor.from_pretrained(DEBERTA_MODEL_NAME).to(device)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # In units of OPTIMIZER steps, not micro-batches -- the LR warmup/decay
    # schedule should track how many times weights actually get updated,
    # not how many forward/backward passes grad accumulation splits that
    # into (see run_epoch).
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    num_training_steps = optimizer_steps_per_epoch * args.epochs
    scheduler = make_lr_scheduler(optimizer, num_training_steps)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    log = []

    for epoch in range(args.epochs):
        if args.freeze_encoder_epoch is not None and epoch == args.freeze_encoder_epoch:
            # The 184M-param encoder is what memorizes 5987 examples; the two
            # heads together are only ~1.4M. Freezing here removes the
            # overfitting engine while keeping DeBERTa's pretrained features,
            # so the (harder, subtler) sigma signal gets many more stable
            # epochs to emerge. TIE freezes at epoch 12/20, but that is ~60%
            # through THEIR run on a 4x larger dataset -- the transferable
            # part is the principle ("freeze once the encoder stops helping"),
            # not the number. On our data val_loss bottoms at epoch 1 and
            # degrades from epoch 2 while train_loss keeps plunging, so our
            # equivalent switch point is epoch 2. See conversation notes.
            for param in model.encoder.parameters():
                param.requires_grad = False
            optimizer = AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=args.lr_frozen, weight_decay=WEIGHT_DECAY
            )
            remaining_steps = optimizer_steps_per_epoch * (args.epochs - epoch)
            scheduler = make_lr_scheduler(optimizer, remaining_steps)
            # Treat post-freeze as a fresh phase: inheriting a patience
            # counter that is already near its limit would early-stop the
            # new regime before it has run a single epoch.
            epochs_without_improvement = 0
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"epoch {epoch:>3}  froze encoder; {trainable:,} trainable params left, lr -> {args.lr_frozen}")

        # Linear interpolation, matching TIE's `3 + (epoch/NUM_EPOCHS)*1.0`.
        sigma_weight = args.sigma_weight_start + (epoch / args.epochs) * (args.sigma_weight_end - args.sigma_weight_start)
        train_loss = run_epoch(
            model, train_loader, device, optimizer, scheduler, args.grad_accum_steps, sigma_weight, args.mu_weight
        )
        # Monitoring keeps sigma's weight fixed (comparability across epochs)
        # but mirrors --mu-weight, so a sigma-only run early-stops on sigma
        # alone rather than on a head that is receiving no gradient at all.
        val_loss = run_epoch(
            model, val_loader, device, sigma_weight=MONITORING_SIGMA_WEIGHT, mu_weight=args.mu_weight
        )
        log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "sigma_weight": sigma_weight})
        print(f"epoch {epoch:>3}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  sigma_weight={sigma_weight:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"early stopping at epoch {epoch} (no val improvement for {EARLY_STOPPING_PATIENCE} epochs)")
                break

    with open(os.path.join(args.output_dir, "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"Done. Best val_loss={best_val_loss:.4f}. Checkpoint + logs in {args.output_dir}")


if __name__ == "__main__":
    main()
