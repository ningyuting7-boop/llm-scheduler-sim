"""DeBERTa-v3-base length-distribution predictor: given a prompt, predicts
(mu_hat, sigma_hat) for a log-t(mu, sigma, nu=3.5) fit of its output-length
distribution (see src/logt_fit.py and docs/Phase2_Predictor_Training_Plan.md
for how the training labels are produced).

Architecture (docs/Phase2_Predictor_Training_Plan.md section 1):

    prompt text
       -> DeBERTa-v3-base encoder (12 layers, hidden=768)
       -> multi-pooling: [CLS, mean-pool, max-pool] concatenated -> 768*3=2304
       -> two independent branches (mu, sigma), each:
            (Linear -> LayerNorm -> GELU -> Dropout(0.2)) x2, hidden=256
            -> regression head: Linear(256->128) -> GELU -> Dropout(0.1) -> Linear(128->1)
       -> two scalars: mu_hat, sigma_hat

CLS-only pooling is not used because it can lose information the mean/max
pools retain (overall semantic tendency vs. the single most salient local
signal, e.g. a phrase in the prompt strongly implying a long answer) -- a
useful prior when predicting a distribution's shape, not just a point.
mu and sigma get separate branches (not a shared trunk) since they measure
different things ("how long" vs. "how unsure am I") and sharing final
layers would let their gradients interfere with each other.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from transformers import AutoModel

DEBERTA_MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 512  # hard limit: deberta-v3-base's max_position_embeddings; see plan doc section 2
BRANCH_HIDDEN = 256
HEAD_HIDDEN = 128


def _masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """(batch, seq, hidden), (batch, seq) -> (batch, hidden); ignores padding."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # (batch, seq, 1)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _masked_max_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """(batch, seq, hidden), (batch, seq) -> (batch, hidden); ignores padding."""
    mask = attention_mask.unsqueeze(-1).to(dtype=torch.bool)  # (batch, seq, 1)
    neg_inf = torch.finfo(last_hidden_state.dtype).min
    masked = last_hidden_state.masked_fill(~mask, neg_inf)
    return masked.max(dim=1).values


def _make_branch(pooled_size: int) -> nn.Module:
    """Two (Linear -> LayerNorm -> GELU -> Dropout(0.2)) blocks, then a
    Linear(256->128) -> GELU -> Dropout(0.1) -> Linear(128->1) regression head.
    """
    return nn.Sequential(
        nn.Linear(pooled_size, BRANCH_HIDDEN),
        nn.LayerNorm(BRANCH_HIDDEN),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(BRANCH_HIDDEN, BRANCH_HIDDEN),
        nn.LayerNorm(BRANCH_HIDDEN),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(BRANCH_HIDDEN, HEAD_HIDDEN),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(HEAD_HIDDEN, 1),
    )


class LengthDistributionPredictor(nn.Module):
    """Wraps a DeBERTa-style encoder (passed in, not constructed here, so
    tests can inject a tiny randomly-initialized encoder -- any encoder
    exposing `.config.hidden_size` and returning `.last_hidden_state` works,
    not just real deberta-v3-base -- see tests/test_predictor_model.py).
    """

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        pooled_size = encoder.config.hidden_size * 3  # CLS + mean + max
        self.mu_branch = _make_branch(pooled_size)
        self.sigma_branch = _make_branch(pooled_size)

    @classmethod
    def from_pretrained(cls, model_name: str = DEBERTA_MODEL_NAME) -> "LengthDistributionPredictor":
        return cls(AutoModel.from_pretrained(model_name))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """input_ids/attention_mask: (batch, seq_len). Returns (mu_hat, sigma_hat), each (batch,)."""
        encoder_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = encoder_out.last_hidden_state  # (batch, seq, hidden)

        cls = last_hidden_state[:, 0, :]
        mean_pool = _masked_mean_pool(last_hidden_state, attention_mask)
        max_pool = _masked_max_pool(last_hidden_state, attention_mask)
        pooled = torch.cat([cls, mean_pool, max_pool], dim=-1)  # (batch, 2304)

        mu_hat = self.mu_branch(pooled).squeeze(-1)
        sigma_hat = self.sigma_branch(pooled).squeeze(-1)
        return mu_hat, sigma_hat


@dataclass
class NormalizeStats:
    """mu: plain z-score. sigma: log1p (compresses its right-skewed scale,
    see plan doc section 3) then z-score. Must be fit once on the TRAIN
    split only and reused as-is for val/test and inference -- refitting on
    a different split silently breaks the train/inference correspondence.
    """

    mu_mean: float
    mu_std: float
    log1p_sigma_mean: float
    log1p_sigma_std: float

    @classmethod
    def fit(cls, mus, sigmas) -> "NormalizeStats":
        import numpy as np

        mus = np.asarray(mus, dtype=float)
        log1p_sigmas = np.log1p(np.asarray(sigmas, dtype=float))
        return cls(
            mu_mean=float(mus.mean()),
            mu_std=float(mus.std()) or 1.0,
            log1p_sigma_mean=float(log1p_sigmas.mean()),
            log1p_sigma_std=float(log1p_sigmas.std()) or 1.0,
        )

    def normalize_mu(self, mu):
        return (mu - self.mu_mean) / self.mu_std

    def denormalize_mu(self, mu_z):
        return mu_z * self.mu_std + self.mu_mean

    def normalize_sigma(self, sigma):
        import numpy as np

        return (np.log1p(sigma) - self.log1p_sigma_mean) / self.log1p_sigma_std

    def denormalize_sigma(self, sigma_z):
        import numpy as np

        return np.expm1(sigma_z * self.log1p_sigma_std + self.log1p_sigma_mean)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "NormalizeStats":
        with open(path) as f:
            return cls(**json.load(f))


def predictor_loss(
    mu_hat_z: torch.Tensor,
    sigma_hat_z: torch.Tensor,
    mu_true_z: torch.Tensor,
    sigma_true_z: torch.Tensor,
    sigma_weight: float = 1.0,
    mu_weight: float = 1.0,
) -> torch.Tensor:
    """`mu_weight * MSE(mu) + sigma_weight * MSE(sigma)`, both heads already
    in normalized space.

    Both default to 1.0 (the plan doc section 4 simplification: the targets
    are z-scored, so their raw scales are already comparable). z-scoring
    equalizes scale, not difficulty -- measured test R^2 is ~0.6-0.77 for mu
    but ~0.05 for sigma across every configuration tried, so these weights
    exist to rebalance. `mu_weight=0.0` trains sigma alone, which isolates
    whether sigma's poor accuracy comes from competing with mu for the
    shared encoder (single-task) or from something else. See conversation
    notes for the full ablation series.
    """
    loss_mu = nn.functional.mse_loss(mu_hat_z, mu_true_z)
    loss_sigma = nn.functional.mse_loss(sigma_hat_z, sigma_true_z)
    return mu_weight * loss_mu + sigma_weight * loss_sigma
