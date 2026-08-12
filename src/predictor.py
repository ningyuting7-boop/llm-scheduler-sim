"""Prediction-error injection shared by every experiment that needs a
predicted (rather than ground-truth) output length: Predicted SJF and ARRS.

Every constant here is traceable to a source -- see docs/Week2_3_Plan.md
section 9.7 for the full derivation and a worked numeric example. Summary:

- REFERENCE_SCALE = 76 words: converted from RMSE=101.29 tokens, the
  fine-tuned prompt-only BGE predictor's measured error on LMSYS-Chat-1M in
  Choi et al. 2025 ("ELIS", arXiv:2505.09142, Table 2), using the standard
  ~0.75 words/token ratio. This is a REAL measured error, not an assumption.
- QUALITY_TIERS: k multipliers are ratios of ELIS Table 2's three reported
  RMSEs to that same reference (iterative predictor / fine-tuned BGE /
  pretrained-not-finetuned BGE), so "low/realistic/high" aren't arbitrary
  labels -- each is pinned to a specific number in that table.
- L_MAX = 3072 words: the same output-length cap used in
  scripts/extract_lmsys_lengths.py (MAX_WORD_COUNT) -- a different quantity
  (max plausible true length) from REFERENCE_SCALE (typical predictor
  error), used only to clip predicted_length into a physically valid range.
  Keep these two constants in sync if either changes.

Noise is UNIFORM, not Gaussian: predicted_length is drawn uniformly from
[true_length - sigma, true_length + sigma], so the error is guaranteed to
never exceed sigma in magnitude (a Gaussian tail could, rarely, exceed it).

`sigma` (and therefore `uncertainty`) varies PER REQUEST via an independent
draw `u ~ Uniform(0, 1)`, deliberately uncorrelated with true_length or any
other property of the request -- this is what makes ARRS's `beta * uncertainty`
term actually discriminate between requests (a constant uncertainty for every
request cancels out of every score comparison and does nothing).
"""

from __future__ import annotations

import math
import random
from typing import Tuple

MIN_PREDICTED_LENGTH = 1.0

# See module docstring: 101.29 token RMSE (Choi et al. 2025, Table 2, "Fine-tuned
# BGE") * 0.75 words/token.
REFERENCE_SCALE = 76.0

# L_max used only to clip predicted_length; must match
# scripts/extract_lmsys_lengths.py's MAX_WORD_COUNT.
L_MAX = 3072.0

# k = (this tier's ELIS-reported RMSE) / (REFERENCE_SCALE's own RMSE, i.e.
# the "realistic" tier) -- ratios of real numbers in Choi et al. 2025 Table 2,
# not independently chosen. "realistic" is 1.0 by construction (it IS the
# tier REFERENCE_SCALE was computed from). Values beyond this dict (e.g. k=5,
# 10, ...) are legitimate for stress-testing but are NOT grounded in ELIS --
# label them as hypothetical/adversarial, not as another measured tier.
QUALITY_TIERS = {
    "low": 34.33 / 101.29,  # iterative predictor (uses partial output, not just the prompt)
    "realistic": 1.0,  # fine-tuned BGE, prompt-only -- the scenario closest to ours
    "high": 224.98 / 101.29,  # pretrained BGE, not fine-tuned (R^2 < 0, worse than guessing the mean)
}


def predict_length(true_length: int, k: float, rng: random.Random, skew: float = 0.0) -> Tuple[float, float]:
    """Return (predicted_length, uncertainty) for a request with true_length,
    under a predictor whose quality tier is `k` (see QUALITY_TIERS; k=1.0
    reproduces the real fine-tuned-BGE-on-LMSYS error scale from ELIS).

    u ~ Uniform(0, 1)              -- per-request confidence draw, independent
                                       of true_length and of every other request
    sigma = u * k * REFERENCE_SCALE -- this request's uncertainty
    predicted_length ~ Uniform(true_length - sigma*(1+skew), true_length + sigma*(1-skew)),
        clipped to [MIN_PREDICTED_LENGTH, L_MAX]
    uncertainty = sigma

    `skew` (default 0.0, symmetric -- the honest default, since we have no
    real evidence that uncertain predictions skew toward underestimating)
    is a DELIBERATELY CONSTRUCTED scenario, not a claim about real predictor
    behavior: skew=1.0 means high-uncertainty requests can only ever be
    underestimated (range becomes [L-2*sigma, L]), never overestimated. This
    is what makes `uncertainty` informative about the *direction* of risk,
    not just its magnitude -- see docs/Week2_3_Plan.md section 9.7 for why a
    symmetric skew=0 (uncertainty uncorrelated with error direction) can
    mathematically never improve average-case latency: penalizing it always
    delays the ~50% of high-uncertainty requests that turn out to be harmless
    overestimates exactly as much as it (correctly) delays the dangerous
    underestimates, canceling out on average.
    """
    u = rng.uniform(0.0, 1.0)
    sigma = u * k * REFERENCE_SCALE
    predicted_length = rng.uniform(true_length - sigma * (1 + skew), true_length + sigma * (1 - skew))
    predicted_length = min(L_MAX, max(MIN_PREDICTED_LENGTH, predicted_length))
    return predicted_length, sigma


# Mean real LMSYS output length (words), from data/lmsys_output_lengths.csv.
# Used only to convert ELIS's absolute RMSE into a relative (log-space) error
# scale below -- an approximation (treating relative RMSE as a lognormal
# coefficient-of-variation proxy), not an exact statistic from any paper.
_MEAN_TRUE_LENGTH = 119.52

# relative_rmse = REFERENCE_SCALE / mean(true_length) = 76 / 119.52 ~= 0.636.
# For a lognormal-ish variable, CV = sqrt(exp(s^2) - 1); solving for s given
# CV = relative_rmse gives the log-space sigma at k=1 ("realistic").
_RELATIVE_RMSE = REFERENCE_SCALE / _MEAN_TRUE_LENGTH
REFERENCE_SCALE_LOG = math.sqrt(math.log(1.0 + _RELATIVE_RMSE ** 2))


def predict_length_lognormal(true_length: int, k: float, rng: random.Random) -> Tuple[float, float]:
    """Alternative to predict_length: noise is symmetric in LOG space, not
    in raw word-count space.

    u ~ Uniform(0, 1)
    sigma_log = u * k * REFERENCE_SCALE_LOG
    eps ~ Uniform(-sigma_log, sigma_log)        -- symmetric in log space
    predicted_length = clip(true_length * exp(eps), MIN_PREDICTED_LENGTH, L_MAX)

    Two things fall out of this for free, without any hand-picked skew
    parameter:

    1. It avoids the multiplicative model's earlier saturation problem
       (see module docstring history / docs/Week2_3_Plan.md section 9.2):
       comparing predicted_i vs predicted_j reduces to comparing
       log(true_i)+eps_i vs log(true_j)+eps_j -- an ADDITIVE relationship in
       log space, so growing sigma_log keeps adding disorder to the ranking
       instead of canceling out.
    2. Because exp() is convex, E[true_length | predicted_length, sigma_log]
       = predicted_length * sinh(sigma_log)/sigma_log > predicted_length,
       and grows with sigma_log (Jensen's inequality) -- i.e. the higher the
       uncertainty, the more the true length is expected to exceed the
       predicted one, *purely from the log transform*, with no separate
       "skew toward underestimation" assumption bolted on. This is the same
       structural property that makes CVaR meaningful in Zheng et al. 2026
       ("TIE", arXiv:2604.00499): a length distribution that's bounded below
       and unbounded above is inherently right-skewed, which is also true of
       our own LMSYS data (measured skewness = 2.945, see data/lmsys_output_lengths.csv).

    `uncertainty` is reported in the same absolute (words) units as
    predicted_length -- the exact expected gap E[true - predicted | ...]
    implied by the model above -- so it plugs into ARRS's existing
    `predicted + beta * uncertainty` score without rescaling beta.
    """
    u = rng.uniform(0.0, 1.0)
    sigma_log = u * k * REFERENCE_SCALE_LOG
    eps = rng.uniform(-sigma_log, sigma_log)
    predicted_length = true_length * math.exp(eps)
    predicted_length = min(L_MAX, max(MIN_PREDICTED_LENGTH, predicted_length))

    expected_ratio = (math.sinh(sigma_log) / sigma_log) if sigma_log > 0 else 1.0
    uncertainty = predicted_length * (expected_ratio - 1.0)
    return predicted_length, uncertainty
