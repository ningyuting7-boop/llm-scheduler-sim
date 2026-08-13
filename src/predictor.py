"""Prediction-error injection shared by every experiment that needs a
predicted (rather than ground-truth) output length: Predicted SJF.

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
other property of the request -- a constant uncertainty for every request
would cancel out of any score comparison that weights it and do nothing.
"""

from __future__ import annotations

import math
import random
from typing import Tuple

from scipy.stats import norm

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


# ---------------------------------------------------------------------------
# Log-normal + CVaR model (approximates TIE's log-t + CVaR; see
# docs/Week2_3_Plan.md section 11 for the full derivation). Log-normal is
# TIE's own second-best-fitting family (Table 1: 60.3% KS pass rate vs
# log-t's 93.1%), chosen here because E[X] and CVaR_alpha[X] have closed
# forms -- no Monte Carlo / numerical integration needed, unlike log-t.
# ---------------------------------------------------------------------------

# Calibrated so a 95/5 Gaussian mixture N(0,SIGMA1^2)/N(0,SIGMA2^2) jointly
# matches Choi et al. 2025 ("ELIS") Table 2's fine-tuned-BGE numbers:
# MAE=71.48, RMSE=101.29 (both real, not independently chosen -- a plain
# Gaussian can't match both at once, since Gaussian forces RMSE/MAE~=1.253,
# but ELIS's ratio is 101.29/71.48~=1.417, i.e. heavier-tailed than Gaussian).
# Solved via scipy.optimize.fsolve; see the two equations in the module
# docstring history / conversation notes.
_MIXTURE_WEIGHT_MAIN = 0.95
_MIXTURE_SIGMA_MAIN = 78.77
_MIXTURE_SIGMA_TAIL = 295.52


def sample_calibrated_error(rng: random.Random) -> float:
    """eps for a "normal" (non-contaminated) request: a 95/5 Gaussian
    mixture jointly calibrated to ELIS's real MAE and RMSE (see constants
    above), not a single Gaussian (which can't match both simultaneously)."""
    if rng.random() < _MIXTURE_WEIGHT_MAIN:
        return rng.gauss(0.0, _MIXTURE_SIGMA_MAIN)
    return rng.gauss(0.0, _MIXTURE_SIGMA_TAIL)


def lognormal_expectation(mu: float, sigma: float) -> float:
    """E[X] for X ~ LogNormal(mu, sigma)."""
    return math.exp(mu + sigma * sigma / 2.0)


def lognormal_cvar(mu: float, sigma: float, alpha: float = 0.9) -> float:
    """CVaR_alpha[X] = E[X | X >= VaR_alpha(X)] for X ~ LogNormal(mu, sigma).
    Closed form: E[X] * Phi(sigma - Phi^-1(alpha)) / (1 - alpha). Verified
    against Monte Carlo (2M samples) to match to within simulation noise."""
    expectation = lognormal_expectation(mu, sigma)
    z = norm.ppf(alpha)
    return expectation * norm.cdf(sigma - z) / (1.0 - alpha)
