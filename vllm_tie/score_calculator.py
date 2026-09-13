"""TIE scheduling score for a log-t(mu, sigma, nu=3.5) output-length
distribution: ``score = E[X] + beta * CVaR_alpha[X]`` (Zheng et al., ICML
2026, Eq. 11).

Adapted from the TIE reference implementation
(``vllm/v1/core/sched/ua_score_calculator.py`` in the authors' vLLM fork).
Trimmed to the log-t case only -- the reference also carries log-normal and
learned-nu variants that phase 2 does not produce parameters for.

Both quantities are **censored**, because vLLM stops generating at
``max_tokens``: a request whose untruncated draw would be 10,000 tokens
still only occupies the GPU for ``MAX_GENERATED_TOKENS`` of decode. Scoring
it by its uncensored mean would overstate how long it actually blocks the
batch. The two censoring limits differ by 20 tokens in the reference
implementation and are kept as-is for fidelity.
"""

from __future__ import annotations

import os

import numpy as np
from scipy import stats

MAX_GENERATED_TOKENS = 2028.0
CVAR_MAX_GENERATED_TOKENS = 2048.0
NU = 3.5
N_SAMPLES = 10_000

# The reference implementation draws from the global numpy RNG, which makes
# scores irreproducible across runs. A dedicated seeded generator costs the
# same and lets a benchmark run be replayed exactly; set TIE_SCORE_SEED to
# an integer to fix it, or leave unset for nondeterministic draws.
_seed_env = os.environ.get("TIE_SCORE_SEED")
_RNG = np.random.default_rng(int(_seed_env) if _seed_env else None)


def _draw(mu: float, sigma: float, n_samples: int) -> np.ndarray:
    """n_samples draws from log-t(mu, sigma, nu=NU)."""
    t_samples = stats.t.rvs(df=NU, size=n_samples, random_state=_RNG)
    return np.exp(mu + sigma * t_samples)


def quantile(mu: float, sigma: float, q: float) -> float:
    """Uncensored q-quantile. q may be given in (0,1) or as a percentage."""
    if q > 1:
        q = q / 100
    if not 0 < q < 1:
        raise ValueError(f"q must be in (0,1) or (0,100), got {q}")
    return float(np.exp(mu + sigma * stats.t.ppf(q, df=NU)))


def mean(mu: float, sigma: float, n_samples: int = N_SAMPLES) -> float:
    """Censored expectation E[min(X, MAX_GENERATED_TOKENS)]."""
    x = np.minimum(_draw(mu, sigma, n_samples), MAX_GENERATED_TOKENS)
    return float(np.mean(x))


def cvar(mu: float, sigma: float, alpha: float, n_samples: int = N_SAMPLES) -> float:
    """Censored CVaR_alpha[X]: the mean of the worst (1-alpha) tail.

    alpha may be given in (0,1) or as a percentage.
    """
    if alpha > 1:
        alpha = alpha / 100
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0,1) or (0,100), got {alpha}")
    x = np.minimum(_draw(mu, sigma, n_samples), CVAR_MAX_GENERATED_TOKENS)
    var = np.percentile(x, alpha * 100)
    exceedances = x[x > var]
    # Empty when censoring has flattened the whole upper tail onto the cap,
    # i.e. VaR is already the cap; the tail mean is then the cap itself.
    if len(exceedances) == 0:
        return float(var)
    return float(np.mean(exceedances))


def tie_score(mu: float, sigma: float, alpha: float, beta: float) -> float:
    """E[X] + beta * CVaR_alpha[X] (paper Eq. 11)."""
    return mean(mu, sigma) + beta * cvar(mu, sigma, alpha)
