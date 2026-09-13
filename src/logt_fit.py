"""Joint MLE fit of a log-t(mu, sigma | nu fixed) distribution to a set of
positive samples, following TIE (Zheng et al. 2026, arXiv:2604.00499) Eq. 5-6.

X is log-t distributed with parameters (mu, sigma, nu) if ln(X) = mu + sigma*T,
T ~ Student-t(df=nu). Per the paper, nu is fixed at 3.5 (not fit per-request);
only (mu, sigma) are estimated from each prompt's sampled output lengths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import optimize, stats

DEFAULT_NU = 3.5

# Used only when every sample is numerically identical (zero variance in
# log-space, e.g. every completion happened to hit the same token cap).
# The true MLE in that degenerate case pushes sigma -> 0 (an unbounded,
# ever-decreasing objective in log_sigma -- see fit_logt), which L-BFGS-B
# can't be trusted to find correctly since there's no local curvature to
# search with. Returning a small floor instead of running the optimizer
# avoids silently reporting an arbitrary, unconverged value.
_DEGENERATE_SIGMA_FLOOR = 0.05

# np.std of N numerically-identical float64 values is essentially never
# exactly 0.0 (mean-subtraction rounding leaves ~1e-16-scale noise), so the
# degenerate-sample check below must use a threshold, not exact equality --
# see docs/Phase2_Predictor_Training_Plan.md / conversation notes: `==0.0`
# silently missed every genuinely-degenerate prompt in the first labeling
# pass, letting the optimizer fit sigma to floating-point noise instead of
# to _DEGENERATE_SIGMA_FLOOR.
_DEGENERATE_STD_THRESHOLD = 1e-8

# Lower bound on log_sigma passed to L-BFGS-B: with nu=3.5 (heavy tails),
# the likelihood surface can have a spurious tall, narrow peak near
# sigma->0 that out-scores a moderate, honest sigma by treating a minority
# of differently-valued samples as extreme tail draws instead of genuine
# dispersion (verified against data/qwen3_8b_length_samples.csv: e.g. 20
# samples split ~15/20 at 2048 and ~5/20 in the 1057-1384 range fit to
# sigma=7.59e-06, i.e. "no uncertainty", which is not a credible summary of
# that data). Bounding log_sigma away from that region prevents the
# optimizer from ever landing there.
_MIN_LOG_SIGMA = math.log(0.01)

# Above this ratio, fitted sigma is considered a credible refinement of the
# raw log-space sample std; below it, treated as evidence L-BFGS-B still
# fell into the near-zero pathology despite the bound above, and we fall
# back to a bias-corrected estimate rather than a fit that ignores most of
# the observed dispersion. (This is a residual safety net, not the primary
# fix -- the bound above should make it rarely fire.)
_MIN_FIT_TO_RAW_STD_RATIO = 0.1


@dataclass
class LogTFit:
    mu: float
    sigma: float
    nu: float = DEFAULT_NU
    log_likelihood: float = 0.0


def _neg_log_likelihood(params: np.ndarray, log_samples: np.ndarray, nu: float) -> float:
    mu, log_sigma = params
    # Reparameterize sigma as exp(log_sigma) so the unconstrained optimizer
    # can't drive sigma <= 0. Use np.exp (returns inf, not an exception, on
    # overflow) since L-BFGS-B's line search can probe arbitrarily large
    # log_sigma while exploring -- math.exp would crash the whole fit on
    # that probe instead of just scoring it as a bad (very high NLL) point.
    sigma = np.exp(log_sigma)
    z = (log_samples - mu) / sigma
    # Eq. 5: ln t_nu(z) - ln(sigma) - ln(x), the last two terms from the
    # change-of-variables Jacobian for X = exp(mu + sigma*T).
    log_pdf_t = stats.t.logpdf(z, df=nu)
    log_likelihood = float(np.sum(log_pdf_t - log_sigma - log_samples))
    return -log_likelihood


def fit_logt(samples: Sequence[float], nu: float = DEFAULT_NU) -> LogTFit:
    """Fit (mu, sigma) by maximizing Eq. 5-6's log-likelihood via L-BFGS-B.

    `samples` must be strictly positive (e.g. observed output token counts).
    """
    samples_arr = np.asarray(samples, dtype=float)
    if samples_arr.size < 2:
        raise ValueError("need at least 2 samples to fit (mu, sigma)")
    if np.any(samples_arr <= 0):
        raise ValueError("log-t fit requires strictly positive samples")

    log_samples = np.log(samples_arr)
    mu0 = float(np.mean(log_samples))
    sample_std = float(np.std(log_samples))
    if sample_std < _DEGENERATE_STD_THRESHOLD:
        return LogTFit(mu=mu0, sigma=_DEGENERATE_SIGMA_FLOOR, nu=nu, log_likelihood=float("nan"))

    x0 = np.array([mu0, math.log(sample_std)])

    result = optimize.minimize(
        _neg_log_likelihood,
        x0,
        args=(log_samples, nu),
        method="L-BFGS-B",
        bounds=[(None, None), (_MIN_LOG_SIGMA, None)],
    )
    mu_hat, log_sigma_hat = result.x
    sigma_hat = float(np.exp(log_sigma_hat))

    # Safety net: nu=3.5's heavy tails can still let L-BFGS-B prefer
    # explaining real dispersion as tail draws over a moderate sigma (see
    # _MIN_LOG_SIGMA comment). If the fit ignores most of the observed
    # spread, fall back to a method-of-moments estimate: for Student-t(nu),
    # Var[T] = nu/(nu-2), so sample_std (which estimates sigma*sqrt(Var[T]))
    # is rescaled by sqrt((nu-2)/nu) to estimate sigma itself.
    if nu > 2 and sigma_hat < sample_std * _MIN_FIT_TO_RAW_STD_RATIO:
        sigma_hat = sample_std * math.sqrt((nu - 2) / nu)

    return LogTFit(
        mu=float(mu_hat),
        sigma=sigma_hat,
        nu=nu,
        log_likelihood=float(-result.fun),
    )
