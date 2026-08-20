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


@dataclass
class LogTFit:
    mu: float
    sigma: float
    nu: float = DEFAULT_NU
    log_likelihood: float = 0.0


def _neg_log_likelihood(params: np.ndarray, log_samples: np.ndarray, nu: float) -> float:
    mu, log_sigma = params
    # Reparameterize sigma as exp(log_sigma) so the unconstrained optimizer
    # can't drive sigma <= 0.
    sigma = math.exp(log_sigma)
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
    if sample_std == 0.0:
        return LogTFit(mu=mu0, sigma=_DEGENERATE_SIGMA_FLOOR, nu=nu, log_likelihood=float("nan"))

    x0 = np.array([mu0, math.log(sample_std)])

    result = optimize.minimize(
        _neg_log_likelihood,
        x0,
        args=(log_samples, nu),
        method="L-BFGS-B",
    )
    mu_hat, log_sigma_hat = result.x
    return LogTFit(
        mu=float(mu_hat),
        sigma=float(math.exp(log_sigma_hat)),
        nu=nu,
        log_likelihood=float(-result.fun),
    )
