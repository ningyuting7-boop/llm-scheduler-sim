"""Regression tests for src/logt_fit.py.

Validates the MLE fitting code itself against synthetic data generated
from a *known* (mu, sigma, nu) log-t distribution -- this needs no GPU
and no real dataset, so it can catch bugs in the optimizer/likelihood
before spending HPC time on the real labeling pipeline.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logt_fit import DEFAULT_NU, fit_logt


def _sample_logt(mu: float, sigma: float, nu: float, size: int, seed: int) -> np.ndarray:
    """X = exp(mu + sigma*T), T ~ Student-t(df=nu)."""
    rng = np.random.default_rng(seed)
    t_samples = stats.t.rvs(df=nu, size=size, random_state=rng)
    return np.exp(mu + sigma * t_samples)


class TestFitLogT(unittest.TestCase):
    def test_recovers_known_parameters_with_many_samples(self) -> None:
        true_mu, true_sigma = 3.0, 0.5
        samples = _sample_logt(true_mu, true_sigma, DEFAULT_NU, size=5000, seed=0)

        fit = fit_logt(samples, nu=DEFAULT_NU)

        self.assertAlmostEqual(fit.mu, true_mu, delta=0.05)
        self.assertAlmostEqual(fit.sigma, true_sigma, delta=0.05)
        self.assertEqual(fit.nu, DEFAULT_NU)

    def test_recovers_known_parameters_at_paper_sample_size(self) -> None:
        # The real pipeline only takes 20 samples per prompt (matching the
        # paper); at this size we expect real estimation noise, so the
        # tolerance is loose -- this just checks the fit lands in the right
        # ballpark and doesn't diverge, not that it's statistically tight.
        true_mu, true_sigma = 4.0, 0.6
        samples = _sample_logt(true_mu, true_sigma, DEFAULT_NU, size=20, seed=1)

        fit = fit_logt(samples, nu=DEFAULT_NU)

        self.assertAlmostEqual(fit.mu, true_mu, delta=0.5)
        self.assertGreater(fit.sigma, 0.0)

    def test_handles_all_identical_samples_without_optimizer(self) -> None:
        # Regression test: this happened for real in Phase 1 pilot data
        # when several prompts' completions all hit the max_tokens cap.
        # The optimizer must not be trusted here (see _DEGENERATE_SIGMA_FLOOR
        # comment) -- check we get a small positive sigma, not an arbitrary
        # unconverged value.
        fit = fit_logt([2048, 2048, 2048])

        self.assertAlmostEqual(fit.mu, math.log(2048), places=9)
        self.assertGreater(fit.sigma, 0.0)
        self.assertLess(fit.sigma, 0.5)

    def test_rejects_non_positive_samples(self) -> None:
        with self.assertRaises(ValueError):
            fit_logt([1.0, 2.0, -3.0, 4.0])

    def test_rejects_too_few_samples(self) -> None:
        with self.assertRaises(ValueError):
            fit_logt([1.0])


if __name__ == "__main__":
    unittest.main()
