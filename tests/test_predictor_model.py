"""Structural tests for src/predictor_model.py -- run locally, no GPU and no
downloaded DeBERTa weights needed (uses a randomly-initialized tiny
DebertaV2 config via AutoModel.from_config). These catch shape/wiring bugs
(wrong pooling, wrong dimensions, broken gradient flow, normalization math),
not model quality -- see docs/Phase2_Predictor_Training_Plan.md's
verification plan, step 1.

Requires torch + transformers; skipped entirely if torch isn't installed
(this repo's own dev environment doesn't have it -- see conversation notes;
intended to run on the HPC training machine).
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    from transformers import AutoModel, DebertaV2Config

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

if _TORCH_AVAILABLE:
    from src.predictor_model import (
        LengthDistributionPredictor,
        NormalizeStats,
        _masked_max_pool,
        _masked_mean_pool,
        predictor_loss,
    )


def _tiny_encoder():
    """A randomly-initialized, tiny DebertaV2 model -- same architecture
    family and interface (`.config.hidden_size`, `.last_hidden_state`) as
    real deberta-v3-base, but ~1000x smaller and requires no download.
    """
    config = DebertaV2Config(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        position_buckets=32,
    )
    return AutoModel.from_config(config)


@unittest.skipUnless(_TORCH_AVAILABLE, "torch not installed in this environment")
class TestPoolingFunctions(unittest.TestCase):
    def test_masked_mean_pool_ignores_padding(self):
        # batch=1, seq=3, hidden=2; last position is padding with an
        # extreme value that must NOT influence the mean.
        hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [1000.0, 1000.0]]])
        mask = torch.tensor([[1, 1, 0]])
        pooled = _masked_mean_pool(hidden, mask)
        self.assertTrue(torch.allclose(pooled, torch.tensor([[2.0, 2.0]])))

    def test_masked_max_pool_ignores_padding(self):
        hidden = torch.tensor([[[1.0, -5.0], [3.0, -1.0], [1000.0, 1000.0]]])
        mask = torch.tensor([[1, 1, 0]])
        pooled = _masked_max_pool(hidden, mask)
        self.assertTrue(torch.allclose(pooled, torch.tensor([[3.0, -1.0]])))


@unittest.skipUnless(_TORCH_AVAILABLE, "torch not installed in this environment")
class TestLengthDistributionPredictor(unittest.TestCase):
    def setUp(self):
        self.model = LengthDistributionPredictor(_tiny_encoder())
        self.batch_size, self.seq_len = 4, 16
        self.input_ids = torch.randint(0, 128, (self.batch_size, self.seq_len))
        self.attention_mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.long)
        self.attention_mask[:, -3:] = 0  # some trailing padding, like real batches

    def test_forward_output_shapes(self):
        mu_hat, sigma_hat = self.model(self.input_ids, self.attention_mask)
        self.assertEqual(mu_hat.shape, (self.batch_size,))
        self.assertEqual(sigma_hat.shape, (self.batch_size,))

    def test_loss_is_finite_scalar(self):
        mu_hat, sigma_hat = self.model(self.input_ids, self.attention_mask)
        mu_true = torch.zeros(self.batch_size)
        sigma_true = torch.zeros(self.batch_size)
        loss = predictor_loss(mu_hat, sigma_hat, mu_true, sigma_true)
        self.assertEqual(loss.shape, ())
        self.assertTrue(torch.isfinite(loss))

    def test_gradients_flow_into_encoder(self):
        """The plan calls for fine-tuning the whole encoder, not just the
        MLP heads -- a frozen/disconnected encoder would silently make
        training a no-op on its parameters. Catch that here.
        """
        mu_hat, sigma_hat = self.model(self.input_ids, self.attention_mask)
        loss = predictor_loss(mu_hat, sigma_hat, torch.zeros(self.batch_size), torch.zeros(self.batch_size))
        loss.backward()

        encoder_params = list(self.model.encoder.parameters())
        self.assertTrue(len(encoder_params) > 0)
        self.assertTrue(any(p.grad is not None and torch.any(p.grad != 0) for p in encoder_params))


@unittest.skipUnless(_TORCH_AVAILABLE, "torch not installed in this environment")
class TestNormalizeStats(unittest.TestCase):
    def test_mu_roundtrip(self):
        stats = NormalizeStats.fit(mus=[1.0, 2.0, 3.0, 4.0, 5.0], sigmas=[0.1, 0.2, 0.3, 0.4, 0.5])
        for mu in [1.0, 3.0, 4.7]:
            self.assertAlmostEqual(stats.denormalize_mu(stats.normalize_mu(mu)), mu, places=6)

    def test_sigma_roundtrip(self):
        stats = NormalizeStats.fit(mus=[1.0, 2.0, 3.0], sigmas=[0.05, 0.5, 1.9])
        for sigma in [0.05, 0.5, 1.2]:
            self.assertAlmostEqual(float(stats.denormalize_sigma(stats.normalize_sigma(sigma))), sigma, places=5)

    def test_save_load_roundtrip(self):
        import tempfile

        stats = NormalizeStats.fit(mus=[1.0, 2.0, 3.0], sigmas=[0.1, 0.2, 0.3])
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            stats.save(path)
            loaded = NormalizeStats.load(path)
            self.assertEqual(stats, loaded)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
