"""Tests for the parts of vllm_tie that do not need vLLM or a GPU.

`vllm_tie.predictor` imports torch and transformers but only touches the GPU
inside load_model(), so the pure functions here are testable anywhere. The
scheduler and queue modules import vLLM itself and are covered by the smoke
test on the cluster instead (plan doc task 8).
"""

from __future__ import annotations

import math

import pytest

from vllm_tie import score_calculator as sc
from vllm_tie.predictor import adaptive_beta, extract_user_prompt


class TestExtractUserPrompt:
    """Getting this wrong is a silent train/serve mismatch: the predictor
    would score chat-template boilerplate instead of the user's prompt, and
    nothing would crash.
    """

    def test_qwen3_template_shape(self):
        # What Qwen3's template decodes to with skip_special_tokens=True.
        text = "user\nWhat is the capital of France?\nassistant\n"
        assert extract_user_prompt(text) == "What is the capital of France?"

    def test_with_thinking_block(self):
        # enable_thinking=False inserts an empty think block after the
        # generation prompt; <think> is not a special token so it survives
        # decoding.
        text = "user\nSummarise this.\nassistant\n<think>\n\n</think>\n\n"
        assert extract_user_prompt(text) == "Summarise this."

    def test_with_system_turn(self):
        text = "system\nYou are helpful.\nuser\nHello there\nassistant\n"
        assert extract_user_prompt(text) == "Hello there"

    def test_prompt_containing_assistant_line(self):
        # The reason extraction anchors on the LAST assistant marker: a
        # first-match search would truncate this prompt to "Rewrite this:".
        text = (
            "user\nRewrite this:\nassistant\nsaid hello\nplease make it "
            "formal\nassistant\n"
        )
        assert extract_user_prompt(text) == (
            "Rewrite this:\nassistant\nsaid hello\nplease make it formal"
        )

    def test_multiline_prompt_preserved(self):
        text = "user\nline one\nline two\n\nline four\nassistant\n"
        assert extract_user_prompt(text) == "line one\nline two\n\nline four"

    def test_untemplated_passthrough(self):
        text = "just a bare prompt with no template"
        assert extract_user_prompt(text) == text

    def test_never_discards_real_content(self):
        # Returning "" for a prompt that had content would tokenize to just
        # [CLS][SEP] and be scored as if it carried no signal at all. An
        # all-whitespace prompt legitimately extracts to nothing, so the
        # invariant is about inputs that actually say something.
        for text in ["user\n\nassistant\n", "assistant", "user\n \nassistant\n"]:
            assert extract_user_prompt(text) != ""

    def test_blank_input_is_handled(self):
        assert extract_user_prompt("   ") == ""


class TestAdaptiveBeta:
    """beta = clip(0.1 * Lq / B, 0.1, 0.5), paper Eq. 12."""

    def test_floor_when_queue_empty(self):
        assert adaptive_beta(0) == pytest.approx(0.1)

    def test_ceiling_under_heavy_load(self):
        assert adaptive_beta(10_000) == pytest.approx(0.5)

    def test_scales_between_bounds(self):
        # B defaults to 32: Lq=64 -> 0.1*64/32 = 0.2
        assert adaptive_beta(64) == pytest.approx(0.2)

    def test_monotone_in_queue_depth(self):
        values = [adaptive_beta(n) for n in range(0, 200, 10)]
        assert values == sorted(values)

    def test_env_override_pins_beta(self, monkeypatch):
        """TIE_BETA=0 is the predicted-SJF arm: it must hold at zero
        regardless of queue depth, or the arm silently becomes TIE again."""
        import importlib

        import vllm_tie.predictor as p

        monkeypatch.setenv("TIE_BETA", "0")
        importlib.reload(p)
        try:
            assert p.adaptive_beta(0) == 0.0
            assert p.adaptive_beta(10_000) == 0.0
            # score must then be the censored mean alone, no CVaR term
            assert p.compute_score(4.5, 0.6, 500) == pytest.approx(
                sc.mean(4.5, 0.6), rel=0.05
            )
        finally:
            monkeypatch.delenv("TIE_BETA")
            importlib.reload(p)


class TestScoreCalculator:
    def test_tail_risk_grows_with_sigma(self):
        """The whole premise of TIE: sigma has to move CVaR relative to the
        mean, or the CVaR term carries no information the mean lacks."""
        ratios = [
            sc.cvar(4.0, s, 0.9) / sc.mean(4.0, s) for s in (0.05, 0.3, 0.8, 1.2)
        ]
        assert ratios == sorted(ratios)
        assert ratios[0] < 1.5, "near-deterministic prompt should get little tail penalty"
        assert ratios[-1] > 3.0, "high-sigma prompt should get a large tail penalty"

    def test_censoring_caps_the_mean(self):
        # A prompt whose uncensored mean would be astronomically large still
        # only occupies the GPU for max_tokens of decode.
        assert sc.mean(12.0, 2.0) <= sc.MAX_GENERATED_TOKENS
        assert sc.cvar(12.0, 2.0, 0.9) <= sc.CVAR_MAX_GENERATED_TOKENS

    def test_score_increases_with_beta(self):
        low = sc.tie_score(4.5, 0.6, alpha=0.9, beta=0.1)
        high = sc.tie_score(4.5, 0.6, alpha=0.9, beta=0.5)
        assert high > low

    def test_quantile_matches_closed_form(self):
        # quantile is the one piece with an exact answer to check against.
        from scipy import stats

        mu, sigma, q = 4.0, 0.5, 0.9
        expected = math.exp(mu + sigma * stats.t.ppf(q, df=sc.NU))
        assert sc.quantile(mu, sigma, q) == pytest.approx(expected)

    def test_percentage_and_fraction_agree(self):
        assert sc.quantile(4.0, 0.5, 90) == pytest.approx(sc.quantile(4.0, 0.5, 0.9))

    def test_rejects_out_of_range_alpha(self):
        with pytest.raises(ValueError):
            sc.cvar(4.0, 0.5, alpha=0.0)
