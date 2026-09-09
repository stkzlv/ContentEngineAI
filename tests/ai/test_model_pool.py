"""The discovered-model filter (#404).

A reasoning model from OpenRouter's free pool answered a description request
with its chain of thought, and the result was scheduled to three platforms.
The entries below are the shapes the live ``/models`` response actually
returns, including the two models named in that failure.
"""

from src.ai.model_pool import model_reject_reason


def _model(**overrides):
    base = {
        "id": "vendor/model:free",
        "architecture": {"output_modalities": ["text"]},
        "reasoning": None,
    }
    base.update(overrides)
    return base


class TestUsableModels:
    """A model that answers in text without narrating is kept."""

    def test_plain_text_model(self):
        assert model_reject_reason(_model()) is None

    def test_reasoning_supported_but_off_by_default(self):
        # google/gemma-4-31b-it:free
        model = _model(reasoning={"mandatory": False, "default_enabled": False})
        assert model_reject_reason(model) is None

    def test_reasoning_declared_without_a_default(self):
        # nvidia/nemotron-3.5-lightning:free
        assert model_reject_reason(_model(reasoning={"mandatory": False})) is None

    def test_unknown_shape_is_not_rejected(self):
        """An unrecognised response must not empty the fallback pool."""
        assert model_reject_reason({"id": "vendor/model"}) is None
        assert model_reject_reason(_model(architecture={})) is None
        assert model_reject_reason(_model(architecture=None)) is None


class TestRejectedModels:
    """Models that cannot produce a description this pipeline can publish."""

    def test_reasoning_on_by_default(self):
        # thinkingmachines/inkling:free, one of the two models in #404.
        model = _model(
            reasoning={
                "mandatory": False,
                "default_enabled": True,
                "default_effort": "high",
            }
        )
        reason = model_reject_reason(model)
        assert reason is not None
        assert "reasons" in reason

    def test_reasoning_mandatory(self):
        # liquid/lfm-2.5-2.6b:free
        reason = model_reject_reason(_model(reasoning={"mandatory": True}))
        assert reason is not None
        assert "cannot be turned off" in reason

    def test_reasoning_effort_defaults_above_none(self):
        """default_effort is a third spelling of "on unless turned off"."""
        # nex-agi/nex-n2.5-pro:free declares no default_enabled at all.
        model = _model(
            reasoning={
                "mandatory": False,
                "supported_efforts": ["high", "medium", "none"],
                "default_effort": "high",
            }
        )
        reason = model_reject_reason(model)
        assert reason is not None
        assert "high" in reason

    def test_effort_of_none_is_kept(self):
        model = _model(reasoning={"mandatory": False, "default_effort": "none"})
        assert model_reject_reason(model) is None

    def test_non_text_output(self):
        # google/lyria-3-pro-preview, the other model in #404: a music model
        # sitting in the free pool. Its output modalities include text, so a
        # filter asking merely whether text is present would keep it.
        model = _model(
            id="google/lyria-3-pro-preview",
            architecture={"output_modalities": ["text", "audio"]},
        )
        reason = model_reject_reason(model)
        assert reason is not None
        assert "not text alone" in reason

    def test_output_modality_is_checked_before_reasoning(self):
        model = _model(
            architecture={"output_modalities": ["image"]},
            reasoning={"mandatory": True},
        )
        assert "not text alone" in (model_reject_reason(model) or "")
