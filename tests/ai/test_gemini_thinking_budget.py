"""Thinking is off for Gemini calls, and the SDK is new enough to say so.

The flash tier spends around a thousand thinking tokens on a task whose
visible output is sixteen. That, not the headline rate, is where the
fortyfold gap between the lite and flash tiers comes from, and it made the
stronger models look unaffordable when they are not.

`google-genai` 1.x could not turn it off: its `ThinkingConfig` exposed only
`include_thoughts`. The pin was a caret on `^1.0`, so the control was
unreachable until the major version moved.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.video.config.llm_settings import LLMSettings

pytestmark = pytest.mark.unit


def _settings(**kwargs) -> LLMSettings:
    return LLMSettings(
        provider="gemini",
        api_key_env_var="GEMINI_API_KEY",
        models=["gemini-2.5-flash-lite"],
        prompt_template_path="src/ai/prompts/video_script.md",
        **kwargs,
    )


class TestTheSdkExposesTheControl:
    def test_thinking_budget_is_a_field(self) -> None:
        """The whole reason the pin had to move."""
        from google.genai import types

        assert "thinking_budget" in types.ThinkingConfig.model_fields

    def test_the_pin_allows_it(self) -> None:
        import tomllib
        from pathlib import Path

        repo = Path(__file__).resolve().parents[2]
        pinned = tomllib.loads((repo / "pyproject.toml").read_text())
        constraint = pinned["tool"]["poetry"]["dependencies"]["google-genai"]

        assert not constraint.startswith(
            "^1."
        ), "a caret on 1.x forbids 2.x, where thinking_budget lives"


class TestTheBudgetReachesTheApi:
    @staticmethod
    async def _config_sent(settings: LLMSettings):
        from src.ai import llm_client

        generate = AsyncMock()
        generate.return_value.text = "some text"

        with patch.object(llm_client.genai, "Client") as client:
            client.return_value.aio.models.generate_content = generate
            await llm_client._call_gemini(
                "prompt", "gemini-2.5-flash-lite", settings, "key"
            )

        return generate.await_args.kwargs["config"]

    @pytest.mark.asyncio
    async def test_zero_is_sent(self) -> None:
        config = await self._config_sent(_settings(thinking_budget=0))

        assert config.thinking_config is not None
        assert config.thinking_config.thinking_budget == 0

    @pytest.mark.asyncio
    async def test_unset_sends_no_block_at_all(self) -> None:
        """Not the same as sending the block with a null budget.

        The field is what the SDK serialises, so an empty block still asks
        the model to accept a control it may not support.
        """
        config = await self._config_sent(_settings(thinking_budget=None))

        assert config.thinking_config is None

    @pytest.mark.asyncio
    async def test_a_positive_budget_is_passed_through(self) -> None:
        """The field is a cap, not a boolean; a caller may want some."""
        config = await self._config_sent(_settings(thinking_budget=512))

        assert config.thinking_config.thinking_budget == 512


class TestTheShippedConfigDisablesIt:
    def test_the_yaml_sets_zero(self) -> None:
        from pathlib import Path

        import yaml

        repo = Path(__file__).resolve().parents[2]
        raw = yaml.safe_load((repo / "config" / "ai_services.yaml").read_text())

        assert raw["llm_settings"]["thinking_budget"] == 0

    def test_it_survives_the_model(self) -> None:
        """A field the loader drops would leave the YAML as decoration."""
        from pathlib import Path

        import yaml

        repo = Path(__file__).resolve().parents[2]
        raw = yaml.safe_load((repo / "config" / "ai_services.yaml").read_text())
        block = dict(raw["llm_settings"])
        block.pop("fallback_provider", None)

        assert LLMSettings(**block).thinking_budget == 0
