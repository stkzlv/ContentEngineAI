"""Tests for narrator profile sharing with platform metadata generators (#86)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.ai.script_generator import apply_prompt_preambles


class TestApplyPromptPreamblesToCaptions:
    """Verify the preamble stacking works for caption prompts the same way
    it works for script prompts (tested in test_script_templates.py).
    """

    def test_narrator_prepended_to_caption_prompt(self) -> None:
        narrator = "You are a calm tech reviewer. No hype."
        prompt = "Write a YouTube description for {product}."
        result = apply_prompt_preambles(prompt, narrator, None, {})
        assert result.startswith(narrator)
        assert prompt in result

    def test_narrator_plus_pillar_stacked(self) -> None:
        narrator = "NARRATOR"
        preamble = "VALUE PILLAR PREAMBLE"
        prompt = "Write a caption."
        result = apply_prompt_preambles(prompt, narrator, "value", {"value": preamble})
        parts = result.split("\n\n")
        assert parts[0] == narrator
        assert parts[1] == preamble
        assert parts[2] == prompt

    def test_empty_narrator_skipped(self) -> None:
        prompt = "Write a caption."
        result = apply_prompt_preambles(prompt, "", None, {})
        assert result == prompt

    def test_unknown_pillar_skipped_gracefully(self) -> None:
        narrator = "NARRATOR"
        prompt = "Write a caption."
        result = apply_prompt_preambles(prompt, narrator, "unknown", {"value": "VALUE"})
        parts = result.split("\n\n")
        assert len(parts) == 2
        assert parts[0] == narrator
        assert parts[1] == prompt


class TestGenerateWithLlmNarratorIntegration:
    """Verify generate_with_llm passes narrator context through to the prompt."""

    @pytest.mark.asyncio
    async def test_narrator_profile_prepended_to_prompt(self) -> None:
        from unittest.mock import MagicMock

        from src.scraper.amazon.models import ProductData
        from src.scraper.base.models import Platform

        product = ProductData(
            title="Test Product",
            price="$10",
            url="https://example.com",
            platform=Platform.AMAZON,
            description="A test product",
        )

        narrator = "You are a calm tech reviewer."
        captured_prompt = {}

        async def fake_llm_call(
            prompt, model, settings, api_key, session, api_settings
        ):
            captured_prompt["value"] = prompt
            return "Generated caption text"

        with (
            patch(
                "src.ai.platform_metadata.utilities.load_prompt_template",
                return_value="Caption for {FULL_PRODUCT_NAME}: {PRODUCT_DESCRIPTION}",
            ),
            patch(
                "src.ai.platform_metadata.utilities.call_llm_api_with_retry",
                side_effect=fake_llm_call,
            ),
            patch(
                "src.ai.platform_metadata.utilities.fetch_and_select_model",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            from pathlib import Path

            from src.ai.platform_metadata.utilities import generate_with_llm
            from src.video.config.llm_settings import LLMSettings

            settings = MagicMock(spec=LLMSettings)
            settings.provider = "gemini"
            settings.models = ["gemini-2.5-flash"]

            result = await generate_with_llm(
                template_path=Path("fake/template.md"),
                product=product,
                settings=settings,
                api_key="fake-key",
                session=MagicMock(),
                narrator_profile=narrator,
                pillar="value",
                pillar_preambles={"value": "Budget-friendly angle."},
            )

            assert result == "Generated caption text"
            prompt = captured_prompt["value"]
            assert prompt.startswith(narrator)
            assert "Budget-friendly angle." in prompt
            assert "Test Product" in prompt
