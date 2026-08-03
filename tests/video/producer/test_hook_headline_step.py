"""Tests for the authored hook headline step wiring (#160, roadmap 1.9).

The headline must survive the resume path. It lives as a top-level
``ctx.state`` key, which the partial-state loader drops, and a product scripted
before the feature existed never had one, so regenerating it when missing is
what keeps the hook from silently reverting to the script's first sentence.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.video.producer.steps import _ensure_hook_headline


def _ctx(*, state=None, hook_enabled=True, max_words=7):
    """Minimal duck-typed context: only the attributes the step reads."""
    return SimpleNamespace(
        state=dict(state or {}),
        script="Some spoken script. Second sentence.",
        product=SimpleNamespace(asin="B0TEST", title="Test product"),
        secrets={"GEMINI_API_KEY": "k"},
        session=None,
        debug_mode=False,
        config=SimpleNamespace(
            llm_settings=SimpleNamespace(
                script_templates=SimpleNamespace(
                    narrator_profile="narrator", pillar_preambles={}
                )
            ),
            api_settings=None,
            video_settings=SimpleNamespace(
                hook_overlay=SimpleNamespace(enabled=hook_enabled, max_words=max_words)
            ),
        ),
    )


@pytest.mark.asyncio
class TestEnsureHookHeadline:
    async def test_generates_when_missing(self) -> None:
        ctx = _ctx()
        with patch(
            "src.video.producer.steps.generate_hook_headline",
            new_callable=AsyncMock,
            return_value="This $15 hub wins",
        ) as mock_gen:
            await _ensure_hook_headline(ctx, None)
        mock_gen.assert_awaited_once()
        assert ctx.state["hook_headline"] == "This $15 hub wins"

    async def test_regenerates_on_resume_when_state_lost(self) -> None:
        """A truncated state file drops the top-level key; regenerate it.

        This is the resume path that previously returned before the headline
        block ever ran, silently reverting the overlay to the script sentence.
        """
        ctx = _ctx(state={"generate_script": {"status": "done"}})
        with patch(
            "src.video.producer.steps.generate_hook_headline",
            new_callable=AsyncMock,
            return_value="Cheap hub beats dock",
        ) as mock_gen:
            await _ensure_hook_headline(ctx, None)
        mock_gen.assert_awaited_once()
        assert ctx.state["hook_headline"] == "Cheap hub beats dock"

    async def test_reuses_existing_headline(self) -> None:
        ctx = _ctx(state={"hook_headline": "Already authored"})
        with patch(
            "src.video.producer.steps.generate_hook_headline",
            new_callable=AsyncMock,
        ) as mock_gen:
            await _ensure_hook_headline(ctx, None)
        mock_gen.assert_not_awaited()
        assert ctx.state["hook_headline"] == "Already authored"

    async def test_skips_llm_call_when_overlay_disabled(self) -> None:
        """A disabled overlay must not cost an LLM round-trip per product."""
        ctx = _ctx(hook_enabled=False)
        with patch(
            "src.video.producer.steps.generate_hook_headline",
            new_callable=AsyncMock,
        ) as mock_gen:
            await _ensure_hook_headline(ctx, None)
        mock_gen.assert_not_awaited()
        assert "hook_headline" not in ctx.state

    async def test_empty_result_leaves_fallback_in_place(self) -> None:
        ctx = _ctx()
        with patch(
            "src.video.producer.steps.generate_hook_headline",
            new_callable=AsyncMock,
            return_value="",
        ):
            await _ensure_hook_headline(ctx, None)
        assert "hook_headline" not in ctx.state

    async def test_passes_configured_max_words(self) -> None:
        ctx = _ctx(max_words=5)
        with patch(
            "src.video.producer.steps.generate_hook_headline",
            new_callable=AsyncMock,
            return_value="short one here",
        ) as mock_gen:
            await _ensure_hook_headline(ctx, "value")
        call = mock_gen.await_args
        assert call is not None
        assert call.kwargs["max_words"] == 5
        assert call.kwargs["pillar"] == "value"
