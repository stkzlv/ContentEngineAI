"""Configured OpenRouter free models that are gone are skipped.

OpenRouter's free list churns. Every configured fallback model of one install
had been withdrawn, so each retry round spent a 404 on each before reaching
discovery; the configured list was appended to the live one unchecked.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.ai import model_pool
from src.ai.model_pool import configured_live, fetch_and_select_model
from tests.ai.test_model_pool_wiring import MODELS_RESPONSE, _FakeSession


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    monkeypatch.setattr(model_pool, "_LIVE_FREE", {})


def _openrouter(models: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        models=models,
        auto_select_free_model=True,
        model_blocklist=[],
        random_model_selection=False,
    )


def test_without_a_fetch_the_configured_list_stands() -> None:
    settings = _openrouter(["gone/model:free", "vendor/good-instruct:free"])

    assert configured_live(settings) == settings.models


def test_another_provider_is_never_filtered() -> None:
    model_pool._LIVE_FREE["https://openrouter.ai/api/v1"] = {"x:free"}
    gemini = SimpleNamespace(provider="gemini", base_url=None, models=["gemini-x"])

    assert configured_live(gemini) == ["gemini-x"]


@pytest.mark.asyncio
async def test_after_a_fetch_withdrawn_free_models_are_skipped() -> None:
    settings = _openrouter(
        ["gone/model:free", "vendor/good-instruct:free", "paid/model"]
    )

    await fetch_and_select_model(settings, "k", _FakeSession(), None)

    # The withdrawn free one goes; a listed one and a paid one stay.
    assert configured_live(settings) == ["vendor/good-instruct:free", "paid/model"]
    assert {m["id"] for m in MODELS_RESPONSE["data"]} >= {"vendor/good-instruct:free"}


def test_every_fallback_site_filters_the_configured_list() -> None:
    """The loops that append configured models use the filter, not the list."""
    import inspect

    from src.ai import description_generator, script_generator
    from src.ai.platform_metadata import instagram, utilities

    for module in (script_generator, description_generator, utilities, instagram):
        source = inspect.getsource(module)
        assert "for m in fb.models" not in source, module.__name__
        assert "for model in settings.models" not in source, module.__name__
        assert "configured_live(" in source, module.__name__
