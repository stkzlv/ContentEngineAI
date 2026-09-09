"""The model filter is actually applied at both discovery sites (#404).

`model_reject_reason` being correct buys nothing if the discovery loops do not
call it, and both generators carry their own byte-identical copy of those
loops. These drive the real functions against a crafted `/models` response
rather than asserting the symbol appears in the source.
"""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from src.video.config.llm_settings import LLMSettings

MODELS_RESPONSE = {
    "data": [
        {
            "id": "vendor/good-instruct:free",
            "context_length": 128000,
            "pricing": {"prompt": "0", "completion": "0"},
            "architecture": {"output_modalities": ["text"]},
            "reasoning": {"mandatory": False, "default_enabled": False},
        },
        {
            "id": "vendor/thinker-instruct:free",
            "context_length": 128000,
            "pricing": {"prompt": "0", "completion": "0"},
            "architecture": {"output_modalities": ["text"]},
            "reasoning": {"mandatory": False, "default_enabled": True},
        },
        {
            "id": "vendor/music-chat:free",
            "context_length": 128000,
            "pricing": {"prompt": "0", "completion": "0"},
            "architecture": {"output_modalities": ["text", "audio"]},
            "reasoning": None,
        },
    ]
}


class _FakeSession:
    """Minimal stand-in for the aiohttp session the discovery loops use."""

    closed = False

    @asynccontextmanager
    async def _response(self):
        yield SimpleNamespace(
            raise_for_status=lambda: None,
            json=self._json,
        )

    async def _json(self):
        return MODELS_RESPONSE

    def get(self, *_args, **_kwargs):
        return self._response()


@pytest.fixture
def settings():
    return LLMSettings(
        provider="openrouter",
        api_key_env_var="OPENROUTER_API_KEY",
        prompt_template_path="src/ai/prompts/video_description.md",
        models=["vendor/configured-but-absent"],
        auto_select_free_model=True,
        model_blocklist=[],
        min_context_length=8000,
    )


@pytest.fixture(params=["description_generator", "script_generator"])
def generator(request):
    import importlib

    return importlib.import_module(f"src.ai.{request.param}")


@pytest.mark.asyncio
async def test_auto_selection_drops_filtered_models(generator, settings):
    found = await generator._fetch_and_select_model(
        settings, "key", _FakeSession(), None
    )

    assert "vendor/good-instruct:free" in found
    assert "vendor/thinker-instruct:free" not in found
    assert "vendor/music-chat:free" not in found


@pytest.mark.asyncio
async def test_last_resort_discovery_drops_filtered_models(generator, settings):
    found = await generator._discover_any_free_model(
        settings, "key", _FakeSession(), None, set()
    )

    assert "vendor/good-instruct:free" in found
    assert "vendor/thinker-instruct:free" not in found
    # The last-resort loop has no instruct/chat name filter, so this is the
    # only thing keeping a music model out of it.
    assert "vendor/music-chat:free" not in found
