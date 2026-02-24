"""Tests for the provider-abstracted LLM client."""

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from src.ai.llm_client import LLMCallError, _call_gemini, _call_openrouter, call_llm
from src.video.config.llm_settings import LLMSettings


def _make_settings(provider: str = "openrouter", **overrides) -> LLMSettings:
    defaults = {
        "api_key_env_var": "TEST_KEY",
        "models": ["test-model"],
        "prompt_template_path": "test.md",
        "max_tokens": 100,
        "temperature": 0.5,
        "timeout_seconds": 10,
        "auto_select_free_model": False,
        "fallback_discover_any_free": False,
    }
    defaults.update(overrides)
    return LLMSettings(provider=provider, **defaults)


class TestCallLlmDispatch:
    """Test that call_llm dispatches to the right provider."""

    async def test_dispatches_to_openrouter(self):
        settings = _make_settings("openrouter", base_url="https://test.api/v1")
        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(
            return_value={"choices": [{"message": {"content": "Generated script"}}]}
        )
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        session = MagicMock(spec=aiohttp.ClientSession)
        session.closed = False
        session.post = MagicMock(return_value=mock_resp)

        result = await call_llm("prompt", "model", settings, "key", session)
        assert result == "Generated script"
        session.post.assert_called_once()

    async def test_dispatches_to_gemini(self):
        settings = _make_settings("gemini")

        mock_response = MagicMock()
        mock_response.text = "Gemini response"

        mock_generate = AsyncMock(return_value=mock_response)

        with patch("src.ai.llm_client.genai") as mock_genai:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = mock_generate
            mock_genai.Client.return_value = mock_client
            mock_genai.types = MagicMock()

            result = await call_llm("prompt", "model", settings, "key")
            assert result == "Gemini response"
            mock_genai.Client.assert_called_once_with(api_key="key")


class TestCallOpenRouter:
    """Test OpenRouter-specific behavior."""

    async def test_empty_response_raises(self):
        settings = _make_settings("openrouter", base_url="https://test.api/v1")
        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(
            return_value={"choices": [{"message": {"content": ""}}]}
        )
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        session = MagicMock(spec=aiohttp.ClientSession)
        session.closed = False
        session.post = MagicMock(return_value=mock_resp)

        with pytest.raises(LLMCallError, match="Empty content"):
            await _call_openrouter("prompt", "model", settings, "key", session)

    async def test_uses_configured_url_and_params(self):
        settings = _make_settings(
            "openrouter",
            base_url="https://custom.api/v1",
            max_tokens=200,
            temperature=0.9,
        )
        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(
            return_value={"choices": [{"message": {"content": "ok"}}]}
        )
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        session = MagicMock(spec=aiohttp.ClientSession)
        session.closed = False
        session.post = MagicMock(return_value=mock_resp)

        await _call_openrouter("test prompt", "gpt-4", settings, "mykey", session)

        call_args = session.post.call_args
        assert "https://custom.api/v1/chat/completions" in call_args[0]
        payload = call_args[1]["json"]
        assert payload["max_tokens"] == 200
        assert payload["temperature"] == 0.9
        assert payload["model"] == "gpt-4"


class TestCallGemini:
    """Test Gemini-specific behavior."""

    async def test_empty_response_raises(self):
        settings = _make_settings("gemini")

        mock_response = MagicMock()
        mock_response.text = ""

        mock_generate = AsyncMock(return_value=mock_response)

        with patch("src.ai.llm_client.genai") as mock_genai:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = mock_generate
            mock_genai.Client.return_value = mock_client
            mock_genai.types = MagicMock()

            with pytest.raises(LLMCallError, match="Empty content"):
                await _call_gemini("prompt", "model", settings, "key")

    async def test_timeout_raises(self):
        import asyncio

        settings = _make_settings("gemini", timeout_seconds=1)

        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(10)

        with patch("src.ai.llm_client.genai") as mock_genai:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = slow_generate
            mock_genai.Client.return_value = mock_client
            mock_genai.types = MagicMock()

            with pytest.raises(LLMCallError, match="timed out"):
                await _call_gemini("prompt", "model", settings, "key")

    async def test_api_error_wraps_in_llm_call_error(self):
        settings = _make_settings("gemini")

        async def failing_generate(*args, **kwargs):
            raise RuntimeError("API quota exceeded")

        with patch("src.ai.llm_client.genai") as mock_genai:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = failing_generate
            mock_genai.Client.return_value = mock_client
            mock_genai.types = MagicMock()

            with pytest.raises(LLMCallError, match="Gemini API error"):
                await _call_gemini("prompt", "model", settings, "key")


class TestProviderValidation:
    """Test that invalid providers are rejected."""

    def test_invalid_provider_rejected(self):
        with pytest.raises(ValueError, match="Input should be"):
            _make_settings("invalid_provider")

    def test_openrouter_accepted(self):
        s = _make_settings("openrouter")
        assert s.provider == "openrouter"

    def test_gemini_accepted(self):
        s = _make_settings("gemini")
        assert s.provider == "gemini"


class TestFallbackProvider:
    """Test fallback_provider field on LLMSettings."""

    def test_fallback_provider_defaults_to_none(self):
        s = _make_settings("gemini")
        assert s.fallback_provider is None

    def test_fallback_provider_nested_settings(self):
        fb = _make_settings(
            "openrouter",
            base_url="https://openrouter.ai/api/v1",
            auto_select_free_model=True,
            fallback_discover_any_free=True,
            models=["model-a:free", "model-b:free"],
        )
        s = _make_settings("gemini", fallback_provider=fb)
        assert s.fallback_provider is not None
        assert s.fallback_provider.provider == "openrouter"
        assert s.fallback_provider.auto_select_free_model is True
        assert len(s.fallback_provider.models) == 2

    def test_fallback_provider_from_dict(self):
        """Test that YAML-style nested dict works with Pydantic."""
        s = LLMSettings(
            provider="gemini",
            api_key_env_var="GEMINI_API_KEY",
            models=["gemini-2.5-flash-lite"],
            prompt_template_path="test.md",
            fallback_provider={
                "provider": "openrouter",
                "api_key_env_var": "OPENROUTER_API_KEY",
                "models": ["free-model:free"],
                "prompt_template_path": "test.md",
                "base_url": "https://openrouter.ai/api/v1",
                "auto_select_free_model": True,
            },
        )
        assert s.fallback_provider.provider == "openrouter"
        assert s.fallback_provider.base_url == "https://openrouter.ai/api/v1"

    def test_no_double_nesting(self):
        """Fallback provider's own fallback should be None by default."""
        s = LLMSettings(
            provider="gemini",
            api_key_env_var="GEMINI_API_KEY",
            models=["gemini-2.5-flash-lite"],
            prompt_template_path="test.md",
            fallback_provider={
                "provider": "openrouter",
                "api_key_env_var": "OPENROUTER_API_KEY",
                "models": ["free-model:free"],
                "prompt_template_path": "test.md",
            },
        )
        assert s.fallback_provider.fallback_provider is None

    async def test_fallback_dispatches_to_openrouter(self):
        """When primary is gemini, fallback settings should dispatch to openrouter."""
        fb_settings = _make_settings(
            "openrouter", base_url="https://openrouter.ai/api/v1"
        )

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(
            return_value={"choices": [{"message": {"content": "Fallback response"}}]}
        )
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        session = MagicMock(spec=aiohttp.ClientSession)
        session.closed = False
        session.post = MagicMock(return_value=mock_resp)

        result = await call_llm(
            "prompt", "fallback-model", fb_settings, "or-key", session
        )
        assert result == "Fallback response"
