"""Provider-abstracted LLM client.

Dispatches to OpenRouter (aiohttp) or Gemini (google-genai SDK)
based on LLMSettings.provider.
"""

import asyncio
import logging

import aiohttp
from google import genai  # type: ignore[import-untyped]

from src.video.config.llm_settings import LLMSettings

logger = logging.getLogger(__name__)


class LLMCallError(Exception):
    """Raised when an LLM API call fails."""


async def call_llm(
    prompt: str,
    model: str,
    settings: LLMSettings,
    api_key: str,
    session: aiohttp.ClientSession | None = None,
    api_settings=None,
) -> str:
    """Call the configured LLM provider and return generated text."""
    if settings.provider == "gemini":
        return await _call_gemini(prompt, model, settings, api_key)
    return await _call_openrouter(prompt, model, settings, api_key, session)


async def _call_openrouter(
    prompt: str,
    model: str,
    settings: LLMSettings,
    api_key: str,
    session: aiohttp.ClientSession | None,
) -> str:
    """Call OpenRouter via aiohttp (OpenAI chat/completions format)."""
    api_url = (
        f"{(settings.base_url or 'https://openrouter.ai/api/v1').rstrip('/')}"
        f"/chat/completions"
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
    }

    if session is None or session.closed:  # type: ignore[attr-defined]
        logger.warning("Session is closed during API call, getting new session")
        from src.utils.connection_pool import get_http_session

        session = await get_http_session()

    async with session.post(
        api_url,
        headers=headers,
        json=payload,
        timeout=settings.timeout_seconds,
    ) as response:
        response.raise_for_status()
        data = await response.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content or not content.strip():
            raise LLMCallError("Empty content in OpenRouter response")
        return str(content)


async def _call_gemini(
    prompt: str,
    model: str,
    settings: LLMSettings,
    api_key: str,
) -> str:
    """Call Google Gemini via google-genai SDK."""
    client = genai.Client(api_key=api_key)

    config = genai.types.GenerateContentConfig(
        max_output_tokens=settings.max_tokens,
        temperature=settings.temperature,
    )

    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            ),
            timeout=settings.timeout_seconds,
        )
    except TimeoutError as e:
        raise LLMCallError(
            f"Gemini API timed out after {settings.timeout_seconds}s"
        ) from e
    except Exception as e:
        raise LLMCallError(f"Gemini API error: {e}") from e

    if not response.text or not response.text.strip():
        raise LLMCallError("Empty content in Gemini response")

    return str(response.text)
