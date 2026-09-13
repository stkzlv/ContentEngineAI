"""Filters for the OpenRouter free-model pool.

The pool is whatever OpenRouter currently lists as free. It is unaudited, it
changes under the account, and two kinds of entry in it cannot do this
pipeline's job:

- A model that does not emit text. The pool carries music and image models
  beside the chat ones, and a request to one either errors or returns
  something no validator downstream is written against.
- A model that reasons in its visible output. Its monologue about writing a
  description is longer and wordier than a description, so every length floor
  a validator applies is cleared by a wide margin (#404).

Both are declared by the models API, so they are decided here from the
response rather than from a blocklist that has to chase a moving pool. The
blocklist stays for the case this cannot see: a model that answers in the
right shape and answers badly.

Unknown is treated as usable throughout. A response shape this does not
recognise must not empty the pool, because the pool is the fallback that keeps
a batch running when the primary provider is rate-limited.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any

from aiohttp.client_exceptions import ClientError

if TYPE_CHECKING:
    import aiohttp

    from src.video.config import LLMSettings

logger = logging.getLogger(__name__)

__all__ = [
    "discover_any_free_model",
    "fetch_and_select_model",
    "model_reject_reason",
]


def _output_modality_reason(model: dict[str, Any]) -> str | None:
    """Reject a model that advertises any output modality other than text."""
    architecture = model.get("architecture")
    if not isinstance(architecture, dict):
        return None
    modalities = architecture.get("output_modalities")
    if not isinstance(modalities, list) or not modalities:
        return None
    if [m for m in modalities if m != "text"]:
        return f"outputs {'/'.join(str(m) for m in modalities)}, not text alone"
    return None


def _reasoning_reason(model: dict[str, Any]) -> str | None:
    """Reject a model whose reasoning is on unless a caller turns it off.

    Three spellings of the same thing appear in the live response, and a model
    matching any of them emits its reasoning by default: ``mandatory``,
    ``default_enabled``, and a ``default_effort`` above ``none``. A model that
    merely *supports* reasoning is kept, because it does not use it unless
    asked to and nothing here asks.
    """
    reasoning = model.get("reasoning")
    if not isinstance(reasoning, dict):
        return None
    if reasoning.get("mandatory") is True:
        return "reasons in its output and cannot be turned off"
    if reasoning.get("default_enabled") is True:
        return "reasons in its output by default"
    effort = reasoning.get("default_effort")
    if isinstance(effort, str) and effort.lower() not in ("", "none"):
        return f"reasons in its output by default (effort '{effort}')"
    return None


def model_reject_reason(model: dict[str, Any]) -> str | None:
    """Return why this discovered model is unusable here, or None if it is.

    ``model`` is one entry of the OpenRouter ``/models`` response.
    """
    return _output_modality_reason(model) or _reasoning_reason(model)


async def fetch_and_select_model(
    settings: LLMSettings, api_key: str, session: aiohttp.ClientSession, api_settings
) -> list[str]:
    """Fetches available models from OpenRouter and returns free models to try.

    Args:
    ----
        settings: LLM settings configuration
        api_key: API key for authentication
        session: HTTP session for API calls
        api_settings: Additional API settings

    Returns:
    -------
        List of free model IDs to try (ordered or shuffled based on settings)

    """
    if not settings.auto_select_free_model:
        logger.info("Auto-selection of free model is disabled in settings.")
        return []

    api_url = (
        f"{(settings.base_url or 'https://openrouter.ai/api/v1').rstrip('/')}/models"
    )
    headers = {"Authorization": f"Bearer {api_key}"}

    logger.info("Fetching available models from OpenRouter...")
    try:
        # Check if session is None or closed and get a new one if needed
        if session is None or session.closed:  # type: ignore[attr-defined]
            logger.warning(
                "Session is closed, getting a new session from connection pool"
            )
            from src.utils.connection_pool import get_http_session

            session = await get_http_session()

        timeout = (
            api_settings.llm_model_fetch_timeout_sec
            if api_settings
            else 30  # Default timeout in seconds
        )
        async with session.get(api_url, headers=headers, timeout=timeout) as response:  # type: ignore[attr-defined]
            response.raise_for_status()
            data = await response.json()

            blocklist = set(settings.model_blocklist)

            # Build set of ALL free model IDs (for checking configured models)
            all_free_ids: set[str] = set()
            # Build set of discoverable free models (instruct/chat only)
            discoverable_free: set[str] = set()

            if "data" in data and isinstance(data["data"], list):
                for model in data["data"]:
                    pricing = model.get("pricing", {})
                    if (
                        pricing.get("prompt") == "0"
                        and pricing.get("completion") == "0"
                    ):
                        model_id = model.get("id")
                        if model_id and model_id not in blocklist:
                            all_free_ids.add(model_id)
                            # Only auto-discover instruct/chat models
                            if "instruct" in model_id or "chat" in model_id:
                                reject = model_reject_reason(model)
                                if reject:
                                    logger.debug(
                                        "Skipping discovered model %s: %s",
                                        model_id,
                                        reject,
                                    )
                                else:
                                    discoverable_free.add(model_id)

            if not all_free_ids:
                logger.warning("No free models found from API. Using fallback list.")
                return []

            # Configured models that are verified free (priority)
            ordered_models = [m for m in settings.models if m in all_free_ids]

            # Additional discoverable free models not in config
            extra_free = [m for m in discoverable_free if m not in ordered_models]

            if settings.random_model_selection:
                # Shuffle for random selection
                combined = ordered_models + extra_free
                random.shuffle(combined)
                logger.info(
                    f"Found {len(combined)} free models (random order): "
                    f"{combined[:3]}..."
                )
                return combined
            else:
                # Keep configured order, append extras at end
                result = ordered_models + extra_free
                logger.info(
                    f"Found {len(result)} free models (priority order): "
                    f"{result[:3]}..."
                )
                return result

    except (TimeoutError, ClientError) as e:
        logger.error(f"Failed to fetch models: {e}. Using fallback list.")
        return []
    except Exception as e:
        logger.error(
            f"Unexpected error fetching models: {e}. Using fallback list.",
            exc_info=True,
        )
        return []


async def discover_any_free_model(
    settings: LLMSettings,
    api_key: str,
    session: aiohttp.ClientSession,
    api_settings,
    already_tried: set[str],
) -> list[str]:
    """Fallback: discover free models from OpenRouter, excluding tiny models.

    This is used as a last resort when all configured/discovered models fail.
    Excludes models smaller than 7B parameters to avoid hallucination issues.

    Args:
    ----
        settings: LLM settings configuration
        api_key: API key for authentication
        session: HTTP session for API calls
        api_settings: Additional API settings
        already_tried: Set of model IDs that have already been attempted

    Returns:
    -------
        List of free model IDs not yet tried (sorted by size descending)

    """
    api_url = (
        f"{(settings.base_url or 'https://openrouter.ai/api/v1').rstrip('/')}/models"
    )
    headers = {"Authorization": f"Bearer {api_key}"}

    blocklist = set(settings.model_blocklist)
    min_ctx = settings.min_context_length

    logger.info("Fallback: discovering available free models (excluding tiny)...")
    try:
        if session is None or session.closed:  # type: ignore[attr-defined]
            from src.utils.connection_pool import get_http_session

            session = await get_http_session()

        timeout = api_settings.llm_model_fetch_timeout_sec if api_settings else 30
        async with session.get(api_url, headers=headers, timeout=timeout) as response:  # type: ignore[attr-defined]
            response.raise_for_status()
            data = await response.json()

            # Collect free models with size/context filtering
            candidates: list[tuple[str, int]] = []
            if "data" in data and isinstance(data["data"], list):
                for model in data["data"]:
                    pricing = model.get("pricing", {})
                    if (
                        pricing.get("prompt") == "0"
                        and pricing.get("completion") == "0"
                    ):
                        model_id = model.get("id")
                        context_length = model.get("context_length", 0)

                        # Skip if already tried, blocklisted, or too small
                        if not model_id:
                            continue
                        if model_id in already_tried:
                            continue
                        if model_id in blocklist:
                            logger.debug(f"Skipping blocklisted model: {model_id}")
                            continue
                        if context_length < min_ctx:
                            logger.debug(
                                f"Skipping small model: {model_id} "
                                f"(context={context_length})"
                            )
                            continue
                        reject = model_reject_reason(model)
                        if reject:
                            logger.debug(
                                "Skipping discovered model %s: %s", model_id, reject
                            )
                            continue

                        candidates.append((model_id, context_length))

            # Sort by context length descending (larger models first)
            candidates.sort(key=lambda x: x[1], reverse=True)
            all_free = [model_id for model_id, _ in candidates]

            if all_free:
                logger.info(
                    f"Fallback discovered {len(all_free)} untried free models: "
                    f"{all_free[:5]}..."
                )
            else:
                logger.warning("Fallback: no additional free models available")

            return all_free

    except Exception as e:
        logger.error(f"Fallback discovery failed: {e}")
        return []
