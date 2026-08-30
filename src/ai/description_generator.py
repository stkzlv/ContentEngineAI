"""Video Description Generator Module

This module handles the generation of video descriptions for social media platforms
using Large Language Models (LLMs). It interfaces with external LLM APIs like OpenRouter
to create engaging, platform-optimized descriptions based on product data.

Key features:
- Template-based prompt formatting with product data injection
- Multiple model support with automatic fallback mechanisms
- Robust error handling and retry logic
- Platform-specific optimization for TikTok, YouTube, Instagram

The generated descriptions include relevant hashtags and are optimized for social media.
"""

import asyncio
import logging
import random
import re
from pathlib import Path

import aiohttp
from aiohttp.client_exceptions import (
    ClientError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Configure module logger
from src.ai.prompt_selection import prompt_path_for
from src.scraper.amazon.scraper import ProductData
from src.utils import ensure_dirs_exist
from src.utils.circuit_breaker import llm_circuit_breaker
from src.video.config.llm_settings import LLMSettings

logger = logging.getLogger(__name__)


class DescriptionGenerationError(Exception):
    """Exception raised for errors during description generation process.

    This custom exception is used to encapsulate various errors that might occur
    during the description generation process, including API failures, model errors,
    or content filtering issues.
    """

    pass


def load_prompt_template(path: Path) -> str:
    """Load a prompt template from a file.

    Args:
    ----
        path: Path to the prompt template file

    Returns:
    -------
        The prompt template as a string

    Raises:
    ------
        FileNotFoundError: If the template file doesn't exist

    """
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def format_prompt(
    template: str,
    product: ProductData,
    video_script: str | None = None,
    extra_placeholders: dict[str, str] | None = None,
) -> str:
    """Format the prompt template with product data.

    Replaces placeholders in the template with actual product data. The template
    should contain the following placeholders:
    - {FULL_PRODUCT_NAME}: The product title
    - {PRODUCT_DESCRIPTION}: The product description
    - {VIDEO_SCRIPT}: Optional. The generated spoken script for the video. When
      a caller passes this, prompts can read the script (e.g. to mirror the
      closing engagement-bait line into the platform caption). Substitutes the
      empty string when not provided.

    Args:
    ----
        template: The prompt template string
        product: Product data object containing title and description
        video_script: Optional generated script text. When provided, the
            `{VIDEO_SCRIPT}` placeholder is substituted with this text;
            otherwise it substitutes the empty string. Extra `str.format`
            kwargs are silently ignored by templates that don't reference
            the placeholder, so this stays backwards compatible.
        extra_placeholders: Optional prompt-specific substitutions, for values
            that come from config rather than from the product (e.g. the hook
            headline's word budget). Without this a prompt has to hardcode the
            number, which then silently diverges from the config field that
            claims to control it.

    Returns:
    -------
        Formatted prompt string ready for LLM submission

    Raises:
    ------
        ValueError: If the template contains placeholders that can't be filled

    """
    try:
        return template.format(
            FULL_PRODUCT_NAME=product.title or "Product",
            PRODUCT_DESCRIPTION=product.description or "No description available",
            VIDEO_SCRIPT=video_script or "",
            # Neutral aliases, matching what the script-template renderer
            # already supplies. Two renderers offering different placeholder
            # sets for the same record is how a topic prompt written against
            # the documented names dies with "Missing placeholder". Passing
            # them always is harmless: `str.format` ignores keys a template
            # does not reference, so no product prompt changes.
            TOPIC_TITLE=product.title or "Topic",
            TOPIC_DETAIL=product.description or "No description available",
            **(extra_placeholders or {}),
        )
    except KeyError as e:
        raise ValueError(f"Missing placeholder in template: {e}") from e


def save_debug_prompt(prompt: str, path: Path):
    """Save the formatted prompt to a file for debugging purposes.

    Args:
    ----
        prompt: The formatted prompt string
        path: Path where the debug prompt should be saved

    """
    try:
        ensure_dirs_exist(path)
        with path.open("w", encoding="utf-8") as f:
            f.write(prompt)
        logger.debug(f"Saved formatted prompt to {path}")
    except Exception as e:
        logger.error(f"Failed to save debug prompt to {path}: {e}", exc_info=True)


async def _fetch_and_select_model(
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


async def _discover_any_free_model(
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


async def _call_llm_api_with_retry(
    prompt: str,
    model: str,
    settings: LLMSettings,
    api_key: str,
    session: aiohttp.ClientSession,
    api_settings=None,
) -> str:
    """Call the LLM API with manual retry logic for async functions."""
    # Get retry settings
    # Exponential backoff multiplier
    multiplier = api_settings.llm_retry_multiplier if api_settings else 2
    # Minimum wait in seconds
    min_wait = api_settings.llm_retry_min_wait_sec if api_settings else 1
    # Maximum wait in seconds
    max_wait = api_settings.llm_retry_max_wait_sec if api_settings else 30
    # Number of retry attempts
    attempts = api_settings.llm_retry_attempts if api_settings else 3

    last_exception = None

    for attempt in range(attempts):
        try:
            return await _call_llm_api(
                prompt, model, settings, api_key, session, api_settings
            )
        except (TimeoutError, ClientError, DescriptionGenerationError) as e:
            last_exception = e
            if attempt < attempts - 1:  # Don't sleep on the last attempt
                wait_time = min(max_wait, min_wait * (multiplier**attempt))
                await asyncio.sleep(wait_time)

    # If we get here, all attempts failed
    if last_exception:
        raise last_exception
    else:
        raise DescriptionGenerationError("All retry attempts failed")


async def _call_llm_api(
    prompt: str,
    model: str,
    settings: LLMSettings,
    api_key: str,
    session: aiohttp.ClientSession,
    api_settings=None,
) -> str:
    """Call the LLM API to generate description content.

    Delegates to the shared llm_client which handles provider dispatch
    (OpenRouter vs Gemini).
    """
    from src.ai.llm_client import LLMCallError, call_llm

    try:
        return await call_llm(prompt, model, settings, api_key, session)
    except LLMCallError as e:
        raise DescriptionGenerationError(str(e)) from e


def validate_description_completeness(description: str) -> tuple[bool, str]:
    """Validate if a description appears complete and well-formed.

    Args:
    ----
        description: The generated description text

    Returns:
    -------
        Tuple of (is_complete: bool, reason: str)

    """
    if not description or not description.strip():
        return False, "Description is empty"

    description = description.strip()

    # Check minimum length (descriptions should be substantial)
    if len(description) < 50:
        return False, f"Description too short ({len(description)} chars, minimum 50)"

    # Check for reasonable word count
    words = description.split()
    if len(words) < 10:
        return False, f"Description too few words ({len(words)}, minimum 10)"

    return (
        True,
        f"Description validation passed ({len(words)} words, {len(description)} chars)",
    )


@llm_circuit_breaker
async def generate_description(
    product: ProductData,
    settings: LLMSettings,
    secrets: dict[str, str],
    session: aiohttp.ClientSession,
    intermediate_paths: dict[str, Path],
    debug_mode: bool,
    api_settings=None,
) -> str | None:
    """Generate a video description for a product using LLM.

    This is the main entry point for description generation. It orchestrates the entire
    process:
    1. Validates API credentials
    2. Selects appropriate LLM models to try
    3. Loads and formats the prompt template with product data
    4. Makes API requests to generate the description
    5. Handles fallback to alternative models if needed
    6. Saves debug information when in debug mode
    7. Sanitizes and returns the final description

    The function implements a fallback mechanism that tries multiple models in sequence
    if earlier attempts fail, providing resilience against model-specific issues.

    Args:
    ----
        product: Product data containing title, description, etc.
        settings: LLM configuration settings
        secrets: Dictionary containing API keys and credentials
        session: Shared HTTP session for API requests
        intermediate_paths: Dictionary of paths for saving intermediate files
        debug_mode: Whether to save debug information
        api_settings: Additional API settings for configuration

    Returns:
    -------
        The generated and sanitized description, or None if generation failed

    Raises:
    ------
        DescriptionGenerationError: If description generation fails for all models

    """
    # Get API key from secrets
    api_key = secrets.get(settings.api_key_env_var)
    if not api_key:
        raise DescriptionGenerationError(
            f"Missing API key from environment variable: {settings.api_key_env_var}"
        )

    # Fetch available free models (OpenRouter only; Gemini uses configured models)
    if settings.provider == "openrouter":
        free_models = await _fetch_and_select_model(
            settings, api_key, session, api_settings
        )
    else:
        free_models = []

    # Build prioritized list: free models first, then fallback to configured list
    models_to_try: list[str] = []
    if free_models:
        models_to_try.extend(free_models)

    # Add configured models as fallback (if not already in list)
    for model in settings.models:
        if model not in models_to_try:
            models_to_try.append(model)

    # Ensure we have at least one model to try
    if not models_to_try:
        raise DescriptionGenerationError("No models available to generate description.")

    logger.info(f"Order of models to attempt: {models_to_try}")

    # Load and format the description prompt template
    # Use absolute path to ensure it works regardless of working directory
    project_root = Path(__file__).parent.parent.parent
    template_path = project_root / prompt_path_for(
        product, "src/ai/prompts/video_description.md"
    )
    try:
        template = load_prompt_template(template_path)
        prompt = format_prompt(template, product)
    except (FileNotFoundError, ValueError) as e:
        raise DescriptionGenerationError(f"Prompt template error: {e}") from e

    if debug_mode and "formatted_prompt" in intermediate_paths:
        save_debug_prompt(prompt, intermediate_paths["formatted_prompt"])

    for model in models_to_try:
        # Try each model up to 2 times to handle incomplete responses
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                attempt_suffix = (
                    f" (attempt {attempt + 1}/{max_attempts})"
                    if max_attempts > 1
                    else ""
                )
                logger.info(f"Trying LLM model: {model}{attempt_suffix}")
                description_text = await _call_llm_api_with_retry(
                    prompt, model, settings, api_key, session, api_settings
                )

                # Clean the description (remove code blocks, etc.)
                clean_description = re.sub(r"```[\w\s]*", "", description_text).strip()

                # Validate description completeness
                is_complete, validation_reason = validate_description_completeness(
                    clean_description
                )
                if is_complete:
                    logger.info(
                        f"Description successfully generated with model: {model} - "
                        f"{validation_reason}"
                    )
                    return clean_description
                else:
                    logger.warning(
                        f"Description incomplete from {model}: {validation_reason}"
                    )
                    # Save incomplete description for debugging if in debug mode
                    if debug_mode and "description" in intermediate_paths:
                        model_safe = model.replace("/", "_")
                        file_name = (
                            f"incomplete_description_{model_safe}_"
                            f"attempt_{attempt + 1}.txt"
                        )
                        incomplete_path = (
                            intermediate_paths["description"].parent / file_name
                        )
                        try:
                            with open(incomplete_path, "w", encoding="utf-8") as f:
                                content = (
                                    f"# INCOMPLETE DESCRIPTION - {validation_reason}\n"
                                    f"# Model: {model}, Attempt: {attempt + 1}\n\n"
                                    f"{clean_description}"
                                )
                                f.write(content)
                            logger.debug(
                                f"Saved incomplete description to {incomplete_path}"
                            )
                        except Exception as save_error:
                            logger.warning(
                                f"Could not save incomplete description: {save_error}"
                            )

                    if attempt < max_attempts - 1:
                        logger.info(
                            f"Retrying with {model} for complete description..."
                        )
                        continue
                    else:
                        logger.warning(
                            f"Model {model} produced incomplete description after "
                            f"{max_attempts} attempts"
                        )
                        break

            except Exception as e:
                logger.warning(f"Model {model} failed{attempt_suffix}: {e}")
                if attempt < max_attempts - 1:
                    logger.info(f"Retrying {model} after error...")
                    continue
                else:
                    break

    # Fallback: try discovering any free model not yet attempted (OpenRouter only)
    if settings.provider == "openrouter" and settings.fallback_discover_any_free:
        already_tried = set(models_to_try)
        fallback_models = await _discover_any_free_model(
            settings, api_key, session, api_settings, already_tried
        )

        for model in fallback_models:
            try:
                logger.info(f"Fallback: trying discovered model {model}")
                description_text = await _call_llm_api_with_retry(
                    prompt, model, settings, api_key, session, api_settings
                )
                clean_description = re.sub(r"```[\w\s]*", "", description_text).strip()
                is_complete, validation_reason = validate_description_completeness(
                    clean_description
                )
                if is_complete:
                    logger.info(f"Fallback success with {model} - {validation_reason}")
                    return clean_description
                else:
                    logger.warning(f"Fallback {model} incomplete: {validation_reason}")
            except Exception as e:
                logger.warning(f"Fallback model {model} failed: {e}")
                continue

    # Provider fallback: try fallback_provider if primary exhausted all models
    if settings.fallback_provider:
        fb = settings.fallback_provider
        fb_api_key = secrets.get(fb.api_key_env_var)
        if fb_api_key:
            logger.info("Primary provider exhausted, falling back to %s", fb.provider)

            if fb.provider == "openrouter":
                fb_free_models = await _fetch_and_select_model(
                    fb, fb_api_key, session, api_settings
                )
            else:
                fb_free_models = []

            fb_models: list[str] = list(fb_free_models)
            for m in fb.models:
                if m not in fb_models:
                    fb_models.append(m)

            for model in fb_models:
                try:
                    logger.info("Fallback provider: trying %s", model)
                    description_text = await _call_llm_api_with_retry(
                        prompt, model, fb, fb_api_key, session, api_settings
                    )
                    clean_description = re.sub(
                        r"```[\w\s]*", "", description_text
                    ).strip()
                    is_complete, reason = validate_description_completeness(
                        clean_description
                    )
                    if is_complete:
                        logger.info("Fallback success with %s - %s", model, reason)
                        return clean_description
                    else:
                        logger.warning("Fallback %s incomplete: %s", model, reason)
                except Exception as e:
                    logger.warning("Fallback model %s failed: %s", model, e)
                    continue

            if fb.provider == "openrouter" and fb.fallback_discover_any_free:
                already_tried_fb = set(fb_models)
                discovered = await _discover_any_free_model(
                    fb, fb_api_key, session, api_settings, already_tried_fb
                )
                for model in discovered:
                    try:
                        logger.info("Fallback discovery: trying %s", model)
                        description_text = await _call_llm_api_with_retry(
                            prompt, model, fb, fb_api_key, session, api_settings
                        )
                        clean_description = re.sub(
                            r"```[\w\s]*", "", description_text
                        ).strip()
                        is_complete, reason = validate_description_completeness(
                            clean_description
                        )
                        if is_complete:
                            logger.info(
                                "Fallback discovery success with %s - %s",
                                model,
                                reason,
                            )
                            return clean_description
                    except Exception as e:
                        logger.warning(
                            "Fallback discovered model %s failed: %s", model, e
                        )
                        continue
        else:
            logger.warning(
                "Fallback provider configured but API key %s not found",
                fb.api_key_env_var,
            )

    logger.error("All models failed to generate a description.")
    return None
