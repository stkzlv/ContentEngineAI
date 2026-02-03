"""Script Generator Module

This module handles the generation of promotional scripts for e-commerce product videos
using Large Language Models (LLMs). It interfaces with external LLM APIs like OpenRouter
to create compelling, persuasive scripts based on product data.

Key features:
- Template-based prompt formatting with product data injection
- Multiple model support with automatic fallback mechanisms
- Robust error handling and retry logic
- Debug output for prompt inspection

The generated scripts are optimized for conversion to speech via TTS systems
and serve as the foundation for the video's voiceover and subtitles.
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

from src.scraper.amazon.scraper import ProductData
from src.utils import ensure_dirs_exist
from src.utils.circuit_breaker import openrouter_circuit_breaker
from src.video.config import LLMSettings, config

# Configure module logger
logger = logging.getLogger(__name__)


class ScriptGenerationError(Exception):
    """Exception raised for errors during script generation process.

    This custom exception is used to encapsulate various errors that might occur
    during the script generation process, including API failures, model errors,
    or content filtering issues.
    """

    pass


# Using LLM parameters from config


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


def format_prompt(template: str, product: ProductData, audience: str) -> str:
    """Format the prompt template with product data and audience information.

    Replaces placeholders in the template with actual product data. The template
    should contain the following placeholders:
    - {FULL_PRODUCT_NAME}: The product title
    - {PRODUCT_DESCRIPTION}: The product description
    - {AUDIENCE}: The target audience for the video

    Args:
    ----
        template: The prompt template string
        product: Product data object containing title and description
        audience: Target audience for the video

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
            AUDIENCE=audience,
            PRICE=product.price or "Price not available",
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

            # Blocklist of models known to produce poor results
            blocklist = {
                "liquid/lfm-2.5-1.2b-instruct:free",  # 1.2B - hallucinates
                "liquid/lfm-2.5-1.2b-instruct",
            }

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

    # Minimum context length to filter out tiny models (proxy for model size)
    MIN_CONTEXT_LENGTH = 8000  # Small models often have small contexts

    # Blocklist of models known to produce poor results
    BLOCKLIST = {
        "liquid/lfm-2.5-1.2b-instruct:free",  # 1.2B - hallucinates
        "liquid/lfm-2.5-1.2b-instruct",
    }

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
                        if model_id in BLOCKLIST:
                            logger.debug(f"Skipping blocklisted model: {model_id}")
                            continue
                        if context_length < MIN_CONTEXT_LENGTH:
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


def _create_retry_decorator(api_settings):
    """Create a retry decorator with configurable parameters."""
    # Exponential backoff multiplier
    multiplier = api_settings.llm_retry_multiplier if api_settings else 2
    # Minimum wait in seconds
    min_wait = api_settings.llm_retry_min_wait_sec if api_settings else 1
    # Maximum wait in seconds
    max_wait = api_settings.llm_retry_max_wait_sec if api_settings else 30
    # Number of retry attempts
    attempts = api_settings.llm_retry_attempts if api_settings else 3

    return retry(  # type: ignore[operator]
        wait=wait_exponential(multiplier=multiplier, min=min_wait, max=max_wait),
        stop=stop_after_attempt(attempts),
        retry=retry_if_exception_type(
            (ClientError, asyncio.TimeoutError, ScriptGenerationError)
        ),
        reraise=True,
    )


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
        except (TimeoutError, ClientError, ScriptGenerationError) as e:
            last_exception = e
            if attempt < attempts - 1:  # Don't sleep on the last attempt
                wait_time = min(max_wait, min_wait * (multiplier**attempt))
                await asyncio.sleep(wait_time)

    # If we get here, all attempts failed
    if last_exception:
        raise last_exception
    else:
        raise ScriptGenerationError("All retry attempts failed")


async def _call_llm_api(
    prompt: str,
    model: str,
    settings: LLMSettings,
    api_key: str,
    session: aiohttp.ClientSession,
    api_settings=None,
) -> str:
    """Call the LLM API to generate script content.

    This function makes an asynchronous request to the LLM API (typically OpenRouter)
    to generate a script based on the provided prompt.

    Args:
    ----
        prompt: The formatted prompt to send to the LLM
        model: The specific LLM model identifier to use
        settings: LLM configuration settings
        api_key: API key for authentication
        session: Shared aiohttp client session for making requests
        api_settings: Additional API settings for configuration

    Returns:
    -------
        The generated script text

    Raises:
    ------
        ScriptGenerationError: If the response is empty or invalid
        ClientError: If there's an HTTP error communicating with the API
        asyncio.TimeoutError: If the API request times out

    """
    # Construct API URL, defaulting to OpenRouter if not specified
    api_url = f"{(settings.base_url or 'https://openrouter.ai/api/v1').rstrip('/')}/chat/completions"

    # Set up authentication and content headers
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Prepare request payload with model parameters
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": config.llm_settings.max_tokens,
        "temperature": config.llm_settings.temperature,
    }

    # Check if session is None or closed and get a new one if needed
    if session is None or session.closed:  # type: ignore[attr-defined]
        logger.warning("Session is closed during API call, getting new session")
        from src.utils.connection_pool import get_http_session

        session = await get_http_session()

    # Make the API request
    async with session.post(
        api_url,
        headers=headers,
        json=payload,
        timeout=config.llm_settings.timeout_seconds,
    ) as response:
        response.raise_for_status()  # Raise exception for HTTP errors
        data = await response.json()

        # Extract the generated content from the response
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content or not content.strip():
            raise ScriptGenerationError("Empty content in response")

        return str(content)


def validate_script_completeness(script: str) -> tuple[bool, str]:
    """Validate if a script appears complete and well-formed.

    Args:
    ----
        script: The generated script text

    Returns:
    -------
        Tuple of (is_complete: bool, reason: str)

    """
    if not script or not script.strip():
        return False, "Script is empty"

    # Check if script ends mid-sentence (common truncation indicators)
    script = script.strip()
    incomplete_endings = [
        # Mid-sentence endings
        " and",
        " the",
        " with",
        " for",
        " to",
        " in",
        " on",
        " at",
        " by",
        " of",
        # Incomplete words
        "ing",
        "ed",
        "er",
        "ly",
        "tion",
        "ness",
        "ment",
        "able",
        "ible",
        # Partial phrases
        "charging",
        "battery",
        "comfort",
        "design",
        "quality",
        "performance",
    ]

    # Check for proper sentence endings
    proper_endings = [".", "!", "?", '"', "'"]
    if not any(script.endswith(ending) for ending in proper_endings):
        # Check if it ends with a problematic fragment
        for incomplete in incomplete_endings:
            if script.lower().endswith(incomplete.lower()):
                return False, f"Script appears truncated (ends with '{incomplete}')"
        return False, "Script doesn't end with proper punctuation"

    # Check minimum length (very short scripts might be incomplete)
    if len(script) < 200:
        return False, f"Script too short ({len(script)} chars, minimum 200)"

    # Check for reasonable word count
    words = script.split()
    if len(words) < 50:
        return False, f"Script too few words ({len(words)}, minimum 50)"

    return True, f"Script validation passed ({len(words)} words, {len(script)} chars)"


@openrouter_circuit_breaker
async def generate_script(
    product: ProductData,
    settings: LLMSettings,
    secrets: dict[str, str],
    session: aiohttp.ClientSession,
    intermediate_paths: dict[str, Path],
    debug_mode: bool,
    api_settings=None,
) -> str | None:
    """Generate a promotional script for a product using LLM.

    This is the main entry point for script generation. It orchestrates the entire
    process:
    1. Validates API credentials
    2. Selects appropriate LLM models to try
    3. Loads and formats the prompt template with product data
    4. Makes API requests to generate the script
    5. Handles fallback to alternative models if needed
    6. Saves debug information when in debug mode
    7. Sanitizes and returns the final script

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
        The generated and sanitized script, or None if generation failed

    Raises:
    ------
        ScriptGenerationError: If script generation fails for all models

    """
    # Get API key from secrets
    api_key = secrets.get(settings.api_key_env_var)
    if not api_key:
        raise ScriptGenerationError(
            f"Missing API key from environment variable: {settings.api_key_env_var}"
        )

    # Fetch available free models (returns ordered or shuffled list)
    free_models = await _fetch_and_select_model(
        settings, api_key, session, api_settings
    )

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
        raise ScriptGenerationError("No models available to generate script.")

    logger.info(f"Order of models to attempt: {models_to_try}")

    try:
        template = load_prompt_template(Path(settings.prompt_template_path))
        prompt = format_prompt(template, product, settings.target_audience)
    except (FileNotFoundError, ValueError) as e:
        raise ScriptGenerationError(f"Prompt template error: {e}") from e

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
                script_text = await _call_llm_api_with_retry(
                    prompt, model, settings, api_key, session, api_settings
                )

                # Clean the script (remove code blocks, etc.)
                clean_script = re.sub(r"```[\w\s]*", "", script_text).strip()

                # Validate script completeness
                is_complete, validation_reason = validate_script_completeness(
                    clean_script
                )
                if is_complete:
                    logger.info(
                        f"Script successfully generated with model: {model} - "
                        f"{validation_reason}"
                    )
                    return clean_script
                else:
                    logger.warning(
                        f"Script incomplete from {model}: {validation_reason}"
                    )
                    # Save incomplete script for debugging if in debug mode
                    if debug_mode and "script" in intermediate_paths:
                        model_safe = model.replace("/", "_")
                        file_name = (
                            f"incomplete_script_{model_safe}_attempt_{attempt + 1}.txt"
                        )
                        incomplete_path = (
                            intermediate_paths["script"].parent / file_name
                        )
                        try:
                            with open(incomplete_path, "w", encoding="utf-8") as f:
                                content = (
                                    f"# INCOMPLETE SCRIPT - {validation_reason}\n"
                                    f"# Model: {model}, Attempt: {attempt + 1}\n\n"
                                    f"{clean_script}"
                                )
                                f.write(content)
                            logger.debug(
                                f"Saved incomplete script to {incomplete_path}"
                            )
                        except Exception as save_error:
                            logger.warning(
                                f"Could not save incomplete script: {save_error}"
                            )

                    if attempt < max_attempts - 1:
                        logger.info(f"Retrying with {model} for complete script...")
                        continue
                    else:
                        logger.warning(
                            f"Model {model} produced incomplete script after "
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

    # Fallback: try discovering any free model not yet attempted
    if settings.fallback_discover_any_free:
        already_tried = set(models_to_try)
        fallback_models = await _discover_any_free_model(
            settings, api_key, session, api_settings, already_tried
        )

        for model in fallback_models:
            try:
                logger.info(f"Fallback: trying discovered model {model}")
                script_text = await _call_llm_api_with_retry(
                    prompt, model, settings, api_key, session, api_settings
                )
                clean_script = re.sub(r"```[\w\s]*", "", script_text).strip()
                is_complete, validation_reason = validate_script_completeness(
                    clean_script
                )
                if is_complete:
                    logger.info(f"Fallback success with {model} - {validation_reason}")
                    return clean_script
                else:
                    logger.warning(f"Fallback {model} incomplete: {validation_reason}")
            except Exception as e:
                logger.warning(f"Fallback model {model} failed: {e}")
                continue

    logger.error("All models failed to generate a script.")
    return None
