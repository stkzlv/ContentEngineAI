"""Shared utilities for platform metadata generation.

This module provides common helper functions for LLM-based metadata generation
across all platforms. It wraps existing utilities from description_generator.py
to reduce code duplication and provides high-level helpers for common patterns.
"""

import json
import logging
from pathlib import Path

import aiohttp

from src.ai.description_generator import (
    _call_llm_api_with_retry,
    _fetch_and_select_model,
    format_prompt,
    load_prompt_template,
)
from src.ai.platform_metadata.models import PlatformMetadata
from src.scraper.amazon.scraper import ProductData
from src.video.config.llm_settings import LLMSettings

logger = logging.getLogger(__name__)

# Re-export these functions for convenience
__all__ = [
    "load_prompt_template",
    "format_prompt",
    "fetch_and_select_model",
    "call_llm_api_with_retry",
    "generate_with_llm",
    "save_metadata_to_file",
]


async def fetch_and_select_model(
    settings: LLMSettings,
    api_key: str,
    session: aiohttp.ClientSession,
    api_settings=None,
) -> list[str]:
    """Fetch available models from OpenRouter and return free models to try.

    This is a thin wrapper around description_generator._fetch_and_select_model()
    that provides a cleaner public interface.

    Args:
    ----
        settings: LLM configuration settings
        api_key: API key for authentication
        session: Aiohttp session for API calls
        api_settings: Optional API-specific settings override

    Returns:
    -------
        List of free model IDs to try (ordered or shuffled based on settings)

    """
    return await _fetch_and_select_model(settings, api_key, session, api_settings)


async def call_llm_api_with_retry(
    prompt: str,
    model: str,
    settings: LLMSettings,
    api_key: str,
    session: aiohttp.ClientSession,
    api_settings=None,
) -> str:
    """Call LLM API with automatic retry logic.

    This is a thin wrapper around description_generator._call_llm_api_with_retry()
    that provides a cleaner public interface.

    Args:
    ----
        prompt: Formatted prompt to send to LLM
        model: Model identifier to use
        settings: LLM configuration settings
        api_key: API key for authentication
        session: Aiohttp session for API calls
        api_settings: Optional API-specific settings override

    Returns:
    -------
        LLM response text

    Raises:
    ------
        Exception: If all retry attempts fail

    """
    return await _call_llm_api_with_retry(
        prompt, model, settings, api_key, session, api_settings
    )


async def generate_with_llm(
    template_path: Path,
    product: ProductData,
    settings: LLMSettings,
    api_key: str,
    session: aiohttp.ClientSession,
    api_settings=None,
    debug_mode: bool = False,
    secrets: dict[str, str] | None = None,
    video_script: str | None = None,
    narrator_profile: str = "",
    pillar: str | None = None,
    pillar_preambles: dict[str, str] | None = None,
    extra_placeholders: dict[str, str] | None = None,
) -> str | None:
    """High-level helper to generate content using LLM with automatic model fallback.

    This function encapsulates the common pattern of:
    1. Load prompt template from file
    2. Format template with product data
    3. Prepend narrator profile and pillar preamble (when provided)
    4. Try auto-selecting free model (if enabled)
    5. Fallback to configured models list
    6. Call LLM API with retry logic

    Args:
    ----
        template_path: Path to prompt template file
        product: Product data for template formatting
        settings: LLM configuration settings
        api_key: API key for authentication
        session: Aiohttp session for API calls
        api_settings: Optional API-specific settings override
        debug_mode: Enable verbose logging if True
        secrets: Dict of env var names to values for fallback provider key lookup
        video_script: Optional generated spoken script for the video. When
            provided, the prompt template's `{VIDEO_SCRIPT}` placeholder
            substitutes with this text, letting caption prompts mirror the
            closing engagement-bait line into the platform caption. Templates
            that don't reference the placeholder are unaffected.
        narrator_profile: Channel-wide voice direction prepended to the prompt.
        pillar: Content pillar name for pillar-specific preamble lookup.
        pillar_preambles: Dict mapping pillar names to preamble text.
        extra_placeholders: Optional prompt-specific template substitutions for
            config-derived values (e.g. `{MAX_WORDS}`), so a prompt doesn't have
            to hardcode a number the config claims to own.

    Returns:
    -------
        LLM-generated text or None if all attempts fail

    """
    try:
        # Step 1: Load template
        template = load_prompt_template(template_path)
        if debug_mode:
            logger.info(f"Loaded template from {template_path}")

        # Step 2: Format with product data
        prompt = format_prompt(
            template,
            product,
            video_script=video_script,
            extra_placeholders=extra_placeholders,
        )

        # Step 2b: Prepend narrator profile and pillar preamble
        if narrator_profile or pillar:
            from src.ai.script_generator import apply_prompt_preambles

            prompt = apply_prompt_preambles(
                prompt,
                narrator_profile,
                pillar,
                pillar_preambles or {},
            )

        if debug_mode:
            logger.info(f"Formatted prompt ({len(prompt)} chars)")

        # Step 3: Fetch available free models (OpenRouter only)
        if settings.provider == "openrouter":
            free_models = await fetch_and_select_model(
                settings, api_key, session, api_settings
            )
        else:
            free_models = []

        # Step 4: Prepare model list (free models first, then configured fallbacks)
        models_to_try: list[str] = []
        if free_models:
            models_to_try.extend(free_models)
            if debug_mode:
                logger.info("Free models to try: %s", free_models[:3])

        # Add configured models as fallback (if not already in list)
        for model in settings.models:
            if model not in models_to_try:
                models_to_try.append(model)

        # Step 5: Try each model until one succeeds
        for model in models_to_try:
            try:
                logger.info(f"Attempting generation with model: {model}")
                response = await call_llm_api_with_retry(
                    prompt, model, settings, api_key, session, api_settings
                )
                logger.info(
                    f"Successfully generated content with {model} "
                    f"({len(response)} chars)"
                )
                return response

            except Exception as e:
                logger.warning(f"Model {model} failed: {e}")
                continue

        # Provider fallback: try fallback_provider if primary exhausted
        if settings.fallback_provider:
            fb = settings.fallback_provider
            fb_api_key_val = (secrets or {}).get(fb.api_key_env_var)
            if fb_api_key_val:
                logger.info(
                    "Primary provider exhausted, falling back to %s", fb.provider
                )

                if fb.provider == "openrouter":
                    fb_free_models = await fetch_and_select_model(
                        fb, fb_api_key_val, session, api_settings
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
                        response = await call_llm_api_with_retry(
                            prompt, model, fb, fb_api_key_val, session, api_settings
                        )
                        logger.info(
                            "Fallback success with %s (%d chars)",
                            model,
                            len(response),
                        )
                        return response
                    except Exception as e:
                        logger.warning("Fallback model %s failed: %s", model, e)
                        continue
            else:
                logger.warning(
                    "Fallback provider configured but API key %s not found",
                    fb.api_key_env_var,
                )

        logger.error("All models failed to generate content")
        return None

    except FileNotFoundError as e:
        logger.error(f"Template file not found: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in generate_with_llm: {e}", exc_info=True)
        return None


def save_metadata_to_file(
    metadata: PlatformMetadata,
    output_path: Path,
    debug_mode: bool = False,
    disclose: bool = True,
) -> bool:
    """Save platform metadata to JSON file.

    Serializes PlatformMetadata object to JSON and writes to the specified path.
    Creates parent directories if they don't exist.

    Args:
    ----
        metadata: PlatformMetadata object to save
        output_path: Path where JSON file will be written
        debug_mode: Enable verbose logging if True
        disclose: Whether this render has a material connection to disclose.
            Recorded in the file so the publisher does not re-derive it, and
            used to drop the `#ad` tag the platform generators append
            unconditionally. Defaults to True.

    Returns:
    -------
        True if save succeeded, False otherwise

    Example:
    -------
        success = save_metadata_to_file(
            metadata,
            Path("outputs/B0TESTID/metadata_youtube.json"),
            debug_mode=True
        )

    """
    try:
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and save as JSON
        metadata_dict = metadata.to_dict()
        # The platform generators append `#ad` unconditionally. Record the
        # render's decision alongside it and drop the tag when there is no
        # material connection, so this path cannot disagree with the on-frame
        # overlay about whether the video is promotional.
        metadata_dict["carries_affiliate_content"] = disclose
        if not disclose:
            metadata_dict["hashtags"] = [
                tag
                for tag in (metadata_dict.get("hashtags") or [])
                if tag.lstrip("#").lower() != "ad"
            ]
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)

        if debug_mode:
            logger.info(f"Saved {metadata.platform} metadata to {output_path}")

        return True

    except Exception as e:
        logger.error(f"Failed to save metadata to {output_path}: {e}", exc_info=True)
        return False


def load_metadata_from_file(file_path: Path) -> PlatformMetadata | None:
    """Load platform metadata from JSON file.

    Reads JSON file and deserializes into PlatformMetadata object.

    Args:
    ----
        file_path: Path to JSON file containing metadata

    Returns:
    -------
        PlatformMetadata object or None if load failed

    Example:
    -------
        metadata = load_metadata_from_file(
            Path("outputs/B0TESTID/metadata_youtube.json")
        )

    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Reconstruct PlatformMetadata from dict
        # Note: This uses the constructor directly since we stored all fields
        metadata = PlatformMetadata(
            platform=data["platform"],
            title=data.get("title"),
            description=data["description"],
            hashtags=data["hashtags"],
            keywords=data["keywords"],
            character_counts=data["character_counts"],
            generated_at=data["generated_at"],
            product_id=data["product_id"],
            validation_status=data["validation_status"],
            validation_messages=data["validation_messages"],
            prompt_variant=data.get("prompt_variant"),
        )

        logger.info(f"Loaded {metadata.platform} metadata from {file_path}")
        return metadata

    except FileNotFoundError:
        logger.warning(f"Metadata file not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Failed to load metadata from {file_path}: {e}", exc_info=True)
        return None
