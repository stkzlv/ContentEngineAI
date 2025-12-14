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
) -> str | None:
    """Fetch available models from OpenRouter and select a free model.

    This is a thin wrapper around description_generator._fetch_and_select_model()
    that provides a cleaner public interface.

    Args:
        settings: LLM configuration settings
        api_key: API key for authentication
        session: Aiohttp session for API calls
        api_settings: Optional API-specific settings override

    Returns:
        Selected model name or None if auto-selection disabled or failed
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
        prompt: Formatted prompt to send to LLM
        model: Model identifier to use
        settings: LLM configuration settings
        api_key: API key for authentication
        session: Aiohttp session for API calls
        api_settings: Optional API-specific settings override

    Returns:
        LLM response text

    Raises:
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
) -> str | None:
    """High-level helper to generate content using LLM with automatic model fallback.

    This function encapsulates the common pattern of:
    1. Load prompt template from file
    2. Format template with product data
    3. Try auto-selecting free model (if enabled)
    4. Fallback to configured models list
    5. Call LLM API with retry logic

    Args:
        template_path: Path to prompt template file
        product: Product data for template formatting
        settings: LLM configuration settings
        api_key: API key for authentication
        session: Aiohttp session for API calls
        api_settings: Optional API-specific settings override
        debug_mode: Enable verbose logging if True

    Returns:
        LLM-generated text or None if all attempts fail

    Example:
        response = await generate_with_llm(
            Path("src/ai/prompts/youtube_metadata.md"),
            product,
            settings,
            api_key,
            session,
            debug_mode=True
        )
    """
    try:
        # Step 1: Load template
        template = load_prompt_template(template_path)
        if debug_mode:
            logger.info(f"Loaded template from {template_path}")

        # Step 2: Format with product data
        prompt = format_prompt(template, product)
        if debug_mode:
            logger.info(f"Formatted prompt ({len(prompt)} chars)")

        # Step 3: Try auto-selecting free model
        selected_model = await fetch_and_select_model(
            settings, api_key, session, api_settings
        )

        # Step 4: Prepare model list (auto-selected first, then fallbacks)
        models_to_try = []
        if selected_model:
            models_to_try.append(selected_model)
            if debug_mode:
                logger.info(f"Auto-selected free model: {selected_model}")
        models_to_try.extend(settings.models)

        # Step 5: Try each model until one succeeds
        for model in models_to_try:
            try:
                logger.info(f"Attempting generation with model: {model}")
                response = await call_llm_api_with_retry(
                    prompt, model, settings, api_key, session, api_settings
                )
                logger.info(
                    f"Successfully generated content with {model} ({len(response)} chars)"
                )
                return response

            except Exception as e:
                logger.warning(f"Model {model} failed: {e}")
                continue

        # All models failed
        logger.error("All models failed to generate content")
        return None

    except FileNotFoundError as e:
        logger.error(f"Template file not found: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in generate_with_llm: {e}", exc_info=True)
        return None


def save_metadata_to_file(
    metadata: PlatformMetadata, output_path: Path, debug_mode: bool = False
) -> bool:
    """Save platform metadata to JSON file.

    Serializes PlatformMetadata object to JSON and writes to the specified path.
    Creates parent directories if they don't exist.

    Args:
        metadata: PlatformMetadata object to save
        output_path: Path where JSON file will be written
        debug_mode: Enable verbose logging if True

    Returns:
        True if save succeeded, False otherwise

    Example:
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
        file_path: Path to JSON file containing metadata

    Returns:
        PlatformMetadata object or None if load failed

    Example:
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
        )

        logger.info(f"Loaded {metadata.platform} metadata from {file_path}")
        return metadata

    except FileNotFoundError:
        logger.warning(f"Metadata file not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Failed to load metadata from {file_path}: {e}", exc_info=True)
        return None
