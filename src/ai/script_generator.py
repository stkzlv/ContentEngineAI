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
import hashlib
import logging
import random
import re
import unicodedata
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
from src.utils.circuit_breaker import llm_circuit_breaker
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


def _normalize_for_llm(text: str) -> str:
    """Normalize text before sending to the LLM.

    Amazon product titles and descriptions sometimes use mathematical-alphabet
    Unicode codepoints for fake bold/italic styling (e.g. 𝐌𝐢𝐠𝐡𝐭𝐲 𝐏𝐨𝐰𝐞𝐫).
    NFKC folds those down to plain ASCII letters, which both reduces token
    waste and keeps the LLM from mimicking the styling in its output.

    Em dashes and en dashes are also replaced. Em dashes especially are a
    strong AI tell in generated text, and Amazon descriptions use them
    liberally. Em dash becomes ", " (matches the comma-pause it usually
    represents); en dash becomes "-" (matches its range/connector use).
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s*—\s*", ", ", text)
    text = re.sub(r"\s*–\s*", "-", text)
    return text


def _short_product_name(full_title: str) -> str:
    """Heuristic short alias for SEO-bloated product titles.

    Cuts at the first SEO-style separator (comma, opening paren, vertical bar,
    or hyphenated descriptor) and caps the result at the first three words.
    Designed to give the LLM a plain "BRAND MODEL" handle without forcing it
    to parse a 30-word Amazon listing title.
    """
    if not full_title:
        return "this product"
    cut = full_title
    for sep in [",", " (", " | ", " - "]:
        idx = cut.find(sep)
        if idx > 0:
            cut = cut[:idx]
    short = " ".join(cut.split()[:3]).strip()
    return short or "this product"


def format_prompt(template: str, product: ProductData, audience: str) -> str:
    """Format the prompt template with product data and audience information.

    Replaces placeholders in the template with actual product data. The template
    should contain the following placeholders:
    - {FULL_PRODUCT_NAME}: The product title
    - {SHORT_PRODUCT_NAME}: A short alias for the product (brand + model).
    - {PRODUCT_DESCRIPTION}: The product description
    - {AUDIENCE}: The target audience for the video

    Product name and description are NFKC-normalized so Amazon's mathematical-
    alphabet bold tricks don't reach the LLM.

    Price is intentionally excluded to avoid stale/incorrect pricing in videos.

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
    full_name = _normalize_for_llm(product.title or "Product")
    short_name = _short_product_name(full_name)
    description = _normalize_for_llm(product.description or "No description available")
    try:
        return template.format(
            FULL_PRODUCT_NAME=full_name,
            SHORT_PRODUCT_NAME=short_name,
            PRODUCT_DESCRIPTION=description,
            AUDIENCE=audience,
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


def select_script_template(
    settings: LLMSettings,
    product_id: str | None = None,
    pillar: str | None = None,
) -> Path:
    """Select a script template, deterministically by product ID.

    When script_templates is enabled, picks from the templates directory.
    When `pillar` is given and the templates config has a pillar map entry
    for it, the pool is narrowed to that pillar's templates before selection.
    Falls back to the single prompt_template_path when disabled.
    """
    templates_cfg = settings.script_templates

    if not templates_cfg.enabled:
        return Path(settings.prompt_template_path)

    # Fixed template override (from config or CLI --script-template)
    if templates_cfg.fixed_template:
        path = Path(templates_cfg.templates_dir) / f"{templates_cfg.fixed_template}.md"
        if path.exists():
            return path
        logger.warning(
            "Fixed template '%s' not found, falling back to default",
            templates_cfg.fixed_template,
        )
        return Path(settings.prompt_template_path)

    # Discover available templates
    templates_dir = Path(templates_cfg.templates_dir)
    if not templates_dir.is_dir():
        logger.warning("Templates dir '%s' not found, falling back", templates_dir)
        return Path(settings.prompt_template_path)

    all_templates = sorted(p.stem for p in templates_dir.glob("*.md"))
    if not all_templates:
        logger.warning("No templates found in '%s'", templates_dir)
        return Path(settings.prompt_template_path)

    # Apply pool filter (empty = all)
    pool = templates_cfg.template_pool or all_templates
    pool = [t for t in pool if t in all_templates]
    if not pool:
        pool = all_templates

    # Apply pillar filter (when pillar is provided and known)
    if pillar and pillar in templates_cfg.pillars:
        pillar_templates = templates_cfg.pillars[pillar]
        narrowed = [t for t in pool if t in pillar_templates]
        if narrowed:
            pool = narrowed
        else:
            logger.warning(
                "Pillar '%s' has no templates intersecting current pool; "
                "using full pool",
                pillar,
            )

    # Deterministic selection using salted product ID hash
    if product_id:
        hash_hex = hashlib.md5(
            f"{product_id}:script_template".encode(),
            usedforsecurity=False,
        ).hexdigest()
        seed = int(hash_hex[:8], 16)
        rng = random.Random(seed)  # noqa: S311
        name = rng.choice(pool)
    else:
        name = random.choice(pool)  # noqa: S311

    logger.info(
        "Selected script template '%s' for product '%s' (pillar=%s)",
        name,
        product_id,
        pillar,
    )
    return templates_dir / f"{name}.md"


def _warn_unknown_pillar(pillar: str, settings: LLMSettings) -> None:
    """Log a hint when a pillar is set but not configured in any of the
    three pillar maps (pillars, pillar_preambles, pillar_audiences).

    Each map fails open: a missing entry just means no filter / no preamble /
    no audience override. That's correct behavior, but it makes typos invisible
    at runtime. This warning surfaces the typo so users know why their pillar
    had no effect.
    """
    cfg = settings.script_templates
    known = set(cfg.pillars) | set(cfg.pillar_preambles) | set(cfg.pillar_audiences)
    if pillar in known:
        return
    logger.info(
        "Pillar '%s' is not configured in pillars, pillar_preambles, or "
        "pillar_audiences. No template filter, preamble, or audience override "
        "will apply for this run. Known pillars: %s",
        pillar,
        sorted(known) or [],
    )


def apply_prompt_preambles(
    prompt: str,
    narrator_profile: str,
    pillar: str | None,
    pillar_preambles: dict[str, str],
) -> str:
    """Stack channel-wide and pillar preambles above the prompt.

    Order in the returned string: narrator_profile, then the pillar's
    preamble (when pillar is set and maps to a non-empty entry), then
    the original prompt, joined by blank lines. Each layer is dropped
    when its source is empty, so this function is safe to call with no
    preambles configured.
    """
    parts: list[str] = []
    if narrator_profile:
        parts.append(narrator_profile)
    if pillar:
        pillar_text = pillar_preambles.get(pillar)
        if pillar_text:
            parts.append(pillar_text)
    parts.append(prompt)
    return "\n\n".join(parts)


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

    Delegates to the shared llm_client which handles provider dispatch
    (OpenRouter vs Gemini).
    """
    from src.ai.llm_client import LLMCallError, call_llm

    try:
        return await call_llm(prompt, model, settings, api_key, session)
    except LLMCallError as e:
        raise ScriptGenerationError(str(e)) from e


def validate_script_completeness(
    script: str, min_chars: int = 200, min_words: int = 50
) -> tuple[bool, str]:
    """Validate if a script appears complete and well-formed.

    Args:
    ----
        script: The generated script text
        min_chars: Minimum character count for a valid script
        min_words: Minimum word count for a valid script

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
    if len(script) < min_chars:
        return False, f"Script too short ({len(script)} chars, minimum {min_chars})"

    # Check for reasonable word count
    words = script.split()
    if len(words) < min_words:
        return False, f"Script too few words ({len(words)}, minimum {min_words})"

    return True, f"Script validation passed ({len(words)} words, {len(script)} chars)"


@llm_circuit_breaker
async def generate_script(
    product: ProductData,
    settings: LLMSettings,
    secrets: dict[str, str],
    session: aiohttp.ClientSession,
    intermediate_paths: dict[str, Path],
    debug_mode: bool,
    api_settings=None,
    product_id: str | None = None,
    pillar: str | None = None,
) -> tuple[str | None, str | None]:
    """Generate a promotional script for a product using LLM.

    Returns (script_text, template_name) tuple. template_name is the stem
    of the selected template file (e.g. "curiosity_hook"), or None on failure.
    """
    # Script validation thresholds from config
    sv = settings.script_validation
    sv_min_chars = sv.min_chars
    sv_min_words = sv.min_words

    # Get API key from secrets
    api_key = secrets.get(settings.api_key_env_var)
    if not api_key:
        raise ScriptGenerationError(
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
        raise ScriptGenerationError("No models available to generate script.")

    logger.info(f"Order of models to attempt: {models_to_try}")

    if debug_mode:
        logger.debug(
            "Product fields: title=%s, price=%s, asin=%s",
            product.title,
            product.price,
            getattr(product, "asin", None),
        )

    if pillar:
        _warn_unknown_pillar(pillar, settings)

    template_path = select_script_template(settings, product_id, pillar)
    template_name = template_path.stem
    audience = settings.target_audience
    if pillar:
        pillar_audience = settings.script_templates.pillar_audiences.get(pillar)
        if pillar_audience:
            audience = pillar_audience
    try:
        template = load_prompt_template(template_path)
        prompt = format_prompt(template, product, audience)
    except (FileNotFoundError, ValueError) as e:
        raise ScriptGenerationError(f"Prompt template error: {e}") from e

    prompt = apply_prompt_preambles(
        prompt,
        settings.script_templates.narrator_profile,
        pillar,
        settings.script_templates.pillar_preambles,
    )

    if "formatted_prompt" in intermediate_paths:
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
                    clean_script, sv_min_chars, sv_min_words
                )
                if is_complete:
                    logger.info(
                        f"Script successfully generated with model: {model} - "
                        f"{validation_reason}"
                    )
                    return clean_script, template_name
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

    # Fallback: try discovering any free model not yet attempted (OpenRouter only)
    if settings.provider == "openrouter" and settings.fallback_discover_any_free:
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
                    clean_script, sv_min_chars, sv_min_words
                )
                if is_complete:
                    logger.info(f"Fallback success with {model} - {validation_reason}")
                    return clean_script, template_name
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

            # Build fallback model list (with free model discovery for OpenRouter)
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
                    script_text = await _call_llm_api_with_retry(
                        prompt, model, fb, fb_api_key, session, api_settings
                    )
                    clean_script = re.sub(r"```[\w\s]*", "", script_text).strip()
                    is_complete, reason = validate_script_completeness(
                        clean_script, sv_min_chars, sv_min_words
                    )
                    if is_complete:
                        logger.info("Fallback success with %s - %s", model, reason)
                        return clean_script, template_name
                    else:
                        logger.warning("Fallback %s incomplete: %s", model, reason)
                except Exception as e:
                    logger.warning("Fallback model %s failed: %s", model, e)
                    continue

            # OpenRouter fallback: discover any free model as last resort
            if fb.provider == "openrouter" and fb.fallback_discover_any_free:
                already_tried_fb = set(fb_models)
                discovered = await _discover_any_free_model(
                    fb, fb_api_key, session, api_settings, already_tried_fb
                )
                for model in discovered:
                    try:
                        logger.info("Fallback discovery: trying %s", model)
                        script_text = await _call_llm_api_with_retry(
                            prompt, model, fb, fb_api_key, session, api_settings
                        )
                        clean_script = re.sub(r"```[\w\s]*", "", script_text).strip()
                        is_complete, reason = validate_script_completeness(
                            clean_script, sv_min_chars, sv_min_words
                        )
                        if is_complete:
                            logger.info(
                                "Fallback discovery success with %s - %s",
                                model,
                                reason,
                            )
                            return clean_script, template_name
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

    logger.error("All models failed to generate a script.")
    return None, None
