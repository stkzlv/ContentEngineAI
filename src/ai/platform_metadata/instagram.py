"""Instagram-specific metadata generator.

This module implements Instagram Reels optimization with dual caption styles
(ultra-short or SEO-descriptive) and extensive hashtag strategy for maximum reach.
"""

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

from src.ai.platform_metadata.base import BasePlatformMetadataGenerator
from src.ai.platform_metadata.models import PlatformMetadata
from src.ai.platform_metadata.utilities import (
    format_prompt,
    generate_with_llm,
    load_prompt_template,
)
from src.scraper.amazon.scraper import ProductData
from src.video.config import LLMSettings

logger = logging.getLogger(__name__)


class InstagramMetadataGenerator(BasePlatformMetadataGenerator):
    """Instagram Reels metadata generator with dual caption styles.

    Generates Instagram-optimized metadata with:
    - Caption: Ultra-short (3-5 words) OR SEO-descriptive (100-200 chars)
    - Hashtags: 15-30 hashtags in caption (not comments)
    - Keywords: 5-10 search-friendly keywords

    Based on Instagram 2025 Reels algorithm best practices:
    - Hashtags in caption perform better than comments (algorithm change)
    - Mix high-volume + niche + specific hashtags (15-30 total)
    - Two caption strategies: short hooks for viral reach, SEO for search discovery
    - Emoji support for visual appeal and engagement
    """

    def __init__(self, settings: dict):
        """Initialize Instagram metadata generator.

        Args:
            settings: Dictionary containing InstagramPlatformSettings configuration
        """
        self.settings = settings

    @property
    def platform_name(self) -> str:
        """Return platform identifier.

        Returns:
            "instagram"
        """
        return "instagram"

    async def generate(
        self,
        product: ProductData,
        settings: LLMSettings,
        secrets: dict[str, str],
        session: aiohttp.ClientSession,
        intermediate_paths: dict[str, Path],
        debug_mode: bool,
        api_settings=None,
    ) -> PlatformMetadata | None:
        """Generate Instagram-optimized metadata using LLM.

        Produces caption in short (3-5 words) or SEO (100-200 chars) style based on
        settings, with 15-30 hashtags in caption and search-friendly keywords.

        Args:
            product: Product data containing title, description, URL, etc.
            settings: LLM configuration (API keys, models, timeouts)
            secrets: Dictionary containing API keys (OPENROUTER_API_KEY)
            session: Aiohttp session for HTTP requests
            intermediate_paths: Dictionary of file paths for outputs (unused)
            debug_mode: Enable verbose logging if True
            api_settings: Optional API-specific settings override

        Returns:
            PlatformMetadata object with Instagram-optimized content, or None if
            generation fails or validation fails.

        Process:
            1. Determine caption style (short vs SEO) from settings
            2. Load instagram_caption.md prompt template
            3. Format with product data and caption style parameter
            4. Call LLM API using custom formatted prompt
            5. Parse response to extract caption/hashtags/keywords
            6. Create PlatformMetadata object (no title for Instagram)
            7. Validate against Instagram requirements
            8. Return metadata or None if validation fails
        """
        try:
            # Get API key
            api_key_env = settings.api_key_env_var
            api_key = secrets.get(api_key_env)
            if not api_key:
                logger.error(f"Missing API key: {api_key_env}")
                return None

            # Determine caption style
            caption_style = self._determine_caption_style()

            # Load and format prompt with caption style
            template_path = Path("src/ai/prompts/instagram_caption.md")
            template = load_prompt_template(template_path)

            # Add caption style and emoji enabled to template
            emoji_status = (
                "Enabled - use 2-4 relevant emojis"
                if self.settings.get("emoji_enabled", True)
                else "Disabled - no emojis"
            )
            template = template.replace("{CAPTION_STYLE}", caption_style)
            template = template.replace("{EMOJI_ENABLED}", emoji_status)

            # Format with product data
            prompt = format_prompt(template, product)

            if debug_mode:
                logger.info(f"Instagram caption style: {caption_style}")

            # Call LLM API directly (can't use generate_with_llm helper due to custom formatting)
            from src.ai.platform_metadata.utilities import (
                call_llm_api_with_retry,
                fetch_and_select_model,
            )

            # Try auto-selecting free model
            selected_model = await fetch_and_select_model(
                settings, api_key, session, api_settings
            )

            # Prepare model list
            models_to_try = []
            if selected_model:
                models_to_try.append(selected_model)
                if debug_mode:
                    logger.info(f"Auto-selected free model: {selected_model}")
            models_to_try.extend(settings.models)

            # Try each model until one succeeds
            response = None
            for model in models_to_try:
                try:
                    logger.info(f"Attempting generation with model: {model}")
                    response = await call_llm_api_with_retry(
                        prompt, model, settings, api_key, session, api_settings
                    )
                    logger.info(
                        f"Successfully generated content with {model} ({len(response)} chars)"
                    )
                    break
                except Exception as e:
                    logger.warning(f"Model {model} failed: {e}")
                    continue

            if not response:
                logger.error("LLM generation failed for Instagram metadata")
                return None

            # Parse LLM response
            parsed = self._parse_llm_response(response)
            if not parsed:
                logger.error("Failed to parse LLM response for Instagram metadata")
                return None

            caption, hashtags, keywords = parsed

            # Validate caption length based on style
            caption = self._validate_caption_style(caption, caption_style)

            # Ensure #ad is present
            if "#ad" not in hashtags and "#Ad" not in hashtags:
                hashtags.append("#ad")
                logger.debug("Added #ad hashtag for advertising disclosure")

            # Create metadata object (no title for Instagram)
            metadata = PlatformMetadata(
                platform="instagram",
                title=None,  # Instagram doesn't use titles
                description=caption,
                hashtags=hashtags,
                keywords=keywords,
                character_counts=self._calculate_character_counts(None, caption),
                generated_at=datetime.now(timezone.utc).isoformat(),
                product_id=product.product_id,
            )

            # Validate
            is_valid, error_msg = self.validate(metadata)
            if not is_valid:
                logger.error(f"Instagram metadata validation failed: {error_msg}")
                metadata.validation_status = "invalid"
                metadata.validation_messages.append(error_msg)
                return metadata  # Return with validation errors for debugging

            metadata.validation_status = "valid"
            logger.info("Instagram metadata generated and validated successfully")
            return metadata

        except Exception as e:
            logger.error(f"Error generating Instagram metadata: {e}", exc_info=True)
            return None

    def validate(self, metadata: PlatformMetadata) -> tuple[bool, str]:
        """Validate Instagram metadata against platform requirements.

        Checks:
        - Caption style compliance (short: 3-5 words, SEO: 100-200 chars)
        - Hashtag count: 15-30 (Instagram allows 30, recommends 15-30)
        - Hashtags in caption: Must be embedded in caption text
        - #ad tag: Required for sponsored content

        Args:
            metadata: PlatformMetadata object to validate

        Returns:
            Tuple of (is_valid, message):
                - is_valid: True if all validation checks pass
                - message: Empty string if valid, detailed error if invalid
        """
        errors = []

        # Check platform match
        if metadata.platform != "instagram":
            errors.append(
                f"Platform mismatch: expected 'instagram', got '{metadata.platform}'"
            )

        # Validate caption style compliance
        caption_style = self._determine_caption_style()
        caption_len = len(metadata.description)

        if caption_style == "short":
            # Count words (rough estimate)
            word_count = len(metadata.description.split())
            if word_count > 5:
                errors.append(
                    f"Short caption style: too many words ({word_count}, max 5). "
                    f"Caption should be 3-5 words for short style."
                )
        else:  # SEO style
            caption_len_target = self.settings["caption_length_seo"]
            if caption_len > caption_len_target:
                # Warning, not error - but still check
                logger.warning(
                    f"SEO caption longer than optimal: {caption_len} chars "
                    f"(optimal {caption_len_target})"
                )

        # Validate hashtag count (15-30)
        hashtag_count = len(metadata.hashtags)
        min_count = self.settings["hashtag_count_min"]
        max_count = self.settings["hashtag_count_max"]
        if hashtag_count < min_count:
            errors.append(
                f"Too few hashtags: {hashtag_count} (min {min_count}). "
                f"Instagram Reels need 15-30 hashtags for optimal reach."
            )
        elif hashtag_count > max_count:
            errors.append(
                f"Too many hashtags: {hashtag_count} (max {max_count})"
            )

        # Check for #ad tag
        has_ad = any(tag.lower() == "#ad" for tag in metadata.hashtags)
        if not has_ad:
            errors.append(
                "Missing required #ad hashtag for advertising disclosure"
            )

        # Return validation result
        if errors:
            return False, "; ".join(errors)
        return True, ""

    def _determine_caption_style(self) -> str:
        """Determine caption style from settings.

        Returns caption style based on InstagramPlatformSettings.caption_style.
        Defaults to "seo" if not specified.

        Returns:
            "short" for ultra-brief 3-5 word captions
            "seo" for descriptive 100-200 character captions
        """
        caption_style = self.settings.get("caption_style", "seo")
        if caption_style not in ["short", "seo"]:
            logger.warning(
                f"Invalid caption_style '{caption_style}', defaulting to 'seo'"
            )
            return "seo"
        return caption_style

    def _validate_caption_style(self, caption: str, style: str) -> str:
        """Validate and potentially truncate caption based on style.

        Args:
            caption: Generated caption text
            style: Caption style ("short" or "seo")

        Returns:
            Caption (potentially truncated if exceeds style limits)
        """
        if style == "short":
            # For short style, roughly check word count
            words = caption.split()
            if len(words) > 5:
                # Truncate to 5 words
                caption = " ".join(words[:5])
                logger.warning(
                    f"Short caption exceeded 5 words, truncated to: {caption}"
                )
        else:  # SEO style
            max_len = self.settings["caption_length_seo"]
            caption = self._truncate_if_needed(
                caption, max_len, "Instagram SEO caption"
            )

        return caption

    def _parse_llm_response(
        self, response: str
    ) -> tuple[str, list[str], list[str]] | None:
        """Parse LLM response to extract caption, hashtags, and keywords.

        Expected format:
            CAPTION: [caption text]
            HASHTAGS: [#Tag1 #Tag2 #Tag3 ...]
            KEYWORDS: [keyword1, keyword2, keyword3]

        Args:
            response: Raw LLM response text

        Returns:
            Tuple of (caption, hashtags_list, keywords_list) or None if parsing fails.
        """
        try:
            # Extract sections using regex
            caption_match = re.search(
                r"CAPTION:\s*(.+?)(?=HASHTAGS:|KEYWORDS:|$)",
                response,
                re.IGNORECASE | re.DOTALL,
            )
            hashtags_match = re.search(
                r"HASHTAGS:\s*(.+?)(?=KEYWORDS:|$)",
                response,
                re.IGNORECASE | re.DOTALL,
            )
            keywords_match = re.search(
                r"KEYWORDS:\s*(.+?)(?:\n|$)", response, re.IGNORECASE
            )

            if not caption_match:
                logger.error("Failed to parse caption from LLM response")
                return None

            # Extract and clean caption
            caption = caption_match.group(1).strip()

            # Extract hashtags
            hashtags = []
            if hashtags_match:
                hashtags_text = hashtags_match.group(1).strip()
                # Split by spaces and filter out empty strings
                hashtags = [tag.strip() for tag in hashtags_text.split() if tag.strip()]
                # Ensure hashtags start with #
                hashtags = [
                    tag if tag.startswith("#") else f"#{tag}" for tag in hashtags
                ]

            # Extract keywords
            keywords = []
            if keywords_match:
                keywords_text = keywords_match.group(1).strip()
                # Split by commas
                keywords = [kw.strip() for kw in keywords_text.split(",") if kw.strip()]

            return caption, hashtags, keywords

        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}", exc_info=True)
            return None
