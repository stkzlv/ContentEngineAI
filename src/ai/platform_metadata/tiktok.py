"""TikTok-specific metadata generator.

This module implements TikTok search optimization with SEO-focused caption generation,
niche hashtag selection, and validation against generic viral tags.
"""

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

from src.ai.platform_metadata.base import BasePlatformMetadataGenerator
from src.ai.platform_metadata.models import PlatformMetadata
from src.ai.platform_metadata.utilities import generate_with_llm
from src.scraper.amazon.scraper import ProductData
from src.video.config import LLMSettings

logger = logging.getLogger(__name__)


class TikTokMetadataGenerator(BasePlatformMetadataGenerator):
    """TikTok metadata generator with search optimization focus.

    Generates TikTok-optimized metadata with:
    - Caption: 100-300 characters optimal (up to 2200 max) with exact search phrases
    - Hashtags: 3-5 niche-specific hashtags, avoiding generic viral tags
    - Keywords: 5-10 search-friendly phrases

    Based on TikTok 2025 algorithm shift to search-engine model:
    - Use exact search phrases users type (not creative hooks)
    - Avoid generic hashtags (#fyp, #foryoupage, #viral) - provide no discovery value
    - Focus on niche community hashtags for targeted reach
    - SEO-optimized language over viral-style captions
    """

    def __init__(self, settings: dict):
        """Initialize TikTok metadata generator.

        Args:
            settings: Dictionary containing TikTokPlatformSettings configuration
        """
        self.settings = settings

    @property
    def platform_name(self) -> str:
        """Return platform identifier.

        Returns:
            "tiktok"
        """
        return "tiktok"

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
        """Generate TikTok-optimized metadata using LLM.

        Produces SEO-focused caption (100-300 chars optimal, up to 2200 max), 3-5
        niche hashtags avoiding generic tags, and search-friendly keywords.

        Args:
            product: Product data containing title, description, URL, etc.
            settings: LLM configuration (API keys, models, timeouts)
            secrets: Dictionary containing API keys (OPENROUTER_API_KEY)
            session: Aiohttp session for HTTP requests
            intermediate_paths: Dictionary of file paths for outputs (unused)
            debug_mode: Enable verbose logging if True
            api_settings: Optional API-specific settings override

        Returns:
            PlatformMetadata object with TikTok-optimized content, or None if
            generation fails or validation fails.

        Process:
            1. Load tiktok_caption.md prompt template
            2. Format with product data
            3. Call LLM API using generate_with_llm() helper
            4. Parse response to extract caption/hashtags/keywords
            5. Create PlatformMetadata object (no title for TikTok)
            6. Validate against TikTok requirements (length, blacklist)
            7. Return metadata or None if validation fails
        """
        try:
            # Get API key
            api_key_env = settings.api_key_env_var
            api_key = secrets.get(api_key_env)
            if not api_key:
                logger.error(f"Missing API key: {api_key_env}")
                return None

            # Generate using LLM helper
            template_path = Path("src/ai/prompts/tiktok_caption.md")
            response = await generate_with_llm(
                template_path,
                product,
                settings,
                api_key,
                session,
                api_settings,
                debug_mode,
            )

            if not response:
                logger.error("LLM generation failed for TikTok metadata")
                return None

            # Parse LLM response
            parsed = self._parse_llm_response(response)
            if not parsed:
                logger.error("Failed to parse LLM response for TikTok metadata")
                return None

            caption, hashtags, keywords = parsed

            # Truncate caption if exceeds maximum (2200 chars)
            caption = self._truncate_if_needed(
                caption, self.settings["caption_length_max"], "TikTok caption"
            )

            # Ensure #ad is present
            if "#ad" not in hashtags and "#Ad" not in hashtags:
                hashtags.append("#ad")
                logger.debug("Added #ad hashtag for advertising disclosure")

            # Create metadata object (no title for TikTok)
            metadata = PlatformMetadata(
                platform="tiktok",
                title=None,  # TikTok doesn't use titles
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
                logger.error(f"TikTok metadata validation failed: {error_msg}")
                metadata.validation_status = "invalid"
                metadata.validation_messages.append(error_msg)
                return metadata  # Return with validation errors for debugging

            metadata.validation_status = "valid"
            logger.info("TikTok metadata generated and validated successfully")
            return metadata

        except Exception as e:
            logger.error(f"Error generating TikTok metadata: {e}", exc_info=True)
            return None

    def validate(self, metadata: PlatformMetadata) -> tuple[bool, str]:
        """Validate TikTok metadata against platform requirements.

        Checks:
        - Caption length: 100-300 optimal (warning if exceeded), 2200 max (error)
        - Hashtag count: 3-5 niche-specific tags
        - Blacklisted hashtags: No generic viral tags from avoid_generic_tags setting
        - #ad tag: Required for sponsored content

        Args:
            metadata: PlatformMetadata object to validate

        Returns:
            Tuple of (is_valid, message):
                - is_valid: True if all validation checks pass
                - message: Empty string if valid, detailed error if invalid
        """
        errors = []
        warnings = []

        # Check platform match
        if metadata.platform != "tiktok":
            errors.append(
                f"Platform mismatch: expected 'tiktok', got '{metadata.platform}'"
            )

        # Validate caption length
        caption_len = len(metadata.description)
        optimal_len = self.settings["caption_length_optimal"]
        max_len = self.settings["caption_length_max"]

        if caption_len > max_len:
            errors.append(
                f"Caption too long: {caption_len} chars (max {max_len})"
            )
        elif caption_len > optimal_len:
            # Warning, not error - caption is functional but not optimal
            warning_msg = (
                f"Caption longer than optimal: {caption_len} chars "
                f"(optimal {optimal_len}), still under max {max_len}"
            )
            warnings.append(warning_msg)
            logger.warning(warning_msg)

        # Validate hashtag count (3-5)
        hashtag_count = len(metadata.hashtags)
        min_count = self.settings["hashtag_count_min"]
        max_count = self.settings["hashtag_count_max"]
        if hashtag_count < min_count:
            errors.append(
                f"Too few hashtags: {hashtag_count} (min {min_count})"
            )
        elif hashtag_count > max_count:
            errors.append(
                f"Too many hashtags: {hashtag_count} (max {max_count})"
            )

        # Check for blacklisted generic hashtags
        avoid_tags = self.settings.get("avoid_generic_tags", [])
        if avoid_tags:
            found_generic = []
            for tag in metadata.hashtags:
                # Normalize tag for comparison (remove # and lowercase)
                normalized_tag = tag.lower().replace("#", "")
                if normalized_tag in avoid_tags:
                    found_generic.append(tag)

            if found_generic:
                errors.append(
                    f"Generic hashtags found (provide no discovery value): {', '.join(found_generic)}. "
                    f"Avoid: {', '.join(avoid_tags)}"
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

    def _parse_llm_response(
        self, response: str
    ) -> tuple[str, list[str], list[str]] | None:
        """Parse LLM response to extract caption, hashtags, and keywords.

        Expected format:
            CAPTION: [caption text]
            HASHTAGS: [#Tag1 #Tag2 #Tag3]
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
                r"HASHTAGS:\s*(.+?)(?:\n|$)", response, re.IGNORECASE
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
