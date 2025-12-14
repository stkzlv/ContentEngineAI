"""YouTube-specific metadata generator.

This module implements YouTube Shorts optimization with SEO-focused title/description
generation, strategic hashtag placement, and validation against YouTube's requirements.
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
from src.video.config.llm_settings import LLMSettings

logger = logging.getLogger(__name__)


class YouTubeMetadataGenerator(BasePlatformMetadataGenerator):
    """YouTube Shorts metadata generator.

    Generates YouTube-optimized metadata with:
    - Title: 50-60 characters with front-loaded keywords
    - Description: Up to 5000 chars, first 150 chars SEO-optimized
    - Hashtags: 3-5 hashtags including #Shorts for vertical videos
    - Keywords: 5-10 search-friendly keywords

    Based on YouTube 2025 algorithm best practices:
    - Front-load keywords in title for search ranking
    - Optimize first 150 chars of description (shown before "Show more")
    - Use 3-5 targeted hashtags (>15 hashtags are ignored by algorithm)
    - Include #Shorts tag for vertical video <60s discovery
    """

    def __init__(self, settings: dict):
        """Initialize YouTube metadata generator.

        Args:
            settings: Dictionary containing YouTubePlatformSettings configuration
        """
        self.settings = settings

    @property
    def platform_name(self) -> str:
        """Return platform identifier.

        Returns:
            "youtube"
        """
        return "youtube"

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
        """Generate YouTube-optimized metadata using LLM.

        Produces title (50-60 chars), description (up to 5000 chars with first 150
        optimized), 3-5 hashtags including #Shorts, and 5-10 SEO keywords.

        Args:
            product: Product data containing title, description, URL, etc.
            settings: LLM configuration (API keys, models, timeouts)
            secrets: Dictionary containing API keys (OPENROUTER_API_KEY)
            session: Aiohttp session for HTTP requests
            intermediate_paths: Dictionary of file paths for outputs (unused)
            debug_mode: Enable verbose logging if True
            api_settings: Optional API-specific settings override

        Returns:
            PlatformMetadata object with YouTube-optimized content, or None if
            generation fails or validation fails.

        Process:
            1. Load youtube_metadata.md prompt template
            2. Format with product data
            3. Call LLM API using generate_with_llm() helper
            4. Parse response to extract title/description/hashtags/keywords
            5. Create PlatformMetadata object
            6. Validate against YouTube requirements
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
            template_path = Path("src/ai/prompts/youtube_metadata.md")
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
                logger.error("LLM generation failed for YouTube metadata")
                return None

            # Parse LLM response
            parsed = self._parse_llm_response(response)
            if not parsed:
                logger.error("Failed to parse LLM response for YouTube metadata")
                return None

            title, description, hashtags, keywords = parsed

            # Truncate title if needed (50-60 chars)
            title = self._truncate_if_needed(
                title, self.settings["title_length_max"], "YouTube title"
            )

            # Truncate description if needed
            description = self._truncate_if_needed(
                description,
                self.settings["description_length_max"],
                "YouTube description",
            )

            # Add #Shorts if enabled and not present
            if self.settings.get("include_shorts_tag", True):
                if "#Shorts" not in hashtags and "#shorts" not in hashtags:
                    hashtags.insert(0, "#Shorts")
                    logger.debug("Added #Shorts hashtag automatically")

            # Ensure #ad is present
            if "#ad" not in hashtags and "#Ad" not in hashtags:
                hashtags.append("#ad")
                logger.debug("Added #ad hashtag for advertising disclosure")

            # Create metadata object with validation
            # Note: We create with default "valid" status first, then validate
            temp_metadata = PlatformMetadata.create(
                platform="youtube",
                title=title,
                description=description,
                hashtags=hashtags,
                keywords=keywords,
                product_id=product.asin,
            )

            # Validate and recreate with proper status if needed
            is_valid, error_msg = self.validate(temp_metadata)
            if not is_valid:
                logger.error(f"YouTube metadata validation failed: {error_msg}")
                # Recreate with error status
                metadata = PlatformMetadata.create(
                    platform="youtube",
                    title=title,
                    description=description,
                    hashtags=hashtags,
                    keywords=keywords,
                    product_id=product.asin,
                    validation_status="invalid",
                    validation_messages=[error_msg],
                )
                return metadata  # Return with validation errors for debugging

            logger.info("YouTube metadata generated and validated successfully")
            return temp_metadata

        except Exception as e:
            logger.error(f"Error generating YouTube metadata: {e}", exc_info=True)
            return None

    def validate(self, metadata: PlatformMetadata) -> tuple[bool, str]:
        """Validate YouTube metadata against platform requirements.

        Checks:
        - Title length: 50-60 characters (optimal for mobile display)
        - Description length: up to 5000 characters
        - Hashtag count: 3-5 (YouTube recommends max 5, >15 ignored)
        - #Shorts tag: required if include_shorts_tag setting is enabled
        - #ad tag: required for sponsored content

        Args:
            metadata: PlatformMetadata object to validate

        Returns:
            Tuple of (is_valid, message):
                - is_valid: True if all validation checks pass
                - message: Empty string if valid, detailed error if invalid
        """
        errors = []

        # Check platform match
        if metadata.platform != "youtube":
            errors.append(
                f"Platform mismatch: expected 'youtube', got '{metadata.platform}'"
            )

        # Validate title length (50-60 chars optimal)
        if metadata.title:
            title_len = len(metadata.title)
            title_max = self.settings["title_length_max"]
            if title_len > title_max:
                errors.append(
                    f"Title too long: {title_len} chars (max {title_max})"
                )
            elif title_len < 50:
                # Warning, not error - but log it
                logger.warning(
                    f"Title shorter than optimal: {title_len} chars (optimal 50-60)"
                )

        # Validate description length
        desc_len = len(metadata.description)
        desc_max = self.settings["description_length_max"]
        if desc_len > desc_max:
            errors.append(
                f"Description too long: {desc_len} chars (max {desc_max})"
            )

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

        # Check for #Shorts tag if required
        if self.settings.get("include_shorts_tag", True):
            has_shorts = any(
                tag.lower() == "#shorts" for tag in metadata.hashtags
            )
            if not has_shorts:
                errors.append(
                    "Missing required #Shorts hashtag for vertical video"
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
    ) -> tuple[str, str, list[str], list[str]] | None:
        """Parse LLM response to extract title, description, hashtags, and keywords.

        Expected format:
            TITLE: [title text]
            DESCRIPTION: [description text]
            HASHTAGS: [#Tag1 #Tag2 #Tag3]
            KEYWORDS: [keyword1, keyword2, keyword3]

        Args:
            response: Raw LLM response text

        Returns:
            Tuple of (title, description, hashtags_list, keywords_list) or None if
            parsing fails.
        """
        try:
            # Extract sections using regex
            title_match = re.search(r"TITLE:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
            desc_match = re.search(
                r"DESCRIPTION:\s*(.+?)(?=HASHTAGS:|KEYWORDS:|$)",
                response,
                re.IGNORECASE | re.DOTALL,
            )
            hashtags_match = re.search(
                r"HASHTAGS:\s*(.+?)(?:\n|$)", response, re.IGNORECASE
            )
            keywords_match = re.search(
                r"KEYWORDS:\s*(.+?)(?:\n|$)", response, re.IGNORECASE
            )

            if not title_match or not desc_match:
                logger.error("Failed to parse required fields from LLM response")
                return None

            # Extract and clean title
            title = title_match.group(1).strip()

            # Extract and clean description
            description = desc_match.group(1).strip()

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

            return title, description, hashtags, keywords

        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}", exc_info=True)
            return None
