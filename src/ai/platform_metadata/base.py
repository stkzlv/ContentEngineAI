"""Abstract base class for platform-specific metadata generators.

This module defines the common interface that all platform-specific metadata
generators (YouTube, TikTok, Instagram) must implement. It provides shared
utilities for LLM-based generation while enforcing implementation of platform-
specific validation and generation logic.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import aiohttp

from src.ai.description_generator import format_prompt, load_prompt_template
from src.ai.platform_metadata.models import PlatformMetadata
from src.scraper.amazon.scraper import ProductData
from src.video.config.llm_settings import LLMSettings

logger = logging.getLogger(__name__)


class BasePlatformMetadataGenerator(ABC):
    """Abstract base class for platform-specific metadata generators.

    This class defines the common interface for all platform metadata generators.
    Subclasses must implement platform-specific generation logic, validation rules,
    and platform identification.

    All generators follow a common pattern:
    1. Load platform-specific prompt template
    2. Format prompt with product data
    3. Call LLM API to generate metadata
    4. Validate generated content against platform rules
    5. Return validated PlatformMetadata object

    Shared utilities from description_generator.py:
    - load_prompt_template(): Load prompt templates from files
    - format_prompt(): Inject product data into templates
    - _fetch_and_select_model(): Auto-select free LLM models
    - _call_llm_api_with_retry(): Call LLM API with retry logic
    """

    @abstractmethod
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
        """Generate platform-specific metadata using LLM.

        This method must be implemented by each platform-specific generator to
        produce optimized titles, descriptions/captions, hashtags, and keywords
        according to platform best practices.

        Args:
            product: Product data containing title, description, URL, etc.
            settings: LLM configuration (API keys, models, timeouts)
            secrets: Dictionary containing API keys and credentials
            session: Aiohttp session for HTTP requests
            intermediate_paths: Dictionary of file paths for outputs
            debug_mode: Enable verbose logging if True
            api_settings: Optional API-specific settings override

        Returns:
            PlatformMetadata object with generated content, or None if generation fails

        Implementation Requirements:
            - Load platform-specific prompt template (e.g., youtube_metadata.md)
            - Format prompt with product data using _format_product_prompt()
            - Call LLM API using _call_llm_with_retry()
            - Parse LLM response to extract title/description/hashtags/keywords
            - Create PlatformMetadata object with extracted data
            - Validate metadata using validate() before returning
            - Return None if validation fails or LLM calls fail after retries
        """
        pass

    @abstractmethod
    def validate(self, metadata: PlatformMetadata) -> tuple[bool, str]:
        """Validate generated metadata against platform-specific rules.

        Each platform has different requirements for character limits, hashtag
        counts, required elements (e.g., #ad, #Shorts), and content formatting.
        This method ensures generated metadata meets platform guidelines.

        Args:
            metadata: PlatformMetadata object to validate

        Returns:
            Tuple of (is_valid, message):
                - is_valid: True if metadata passes all validation rules
                - message: Empty string if valid, detailed error message if invalid

        Validation Requirements:
            - Check character limits (title, description/caption)
            - Verify hashtag count within platform min/max range
            - Ensure required hashtags are present (e.g., #ad for sponsored content)
            - Validate content formatting (e.g., no prohibited characters)
            - Check keyword count if applicable
            - Return specific error messages for failed validations
        """
        pass

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return the platform identifier for this generator.

        Returns:
            Platform identifier: "youtube", "tiktok", or "instagram"

        This property is used for:
            - Logging and debugging
            - Selecting correct prompt template
            - Naming output files (e.g., metadata_youtube.json)
            - Platform-specific configuration lookup
        """
        pass

    # --- Protected Shared Utilities ---
    # These methods can be used by subclasses but are not part of the public API

    def _load_prompt_template(self, template_path: Path) -> str:
        """Load platform-specific prompt template from file.

        This is a convenience wrapper around description_generator.load_prompt_template()
        that adds platform-specific error handling and logging.

        Args:
            template_path: Path to the prompt template file

        Returns:
            Loaded prompt template as string

        Raises:
            FileNotFoundError: If template file doesn't exist

        Example:
            template = self._load_prompt_template(
                Path("src/ai/prompts/youtube_metadata.md")
            )
        """
        logger.debug(
            f"Loading prompt template for {self.platform_name}: {template_path}"
        )
        return load_prompt_template(template_path)

    def _format_product_prompt(self, template: str, product: ProductData) -> str:
        """Format prompt template with product data.

        This is a convenience wrapper around description_generator.format_prompt()
        that adds platform-specific logging.

        Args:
            template: Prompt template string with placeholders
            product: Product data to inject into template

        Returns:
            Formatted prompt ready for LLM API

        Placeholders replaced:
            - {FULL_PRODUCT_NAME}: product.title
            - {PRODUCT_DESCRIPTION}: product.description
            - {PRODUCT_URL}: product.shortened_affiliate_link or product.url

        Example:
            prompt = self._format_product_prompt(template, product)
        """
        logger.debug(f"Formatting prompt for {self.platform_name}")
        return format_prompt(template, product)

    def _calculate_character_counts(
        self, title: str | None, description: str
    ) -> dict[str, int]:
        """Calculate character counts for title and description.

        Helper method to populate the character_counts field in PlatformMetadata.
        Used for validation and analytics.

        Args:
            title: Optional title (YouTube only)
            description: Description or caption text

        Returns:
            Dictionary with character counts, e.g.:
                {"title": 58, "description": 487} or
                {"description": 150} if no title

        Example:
            counts = self._calculate_character_counts(
                title="Best Wireless Earbuds",
                description="Check out these amazing earbuds! #ad"
            )
        """
        counts = {"description": len(description)}
        if title:
            counts["title"] = len(title)
        return counts

    def _truncate_if_needed(
        self, text: str, max_length: int, label: str = "text"
    ) -> str:
        """Truncate text to maximum length if exceeded, adding ellipsis.

        Gracefully handles character limit violations by truncating with "..."
        and logging a warning. Used when LLM generates content slightly over limits.

        Args:
            text: Text to potentially truncate
            max_length: Maximum allowed character count
            label: Description for logging (e.g., "YouTube title", "TikTok caption")

        Returns:
            Original text if within limit, or truncated text with "..." appended

        Example:
            title = self._truncate_if_needed(
                llm_title, max_length=60, label="YouTube title"
            )
        """
        if len(text) <= max_length:
            return text

        truncated = text[: max_length - 3] + "..."
        logger.warning(
            f"{self.platform_name} {label} exceeded {max_length} chars "
            f"(was {len(text)}), truncated to: {truncated}"
        )
        return truncated
