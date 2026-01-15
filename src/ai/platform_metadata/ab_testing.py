"""A/B testing support for platform metadata prompt templates.

This module provides deterministic prompt variant selection for comparing
metadata quality across different prompt templates. Variants are selected
based on product_id hash for reproducibility.

Features:
    - Deterministic variant selection (same product always gets same variant)
    - Multiple variants per platform
    - Variant tracking in metadata for analysis
    - Configurable variant weights for traffic splitting
"""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PromptVariant(BaseModel):
    """Configuration for a single prompt variant.

    Attributes
    ----------
        name: Unique identifier for this variant (e.g., "control", "variant_a")
        template_path: Path to prompt template file (relative to project root)
        weight: Traffic weight for this variant (0-100)
        description: Human-readable description of what this variant tests

    """

    name: str = Field(..., description="Unique variant identifier")
    template_path: str = Field(..., description="Path to prompt template file")
    weight: int = Field(
        50,
        ge=0,
        le=100,
        description="Traffic weight (0-100, higher = more traffic)",
    )
    description: str = Field("", description="Description of variant purpose")


class PlatformABConfig(BaseModel):
    """A/B testing configuration for a single platform.

    Attributes
    ----------
        enabled: Enable A/B testing for this platform
        variants: List of prompt variants to test

    """

    enabled: bool = Field(True, description="Enable A/B testing for this platform")
    variants: list[PromptVariant] = Field(
        default_factory=list,
        description="List of prompt variants to test",
    )


class ABTestingSettings(BaseModel):
    """Global A/B testing configuration for platform metadata.

    Attributes
    ----------
        enabled: Global enable/disable for A/B testing
        youtube: YouTube-specific A/B test configuration
        tiktok: TikTok-specific A/B test configuration
        instagram: Instagram-specific A/B test configuration

    """

    enabled: bool = Field(True, description="Enable A/B testing globally")
    youtube: PlatformABConfig = Field(
        default_factory=lambda: PlatformABConfig(
            enabled=True,
            variants=[
                PromptVariant(
                    name="control",
                    template_path="src/ai/prompts/youtube_metadata.md",
                    weight=50,
                    description="Original YouTube metadata prompt",
                ),
            ],
        ),
        description="YouTube A/B test configuration",
    )
    tiktok: PlatformABConfig = Field(
        default_factory=lambda: PlatformABConfig(
            enabled=True,
            variants=[
                PromptVariant(
                    name="control",
                    template_path="src/ai/prompts/tiktok_caption.md",
                    weight=50,
                    description="Original TikTok caption prompt",
                ),
            ],
        ),
        description="TikTok A/B test configuration",
    )
    instagram: PlatformABConfig = Field(
        default_factory=lambda: PlatformABConfig(
            enabled=True,
            variants=[
                PromptVariant(
                    name="control",
                    template_path="src/ai/prompts/instagram_caption.md",
                    weight=50,
                    description="Original Instagram caption prompt",
                ),
            ],
        ),
        description="Instagram A/B test configuration",
    )


@dataclass
class VariantSelection:
    """Result of variant selection for a product.

    Attributes
    ----------
        variant_name: Name of selected variant
        template_path: Path to prompt template
        selection_hash: Hash used for selection (for debugging)
        platform: Platform this selection is for

    """

    variant_name: str
    template_path: Path
    selection_hash: str
    platform: str

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/tracking."""
        return {
            "variant_name": self.variant_name,
            "template_path": str(self.template_path),
            "selection_hash": self.selection_hash,
            "platform": self.platform,
        }


class PromptVariantSelector:
    """Deterministic prompt variant selector for A/B testing.

    This class selects prompt variants based on a hash of the product_id,
    ensuring reproducibility - the same product always gets the same variant.

    The selection is weighted, allowing traffic splitting between variants.
    For example, with two variants at 50/50 weight, approximately half of
    products will get each variant.

    Example usage:
        selector = PromptVariantSelector(ab_settings)

        # Select variant for a product
        selection = selector.select_variant("B0TESTASIN", "youtube")

        # Use selected template
        template_path = selection.template_path
        variant_name = selection.variant_name

        # Track in metadata
        metadata.prompt_variant = variant_name

    """

    def __init__(
        self,
        settings: ABTestingSettings,
        project_root: Path | None = None,
    ):
        """Initialize variant selector.

        Args:
        ----
            settings: A/B testing configuration
            project_root: Project root for resolving template paths

        """
        self.settings = settings
        self.project_root = project_root or Path.cwd()

    def select_variant(
        self,
        product_id: str,
        platform: str,
    ) -> VariantSelection:
        """Select a prompt variant for a product.

        Selection is deterministic based on product_id hash, ensuring
        the same product always gets the same variant for reproducibility.

        Args:
        ----
            product_id: Product identifier (e.g., ASIN)
            platform: Platform name (youtube, tiktok, instagram)

        Returns:
        -------
            VariantSelection with selected variant details

        Raises:
        ------
            ValueError: If platform is not configured or has no variants

        """
        # Get platform config
        platform_config = self._get_platform_config(platform)

        if not platform_config.enabled or not platform_config.variants:
            raise ValueError(
                f"A/B testing not configured for platform '{platform}'. "
                f"Add variants to settings.{platform}.variants"
            )

        # Compute deterministic hash for selection
        selection_hash = self._compute_selection_hash(product_id, platform)
        hash_value = int(selection_hash[:8], 16)  # Use first 8 hex chars

        # Select variant based on weighted distribution
        selected = self._weighted_select(platform_config.variants, hash_value)

        # Resolve template path
        template_path = self.project_root / selected.template_path

        logger.info(
            f"A/B test: Selected variant '{selected.name}' for "
            f"{platform}/{product_id} (hash: {selection_hash[:8]})"
        )

        return VariantSelection(
            variant_name=selected.name,
            template_path=template_path,
            selection_hash=selection_hash[:8],
            platform=platform,
        )

    def get_default_template(self, platform: str) -> Path:
        """Get the default (control) template for a platform.

        Used when A/B testing is disabled or as fallback.

        Args:
        ----
            platform: Platform name

        Returns:
        -------
            Path to default template

        """
        default_templates = {
            "youtube": "src/ai/prompts/youtube_metadata.md",
            "tiktok": "src/ai/prompts/tiktok_caption.md",
            "instagram": "src/ai/prompts/instagram_caption.md",
        }

        template_path = default_templates.get(platform)
        if not template_path:
            raise ValueError(f"Unknown platform: {platform}")

        return self.project_root / template_path

    def is_enabled(self, platform: str) -> bool:
        """Check if A/B testing is enabled for a platform.

        Args:
        ----
            platform: Platform name

        Returns:
        -------
            True if A/B testing is enabled and configured

        """
        if not self.settings.enabled:
            return False

        try:
            platform_config = self._get_platform_config(platform)
            return platform_config.enabled and len(platform_config.variants) > 0
        except ValueError:
            return False

    def get_variant_names(self, platform: str) -> list[str]:
        """Get list of variant names for a platform.

        Args:
        ----
            platform: Platform name

        Returns:
        -------
            List of variant names

        """
        try:
            platform_config = self._get_platform_config(platform)
            return [v.name for v in platform_config.variants]
        except ValueError:
            return []

    def _get_platform_config(self, platform: str) -> PlatformABConfig:
        """Get A/B config for a platform.

        Args:
        ----
            platform: Platform name

        Returns:
        -------
            PlatformABConfig for the platform

        Raises:
        ------
            ValueError: If platform is not recognized

        """
        configs = {
            "youtube": self.settings.youtube,
            "tiktok": self.settings.tiktok,
            "instagram": self.settings.instagram,
        }

        config = configs.get(platform)
        if config is None:
            raise ValueError(
                f"Unknown platform '{platform}'. "
                f"Supported: {', '.join(configs.keys())}"
            )

        return config

    @staticmethod
    def _compute_selection_hash(product_id: str, platform: str) -> str:
        """Compute deterministic hash for variant selection.

        The hash combines product_id and platform to ensure:
        - Same product gets same variant (reproducibility)
        - Different platforms can have different variants for same product

        Args:
        ----
            product_id: Product identifier
            platform: Platform name

        Returns:
        -------
            SHA-256 hash string

        """
        hash_input = f"{product_id}:{platform}:ab_test"
        return hashlib.sha256(hash_input.encode()).hexdigest()

    @staticmethod
    def _weighted_select(
        variants: list[PromptVariant],
        hash_value: int,
    ) -> PromptVariant:
        """Select variant using weighted distribution.

        Distributes traffic according to variant weights. Higher weights
        get more traffic proportionally.

        Args:
        ----
            variants: List of variants with weights
            hash_value: Integer hash value for selection

        Returns:
        -------
            Selected PromptVariant

        """
        if len(variants) == 1:
            return variants[0]

        # Calculate total weight and normalize
        total_weight = sum(v.weight for v in variants)
        if total_weight == 0:
            # Equal distribution if all weights are 0
            return variants[hash_value % len(variants)]

        # Map hash to weight space
        position = hash_value % total_weight

        # Select based on cumulative weight
        cumulative = 0
        for variant in variants:
            cumulative += variant.weight
            if position < cumulative:
                return variant

        # Fallback to last variant (shouldn't happen)
        return variants[-1]
