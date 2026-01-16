"""Platform-specific metadata generation for YouTube, TikTok, and Instagram.

This package provides platform-optimized metadata generation including titles,
descriptions, captions, and hashtags tailored to each platform's best practices.
"""

# Import for type hints and async
import asyncio
import logging
from pathlib import Path

import aiohttp

from src.ai.platform_metadata.ab_testing import (
    ABTestingSettings,
    PlatformABConfig,
    PromptVariant,
    PromptVariantSelector,
    VariantSelection,
)
from src.ai.platform_metadata.base import BasePlatformMetadataGenerator
from src.ai.platform_metadata.batch import (
    BatchGenerationResult,
    BatchMetadataGenerator,
    ProductGenerationResult,
    ProgressCallback,
)
from src.ai.platform_metadata.cache import MetadataCache
from src.ai.platform_metadata.export import (
    ExportFormat,
    ExportResult,
    MetadataExporter,
)
from src.ai.platform_metadata.instagram import InstagramMetadataGenerator
from src.ai.platform_metadata.models import (
    BatchGenerationSettings,
    ExportSettings,
    InstagramPlatformSettings,
    MetadataCacheSettings,
    PlatformMetadata,
    PlatformMetadataSettings,
    TikTokPlatformSettings,
    YouTubePlatformSettings,
)
from src.ai.platform_metadata.tiktok import TikTokMetadataGenerator
from src.ai.platform_metadata.utilities import (
    call_llm_api_with_retry,
    fetch_and_select_model,
    format_prompt,
    generate_with_llm,
    load_metadata_from_file,
    load_prompt_template,
    save_metadata_to_file,
)
from src.ai.platform_metadata.youtube import YouTubeMetadataGenerator
from src.scraper.amazon.scraper import ProductData
from src.video.config.llm_settings import LLMSettings

logger = logging.getLogger(__name__)


class PlatformMetadataFactory:
    """Factory for creating and managing platform-specific metadata generators.

    This factory provides two modes of operation:
    1. Single platform: create() returns a specific generator instance
    2. Multi-platform: generate_multi_platform() runs all platforms in parallel

    The factory uses a dictionary mapping to enable easy extensibility for future
    platforms. All generators share the same session and secrets for efficiency.

    Example usage:
        # Single platform
        generator = PlatformMetadataFactory.create("youtube", youtube_settings)
        metadata = await generator.generate(product, settings, secrets, session, ...)

        # Multi-platform (parallel)
        results = await PlatformMetadataFactory.generate_multi_platform(
            product, settings, secrets, session, platform_settings, ...
        )
        # results = {"youtube": metadata, "tiktok": metadata, "instagram": metadata}
    """

    # Platform generator mapping for extensibility
    _PLATFORM_GENERATORS: dict[str, type[BasePlatformMetadataGenerator]] = {
        "youtube": YouTubeMetadataGenerator,
        "tiktok": TikTokMetadataGenerator,
        "instagram": InstagramMetadataGenerator,
    }

    @staticmethod
    def create(platform: str, platform_settings: dict) -> BasePlatformMetadataGenerator:
        """Create a platform-specific metadata generator instance.

        Args:
        ----
            platform: Platform identifier ("youtube", "tiktok", "instagram")
            platform_settings: Platform-specific configuration dict

        Returns:
        -------
            Instance of the appropriate generator class

        Raises:
        ------
            ValueError: If platform name is not recognized

        Example:
        -------
            youtube_gen = PlatformMetadataFactory.create(
                "youtube",
                {"title_length_max": 60, "hashtag_count_min": 3, ...}
            )

        """
        if platform not in PlatformMetadataFactory._PLATFORM_GENERATORS:
            known_platforms = ", ".join(
                PlatformMetadataFactory._PLATFORM_GENERATORS.keys()
            )
            raise ValueError(
                f"Unknown platform '{platform}'. "
                f"Supported platforms: {known_platforms}"
            )

        generator_class = PlatformMetadataFactory._PLATFORM_GENERATORS[platform]
        return generator_class(platform_settings)  # type: ignore[call-arg]

    @staticmethod
    async def generate_multi_platform(
        product: ProductData,
        settings: LLMSettings,
        secrets: dict[str, str],
        session: aiohttp.ClientSession,
        platform_settings: dict[str, dict],
        intermediate_paths: dict[str, Path],
        debug_mode: bool = False,
        api_settings=None,
        cache: MetadataCache | None = None,
    ) -> dict[str, PlatformMetadata | None]:
        """Generate metadata for all platforms in parallel using asyncio.gather().

        This method runs YouTube, TikTok, and Instagram generators concurrently,
        maximizing throughput. Errors in one platform don't block others due to
        return_exceptions=True.

        Supports optional caching to avoid regenerating metadata for unchanged
        products. When cache is provided, cached entries are returned immediately
        and only missing/expired entries trigger LLM generation.

        Args:
        ----
            product: Product data for metadata generation
            settings: LLM configuration (API keys, models, timeouts)
            secrets: Dictionary containing API keys
            session: Shared aiohttp session for all generators
            platform_settings: Dict mapping platform names to their settings
                Example: {
                    "youtube": {"title_length_max": 60, ...},
                    "tiktok": {"caption_length_optimal": 150, ...},
                    "instagram": {"caption_style": "seo", ...}
                }
            intermediate_paths: Dictionary of file paths for outputs
            debug_mode: Enable verbose logging if True
            api_settings: Optional API-specific settings override
            cache: Optional MetadataCache for caching generated metadata

        Returns:
        -------
            Dictionary mapping platform names to PlatformMetadata objects or None:
                {
                    "youtube": PlatformMetadata(...),
                    "tiktok": PlatformMetadata(...),
                    "instagram": PlatformMetadata(...)
                }
            If a platform fails, its value will be None.

        Example:
        -------
            # Without cache
            results = await PlatformMetadataFactory.generate_multi_platform(
                product, llm_settings, secrets, session,
                platform_settings, intermediate_paths, debug_mode=True
            )

            # With cache (recommended)
            cache = MetadataCache(cache_settings)
            results = await PlatformMetadataFactory.generate_multi_platform(
                product, llm_settings, secrets, session,
                platform_settings, intermediate_paths,
                debug_mode=True, cache=cache
            )

            if results["youtube"]:
                print(f"YouTube title: {results['youtube'].title}")

        """
        logger.info("Starting multi-platform metadata generation in parallel")

        asin = getattr(product, "asin", None)
        product_id: str = asin or getattr(product, "id", "unknown") or "unknown"

        # Check cache for existing metadata
        results: dict[str, PlatformMetadata | None] = {}
        platforms_to_generate: list[str] = []

        for platform in platform_settings:
            if cache:
                cached_metadata = cache.get(product_id, platform, product)
                if cached_metadata:
                    results[platform] = cached_metadata
                    logger.info(f"Using cached {platform} metadata for {product_id}")
                    continue
            platforms_to_generate.append(platform)

        # If all platforms were cached, return early
        if not platforms_to_generate:
            logger.info(f"All platforms served from cache for {product_id}")
            return results

        # Create generators for platforms that need generation
        generators = {}
        for platform in platforms_to_generate:
            p_settings = platform_settings[platform]
            try:
                generators[platform] = PlatformMetadataFactory.create(
                    platform, p_settings
                )
                logger.debug(f"Created {platform} generator")
            except ValueError as e:
                logger.warning(f"Skipping unknown platform '{platform}': {e}")
                results[platform] = None
                continue

        # Build async tasks for parallel execution
        tasks = []
        task_platforms = []
        for platform, generator in generators.items():
            task = generator.generate(
                product,
                settings,
                secrets,
                session,
                intermediate_paths,
                debug_mode,
                api_settings,
            )
            tasks.append(task)
            task_platforms.append(platform)

        # Run all generators in parallel with fault tolerance
        # return_exceptions=True ensures one failure doesn't block others
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Map results back to platform names and cache successful results
        for platform, result in zip(task_platforms, results_list, strict=False):
            if isinstance(result, BaseException):
                logger.error(
                    f"Error generating {platform} metadata: {result}",
                    exc_info=result,
                )
                results[platform] = None
            else:
                results[platform] = result
                status = "success" if result else "failed"
                logger.info(f"{platform.capitalize()} metadata generation: {status}")

                # Cache successful results
                if result and cache:
                    cache.set(result, product)

        success_count = sum(1 for v in results.values() if v is not None)
        total_count = len(results)
        cached_count = len(platform_settings) - len(platforms_to_generate)
        generated_count = success_count - cached_count

        logger.info(
            f"Multi-platform generation complete. "
            f"Success: {success_count}/{total_count} "
            f"(cached: {cached_count}, generated: {generated_count})"
        )

        return results


__all__ = [
    # Abstract base class
    "BasePlatformMetadataGenerator",
    # Data models
    "PlatformMetadata",
    "PlatformMetadataSettings",
    "YouTubePlatformSettings",
    "TikTokPlatformSettings",
    "InstagramPlatformSettings",
    # Caching
    "MetadataCache",
    "MetadataCacheSettings",
    # A/B Testing
    "ABTestingSettings",
    "PlatformABConfig",
    "PromptVariant",
    "PromptVariantSelector",
    "VariantSelection",
    # Batch Generation
    "BatchMetadataGenerator",
    "BatchGenerationSettings",
    "BatchGenerationResult",
    "ProductGenerationResult",
    "ProgressCallback",
    # Export
    "MetadataExporter",
    "ExportSettings",
    "ExportFormat",
    "ExportResult",
    # Platform generators
    "YouTubeMetadataGenerator",
    "TikTokMetadataGenerator",
    "InstagramMetadataGenerator",
    # Factory
    "PlatformMetadataFactory",
    # Shared utilities (LLM integration)
    "load_prompt_template",
    "format_prompt",
    "fetch_and_select_model",
    "call_llm_api_with_retry",
    "generate_with_llm",
    # Shared utilities (file I/O)
    "save_metadata_to_file",
    "load_metadata_from_file",
]
