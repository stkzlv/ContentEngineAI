"""Batch metadata generation for multiple products concurrently.

This module provides efficient batch processing of platform metadata generation,
supporting concurrent execution with rate limiting and progress tracking.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import aiohttp

from src.ai.platform_metadata.cache import MetadataCache
from src.ai.platform_metadata.models import PlatformMetadata
from src.scraper.amazon.scraper import ProductData
from src.video.config.llm_settings import LLMSettings

logger = logging.getLogger(__name__)


# Type alias for progress callback
ProgressCallback = Callable[[int, int, str, str | None], None]


@dataclass
class ProductGenerationResult:
    """Result of metadata generation for a single product.

    Attributes
    ----------
        product_id: Product identifier (ASIN or similar)
        success: Whether generation succeeded for all platforms
        metadata: Dictionary mapping platform names to generated metadata
        errors: Dictionary mapping platform names to error messages
        duration_seconds: Time taken to generate metadata for this product
        from_cache: Dictionary mapping platform names to cache hit status

    """

    product_id: str
    success: bool
    metadata: dict[str, PlatformMetadata | None] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    duration_seconds: float = 0.0
    from_cache: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert result to dictionary for JSON serialization."""
        return {
            "product_id": self.product_id,
            "success": self.success,
            "metadata": {
                k: v.to_dict() if v else None for k, v in self.metadata.items()
            },
            "errors": self.errors,
            "duration_seconds": self.duration_seconds,
            "from_cache": self.from_cache,
        }


@dataclass
class BatchGenerationResult:
    """Aggregated results from batch metadata generation.

    Attributes
    ----------
        total_products: Total number of products processed
        successful_products: Number of products with all platforms successful
        failed_products: Number of products with at least one platform failure
        results: List of per-product generation results
        total_duration_seconds: Total time for batch generation
        started_at: ISO 8601 timestamp of batch start
        completed_at: ISO 8601 timestamp of batch completion

    """

    total_products: int
    successful_products: int
    failed_products: int
    results: list[ProductGenerationResult]
    total_duration_seconds: float
    started_at: str
    completed_at: str

    def to_dict(self) -> dict:
        """Convert batch result to dictionary for JSON serialization."""
        return {
            "total_products": self.total_products,
            "successful_products": self.successful_products,
            "failed_products": self.failed_products,
            "results": [r.to_dict() for r in self.results],
            "total_duration_seconds": self.total_duration_seconds,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.total_products == 0:
            return 0.0
        return (self.successful_products / self.total_products) * 100


class BatchMetadataGenerator:
    """Generates platform metadata for multiple products concurrently.

    This class provides efficient batch processing with:
    - Concurrent execution using asyncio.gather()
    - Rate limiting via semaphore to respect API limits
    - Progress tracking with customizable callbacks
    - Per-product error isolation (one failure doesn't block others)
    - Optional caching to avoid regenerating unchanged products

    Example usage:
        generator = BatchMetadataGenerator(
            max_concurrent=5,
            progress_callback=lambda n, total, pid, status: print(
                f"[{n}/{total}] {pid}: {status}"
            ),
        )

        results = await generator.generate_batch(
            products, settings, secrets, session, platform_settings, paths
        )

        print(f"Success rate: {results.success_rate:.1f}%")

    """

    def __init__(
        self,
        max_concurrent: int = 3,
        progress_callback: ProgressCallback | None = None,
    ):
        """Initialize batch generator.

        Args:
        ----
            max_concurrent: Maximum number of concurrent product generations (1-20)
            progress_callback: Optional callback for progress updates with signature:
                (current: int, total: int, product_id: str, status: str | None) -> None

        """
        self.max_concurrent = max(1, min(20, max_concurrent))
        self.progress_callback = progress_callback
        self._semaphore: asyncio.Semaphore | None = None

    async def generate_batch(
        self,
        products: list[ProductData],
        settings: LLMSettings,
        secrets: dict[str, str],
        session: aiohttp.ClientSession,
        platform_settings: dict[str, dict],
        intermediate_paths: dict[str, Path],
        debug_mode: bool = False,
        api_settings=None,
        cache: MetadataCache | None = None,
    ) -> BatchGenerationResult:
        """Generate metadata for multiple products concurrently.

        This method processes products in parallel while respecting rate limits.
        Each product is processed independently, so failures in one product
        don't affect others.

        Args:
        ----
            products: List of product data objects to generate metadata for
            settings: LLM configuration (API keys, models, timeouts)
            secrets: Dictionary containing API keys
            session: Shared aiohttp session for all requests
            platform_settings: Dict mapping platform names to their settings
            intermediate_paths: Dictionary of file paths for outputs
            debug_mode: Enable verbose logging if True
            api_settings: Optional API-specific settings override
            cache: Optional MetadataCache for caching generated metadata

        Returns:
        -------
            BatchGenerationResult with aggregated results and per-product details

        Example:
        -------
            products = [product1, product2, product3]
            results = await generator.generate_batch(
                products, settings, secrets, session,
                {"youtube": {...}, "tiktok": {...}},
                intermediate_paths, debug_mode=True, cache=cache
            )

            for result in results.results:
                if result.success:
                    print(f"{result.product_id}: Generated successfully")
                else:
                    print(f"{result.product_id}: Errors: {result.errors}")

        """
        if not products:
            logger.warning("No products provided for batch generation")
            now = datetime.now(UTC).isoformat()
            return BatchGenerationResult(
                total_products=0,
                successful_products=0,
                failed_products=0,
                results=[],
                total_duration_seconds=0.0,
                started_at=now,
                completed_at=now,
            )

        started_at = datetime.now(UTC)
        logger.info(
            f"Starting batch metadata generation for {len(products)} products "
            f"(max_concurrent={self.max_concurrent})"
        )

        # Initialize semaphore for rate limiting
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

        # Import factory here to avoid circular imports
        from src.ai.platform_metadata import PlatformMetadataFactory

        # Create tasks for all products
        tasks = []
        for idx, product in enumerate(products):
            task = self._generate_single_product(
                idx=idx,
                total=len(products),
                product=product,
                factory=PlatformMetadataFactory,
                settings=settings,
                secrets=secrets,
                session=session,
                platform_settings=platform_settings,
                intermediate_paths=intermediate_paths,
                debug_mode=debug_mode,
                api_settings=api_settings,
                cache=cache,
            )
            tasks.append(task)

        # Execute all tasks concurrently (semaphore limits actual concurrency)
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        results: list[ProductGenerationResult] = []
        successful = 0
        failed = 0

        for idx, result in enumerate(results_list):
            if isinstance(result, BaseException):
                # Task raised an unexpected exception
                product = products[idx]
                product_id = getattr(product, "asin", None) or getattr(
                    product, "id", f"product_{idx}"
                )
                logger.error(
                    f"Unexpected error for product {product_id}: {result}",
                    exc_info=result,
                )
                results.append(
                    ProductGenerationResult(
                        product_id=str(product_id),
                        success=False,
                        errors={"_batch": str(result)},
                    )
                )
                failed += 1
            else:
                results.append(result)
                if result.success:
                    successful += 1
                else:
                    failed += 1

        completed_at = datetime.now(UTC)
        total_duration = (completed_at - started_at).total_seconds()

        logger.info(
            f"Batch generation complete: {successful}/{len(products)} successful "
            f"({failed} failed) in {total_duration:.1f}s"
        )

        return BatchGenerationResult(
            total_products=len(products),
            successful_products=successful,
            failed_products=failed,
            results=results,
            total_duration_seconds=total_duration,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
        )

    async def _generate_single_product(
        self,
        idx: int,
        total: int,
        product: ProductData,
        factory: type,  # PlatformMetadataFactory
        settings: LLMSettings,
        secrets: dict[str, str],
        session: aiohttp.ClientSession,
        platform_settings: dict[str, dict],
        intermediate_paths: dict[str, Path],
        debug_mode: bool,
        api_settings,
        cache: MetadataCache | None,
    ) -> ProductGenerationResult:
        """Generate metadata for a single product with rate limiting.

        This method is called for each product and uses a semaphore to limit
        concurrent execution. Errors are caught and returned in the result
        rather than propagating to prevent one failure from blocking others.

        """
        product_id = getattr(product, "asin", None) or getattr(
            product, "id", f"product_{idx}"
        )
        product_id_str = str(product_id) if product_id else f"product_{idx}"

        # Acquire semaphore to respect rate limits
        async with self._semaphore:  # type: ignore[union-attr]
            start_time = datetime.now(UTC)

            # Report progress: starting
            self._report_progress(idx + 1, total, product_id_str, "starting")

            try:
                # Check cache first for all platforms
                metadata_results: dict[str, PlatformMetadata | None] = {}
                from_cache: dict[str, bool] = {}
                platforms_to_generate: list[str] = []

                for platform in platform_settings:
                    if cache:
                        cached = cache.get(product_id_str, platform, product)
                        if cached:
                            metadata_results[platform] = cached
                            from_cache[platform] = True
                            logger.debug(
                                f"[{idx + 1}/{total}] Cache hit for "
                                f"{product_id_str}/{platform}"
                            )
                            continue
                    platforms_to_generate.append(platform)
                    from_cache[platform] = False

                # Generate for platforms not in cache
                if platforms_to_generate:
                    # Filter platform settings to only include platforms we need
                    filtered_settings = {
                        p: platform_settings[p] for p in platforms_to_generate
                    }

                    generated = await factory.generate_multi_platform(  # type: ignore[attr-defined]
                        product=product,
                        settings=settings,
                        secrets=secrets,
                        session=session,
                        platform_settings=filtered_settings,
                        intermediate_paths=intermediate_paths,
                        debug_mode=debug_mode,
                        api_settings=api_settings,
                        cache=cache,  # Pass cache so factory can also cache
                    )

                    # Merge generated results
                    for platform, meta in generated.items():
                        metadata_results[platform] = meta

                # Determine success (all platforms succeeded)
                errors: dict[str, str] = {}
                for platform, meta in metadata_results.items():
                    if meta is None:
                        errors[platform] = "Generation failed"

                success = len(errors) == 0

                end_time = datetime.now(UTC)
                duration = (end_time - start_time).total_seconds()

                # Report progress: completed
                status = "success" if success else f"partial ({len(errors)} errors)"
                self._report_progress(idx + 1, total, product_id_str, status)

                logger.info(
                    f"[{idx + 1}/{total}] {product_id_str}: {status} "
                    f"(duration: {duration:.1f}s)"
                )

                return ProductGenerationResult(
                    product_id=product_id_str,
                    success=success,
                    metadata=metadata_results,
                    errors=errors,
                    duration_seconds=duration,
                    from_cache=from_cache,
                )

            except Exception as e:
                logger.error(
                    f"[{idx + 1}/{total}] Error generating metadata for "
                    f"{product_id_str}: {e}",
                    exc_info=e,
                )

                end_time = datetime.now(UTC)
                duration = (end_time - start_time).total_seconds()

                # Report progress: error
                self._report_progress(idx + 1, total, product_id_str, "error")

                return ProductGenerationResult(
                    product_id=product_id_str,
                    success=False,
                    errors={"_generation": str(e)},
                    duration_seconds=duration,
                )

    def _report_progress(
        self,
        current: int,
        total: int,
        product_id: str,
        status: str | None,
    ) -> None:
        """Report progress via callback if configured."""
        if self.progress_callback:
            try:
                self.progress_callback(current, total, product_id, status)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
