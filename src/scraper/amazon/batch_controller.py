"""Batch processing orchestration for Amazon scraper.

This module provides the BatchController class for coordinating batch scraping
operations across multiple product IDs and keywords.
"""

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .scraper import BotasaurusAmazonScraper

from .config import get_batch_logging_config
from .models import BatchConfig, BatchSummary, ProductData, ProductResult
from .utils import validate_asin_format


class BatchController:
    """Orchestrates batch processing of product IDs and keywords.

    Coordinates sequential scraping of multiple products, handles
    deduplication, progress tracking, and error handling with
    fail-fast support.
    """

    def __init__(self, scraper: "BotasaurusAmazonScraper", config: BatchConfig):
        """Initialize batch controller.

        Args:
        ----
            scraper: BotasaurusAmazonScraper instance for delegating scraping
            config: BatchConfig with product IDs, keywords, and settings

        """
        self.scraper = scraper
        self.config = config
        self.logger = scraper.logger
        self.results: list[ProductResult] = []
        self.seen_asins: set[str] = set()

        # Load logging configuration from YAML
        self.log_config = get_batch_logging_config()
        separator_char = str(self.log_config["separator_char"])
        separator_width = int(self.log_config["separator_width"])
        self.separator = separator_char * separator_width

    def run_batch(self) -> BatchSummary:
        """Execute complete batch processing workflow.

        Processes product IDs first, then keywords, with deduplication
        and comprehensive summary reporting.

        Returns
        -------
            BatchSummary with detailed statistics and results

        """
        start_time = time.time()

        self.logger.info(self.separator)
        self.logger.info("STARTING BATCH SCRAPING")
        self.logger.info("Product IDs: %d", len(self.config.product_ids))
        self.logger.info("Keywords: %d", len(self.config.keywords))
        self.logger.info("Fail-fast: %s", self.config.fail_fast)
        self.logger.info(self.separator)

        # Process product IDs first
        product_id_results = self._process_product_ids()

        # Process keywords (if max products not reached)
        keyword_results = self._process_keywords()

        # Combine and deduplicate
        all_results = product_id_results + keyword_results
        deduplicated_results = self._deduplicate_products(all_results)

        # Generate summary
        duration_sec = time.time() - start_time
        summary = self._generate_summary(
            deduplicated_results,
            len(self.config.product_ids),
            len(self.config.keywords),
            duration_sec,
        )

        # Log final summary
        self._log_summary(summary)

        return summary

    def _process_product_ids(self) -> list[ProductResult]:
        """Process list of product IDs.

        Returns
        -------
            List of ProductResult objects for each product ID

        """
        results: list[ProductResult] = []

        if not self.config.product_ids:
            return results

        self.logger.info(
            "\n%s\nPROCESSING PRODUCT IDS (%d total)\n%s",
            self.separator,
            len(self.config.product_ids),
            self.separator,
        )

        for i, product_id in enumerate(self.config.product_ids, 1):
            # URLs are passed through directly; ASINs are validated
            is_url = product_id.startswith(("http://", "https://"))
            if not is_url and not validate_asin_format(product_id):
                self.logger.warning(
                    "[%d/%d] ⚠️  Invalid ASIN format: %s - Skipping",
                    i,
                    len(self.config.product_ids),
                    product_id,
                )
                results.append(
                    ProductResult(
                        product_id=product_id,
                        success=False,
                        data=None,
                        error="Invalid ASIN format",
                        source="product_id",
                    )
                )
                continue

            self.logger.info(
                "[%d/%d] Scraping product: %s",
                i,
                len(self.config.product_ids),
                product_id,
            )

            try:
                # Delegate to existing scraper with products_per_keyword limit
                # (single product scraping via keyword/ASIN)
                products = self.scraper.scrape_products_unified(
                    keyword=product_id,
                    search_params=self.config.search_params,
                    max_products=self.config.products_per_keyword,
                )

                if products and len(products) > 0:
                    product_data = products[0]
                    self.logger.info(
                        "[%d/%d] ✅ Successfully scraped: %s",
                        i,
                        len(self.config.product_ids),
                        product_id,
                    )
                    results.append(
                        ProductResult(
                            product_id=product_id,
                            success=True,
                            data=product_data,
                            error=None,
                            source="product_id",
                        )
                    )
                else:
                    self.logger.warning(
                        "[%d/%d] ⚠️  No data found for: %s",
                        i,
                        len(self.config.product_ids),
                        product_id,
                    )
                    results.append(
                        ProductResult(
                            product_id=product_id,
                            success=False,
                            data=None,
                            error="No data found",
                            source="product_id",
                        )
                    )

            except Exception as e:
                error_msg = str(e)
                self.logger.error(
                    "[%d/%d] ❌ Failed to scrape %s: %s",
                    i,
                    len(self.config.product_ids),
                    product_id,
                    error_msg,
                )
                results.append(
                    ProductResult(
                        product_id=product_id,
                        success=False,
                        data=None,
                        error=error_msg,
                        source="product_id",
                    )
                )

                # Fail-fast: stop on first error
                if self.config.fail_fast:
                    self.logger.error(
                        "❌ Fail-fast enabled: " "Stopping batch after first failure"
                    )
                    break

        return results

    def _process_keywords(self) -> list[ProductResult]:
        """Process list of keywords for product search.

        Returns
        -------
            List of ProductResult objects for products found via keywords

        """
        results: list[ProductResult] = []

        if not self.config.keywords:
            return results

        self.logger.info(
            "\n%s\nPROCESSING KEYWORDS (%d total)\n%s",
            self.separator,
            len(self.config.keywords),
            self.separator,
        )

        for i, keyword in enumerate(self.config.keywords, 1):
            self.logger.info(
                "[%d/%d] Searching keyword: %s", i, len(self.config.keywords), keyword
            )

            try:
                # Delegate to existing scraper with products_per_keyword limit
                products = self.scraper.scrape_products_unified(
                    keyword=keyword,
                    search_params=self.config.search_params,
                    max_products=self.config.products_per_keyword,
                )

                if products:
                    self.logger.info(
                        "[%d/%d] ✅ Found %d products for: %s",
                        i,
                        len(self.config.keywords),
                        len(products),
                        keyword,
                    )

                    # Add each product as a result
                    for product in products:
                        product_id = product.asin or product.title or "unknown"
                        results.append(
                            ProductResult(
                                product_id=product_id,
                                success=True,
                                data=product,
                                error=None,
                                source="keyword",
                            )
                        )

                    # Check if max_products limit reached
                    if len(results) >= self.config.max_products:
                        self.logger.info(
                            "✅ Reached max_products limit (%d). "
                            "Stopping keyword processing.",
                            self.config.max_products,
                        )
                        break

                else:
                    self.logger.warning(
                        "[%d/%d] ⚠️  No products found for: %s",
                        i,
                        len(self.config.keywords),
                        keyword,
                    )

            except Exception as e:
                error_msg = str(e)
                self.logger.error(
                    "[%d/%d] ❌ Failed to search %s: %s",
                    i,
                    len(self.config.keywords),
                    keyword,
                    error_msg,
                )

                # Fail-fast: stop on first error
                if self.config.fail_fast:
                    self.logger.error(
                        "❌ Fail-fast enabled: " "Stopping batch after first failure"
                    )
                    break

        return results

    def _deduplicate_products(
        self, results: list[ProductResult]
    ) -> list[ProductResult]:
        """Remove duplicate products by ASIN.

        Product IDs take precedence over keyword results for duplicates.

        Args:
        ----
            results: List of ProductResult objects

        Returns:
        -------
            Deduplicated list of ProductResult objects

        """
        seen_asins: set[str] = set()
        deduplicated: list[ProductResult] = []

        for result in results:
            # Extract ASIN from result
            asin = None
            if result.data and result.data.asin:
                asin = result.data.asin
            elif result.product_id and validate_asin_format(result.product_id):
                asin = result.product_id

            # Skip if already seen
            if asin and asin in seen_asins:
                self.logger.debug("Skipping duplicate ASIN: %s", asin)
                continue

            # Add to results
            deduplicated.append(result)
            if asin:
                seen_asins.add(asin)

        duplicates_removed = len(results) - len(deduplicated)
        if duplicates_removed > 0:
            self.logger.info(
                "Deduplication: Removed %d duplicate product(s)", duplicates_removed
            )

        return deduplicated

    def _generate_summary(
        self,
        results: list[ProductResult],
        product_ids_count: int,
        keywords_count: int,
        duration_sec: float,
    ) -> BatchSummary:
        """Generate batch processing summary.

        Args:
        ----
            results: List of all ProductResult objects
            product_ids_count: Number of product IDs attempted
            keywords_count: Number of keywords attempted
            duration_sec: Total execution time

        Returns:
        -------
            BatchSummary with comprehensive statistics

        """
        successful = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        failed_products = [r.product_id for r in results if not r.success]

        # Calculate media statistics
        total_images = 0
        total_videos = 0
        for result in results:
            if result.success and result.data:
                total_images += len(result.data.images or [])
                total_videos += len(result.data.videos or [])

        # Use configured decimal places for rounding
        decimal_places = int(self.log_config["media_stats_decimal_places"])
        duration_places = int(self.log_config["duration_decimal_places"])

        media_stats: dict[str, int | float] = {
            "total_images": total_images,
            "total_videos": total_videos,
            "avg_images_per_product": (
                round(total_images / successful, decimal_places)
                if successful > 0
                else 0
            ),
            "avg_videos_per_product": (
                round(total_videos / successful, decimal_places)
                if successful > 0
                else 0
            ),
        }

        return BatchSummary(
            total_attempted=len(results),
            product_ids_attempted=product_ids_count,
            keywords_attempted=keywords_count,
            successful=successful,
            failed=failed,
            failed_products=failed_products,
            media_stats=media_stats,
            duration_sec=round(duration_sec, duration_places),
        )

    def _log_summary(self, summary: BatchSummary):
        """Log batch summary to console.

        Args:
        ----
            summary: BatchSummary to log

        """
        duration_places = int(self.log_config["duration_decimal_places"])

        self.logger.info("\n" + self.separator)
        self.logger.info("BATCH SCRAPING SUMMARY")
        self.logger.info(self.separator)
        self.logger.info("Total Attempted: %d", summary.total_attempted)
        self.logger.info("  - Product IDs: %d", summary.product_ids_attempted)
        self.logger.info("  - Keywords: %d", summary.keywords_attempted)
        self.logger.info("Successful: %d", summary.successful)
        self.logger.info("Failed: %d", summary.failed)

        if summary.failed_products:
            self.logger.info("Failed Products: %s", ", ".join(summary.failed_products))

        self.logger.info("\nMedia Collection Statistics:")
        for key, value in summary.media_stats.items():
            self.logger.info("  - %s: %s", key, value)

        self.logger.info(
            "\nDuration: %.*f seconds", duration_places, summary.duration_sec
        )
        self.logger.info(self.separator)
