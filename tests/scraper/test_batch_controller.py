"""Unit tests for BatchController class.

Tests validate batch processing orchestration, deduplication logic,
progress tracking, fail-fast behavior, and summary generation.
All tests use mocked scraper to ensure isolation.
"""

import logging
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from src.scraper.amazon.batch_controller import BatchController
from src.scraper.amazon.models import (
    BatchConfig,
    BatchSummary,
    ProductData,
    ProductResult,
    SearchParameters,
)
from src.scraper.base import Platform

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_scraper():
    """Create mock BotasaurusAmazonScraper instance."""
    scraper = Mock()
    scraper.logger = logging.getLogger("test")
    return scraper


@pytest.fixture
def sample_search_params():
    """Create sample search parameters."""
    return SearchParameters(
        min_price=10.0,
        max_price=100.0,
        min_rating=4.0,
        prime_only=True,
    )


@pytest.fixture
def sample_product_data():
    """Create sample ProductData for testing."""
    return ProductData(
        title="Test Product",
        price="$29.99",
        description="Test description",
        images=["img1.jpg", "img2.jpg"],
        videos=["vid1.mp4"],
        affiliate_link="https://amazon.com/test",
        url="https://amazon.com/test",
        platform=Platform.AMAZON,
        asin="B0TEST1234",  # Valid ASIN: B0 + 8 chars
        keyword="test",
        serp_rating="4.5",
        serp_reviews_count="100",
    )


class TestBatchControllerInit:
    """Test BatchController initialization."""

    def test_initialization(self, mock_scraper, sample_search_params):
        """Test BatchController initializes correctly."""
        config = BatchConfig(
            product_ids=["B0TEST123A"],
            keywords=["test"],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        controller = BatchController(mock_scraper, config)

        assert controller.scraper is mock_scraper
        assert controller.config is config
        assert controller.logger is mock_scraper.logger
        assert controller.results == []
        assert controller.seen_asins == set()


class TestProductIDProcessing:
    """Test product ID list processing."""

    def test_process_product_ids_success(
        self, mock_scraper, sample_search_params, sample_product_data
    ):
        """Test successful processing of product IDs."""
        config = BatchConfig(
            product_ids=["B0TEST123A", "B0TEST456X"],
            keywords=[],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        # Mock scraper to return products
        mock_scraper.scrape_products_unified.side_effect = [
            [sample_product_data],
            [sample_product_data],
        ]

        controller = BatchController(mock_scraper, config)
        results = controller._process_product_ids()

        assert len(results) == 2
        assert all(r.success for r in results)
        assert results[0].source == "product_id"
        assert mock_scraper.scrape_products_unified.call_count == 2

    def test_process_product_ids_invalid_asin(self, mock_scraper, sample_search_params):
        """Test processing skips invalid ASIN format."""
        config = BatchConfig(
            product_ids=["INVALID", "B0TEST123A"],
            keywords=[],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        mock_scraper.scrape_products_unified.return_value = [
            ProductData(
                title="Test",
                price="$10",
                description="Test",
                images=[],
                videos=[],
                affiliate_link="",
                url="",
                platform=Platform.AMAZON,
                asin="B0TEST123A",
                keyword="test",
            )
        ]

        controller = BatchController(mock_scraper, config)
        results = controller._process_product_ids()

        # First result should be invalid, second should succeed
        assert len(results) == 2
        assert not results[0].success
        assert results[0].error == "Invalid ASIN format"
        assert results[1].success

    def test_process_product_ids_no_data_found(
        self, mock_scraper, sample_search_params
    ):
        """Test processing when no data found for product ID."""
        config = BatchConfig(
            product_ids=["B0TEST123A"],
            keywords=[],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        # Mock scraper to return empty list
        mock_scraper.scrape_products_unified.return_value = []

        controller = BatchController(mock_scraper, config)
        results = controller._process_product_ids()

        assert len(results) == 1
        assert not results[0].success
        assert results[0].error == "No data found"

    def test_process_product_ids_fail_fast(self, mock_scraper, sample_search_params):
        """Test fail-fast stops processing on first error."""
        config = BatchConfig(
            product_ids=["B0TEST123A", "B0TEST456X", "B0TEST789Z"],
            keywords=[],
            fail_fast=True,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        # First product succeeds, second fails
        def side_effect(*args, **kwargs):
            if mock_scraper.scrape_products_unified.call_count == 1:
                return [
                    ProductData(
                        title="Test",
                        price="$10",
                        description="Test",
                        images=[],
                        videos=[],
                        affiliate_link="",
                        url="",
                        platform=Platform.AMAZON,
                        asin="B0TEST123A",
                        keyword="test",
                    )
                ]
            else:
                raise Exception("Scraping failed")

        mock_scraper.scrape_products_unified.side_effect = side_effect

        controller = BatchController(mock_scraper, config)
        results = controller._process_product_ids()

        # Should process only 2 products (1 success, 1 failure), then stop
        assert len(results) == 2
        assert results[0].success
        assert not results[1].success
        assert mock_scraper.scrape_products_unified.call_count == 2

    def test_process_product_ids_progress_logging(
        self, mock_scraper, sample_search_params, sample_product_data
    ):
        """Test progress logging format [N/total]."""
        config = BatchConfig(
            product_ids=["B0TEST123A", "B0TEST456X"],
            keywords=[],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        mock_scraper.scrape_products_unified.return_value = [sample_product_data]

        controller = BatchController(mock_scraper, config)
        with patch.object(controller.logger, "info") as mock_log:
            controller._process_product_ids()

            # Verify progress logging uses [N/total] format via lazy formatting
            # Logger receives format string + args, not pre-formatted string
            progress_calls = [
                c
                for c in mock_log.call_args_list
                if c.args and isinstance(c.args[0], str) and "[%d/%d]" in c.args[0]
            ]
            assert len(progress_calls) >= 2
            # Verify correct counter values in args
            counters = [(c.args[1], c.args[2]) for c in progress_calls]
            assert (1, 2) in counters
            assert (2, 2) in counters


class TestKeywordProcessing:
    """Test keyword list processing."""

    def test_process_keywords_success(
        self, mock_scraper, sample_search_params, sample_product_data
    ):
        """Test successful processing of keywords."""
        config = BatchConfig(
            product_ids=[],
            keywords=["keyword1", "keyword2"],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        mock_scraper.scrape_products_unified.return_value = [sample_product_data]

        controller = BatchController(mock_scraper, config)
        results = controller._process_keywords()

        assert len(results) == 2
        assert all(r.success for r in results)
        assert all(r.source == "keyword" for r in results)

    def test_process_keywords_sets_product_pillar(
        self, mock_scraper, sample_search_params, sample_product_data
    ):
        """Keyword-sourced products carry the pillar from the config map."""
        config = BatchConfig(
            product_ids=[],
            keywords=["smart plug"],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
            keyword_pillar_map={"smart plug": "value"},
        )

        mock_scraper.scrape_products_unified.return_value = [sample_product_data]

        controller = BatchController(mock_scraper, config)
        results = controller._process_keywords()

        assert len(results) == 1
        assert results[0].data is not None
        assert results[0].data.pillar == "value"

    def test_process_keywords_no_pillar_when_unmapped(
        self, mock_scraper, sample_search_params, sample_product_data
    ):
        """Keywords not in the pillar map leave product.pillar as None."""
        config = BatchConfig(
            product_ids=[],
            keywords=["unknown keyword"],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        mock_scraper.scrape_products_unified.return_value = [sample_product_data]

        controller = BatchController(mock_scraper, config)
        results = controller._process_keywords()

        assert len(results) == 1
        assert results[0].data is not None
        assert results[0].data.pillar is None

    def test_process_keywords_multiple_products(
        self, mock_scraper, sample_search_params, sample_product_data
    ):
        """Test processing keywords that return multiple products."""
        config = BatchConfig(
            product_ids=[],
            keywords=["keyword1"],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        # Return 3 products for single keyword
        product1 = sample_product_data
        product2 = ProductData(
            title="Product 2",
            price="$20",
            description="Test",
            images=[],
            videos=[],
            affiliate_link="",
            url="",
            platform=Platform.AMAZON,
            asin="B0TEST456X",
            keyword="test",
        )
        product3 = ProductData(
            title="Product 3",
            price="$30",
            description="Test",
            images=[],
            videos=[],
            affiliate_link="",
            url="",
            platform=Platform.AMAZON,
            asin="B0TEST789Z",
            keyword="test",
        )

        mock_scraper.scrape_products_unified.return_value = [
            product1,
            product2,
            product3,
        ]

        controller = BatchController(mock_scraper, config)
        results = controller._process_keywords()

        assert len(results) == 3
        assert all(r.success for r in results)

    def test_process_keywords_max_products_limit(
        self, mock_scraper, sample_search_params, sample_product_data
    ):
        """Test processing stops when max_products reached."""
        config = BatchConfig(
            product_ids=[],
            keywords=["keyword1", "keyword2"],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=2,  # Limit to 2 products
            products_per_keyword=5,
        )

        # Each keyword returns 2 products
        mock_scraper.scrape_products_unified.return_value = [
            sample_product_data,
            sample_product_data,
        ]

        controller = BatchController(mock_scraper, config)
        results = controller._process_keywords()

        # Should stop after first keyword (2 products)
        assert len(results) == 2
        assert mock_scraper.scrape_products_unified.call_count == 1

    def test_process_keywords_no_products_found(
        self, mock_scraper, sample_search_params
    ):
        """Test processing when keyword returns no products."""
        config = BatchConfig(
            product_ids=[],
            keywords=["keyword1"],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        mock_scraper.scrape_products_unified.return_value = []

        controller = BatchController(mock_scraper, config)
        results = controller._process_keywords()

        assert len(results) == 0

    def test_process_keywords_fail_fast(self, mock_scraper, sample_search_params):
        """Test fail-fast stops keyword processing on error."""
        config = BatchConfig(
            product_ids=[],
            keywords=["keyword1", "keyword2", "keyword3"],
            fail_fast=True,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        # First keyword succeeds, second fails
        def side_effect(*args, **kwargs):
            if mock_scraper.scrape_products_unified.call_count == 1:
                return [
                    ProductData(
                        title="Test",
                        price="$10",
                        description="Test",
                        images=[],
                        videos=[],
                        affiliate_link="",
                        url="",
                        platform=Platform.AMAZON,
                        asin="B0TEST123A",
                        keyword="test",
                    )
                ]
            else:
                raise Exception("Keyword search failed")

        mock_scraper.scrape_products_unified.side_effect = side_effect

        controller = BatchController(mock_scraper, config)
        results = controller._process_keywords()

        # Should process only 1 keyword successfully, then stop on error
        assert len(results) == 1
        assert results[0].success
        assert mock_scraper.scrape_products_unified.call_count == 2


class TestDeduplication:
    """Test product deduplication logic."""

    def test_deduplicate_by_asin(self, mock_scraper, sample_search_params):
        """Test deduplication removes products with duplicate ASINs."""
        config = BatchConfig(
            product_ids=[],
            keywords=[],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        # Create results with duplicate ASINs
        product1 = ProductData(
            title="Product 1",
            price="$10",
            description="Test",
            images=[],
            videos=[],
            affiliate_link="",
            url="",
            platform=Platform.AMAZON,
            asin="B0TEST123A",
            keyword="test",
        )
        product2 = ProductData(
            title="Product 2",
            price="$20",
            description="Test",
            images=[],
            videos=[],
            affiliate_link="",
            url="",
            platform=Platform.AMAZON,
            asin="B0TEST123A",  # Duplicate ASIN
            keyword="test",
        )
        product3 = ProductData(
            title="Product 3",
            price="$30",
            description="Test",
            images=[],
            videos=[],
            affiliate_link="",
            url="",
            platform=Platform.AMAZON,
            asin="B0TEST456X",  # Different ASIN
            keyword="test",
        )

        results = [
            ProductResult(
                product_id="B0TEST123A",
                success=True,
                data=product1,
                error=None,
                source="product_id",
            ),
            ProductResult(
                product_id="B0TEST123A",
                success=True,
                data=product2,
                error=None,
                source="keyword",
            ),
            ProductResult(
                product_id="B0TEST456X",
                success=True,
                data=product3,
                error=None,
                source="keyword",
            ),
        ]

        controller = BatchController(mock_scraper, config)
        deduplicated = controller._deduplicate_products(results)

        # Should keep first occurrence and remove duplicate
        assert len(deduplicated) == 2
        assert deduplicated[0].data is not None
        assert deduplicated[0].data.asin == "B0TEST123A"
        assert deduplicated[1].data is not None
        assert deduplicated[1].data.asin == "B0TEST456X"

    def test_deduplicate_preserves_order(self, mock_scraper, sample_search_params):
        """Test deduplication preserves order of first occurrences."""
        config = BatchConfig(
            product_ids=[],
            keywords=[],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        results = []
        for i, asin in enumerate(
            ["B0AAAAAAAA", "B0BBBBBBBB", "B0AAAAAAAA", "B0CCCCCCCC", "B0BBBBBBBB"]
        ):
            product = ProductData(
                title=f"Product {i}",
                price="$10",
                description="Test",
                images=[],
                videos=[],
                affiliate_link="",
                url="",
                platform=Platform.AMAZON,
                asin=asin,
                keyword="test",
            )
            results.append(
                ProductResult(
                    product_id=asin,
                    success=True,
                    data=product,
                    error=None,
                    source="product_id",
                )
            )

        controller = BatchController(mock_scraper, config)
        deduplicated = controller._deduplicate_products(results)

        # Should keep first occurrences in order: B0AAAAAAAA, B0BBBBBBBB, B0CCCCCCCC
        assert len(deduplicated) == 3
        assert deduplicated[0].data is not None
        assert deduplicated[0].data.asin == "B0AAAAAAAA"
        assert deduplicated[0].data.title == "Product 0"
        assert deduplicated[1].data is not None
        assert deduplicated[1].data.asin == "B0BBBBBBBB"
        assert deduplicated[1].data.title == "Product 1"
        assert deduplicated[2].data is not None
        assert deduplicated[2].data.asin == "B0CCCCCCCC"

    def test_deduplicate_handles_no_asin(self, mock_scraper, sample_search_params):
        """Test deduplication handles results without ASIN."""
        config = BatchConfig(
            product_ids=[],
            keywords=[],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        # Create result without ASIN
        product = ProductData(
            title="No ASIN Product",
            price="$10",
            description="Test",
            images=[],
            videos=[],
            affiliate_link="",
            url="",
            platform=Platform.AMAZON,
            asin=None,
            keyword="test",
        )

        results = [
            ProductResult(
                product_id="unknown",
                success=True,
                data=product,
                error=None,
                source="keyword",
            )
        ]

        controller = BatchController(mock_scraper, config)
        deduplicated = controller._deduplicate_products(results)

        # Should keep result even without ASIN
        assert len(deduplicated) == 1


class TestSummaryGeneration:
    """Test batch summary generation."""

    def test_generate_summary_statistics(self, mock_scraper, sample_search_params):
        """Test summary generates correct statistics."""
        config = BatchConfig(
            product_ids=["B0TEST123A"],
            keywords=["keyword1"],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        # Create successful and failed results
        success_product = ProductData(
            title="Success",
            price="$10",
            description="Test",
            images=["img1.jpg", "img2.jpg", "img3.jpg"],
            videos=["vid1.mp4"],
            affiliate_link="",
            url="",
            platform=Platform.AMAZON,
            asin="B0TEST123A",
            keyword="test",
        )

        results = [
            ProductResult(
                product_id="B0TEST123A",
                success=True,
                data=success_product,
                error=None,
                source="product_id",
            ),
            ProductResult(
                product_id="B0TEST456X",
                success=False,
                data=None,
                error="No data found",
                source="keyword",
            ),
        ]

        controller = BatchController(mock_scraper, config)
        summary = controller._generate_summary(results, 1, 1, 10.5)

        assert summary.total_attempted == 2
        assert summary.product_ids_attempted == 1
        assert summary.keywords_attempted == 1
        assert summary.successful == 1
        assert summary.failed == 1
        assert summary.failed_products == ["B0TEST456X"]
        assert summary.duration_sec == 10.5

    def test_generate_summary_media_statistics(
        self, mock_scraper, sample_search_params
    ):
        """Test summary generates correct media statistics."""
        config = BatchConfig(
            product_ids=[],
            keywords=[],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        # Create products with different media counts
        product1 = ProductData(
            title="Product 1",
            price="$10",
            description="Test",
            images=["img1.jpg", "img2.jpg"],
            videos=["vid1.mp4"],
            affiliate_link="",
            url="",
            platform=Platform.AMAZON,
            asin="B0TEST123A",
            keyword="test",
        )
        product2 = ProductData(
            title="Product 2",
            price="$20",
            description="Test",
            images=["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"],
            videos=[],
            affiliate_link="",
            url="",
            platform=Platform.AMAZON,
            asin="B0TEST456X",
            keyword="test",
        )

        results = [
            ProductResult(
                product_id="B0TEST123A",
                success=True,
                data=product1,
                error=None,
                source="product_id",
            ),
            ProductResult(
                product_id="B0TEST456X",
                success=True,
                data=product2,
                error=None,
                source="product_id",
            ),
        ]

        controller = BatchController(mock_scraper, config)
        summary = controller._generate_summary(results, 2, 0, 5.0)

        assert summary.media_stats["total_images"] == 6
        assert summary.media_stats["total_videos"] == 1
        assert summary.media_stats["avg_images_per_product"] == 3.0
        assert summary.media_stats["avg_videos_per_product"] == 0.5

    def test_generate_summary_no_products(self, mock_scraper, sample_search_params):
        """Test summary with no products."""
        config = BatchConfig(
            product_ids=[],
            keywords=[],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        controller = BatchController(mock_scraper, config)
        summary = controller._generate_summary([], 0, 0, 1.0)

        assert summary.total_attempted == 0
        assert summary.successful == 0
        assert summary.failed == 0
        assert summary.media_stats["avg_images_per_product"] == 0
        assert summary.media_stats["avg_videos_per_product"] == 0


class TestRunBatch:
    """Test complete batch execution."""

    def test_run_batch_complete_workflow(
        self, mock_scraper, sample_search_params, sample_product_data
    ):
        """Test complete batch workflow execution."""
        config = BatchConfig(
            product_ids=["B0TEST123A"],
            keywords=["keyword1"],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        mock_scraper.scrape_products_unified.return_value = [sample_product_data]

        controller = BatchController(mock_scraper, config)
        summary = controller.run_batch()

        # Verify summary
        assert isinstance(summary, BatchSummary)
        assert summary.total_attempted >= 1
        assert summary.product_ids_attempted == 1
        assert summary.keywords_attempted == 1

    def test_run_batch_mixed_input(
        self, mock_scraper, sample_search_params, sample_product_data
    ):
        """Test batch with both product IDs and keywords."""
        config = BatchConfig(
            product_ids=["B0TEST123A", "B0TEST456X"],
            keywords=["keyword1"],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        mock_scraper.scrape_products_unified.return_value = [sample_product_data]

        controller = BatchController(mock_scraper, config)
        summary = controller.run_batch()

        assert summary.product_ids_attempted == 2
        assert summary.keywords_attempted == 1

    def test_run_batch_deduplication(
        self, mock_scraper, sample_search_params, sample_product_data
    ):
        """Test batch deduplicates across product IDs and keywords."""
        config = BatchConfig(
            product_ids=["B0TEST123AC"],
            keywords=["keyword1"],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        # Both product ID and keyword return same product
        mock_scraper.scrape_products_unified.return_value = [sample_product_data]

        controller = BatchController(mock_scraper, config)
        summary = controller.run_batch()

        # Should deduplicate and only count once
        assert summary.successful == 1


class TestGracefulDegradation:
    """Test graceful degradation (continue on failure without fail-fast)."""

    def test_continues_after_product_id_failure(
        self, mock_scraper, sample_search_params, sample_product_data
    ):
        """Test processing continues after individual product ID failures."""
        config = BatchConfig(
            product_ids=["B0SUCCES01", "B0FAILPRD1", "B0SUCCES02"],
            keywords=[],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        def side_effect(*args, **kwargs):
            keyword = kwargs.get("keyword", "")
            if "FAIL" in keyword:
                raise Exception("Individual failure")
            # Return product with unique ASIN to avoid deduplication
            return [
                ProductData(
                    title=f"Product {keyword}",
                    price="$10",
                    description="Test",
                    images=[],
                    videos=[],
                    affiliate_link="",
                    url="",
                    platform=Platform.AMAZON,
                    asin=keyword,  # Unique ASIN per product
                    keyword="test",
                )
            ]

        mock_scraper.scrape_products_unified.side_effect = side_effect

        controller = BatchController(mock_scraper, config)
        summary = controller.run_batch()

        # All items should be attempted
        assert summary.product_ids_attempted == 3
        assert summary.successful == 2
        assert summary.failed == 1
        assert mock_scraper.scrape_products_unified.call_count == 3

    def test_continues_after_keyword_failure(
        self, mock_scraper, sample_search_params, sample_product_data
    ):
        """Test processing continues after keyword search failures."""
        config = BatchConfig(
            product_ids=[],
            keywords=["success1", "fail_keyword", "success2"],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        call_index = [0]  # Use list for closure

        def side_effect(*args, **kwargs):
            keyword = kwargs.get("keyword", "")
            if "fail" in keyword:
                raise Exception("Keyword search failed")
            # Return product with unique ASIN per call
            call_index[0] += 1
            return [
                ProductData(
                    title=f"Product {call_index[0]}",
                    price="$10",
                    description="Test",
                    images=[],
                    videos=[],
                    affiliate_link="",
                    url="",
                    platform=Platform.AMAZON,
                    asin=f"B0KEYWRD{call_index[0]:02d}",  # Unique ASIN
                    keyword=keyword,
                )
            ]

        mock_scraper.scrape_products_unified.side_effect = side_effect

        controller = BatchController(mock_scraper, config)
        summary = controller.run_batch()

        # All keywords should be attempted
        assert summary.keywords_attempted == 3
        assert mock_scraper.scrape_products_unified.call_count == 3

    def test_multiple_failures_tracked_in_summary(
        self, mock_scraper, sample_search_params, sample_product_data
    ):
        """Test all failures are tracked in summary."""
        config = BatchConfig(
            product_ids=["B0FAIL0001", "B0SUCCES01", "B0FAIL0002", "B0FAIL0003"],
            keywords=[],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        def side_effect(*args, **kwargs):
            keyword = kwargs.get("keyword", "")
            if "FAIL" in keyword:
                raise Exception(f"Failed: {keyword}")
            # Return product with unique ASIN
            return [
                ProductData(
                    title=f"Product {keyword}",
                    price="$10",
                    description="Test",
                    images=[],
                    videos=[],
                    affiliate_link="",
                    url="",
                    platform=Platform.AMAZON,
                    asin=keyword,  # Unique ASIN
                    keyword="test",
                )
            ]

        mock_scraper.scrape_products_unified.side_effect = side_effect

        controller = BatchController(mock_scraper, config)
        summary = controller.run_batch()

        assert summary.successful == 1
        assert summary.failed == 3
        assert len(summary.failed_products) == 3

    def test_successful_results_not_corrupted_by_failures(
        self, mock_scraper, sample_search_params
    ):
        """Test that failures don't affect already successful results."""
        config = BatchConfig(
            product_ids=["B0FIRST001", "B0FAILMID1", "B0LAST0001"],
            keywords=[],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        collected_asins = []

        def side_effect(*args, **kwargs):
            keyword = kwargs.get("keyword", "")
            if "FAIL" in keyword:
                raise Exception("Mid-batch failure")
            product = ProductData(
                title=f"Product {keyword}",
                price="$10",
                description="Test",
                images=[],
                videos=[],
                affiliate_link="",
                url="",
                platform=Platform.AMAZON,
                asin=keyword,
                keyword="test",
            )
            collected_asins.append(keyword)
            return [product]

        mock_scraper.scrape_products_unified.side_effect = side_effect

        controller = BatchController(mock_scraper, config)
        summary = controller.run_batch()

        assert summary.successful == 2
        assert len(collected_asins) == 2
        assert "B0FIRST001" in collected_asins
        assert "B0LAST0001" in collected_asins


class TestParametrizedEdgeCases:
    """Parametrized tests for edge cases and ASIN validation."""

    @pytest.mark.parametrize(
        "invalid_asin,expected_error",
        [
            ("TOOLONG12345", "Invalid ASIN format"),  # Too long (12 chars)
            ("SHORT", "Invalid ASIN format"),  # Too short (5 chars)
            ("12345-abcd", "Invalid ASIN format"),  # Has hyphen and lowercase
            ("", "Invalid ASIN format"),  # Empty
            ("   ", "Invalid ASIN format"),  # Whitespace only
            ("B0invalid!", "Invalid ASIN format"),  # Has lowercase and special char
        ],
    )
    def test_invalid_asin_formats(
        self,
        mock_scraper,
        sample_search_params,
        invalid_asin: str,
        expected_error: str,
    ):
        """Test various invalid ASIN formats are rejected."""
        config = BatchConfig(
            product_ids=[invalid_asin],
            keywords=[],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        controller = BatchController(mock_scraper, config)
        results = controller._process_product_ids()

        assert len(results) == 1
        assert not results[0].success
        assert results[0].error is not None
        assert expected_error in results[0].error

    @pytest.mark.parametrize(
        "valid_asin",
        [
            "B0VALIDASN",  # Standard format
            "B012345678",  # Numeric suffix
            "B0ABCDEFGH",  # Mixed alphanumeric
            "B0XXXXXXXX",  # All same letter
        ],
    )
    def test_valid_asin_formats(
        self,
        mock_scraper,
        sample_search_params,
        sample_product_data,
        valid_asin: str,
    ):
        """Test valid ASIN formats are accepted."""
        config = BatchConfig(
            product_ids=[valid_asin],
            keywords=[],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        mock_scraper.scrape_products_unified.return_value = [sample_product_data]

        controller = BatchController(mock_scraper, config)
        controller._process_product_ids()

        # Should attempt to scrape (validation passed)
        assert mock_scraper.scrape_products_unified.called

    @pytest.mark.parametrize(
        "products_per_keyword,available_products,expected_count",
        [
            # Per-keyword limit: scraper limited to 3, only get 3
            (3, 10, 3),
            # Available less than limit: get what's available
            (10, 5, 5),
            # Limit matches available: get all
            (5, 5, 5),
        ],
    )
    def test_products_per_keyword_limit(
        self,
        mock_scraper,
        sample_search_params,
        products_per_keyword: int,
        available_products: int,
        expected_count: int,
    ):
        """Test products_per_keyword limits what scraper returns."""
        config = BatchConfig(
            product_ids=[],
            keywords=["test"],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=100,  # High global limit, not the limiting factor
            products_per_keyword=products_per_keyword,
        )

        def mock_scrape(*args, **kwargs):
            # Scraper respects max_products kwarg like real implementation
            scraper_limit = kwargs.get("max_products", available_products)
            count = min(scraper_limit, available_products)
            return [
                ProductData(
                    title=f"Product {i}",
                    price="$10",
                    description="Test",
                    images=[],
                    videos=[],
                    affiliate_link="",
                    url="",
                    platform=Platform.AMAZON,
                    asin=f"B0TEST{i:05d}",
                    keyword="test",
                )
                for i in range(count)
            ]

        mock_scraper.scrape_products_unified.side_effect = mock_scrape

        controller = BatchController(mock_scraper, config)
        results = controller._process_keywords()

        assert len(results) == expected_count

    def test_max_products_stops_keyword_processing(
        self,
        mock_scraper,
        sample_search_params,
    ):
        """Test max_products global limit stops processing more keywords."""
        config = BatchConfig(
            product_ids=[],
            keywords=["keyword1", "keyword2", "keyword3"],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=5,  # Should stop after first keyword returns >= 5
            products_per_keyword=10,
        )

        call_count = [0]

        def mock_scrape(*args, **kwargs):
            call_count[0] += 1
            keyword = kwargs.get("keyword", "")
            # Each keyword returns 6 products
            return [
                ProductData(
                    title=f"Product {i} from {keyword}",
                    price="$10",
                    description="Test",
                    images=[],
                    videos=[],
                    affiliate_link="",
                    url="",
                    platform=Platform.AMAZON,
                    asin=f"B0{keyword[:4].upper()}{i:04d}",
                    keyword=keyword,
                )
                for i in range(6)
            ]

        mock_scraper.scrape_products_unified.side_effect = mock_scrape

        controller = BatchController(mock_scraper, config)
        results = controller._process_keywords()

        # First keyword returns 6 products, exceeds max_products=5
        # Implementation adds all 6, then stops processing more keywords
        assert len(results) == 6
        # Only 1 keyword processed (stopped after hitting limit)
        assert call_count[0] == 1


class TestMixedInputProcessing:
    """Tests for combined product IDs and keywords processing."""

    def test_product_ids_processed_before_keywords(
        self, mock_scraper, sample_search_params, sample_product_data
    ):
        """Test that product IDs are processed before keywords."""
        config = BatchConfig(
            product_ids=["B0PRODUCT1"],
            keywords=["search term"],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        call_order = []
        call_index = [0]

        def track_calls(*args, **kwargs):
            keyword = kwargs.get("keyword", "")
            # Product IDs start with B0, keywords don't
            if keyword.startswith("B0"):
                call_order.append(("product_id", keyword))
            else:
                call_order.append(("keyword", keyword))
            # Return unique products to avoid deduplication
            call_index[0] += 1
            return [
                ProductData(
                    title=f"Product {call_index[0]}",
                    price="$10",
                    description="Test",
                    images=[],
                    videos=[],
                    affiliate_link="",
                    url="",
                    platform=Platform.AMAZON,
                    asin=f"B0UNIQUE{call_index[0]:02d}",
                    keyword="test",
                )
            ]

        mock_scraper.scrape_products_unified.side_effect = track_calls

        controller = BatchController(mock_scraper, config)
        controller.run_batch()

        # Product IDs should be called first
        assert len(call_order) >= 1
        assert call_order[0][0] == "product_id"

    def test_keyword_duplicates_product_id_deduplicated(
        self, mock_scraper, sample_search_params
    ):
        """Test that keyword results don't duplicate already-processed product IDs."""
        config = BatchConfig(
            product_ids=["B0ALREADY1"],
            keywords=["find same"],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        # Both return same ASIN
        product = ProductData(
            title="Same Product",
            price="$10",
            description="Test",
            images=[],
            videos=[],
            affiliate_link="",
            url="",
            platform=Platform.AMAZON,
            asin="B0ALREADY1",
            keyword="test",
        )
        mock_scraper.scrape_products_unified.return_value = [product]

        controller = BatchController(mock_scraper, config)
        summary = controller.run_batch()

        # Should deduplicate - only 1 successful
        assert summary.successful == 1


class TestEmptyInputHandling:
    """Tests for empty input scenarios."""

    def test_empty_product_ids_and_keywords(self, mock_scraper, sample_search_params):
        """Test handling of completely empty config."""
        config = BatchConfig(
            product_ids=[],
            keywords=[],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        controller = BatchController(mock_scraper, config)
        summary = controller.run_batch()

        assert summary.total_attempted == 0
        assert summary.successful == 0
        assert summary.failed == 0
        assert mock_scraper.scrape_products_unified.call_count == 0

    def test_only_product_ids_no_keywords(
        self, mock_scraper, sample_search_params, sample_product_data
    ):
        """Test batch with only product IDs."""
        config = BatchConfig(
            product_ids=["B0ONLYPRD1", "B0ONLYPRD2"],
            keywords=[],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        mock_scraper.scrape_products_unified.return_value = [sample_product_data]

        controller = BatchController(mock_scraper, config)
        summary = controller.run_batch()

        assert summary.product_ids_attempted == 2
        assert summary.keywords_attempted == 0

    def test_only_keywords_no_product_ids(
        self, mock_scraper, sample_search_params, sample_product_data
    ):
        """Test batch with only keywords."""
        config = BatchConfig(
            product_ids=[],
            keywords=["keyword1", "keyword2"],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        mock_scraper.scrape_products_unified.return_value = [sample_product_data]

        controller = BatchController(mock_scraper, config)
        summary = controller.run_batch()

        assert summary.product_ids_attempted == 0
        assert summary.keywords_attempted == 2


class TestSummaryLogging:
    """Tests for summary logging output."""

    def test_summary_logged(
        self, mock_scraper, sample_search_params, sample_product_data
    ):
        """Test that summary is logged at end of batch."""
        config = BatchConfig(
            product_ids=["B0LOGSUMRY"],
            keywords=[],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        mock_scraper.scrape_products_unified.return_value = [sample_product_data]

        controller = BatchController(mock_scraper, config)
        with patch.object(controller.logger, "info") as mock_log:
            controller.run_batch()

            # Summary should be logged (case-insensitive check)
            log_calls = [str(call).lower() for call in mock_log.call_args_list]
            assert any("summary" in call or "completed" in call for call in log_calls)

    def test_duration_tracked(
        self, mock_scraper, sample_search_params, sample_product_data
    ):
        """Test that duration is tracked in summary."""
        config = BatchConfig(
            product_ids=["B0DURATN01"],
            keywords=[],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
            products_per_keyword=5,
        )

        mock_scraper.scrape_products_unified.return_value = [sample_product_data]

        controller = BatchController(mock_scraper, config)
        summary = controller.run_batch()

        assert summary.duration_sec >= 0


@pytest.mark.unit
class TestLostKeywordsAreRecorded:
    """The keyword arm records no per-product failure, so a lost keyword
    would otherwise leave no trace in the summary at all.
    """

    @staticmethod
    def _controller(keywords):
        from src.scraper.amazon.batch_controller import BatchController
        from src.scraper.amazon.models import BatchConfig, SearchParameters

        scraper = MagicMock()
        config = BatchConfig(
            product_ids=[],
            keywords=keywords,
            fail_fast=False,
            search_params=SearchParameters(),
            max_products=10,
            products_per_keyword=1,
        )
        return BatchController(scraper, config), scraper

    def test_a_keyword_that_returns_nothing_is_recorded(self):
        controller, scraper = self._controller(["a keyword"])
        scraper.scrape_products_unified.return_value = []

        controller._process_keywords()

        assert controller._failed_keywords == ["a keyword"]

    def test_a_keyword_whose_search_raises_is_recorded(self):
        controller, scraper = self._controller(["a keyword"])
        scraper.scrape_products_unified.side_effect = RuntimeError("blocked")

        controller._process_keywords()

        assert controller._failed_keywords == ["a keyword"]

    def test_the_summary_carries_them(self):
        controller, scraper = self._controller(["good", "bad"])
        controller._failed_keywords = ["bad"]

        summary = controller._generate_summary([], 0, 2, 1.0)

        assert summary.failed_keywords == ["bad"]
