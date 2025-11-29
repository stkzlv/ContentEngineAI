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

    def test_process_product_ids_invalid_asin(
        self, mock_scraper, sample_search_params
    ):
        """Test processing skips invalid ASIN format."""
        config = BatchConfig(
            product_ids=["INVALID", "B0TEST123A"],
            keywords=[],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
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
        )

        mock_scraper.scrape_products_unified.return_value = [sample_product_data]

        controller = BatchController(mock_scraper, config)
        with patch.object(controller.logger, "info") as mock_log:
            controller._process_product_ids()

            # Verify progress logging format
            calls = [str(call) for call in mock_log.call_args_list]
            assert any("[1/2]" in str(call) for call in calls)
            assert any("[2/2]" in str(call) for call in calls)


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
        )

        mock_scraper.scrape_products_unified.return_value = [sample_product_data]

        controller = BatchController(mock_scraper, config)
        results = controller._process_keywords()

        assert len(results) == 2
        assert all(r.success for r in results)
        assert all(r.source == "keyword" for r in results)

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
        assert deduplicated[0].data.asin == "B0TEST123A"
        assert deduplicated[1].data.asin == "B0TEST456X"

    def test_deduplicate_preserves_order(self, mock_scraper, sample_search_params):
        """Test deduplication preserves order of first occurrences."""
        config = BatchConfig(
            product_ids=[],
            keywords=[],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
        )

        results = []
        for i, asin in enumerate(["B0AAAAAAAA", "B0BBBBBBBB", "B0AAAAAAAA", "B0CCCCCCCC", "B0BBBBBBBB"]):
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
        assert deduplicated[0].data.asin == "B0AAAAAAAA"
        assert deduplicated[0].data.title == "Product 0"
        assert deduplicated[1].data.asin == "B0BBBBBBBB"
        assert deduplicated[1].data.title == "Product 1"
        assert deduplicated[2].data.asin == "B0CCCCCCCC"

    def test_deduplicate_handles_no_asin(self, mock_scraper, sample_search_params):
        """Test deduplication handles results without ASIN."""
        config = BatchConfig(
            product_ids=[],
            keywords=[],
            fail_fast=False,
            search_params=sample_search_params,
            max_products=10,
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
        )

        # Both product ID and keyword return same product
        mock_scraper.scrape_products_unified.return_value = [sample_product_data]

        controller = BatchController(mock_scraper, config)
        summary = controller.run_batch()

        # Should deduplicate and only count once
        assert summary.successful == 1
