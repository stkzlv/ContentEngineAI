"""Integration tests for end-to-end batch scraping.

These tests validate the complete batch scraping workflow including:
- Product ID list processing
- Keyword list processing with filters
- Mixed input (product IDs + keywords)
- CLI vs YAML configuration precedence

Tests use mocked scraper responses to ensure repeatability and avoid
dependency on external Amazon availability.

Run with: pytest tests/scraper/test_batch_integration.py -v
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from src.scraper.amazon.batch_controller import BatchController
from src.scraper.amazon.config import load_batch_config
from src.scraper.amazon.models import BatchConfig, ProductData, SearchParameters
from src.scraper.base import Platform

# Test markers
pytestmark = pytest.mark.integration


@pytest.fixture
def temp_config_dir():
    """Create temporary directory for test configuration files."""
    with tempfile.TemporaryDirectory(prefix="test_batch_config_") as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_yaml_config(temp_config_dir):
    """Create sample YAML configuration file for testing."""
    config_file = temp_config_dir / "scraper.yaml"
    config_data = {
        "batch": {
            "product_ids": ["B0TESTAA11", "B0TESTBB22"],
            "keywords": ["test keyword 1", "test keyword 2"],
        }
    }
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
    return config_file


@pytest.fixture
def mock_scraper_with_products():
    """Create mock scraper that returns realistic product data."""
    scraper = Mock()
    scraper.logger = Mock()

    def create_product(asin, title, keyword="test"):
        return ProductData(
            title=title,
            price="$29.99",
            description=f"Test product description for {asin}",
            images=["img1.jpg", "img2.jpg"],
            videos=["video1.mp4"],
            affiliate_link=f"https://amazon.com/dp/{asin}",
            url=f"https://amazon.com/dp/{asin}",
            platform=Platform.AMAZON,
            asin=asin,
            keyword=keyword,
            serp_rating="4.5",
            serp_reviews_count="100",
        )

    # Configure mock to return different products based on input
    def scrape_side_effect(
        keyword=None, search_params=None, max_products=None, **kwargs
    ):
        if keyword == "B0TESTAA11":
            return [create_product("B0TESTAA11", "Test Product AA", keyword)]
        elif keyword == "B0TESTBB22":
            return [create_product("B0TESTBB22", "Test Product BB", keyword)]
        elif keyword == "test keyword 1":
            return [create_product("B0TESTCC33", "Keyword Product 1", keyword)]
        elif keyword == "test keyword 2":
            return [create_product("B0TESTDD44", "Keyword Product 2", keyword)]
        else:
            return []

    scraper.scrape_products_unified.side_effect = scrape_side_effect
    return scraper


class TestProductIDListBatch:
    """Test batch processing with product ID lists."""

    def test_product_id_list_complete_flow(self, mock_scraper_with_products):
        """Test end-to-end batch flow with product ID list."""
        config = BatchConfig(
            product_ids=["B0TESTAA11", "B0TESTBB22"],
            keywords=[],
            fail_fast=False,
            search_params=SearchParameters(),
            max_products=10,
            products_per_keyword=5,
        )

        controller = BatchController(mock_scraper_with_products, config)
        summary = controller.run_batch()

        # Verify summary statistics
        assert summary.total_attempted == 2
        assert summary.product_ids_attempted == 2
        assert summary.keywords_attempted == 0
        assert summary.successful == 2
        assert summary.failed == 0
        assert len(summary.failed_products) == 0

        # Verify scraper was called correctly
        assert mock_scraper_with_products.scrape_products_unified.call_count == 2

    def test_product_id_list_with_failures(self, mock_scraper_with_products):
        """Test product ID list with some failures."""

        # Configure mock to fail for second product
        def failing_side_effect(
            keyword=None, search_params=None, max_products=None, **kwargs
        ):
            if keyword == "B0TESTAA11":
                return [
                    ProductData(
                        title="Test Product AA",
                        price="$29.99",
                        description="Test",
                        images=[],
                        videos=[],
                        affiliate_link="",
                        url="",
                        platform=Platform.AMAZON,
                        asin="B0TESTAA11",
                        keyword=keyword,
                    )
                ]
            else:
                raise Exception("Scraping failed")

        mock_scraper_with_products.scrape_products_unified.side_effect = (
            failing_side_effect
        )

        config = BatchConfig(
            product_ids=["B0TESTAA11", "B0TESTBB22"],
            keywords=[],
            fail_fast=False,
            search_params=SearchParameters(),
            max_products=10,
            products_per_keyword=5,
        )

        controller = BatchController(mock_scraper_with_products, config)
        summary = controller.run_batch()

        # Verify mixed success/failure
        assert summary.total_attempted == 2
        assert summary.successful == 1
        assert summary.failed == 1
        assert "B0TESTBB22" in summary.failed_products


class TestKeywordListBatch:
    """Test batch processing with keyword lists."""

    def test_keyword_list_complete_flow(self, mock_scraper_with_products):
        """Test end-to-end batch flow with keyword list."""
        config = BatchConfig(
            product_ids=[],
            keywords=["test keyword 1", "test keyword 2"],
            fail_fast=False,
            search_params=SearchParameters(),
            max_products=10,
            products_per_keyword=5,
        )

        controller = BatchController(mock_scraper_with_products, config)
        summary = controller.run_batch()

        # Verify summary statistics
        assert summary.total_attempted == 2
        assert summary.product_ids_attempted == 0
        assert summary.keywords_attempted == 2
        assert summary.successful == 2
        assert summary.failed == 0

    def test_keyword_list_with_search_filters(self, mock_scraper_with_products):
        """Test keyword list with search parameter filters."""
        search_params = SearchParameters(
            min_price=10.0,
            max_price=100.0,
            min_rating=4.0,
            prime_only=True,
        )

        config = BatchConfig(
            product_ids=[],
            keywords=["test keyword 1"],
            fail_fast=False,
            search_params=search_params,
            max_products=10,
            products_per_keyword=5,
        )

        controller = BatchController(mock_scraper_with_products, config)
        summary = controller.run_batch()

        # Verify filters were passed to scraper
        call_args = mock_scraper_with_products.scrape_products_unified.call_args
        assert call_args[1]["search_params"] == search_params
        assert summary.successful == 1


class TestMixedInputBatch:
    """Test batch processing with both product IDs and keywords."""

    def test_mixed_input_complete_flow(self, mock_scraper_with_products):
        """Test end-to-end batch flow with mixed product IDs and keywords."""
        config = BatchConfig(
            product_ids=["B0TESTAA11"],
            keywords=["test keyword 1"],
            fail_fast=False,
            search_params=SearchParameters(),
            max_products=10,
            products_per_keyword=5,
        )

        controller = BatchController(mock_scraper_with_products, config)
        summary = controller.run_batch()

        # Verify summary includes both sources
        assert summary.total_attempted == 2
        assert summary.product_ids_attempted == 1
        assert summary.keywords_attempted == 1
        assert summary.successful == 2

    def test_mixed_input_deduplication(self, mock_scraper_with_products):
        """Test that mixed input correctly deduplicates across sources."""

        # Configure mock to return same ASIN from both sources
        def duplicate_side_effect(
            keyword=None, search_params=None, max_products=None, **kwargs
        ):
            # Same product returned for both product ID and keyword
            return [
                ProductData(
                    title="Duplicate Product",
                    price="$29.99",
                    description="Test",
                    images=["img1.jpg"],
                    videos=[],
                    affiliate_link="",
                    url="",
                    platform=Platform.AMAZON,
                    asin="B0TESTAA11",  # Same ASIN (deduplicate test)
                    keyword=keyword,
                )
            ]

        mock_scraper_with_products.scrape_products_unified.side_effect = (
            duplicate_side_effect
        )

        config = BatchConfig(
            product_ids=["B0TESTAA11"],
            keywords=["test keyword 1"],
            fail_fast=False,
            search_params=SearchParameters(),
            max_products=10,
            products_per_keyword=5,
        )

        controller = BatchController(mock_scraper_with_products, config)
        summary = controller.run_batch()

        # Should deduplicate to single product
        assert summary.total_attempted == 1  # After deduplication
        assert summary.successful == 1


class TestConfigurationPrecedence:
    """Test CLI vs YAML configuration precedence."""

    def test_cli_overrides_yaml_product_ids(self):
        """Test that CLI product IDs override YAML configuration."""
        # Mock CONFIG dict with batch section
        mock_config = {
            "batch": {
                "product_ids": ["B0TESTAA11", "B0TESTBB22"],
                "keywords": ["test keyword 1", "test keyword 2"],
            }
        }
        with patch("src.scraper.amazon.config.CONFIG", mock_config):
            # CLI product_ids should override YAML
            batch_config = load_batch_config(
                cli_product_ids=["B0CLI00001"],
                cli_keywords=None,
                cli_fail_fast=False,
            )

            assert batch_config.product_ids == ["B0CLI00001"]
            # Keywords from YAML should still be used
            assert "test keyword 1" in batch_config.keywords

    def test_cli_overrides_yaml_keywords(self):
        """Test that CLI keywords override YAML configuration."""
        # Mock CONFIG dict with batch section
        mock_config = {
            "batch": {
                "product_ids": ["B0TESTAA11", "B0TESTBB22"],
                "keywords": ["test keyword 1", "test keyword 2"],
            }
        }
        with patch("src.scraper.amazon.config.CONFIG", mock_config):
            # CLI keywords should override YAML
            batch_config = load_batch_config(
                cli_product_ids=None,
                cli_keywords=["cli keyword"],
                cli_fail_fast=False,
            )

            assert batch_config.keywords == ["cli keyword"]
            # Product IDs from YAML should still be used
            assert "B0TESTAA11" in batch_config.product_ids

    def test_cli_overrides_yaml_fail_fast(self):
        """Test that CLI fail-fast overrides YAML configuration."""
        # Mock CONFIG dict with batch section
        mock_config = {
            "batch": {
                "product_ids": ["B0TESTAA11"],
                "keywords": [],
                "fail_fast": False,
            }
        }
        with patch("src.scraper.amazon.config.CONFIG", mock_config):
            # CLI fail_fast should override YAML
            batch_config = load_batch_config(
                cli_product_ids=None,
                cli_keywords=None,
                cli_fail_fast=True,
            )

            assert batch_config.fail_fast is True

    def test_yaml_defaults_when_no_cli(self):
        """Test that YAML configuration is used when no CLI args provided."""
        # Mock CONFIG dict with batch section
        mock_config = {
            "batch": {
                "product_ids": ["B0TESTAA11", "B0TESTBB22"],
                "keywords": ["test keyword 1", "test keyword 2"],
            }
        }
        with patch("src.scraper.amazon.config.CONFIG", mock_config):
            batch_config = load_batch_config(
                cli_product_ids=None,
                cli_keywords=None,
                cli_fail_fast=False,
            )

            # Should use YAML values
            assert "B0TESTAA11" in batch_config.product_ids
            assert "B0TESTBB22" in batch_config.product_ids
            assert "test keyword 1" in batch_config.keywords
            assert "test keyword 2" in batch_config.keywords

    def test_defaults_when_no_yaml_or_cli(self):
        """Test that defaults are used when neither YAML nor CLI provided."""
        # Mock CONFIG dict without batch section
        mock_config: dict[str, dict[str, dict[str, int]]] = {}
        with patch("src.scraper.amazon.config.CONFIG", mock_config):
            batch_config = load_batch_config(
                cli_product_ids=None,
                cli_keywords=None,
                cli_fail_fast=False,
            )

            # Should use empty defaults
            assert batch_config.product_ids == []
            assert batch_config.keywords == []
            assert batch_config.fail_fast is False


class TestBatchScraperIntegration:
    """Test integration with actual scraper instantiation patterns."""

    def test_batch_controller_with_search_params(self, mock_scraper_with_products):
        """Test that BatchController correctly passes search parameters."""
        search_params = SearchParameters(
            min_price=20.0,
            max_price=50.0,
            min_rating=4.5,
            prime_only=True,
            max_results=5,
        )

        config = BatchConfig(
            product_ids=["B0TESTAA11"],
            keywords=[],
            fail_fast=False,
            search_params=search_params,
            max_products=10,
            products_per_keyword=5,
        )

        controller = BatchController(mock_scraper_with_products, config)
        controller.run_batch()

        # Verify search params were passed to scraper
        call_args = mock_scraper_with_products.scrape_products_unified.call_args
        assert call_args[1]["search_params"].min_price == 20.0
        assert call_args[1]["search_params"].max_price == 50.0
        assert call_args[1]["search_params"].min_rating == 4.5
        assert call_args[1]["search_params"].prime_only is True

    def test_batch_media_statistics_collection(self, mock_scraper_with_products):
        """Test that media statistics are correctly aggregated."""
        config = BatchConfig(
            product_ids=["B0TESTAA11", "B0TESTBB22"],
            keywords=[],
            fail_fast=False,
            search_params=SearchParameters(),
            max_products=10,
            products_per_keyword=5,
        )

        controller = BatchController(mock_scraper_with_products, config)
        summary = controller.run_batch()

        # Verify media stats
        assert summary.media_stats["total_images"] == 4  # 2 images per product
        assert summary.media_stats["total_videos"] == 2  # 1 video per product
        assert summary.media_stats["avg_images_per_product"] == 2.0
        assert summary.media_stats["avg_videos_per_product"] == 1.0

    def test_batch_fail_fast_integration(self, mock_scraper_with_products):
        """Test fail-fast behavior in complete workflow."""
        # Configure mock to fail on second product
        call_count = {"count": 0}

        def fail_on_second(
            keyword=None, search_params=None, max_products=None, **kwargs
        ):
            call_count["count"] += 1
            if call_count["count"] == 1:
                return [
                    ProductData(
                        title="First Product",
                        price="$29.99",
                        description="Test",
                        images=[],
                        videos=[],
                        affiliate_link="",
                        url="",
                        platform=Platform.AMAZON,
                        asin="B0TESTAA11",
                        keyword="test",
                    )
                ]
            else:
                raise Exception("Intentional failure")

        mock_scraper_with_products.scrape_products_unified.side_effect = fail_on_second

        config = BatchConfig(
            product_ids=["B0TESTAA11", "B0TESTBB22", "B0TESTCC33"],
            keywords=[],
            fail_fast=True,
            search_params=SearchParameters(),
            max_products=10,
            products_per_keyword=5,
        )

        controller = BatchController(mock_scraper_with_products, config)
        summary = controller.run_batch()

        # Should stop after first failure
        assert summary.total_attempted == 2  # 1 success + 1 failure
        assert summary.successful == 1
        assert summary.failed == 1
        assert mock_scraper_with_products.scrape_products_unified.call_count == 2
