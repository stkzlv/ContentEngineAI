"""Integration tests for the Amazon scraper workflow.

This module tests the complete scrape -> download -> validate pipeline using
mocked browser interactions and file system operations.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.scraper.amazon.models import ProductData
from src.scraper.amazon.scraper import BotasaurusAmazonScraper
from src.scraper.base import Platform

pytestmark = [pytest.mark.integration]


@pytest.fixture
def mock_outputs_root(tmp_path):
    """Patch get_outputs_root to use tmp_path."""
    with patch("src.utils.outputs_paths.get_outputs_root", return_value=tmp_path):
        yield tmp_path


@pytest.fixture
def scraper(mock_outputs_root):
    """Initialize scraper with mocked config."""
    # Mock config to avoid loading real files or env vars
    with patch("src.scraper.amazon.scraper.CONFIG") as mock_config:
        # Minimal valid config
        mock_config.get.return_value = {
            "amazon": {
                "max_products": 5,
                "domain": "com",
            },
            "global_settings": {
                "validation_config": {
                    "min_total_media": 1,
                    "min_images_if_no_video": 1,
                    "min_images_with_video": 1,
                },
                "count_products_with_media": False,
            },
        }
        scraper_instance = BotasaurusAmazonScraper()
        # Disable logging to avoid clutter
        scraper_instance.logger = MagicMock()
        yield scraper_instance


def test_full_scrape_workflow(scraper, mock_outputs_root):
    """Test the complete scraping workflow with mocked browser and downloads."""
    # 1. Prepare mock raw data (what Botasaurus would return)
    mock_raw_data = [
        {
            "title": "Test Product 1",
            "price": "$29.99",
            "description": "A great test product",
            "images": ["https://example.com/img1.jpg", "https://example.com/img2.jpg"],
            "videos": ["https://example.com/video1.mp4"],
            "affiliate_link": "https://amzn.to/test1",
            "url": "https://amazon.com/dp/B0TEST0001",
            "asin": "B0TEST0001",
            "keyword": "test keyword",
            "serp_rating": "4.5",
            "serp_reviews_count": "100",
        }
    ]

    # 2. Mock the browser creation and execution
    # BotasaurusAmazonScraper._scrape_single_pass calls create_dynamic_browser_function
    # then calls self._scrape_with_retry(browser_func, data)

    with (
        patch(
            "src.scraper.amazon.scraper.create_dynamic_browser_function"
        ) as mock_create_func,
        patch("src.scraper.amazon.scraper.download_media_files") as mock_download,
    ):
        # Setup browser mock
        mock_browser_func = MagicMock()
        mock_create_func.return_value = mock_browser_func

        # Setup _scrape_with_retry to return our raw data
        # We patch the instance method directly or relying on the flow
        # scraper._scrape_with_retry is defined in BaseScraper but used in _scrape_single_pass
        with patch.object(scraper, "_scrape_with_retry", return_value=mock_raw_data):
            # 3. Setup download mock
            # The scraper calls download_media_files with a list of tasks
            # We simulate successful download and FILE CREATION (important for verification)
            def side_effect_download(tasks):
                results = []
                for task in tasks:
                    asin = task["asin"]

                    # Create dummy files in tmp_path (simulating download)
                    product_dir = mock_outputs_root / asin
                    images_dir = product_dir / "images"
                    videos_dir = product_dir / "videos"
                    images_dir.mkdir(parents=True, exist_ok=True)
                    videos_dir.mkdir(parents=True, exist_ok=True)

                    # Create files
                    (images_dir / "img1.jpg").write_text("mock image")
                    (videos_dir / "vid1.mp4").write_text("mock video")

                    # Return result structure expected by scraper
                    results.append(
                        {
                            "asin": asin,
                            "downloaded_images": [str(images_dir / "img1.jpg")],
                            "downloaded_videos": [str(videos_dir / "vid1.mp4")],
                            "total_images": 1,
                            "total_videos": 1,
                        }
                    )
                return results

            mock_download.side_effect = side_effect_download

            # 4. Execute the scrape
            results = scraper.scrape_products(["test keyword"])

            # 5. Verify results
            assert len(results) == 1
            product = results[0]

            assert isinstance(product, ProductData)
            assert product.asin == "B0TEST0001"
            assert product.title == "Test Product 1"
            assert product.platform == Platform.AMAZON

            # Verify media paths are set (relative or absolute depending on implementation)
            # The scraper sets downloaded_images from the result dict
            assert len(product.downloaded_images) == 1
            assert len(product.downloaded_videos) == 1

            # Verify data.json was saved (scrape_products calls _save_products)
            data_json_path = mock_outputs_root / "B0TEST0001" / "data.json"
            assert data_json_path.exists()

            saved_data = json.loads(data_json_path.read_text())
            # data.json contains a list of products [product_dict]
            assert isinstance(saved_data, list)
            assert saved_data[0]["asin"] == "B0TEST0001"
            assert saved_data[0]["title"] == "Test Product 1"


def test_scrape_graceful_failure(scraper, mock_outputs_root):
    """Test workflow when browser scraping fails."""
    with (
        patch("src.scraper.amazon.scraper.create_dynamic_browser_function"),
        patch.object(
            scraper, "_scrape_with_retry", side_effect=Exception("Browser crashed")
        ),
    ):
        results = scraper.scrape_products(["fail keyword"])

        # Should handle error gracefully and return empty list (or list of failed results depending on impl)
        # scrape_products_unified catches Exception and returns []
        assert results == []

        # Verify error was logged
        scraper.logger.error.assert_called()
