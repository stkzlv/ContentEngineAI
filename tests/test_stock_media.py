"""Unit tests for the stock media fetcher functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.video.stock_media import StockMediaFetcher
from src.video.video_config import VideoConfig


class TestStockMediaFetcher:
    """Test the stock media fetcher functionality."""

    @pytest.fixture
    def stock_media_fetcher(self, mock_config: VideoConfig) -> StockMediaFetcher:
        """Create a stock media fetcher instance for testing."""
        return StockMediaFetcher(
            settings=mock_config.stock_media_settings,
            secrets={"PEXELS_API_KEY": "test_key"},
            media_settings=mock_config.media_settings,
            api_settings=mock_config.api_settings,
        )

    @pytest.mark.unit
    def test_initialization(self, stock_media_fetcher: StockMediaFetcher):
        """Test that StockMediaFetcher initializes correctly."""
        assert stock_media_fetcher is not None
        assert hasattr(stock_media_fetcher, "fetch_and_download_stock")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_fetch_and_download_stock_basic(
        self, stock_media_fetcher: StockMediaFetcher, temp_dir: Path
    ):
        """Test basic fetch_and_download_stock functionality with mocked API."""
        output_dir = temp_dir / "stock_media"

        # Mock the internal _search_and_select_pexels method to avoid actual API calls
        with patch.object(
            stock_media_fetcher, "_search_and_select_pexels"
        ) as mock_search:
            mock_search.return_value = []  # Return empty list to simulate no results

            # Mock aiohttp session for the test
            mock_session = MagicMock()

            result = await stock_media_fetcher.fetch_and_download_stock(
                keywords=["test", "product"],
                image_count=2,
                video_count=1,
                download_dir=output_dir,
                session=mock_session,
            )

            # Should return empty list when no stock media is found
            assert isinstance(result, list)
            assert len(result) == 0

            # Verify the method was called
            mock_search.assert_called()
