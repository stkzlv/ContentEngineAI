"""Unit tests for GlobalPipelineOrchestrator.

Tests orchestrator logic in complete isolation by mocking all external
dependencies (scraper, producer, file system, configuration).
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.pipeline.config import (
    GlobalBatchConfig,
    PipelineSummary,
    ProductionPhaseSummary,
    ScrapingPhaseSummary,
)
from src.pipeline.global_batch import GlobalPipelineOrchestrator
from src.scraper.amazon.models import ProductData, SearchParameters
from src.scraper.base.models import Platform

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def base_config():
    """Create base configuration for testing."""
    return GlobalBatchConfig(
        product_ids=["B0ABC123", "B0DEF456"],
        keywords=[],
        max_products=10,
        scraper_filters=SearchParameters(),
        profile="slideshow_images1",
        random_profile=False,
        profile_pool=[],
        fail_fast=False,
        process_all_products=False,
        outputs_dir=Path("outputs"),
        debug=False,
        skip_publish=True,  # Skip publishing by default in tests
        platforms=None,
        schedule_time=None,
        fail_fast_publish=False,
        clean=False,
    )


@pytest.fixture
def mock_product_data():
    """Create mock ProductData instances for testing."""
    return [
        ProductData(
            asin="B0ABC123",
            title="Test Product 1",
            price="$29.99",
            url="https://amazon.com/dp/B0ABC123",
            platform=Platform.AMAZON,
            images=["img1.jpg", "img2.jpg"],
            videos=["vid1.mp4"],
        ),
        ProductData(
            asin="B0DEF456",
            title="Test Product 2",
            price="$49.99",
            url="https://amazon.com/dp/B0DEF456",
            platform=Platform.AMAZON,
            images=["img3.jpg", "img4.jpg", "img5.jpg"],
            videos=[],
        ),
    ]


@pytest.fixture
def orchestrator(base_config):
    """Create orchestrator instance with base config."""
    return GlobalPipelineOrchestrator(base_config)


@pytest.fixture
def mock_video_config():
    """Create mock video config with all required settings."""
    from types import SimpleNamespace

    return SimpleNamespace(
        pipeline_timeout_sec=900,
        llm_settings=SimpleNamespace(api_key_env_var=None),
        stock_media_settings=SimpleNamespace(pexels_api_key_env_var=None),
        audio_settings=SimpleNamespace(
            freesound_api_key_env_var=None,
            freesound_client_id_env_var=None,
            freesound_client_secret_env_var=None,
            freesound_refresh_token_env_var=None,
        ),
    )


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================


def test_orchestrator_initialization(base_config):
    """Test orchestrator initializes with correct configuration."""
    orchestrator = GlobalPipelineOrchestrator(base_config)

    assert orchestrator.config == base_config
    assert orchestrator.config.product_ids == ["B0ABC123", "B0DEF456"]
    assert orchestrator.config.profile == "slideshow_images1"
    assert orchestrator.config.fail_fast is False


# ============================================================================
# SCRAPING PHASE TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_scraping_phase_success(orchestrator, mock_product_data):
    """Test scraping phase executes successfully with all products."""
    with patch(
        "src.scraper.amazon.scraper.BotasaurusAmazonScraper"
    ) as mock_scraper_class:
        # Set up mock scraper (two-phase approach)
        mock_scraper = Mock()
        mock_scraper_class.return_value = mock_scraper
        mock_scraper_class.return_value.amazon_config = {}

        mock_scraper.scrape_batch_browser.return_value = [
            {"input": "B0ABC123", "products": [{"fake": True}]},
            {"input": "B0DEF456", "products": [{"fake": True}]},
        ]
        mock_scraper.process_raw_products.side_effect = [
            [mock_product_data[0]],
            [mock_product_data[1]],
        ]

        # Execute scraping phase
        summary = await orchestrator._execute_scraping_phase()

        # Verify results
        assert summary.total_attempted == 2
        assert summary.successful == 2
        assert summary.failed == 0
        assert summary.media_stats["total_images"] == 5  # 2 + 3
        assert summary.media_stats["total_videos"] == 1
        assert summary.duration_sec > 0


@pytest.mark.asyncio
async def test_scraping_phase_partial_failure(orchestrator):
    """Test scraping phase handles partial failures correctly."""
    with patch(
        "src.scraper.amazon.scraper.BotasaurusAmazonScraper"
    ) as mock_scraper_class:
        mock_scraper = Mock()
        mock_scraper_class.return_value = mock_scraper
        mock_scraper_class.return_value.amazon_config = {}

        mock_scraper.scrape_batch_browser.return_value = [
            {"input": "B0ABC123", "products": [{"fake": True}]},
            {"input": "B0DEF456", "products": [{"fake": True}]},
        ]
        mock_scraper.process_raw_products.side_effect = [
            [
                ProductData(
                    asin="B0ABC123",
                    title="Product 1",
                    price="$29.99",
                    url="https://amazon.com/dp/B0ABC123",
                    platform=Platform.AMAZON,
                )
            ],  # Success
            RuntimeError("Scraping failed"),  # Failure
        ]

        summary = await orchestrator._execute_scraping_phase()

        assert summary.total_attempted == 2
        assert summary.successful == 1
        assert summary.failed == 1
        assert len(summary.failed_products) == 1
        assert "B0DEF456" in summary.failed_products


@pytest.mark.asyncio
async def test_scraping_phase_with_fail_fast(orchestrator):
    """Test scraping phase stops on first failure with fail-fast enabled."""
    orchestrator.config.fail_fast = True

    with patch(
        "src.scraper.amazon.scraper.BotasaurusAmazonScraper"
    ) as mock_scraper_class:
        mock_scraper = Mock()
        mock_scraper_class.return_value = mock_scraper
        mock_scraper_class.return_value.amazon_config = {}
        mock_scraper.scrape_batch_browser.side_effect = RuntimeError("Scraping failed")

        with pytest.raises(RuntimeError):
            await orchestrator._execute_scraping_phase()


@pytest.mark.asyncio
async def test_scraping_phase_with_exception(orchestrator):
    """Test scraping phase propagates exceptions in fail-fast mode."""
    orchestrator.config.fail_fast = True

    with patch(
        "src.scraper.amazon.scraper.BotasaurusAmazonScraper"
    ) as mock_scraper_class:
        mock_scraper = Mock()
        mock_scraper_class.return_value = mock_scraper
        mock_scraper_class.return_value.amazon_config = {}
        mock_scraper.scrape_batch_browser.side_effect = ValueError("Invalid ASIN")

        with pytest.raises(ValueError):
            await orchestrator._execute_scraping_phase()


# ============================================================================
# HANDOFF PHASE TESTS
# ============================================================================


def test_handoff_phase_discovers_products(orchestrator):
    """Test handoff phase correctly discovers ready products."""
    mock_products = [
        (
            Path("outputs/B0ABC123"),
            ProductData(
                asin="B0ABC123",
                title="Product 1",
                price="$29.99",
                url="https://amazon.com/dp/B0ABC123",
                platform=Platform.AMAZON,
            ),
        ),
        (
            Path("outputs/B0DEF456"),
            ProductData(
                asin="B0DEF456",
                title="Product 2",
                price="$49.99",
                url="https://amazon.com/dp/B0DEF456",
                platform=Platform.AMAZON,
            ),
        ),
    ]

    with patch("src.video.producer.cli.discover_products_for_batch") as mock_discover:
        mock_discover.return_value = mock_products

        # Pass scraped product IDs matching the mocked products
        products = orchestrator._execute_handoff_phase(["B0ABC123", "B0DEF456"])

        assert len(products) == 2
        assert products[0][1].asin == "B0ABC123"
        assert products[1][1].asin == "B0DEF456"


def test_handoff_phase_with_no_products(orchestrator):
    """Test handoff phase handles no ready products gracefully."""
    with patch("src.video.producer.cli.discover_products_for_batch") as mock_discover:
        mock_discover.return_value = []

        products = orchestrator._execute_handoff_phase([])

        assert len(products) == 0


# ============================================================================
# PRODUCTION PHASE TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_production_phase_success(orchestrator, mock_video_config):
    """Test production phase successfully creates videos."""
    mock_products = [
        (
            Path("outputs/B0ABC123"),
            ProductData(
                asin="B0ABC123",
                title="Product 1",
                price="$29.99",
                url="https://amazon.com/dp/B0ABC123",
                platform=Platform.AMAZON,
            ),
        )
    ]

    with (
        patch("src.video.config.load_video_config") as mock_load_config,
        patch("aiohttp.ClientSession") as mock_session_class,
        patch(
            "src.video.producer.orchestration.create_video_for_product"
        ) as mock_create_video,
    ):
        # Set up mocks
        mock_load_config.return_value = mock_video_config
        mock_session = AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session
        mock_create_video.return_value = Path("outputs/B0ABC123/video.mp4")

        summary, produced_videos = await orchestrator._execute_production_phase(
            mock_products
        )

        assert summary.total_attempted == 1
        assert summary.successful == 1
        assert summary.failed == 0
        assert summary.skipped == 0


@pytest.mark.asyncio
async def test_production_phase_with_skipped_products(orchestrator, mock_video_config):
    """Test production phase handles skipped products correctly."""
    mock_products = [
        (
            Path("outputs/B0ABC123"),
            ProductData(
                asin="B0ABC123",
                title="Product 1",
                price="$29.99",
                url="https://amazon.com/dp/B0ABC123",
                platform=Platform.AMAZON,
            ),
        )
    ]

    with (
        patch("src.video.config.load_video_config") as mock_load_config,
        patch("aiohttp.ClientSession") as mock_session_class,
        patch(
            "src.video.producer.orchestration.create_video_for_product"
        ) as mock_create_video,
    ):
        mock_load_config.return_value = mock_video_config
        mock_session = AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session
        mock_create_video.return_value = None  # Skipped

        summary, produced_videos = await orchestrator._execute_production_phase(
            mock_products
        )

        assert summary.total_attempted == 1
        assert summary.successful == 0
        assert summary.failed == 0
        assert summary.skipped == 1
        assert "B0ABC123" in summary.skipped_products


@pytest.mark.asyncio
async def test_production_phase_with_timeout(orchestrator, mock_video_config):
    """Test production phase handles timeout errors."""
    import asyncio

    mock_products = [
        (
            Path("outputs/B0ABC123"),
            ProductData(
                asin="B0ABC123",
                title="Product 1",
                price="$29.99",
                url="https://amazon.com/dp/B0ABC123",
                platform=Platform.AMAZON,
            ),
        )
    ]

    with (
        patch("src.video.config.load_video_config") as mock_load_config,
        patch("aiohttp.ClientSession") as mock_session_class,
        patch("asyncio.wait_for") as mock_wait_for,
    ):
        mock_load_config.return_value = mock_video_config
        mock_session = AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session
        mock_wait_for.side_effect = TimeoutError()

        summary, produced_videos = await orchestrator._execute_production_phase(
            mock_products
        )

        assert summary.total_attempted == 1
        assert summary.successful == 0
        assert summary.failed == 1
        assert "B0ABC123" in summary.failed_products


@pytest.mark.asyncio
async def test_production_phase_with_exception(orchestrator, mock_video_config):
    """Test production phase handles general exceptions."""
    mock_products = [
        (
            Path("outputs/B0ABC123"),
            ProductData(
                asin="B0ABC123",
                title="Product 1",
                price="$29.99",
                url="https://amazon.com/dp/B0ABC123",
                platform=Platform.AMAZON,
            ),
        )
    ]

    with (
        patch("src.video.config.load_video_config") as mock_load_config,
        patch("aiohttp.ClientSession") as mock_session_class,
        patch(
            "src.video.producer.orchestration.create_video_for_product"
        ) as mock_create_video,
    ):
        mock_load_config.return_value = mock_video_config
        mock_session = AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session
        mock_create_video.side_effect = RuntimeError("Video creation failed")

        summary, produced_videos = await orchestrator._execute_production_phase(
            mock_products
        )

        assert summary.total_attempted == 1
        assert summary.successful == 0
        assert summary.failed == 1


@pytest.mark.asyncio
async def test_production_phase_with_fail_fast(orchestrator, mock_video_config):
    """Test production phase stops on first failure with fail-fast."""
    orchestrator.config.fail_fast = True
    mock_products = [
        (
            Path("outputs/B0ABC123"),
            ProductData(
                asin="B0ABC123",
                title="Product 1",
                price="$29.99",
                url="https://amazon.com/dp/B0ABC123",
                platform=Platform.AMAZON,
            ),
        ),
        (
            Path("outputs/B0DEF456"),
            ProductData(
                asin="B0DEF456",
                title="Product 2",
                price="$49.99",
                url="https://amazon.com/dp/B0DEF456",
                platform=Platform.AMAZON,
            ),
        ),
    ]

    with (
        patch("src.video.config.load_video_config") as mock_load_config,
        patch("aiohttp.ClientSession") as mock_session_class,
        patch(
            "src.video.producer.orchestration.create_video_for_product"
        ) as mock_create_video,
    ):
        mock_load_config.return_value = mock_video_config
        mock_session = AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session
        mock_create_video.side_effect = RuntimeError("Video creation failed")

        with pytest.raises(RuntimeError):
            await orchestrator._execute_production_phase(mock_products)


@pytest.mark.asyncio
async def test_production_phase_random_profile_mode(orchestrator, mock_video_config):
    """Test production phase with random profile selection."""
    orchestrator.config.random_profile = True
    orchestrator.config.profile = None
    orchestrator.config.profile_pool = ["slideshow_images1", "video_sequential"]

    mock_products = [
        (
            Path("outputs/B0ABC123"),
            ProductData(
                asin="B0ABC123",
                title="Product 1",
                price="$29.99",
                url="https://amazon.com/dp/B0ABC123",
                platform=Platform.AMAZON,
            ),
        )
    ]

    with (
        patch("src.video.config.load_video_config") as mock_load_config,
        patch("aiohttp.ClientSession") as mock_session_class,
        patch(
            "src.video.producer.orchestration.create_video_for_product"
        ) as mock_create_video,
        patch(
            "src.video.producer.utils.select_profile_for_product"
        ) as mock_select_profile,
        patch("src.video.producer.utils.ProfileUsageTracker") as mock_tracker_class,
    ):
        mock_load_config.return_value = mock_video_config
        mock_session = AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session
        mock_create_video.return_value = Path("outputs/B0ABC123/video.mp4")
        mock_select_profile.return_value = "slideshow_images1"
        mock_tracker = Mock()
        mock_tracker.get_counts.return_value = {"slideshow_images1": 1}
        mock_tracker_class.return_value = mock_tracker

        summary, produced_videos = await orchestrator._execute_production_phase(
            mock_products
        )

        assert summary.successful == 1
        assert "slideshow_images1" in summary.profile_distribution
        mock_select_profile.assert_called_once()
        mock_tracker.record_usage.assert_called_once_with("slideshow_images1")


# ============================================================================
# END-TO-END PIPELINE TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_complete_pipeline_success(orchestrator, mock_video_config):
    """Test complete pipeline executes all phases successfully."""
    mock_product_data = ProductData(
        asin="B0ABC123",
        title="Test Product",
        price="$29.99",
        url="https://amazon.com/dp/B0ABC123",
        platform=Platform.AMAZON,
        images=["img1.jpg"],
        videos=[],
    )

    with (
        patch(
            "src.scraper.amazon.scraper.BotasaurusAmazonScraper"
        ) as mock_scraper_class,
        patch("src.video.producer.cli.discover_products_for_batch") as mock_discover,
        patch("src.video.config.load_video_config") as mock_load_config,
        patch("aiohttp.ClientSession") as mock_session_class,
        patch(
            "src.video.producer.orchestration.create_video_for_product"
        ) as mock_create_video,
        patch("src.publisher.create_publisher") as mock_create_publisher,
        patch.dict("os.environ", {"LATE_API_KEY": "test-key"}),
    ):
        # Mock scraping phase (two-phase approach)
        mock_scraper = Mock()
        mock_scraper_class.return_value = mock_scraper
        mock_scraper_class.return_value.amazon_config = {}
        mock_scraper.scrape_batch_browser.return_value = [
            {"input": "B0ABC123", "products": [{"fake": True}]},
            {"input": "B0DEF456", "products": [{"fake": True}]},
        ]
        mock_scraper.process_raw_products.side_effect = [
            [mock_product_data],
            [mock_product_data],
        ]

        # Mock handoff phase
        mock_discover.return_value = [(Path("outputs/B0ABC123"), mock_product_data)]

        # Mock production phase
        mock_load_config.return_value = mock_video_config
        mock_session = AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session
        mock_create_video.return_value = Path("outputs/B0ABC123/video.mp4")

        # Mock publishing phase
        mock_publisher = AsyncMock()
        mock_publisher.authenticate = AsyncMock()
        mock_publisher.get_accounts = AsyncMock(return_value=[{"id": "test"}])
        mock_publisher.publish_video = AsyncMock(
            return_value={
                "success": True,
                "urls": {"youtube": "https://youtube.com/test"},
            }
        )
        mock_create_publisher.return_value = mock_publisher

        # Execute pipeline
        summary = await orchestrator.run_pipeline()

        # Verify all phases completed - 2 scraping calls, 1 production
        assert summary.scraping.successful == 2
        assert summary.production.successful == 1
        assert summary.end_to_end_success == 1
        assert summary.total_failures == 0


@pytest.mark.asyncio
async def test_pipeline_with_no_ready_products(orchestrator, mock_video_config):
    """Test pipeline handles case where scraping succeeds
    but no products ready for production.
    """
    with (
        patch(
            "src.scraper.amazon.scraper.BotasaurusAmazonScraper"
        ) as mock_scraper_class,
        patch("src.video.producer.cli.discover_products_for_batch") as mock_discover,
    ):
        mock_scraper = Mock()
        mock_scraper_class.return_value = mock_scraper
        mock_scraper_class.return_value.amazon_config = {}
        mock_scraper.scrape_batch_browser.return_value = [
            {"input": "B0ABC123", "products": [{"fake": True}]},
            {"input": "B0DEF456", "products": [{"fake": True}]},
        ]
        mock_scraper.process_raw_products.side_effect = [
            [
                ProductData(
                    asin="B0ABC123",
                    title="Product 1",
                    price="$29.99",
                    url="https://amazon.com/dp/B0ABC123",
                    platform=Platform.AMAZON,
                )
            ],
            [
                ProductData(
                    asin="B0DEF456",
                    title="Product 2",
                    price="$49.99",
                    url="https://amazon.com/dp/B0DEF456",
                    platform=Platform.AMAZON,
                )
            ],
        ]
        mock_discover.return_value = []  # No ready products

        summary = await orchestrator.run_pipeline()

        assert summary.scraping.successful == 2
        assert summary.production.total_attempted == 0
        assert summary.end_to_end_success == 0


# ============================================================================
# SUMMARY GENERATION TESTS
# ============================================================================


def test_generate_final_summary():
    """Test final summary generation with correct calculations."""
    scraping_summary = ScrapingPhaseSummary(
        total_attempted=3,
        successful=2,
        failed=1,
        successful_products=["B0ABC123", "B0DEF456"],
        failed_products=["B0FAIL1"],
        media_stats={"total_images": 10, "total_videos": 2},
        duration_sec=15.5,
    )

    production_summary = ProductionPhaseSummary(
        total_attempted=2,
        successful=1,
        failed=1,
        skipped=0,
        failed_products=["B0FAIL2"],
        skipped_products=[],
        profile_distribution={},
        duration_sec=45.2,
    )

    summary = PipelineSummary(
        scraping=scraping_summary,
        production=production_summary,
        publishing=None,
        end_to_end_success=1,
        partial_success=1,
        total_failures=2,
        total_duration_sec=60.7,
    )

    assert summary.end_to_end_success == 1
    assert summary.partial_success == 1
    assert summary.total_failures == 2
    assert summary.total_duration_sec == 60.7


def test_summary_format_method():
    """Test summary formatting produces readable output."""
    scraping_summary = ScrapingPhaseSummary(
        total_attempted=2,
        successful=2,
        failed=0,
        successful_products=["B0ABC123", "B0DEF456"],
        failed_products=[],
        media_stats={"total_images": 5, "total_videos": 1},
        duration_sec=10.0,
    )

    production_summary = ProductionPhaseSummary(
        total_attempted=2,
        successful=2,
        failed=0,
        skipped=0,
        failed_products=[],
        skipped_products=[],
        profile_distribution={"slideshow_images1": 2},
        duration_sec=30.0,
    )

    summary = PipelineSummary(
        scraping=scraping_summary,
        production=production_summary,
        publishing=None,
        end_to_end_success=2,
        partial_success=0,
        total_failures=0,
        total_duration_sec=40.0,
    )

    formatted = summary.format()

    assert "GLOBAL PIPELINE SUMMARY" in formatted
    assert "SCRAPING PHASE:" in formatted
    assert "VIDEO PRODUCTION PHASE:" in formatted
    assert "END-TO-END RESULTS:" in formatted
    assert "Total Images: 5" in formatted
    assert "Total Videos: 1" in formatted
    assert "slideshow_images1: 2 (100.0%)" in formatted
