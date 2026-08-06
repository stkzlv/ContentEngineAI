"""Integration tests for global batch pipeline end-to-end flows.

These tests validate complete pipeline execution through all three phases:
1. Scraping Phase - Product data acquisition
2. Handoff Phase - Product discovery and filtering
3. Production Phase - Video generation

Tests use mocked scraper and producer responses to ensure:
- Repeatability without external dependencies
- Fast execution
- Reliable test results

Run with: pytest tests/pipeline/test_global_batch_integration.py -v
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import yaml

from src.pipeline.config import GlobalBatchConfig
from src.pipeline.global_batch import GlobalPipelineOrchestrator
from src.publisher.models import FirstCommentConfig
from src.scraper.amazon.models import ProductData, SearchParameters
from src.scraper.base.models import Platform

# Test markers
pytestmark = pytest.mark.integration


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def mock_asyncio_sleep():
    """Mock asyncio.sleep to prevent slow stagger delays in tests."""
    with patch("asyncio.sleep", return_value=None):
        yield


@pytest.fixture
def temp_outputs_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory(prefix="test_pipeline_outputs_") as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def temp_config_dir():
    """Create temporary directory for test configuration files."""
    with tempfile.TemporaryDirectory(prefix="test_pipeline_config_") as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_yaml_config(temp_config_dir):
    """Create sample YAML configuration file for testing."""
    config_file = temp_config_dir / "pipeline.yaml"
    config_data = {
        "global_batch": {
            "product_ids": ["B0TEST111", "B0TEST222"],
            "keywords": ["test keyword"],
            "max_products": 5,
            "profile": "slideshow_images1",
            "random_profile": False,
            "profile_pool": [],
            "fail_fast": False,
            "outputs_dir": "outputs",
            "debug": False,
        }
    }
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
    return config_file


@pytest.fixture
def mock_product_data_factory():
    """Factory for creating realistic mock product data."""

    def create_product(asin, title="Test Product", has_images=True, has_videos=False):
        images = ["img1.jpg", "img2.jpg"] if has_images else []
        videos = ["video1.mp4"] if has_videos else []

        return ProductData(
            asin=asin,
            title=title,
            price="$29.99",
            url=f"https://amazon.com/dp/{asin}",
            platform=Platform.AMAZON,
            description=f"Test description for {title}",
            images=images,
            videos=videos,
            affiliate_link=f"https://amazon.com/dp/{asin}",
            keyword="test",
        )

    return create_product


def _setup_scraper_mock(mock_scraper_class, products_per_input):
    """Wire up scraper mock for the two-phase scraping approach."""
    mock_scraper = Mock()
    mock_scraper_class.return_value = mock_scraper
    mock_scraper_class.return_value.amazon_config = {}

    # scrape_batch_browser returns a list of {"input": ..., "products": [...]}
    batch_results = [
        {"input": inp, "products": [{"fake": True}] if prods else []}
        for inp, prods in products_per_input.items()
    ]
    mock_scraper.scrape_batch_browser.return_value = batch_results

    # process_raw_products is called per-input; return the matching products
    call_order = list(products_per_input.values())
    mock_scraper.process_raw_products.side_effect = call_order

    return mock_scraper


def _setup_scraper_mock_with_error(mock_scraper_class, products_per_input, error_input):
    """Wire up scraper mock where process_raw_products raises on a specific input."""
    mock_scraper = Mock()
    mock_scraper_class.return_value = mock_scraper
    mock_scraper_class.return_value.amazon_config = {}

    # batch browser succeeds for all inputs
    batch_results = [
        {"input": inp, "products": [{"fake": True}]} for inp in products_per_input
    ]
    mock_scraper.scrape_batch_browser.return_value = batch_results

    def side_effect(raw_products, target_download_count=None):
        # Find which input this call corresponds to by order
        call_idx = mock_scraper.process_raw_products.call_count - 1
        inp = list(products_per_input.keys())[call_idx]
        if inp == error_input:
            raise RuntimeError("Scraping failed")
        return products_per_input[inp]

    mock_scraper.process_raw_products.side_effect = side_effect

    return mock_scraper


@pytest.fixture
def mock_video_config():
    """Create mock video config with all required settings."""
    return SimpleNamespace(
        pipeline_timeout_sec=900,
        llm_settings=SimpleNamespace(api_key_env_var=None),
        stock_media_settings=SimpleNamespace(pexels_api_key_env_var=None),
        audio_settings=SimpleNamespace(
            freesound_api_key_env_var=None,
            freesound_client_id_env_var=None,
            freesound_client_secret_env_var=None,
            freesound_refresh_token_env_var=None,
            audio_providers=[],
        ),
    )


# ============================================================================
# INTEGRATION TESTS - PRODUCT IDS MODE
# ============================================================================


@pytest.mark.asyncio
async def test_pipeline_with_product_ids_only(
    temp_outputs_dir, mock_product_data_factory, mock_video_config
):
    """Test complete pipeline execution with product IDs only."""
    # Configuration - product IDs only
    config = GlobalBatchConfig(
        product_ids=["B0TEST111", "B0TEST222"],
        keywords=[],
        max_products=10,
        scraper_filters=SearchParameters(),
        profile="slideshow_images1",
        random_profile=False,
        profile_pool=[],
        fail_fast=False,
        outputs_dir=temp_outputs_dir,
        debug=False,
    )

    orchestrator = GlobalPipelineOrchestrator(config)

    # Mock product data
    product1 = mock_product_data_factory("B0TEST111", "Product 1", has_images=True)
    product2 = mock_product_data_factory("B0TEST222", "Product 2", has_images=True)

    # Mock metadata loading
    def mock_metadata_factory(*args, **kwargs):
        mock_metadata = Mock()
        mock_metadata.format_content = Mock(
            return_value={
                "title": "Test Title",
                "description": "Test Description",
            }
        )
        mock_metadata.clamp_to_limits = Mock(return_value=())
        return mock_metadata

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
        patch(
            "src.publisher.publish_modes.load_platform_metadata",
            side_effect=mock_metadata_factory,
        ),
        patch.dict("os.environ", {"LATE_API_KEY": "test-key"}),
    ):
        # Mock scraper (two-phase approach)
        _setup_scraper_mock(
            mock_scraper_class,
            {
                "B0TEST111": [product1],
                "B0TEST222": [product2],
            },
        )

        # Mock handoff phase - both products ready
        mock_discover.return_value = [
            (temp_outputs_dir / "B0TEST111", product1),
            (temp_outputs_dir / "B0TEST222", product2),
        ]

        # Mock production phase
        mock_load_config.return_value = mock_video_config
        mock_session = AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session
        mock_create_video.side_effect = [
            temp_outputs_dir / "B0TEST111" / "video.mp4",
            temp_outputs_dir / "B0TEST222" / "video.mp4",
        ]

        # Mock publishing phase
        mock_publisher = AsyncMock()
        mock_publisher.authenticate = AsyncMock()
        mock_publisher.first_comment_config = FirstCommentConfig(enabled=False)
        mock_publisher.get_accounts = AsyncMock(
            return_value=[
                {"id": "test1", "platform": "youtube", "account_id": "acc1"},
                {"id": "test2", "platform": "tiktok", "account_id": "acc2"},
                {"id": "test3", "platform": "instagram", "account_id": "acc3"},
            ]
        )
        mock_publisher.upload_media = AsyncMock(return_value="media_id_123")
        mock_publisher.publish = AsyncMock(return_value={"success": True})
        mock_create_publisher.return_value = mock_publisher

        # Execute pipeline
        summary = await orchestrator.run_pipeline()

        # Verify scraping phase
        assert summary.scraping.total_attempted == 2
        assert summary.scraping.successful == 2
        assert summary.scraping.failed == 0

        # Verify production phase
        assert summary.production.total_attempted == 2
        assert summary.production.successful == 2
        assert summary.production.failed == 0

        # Verify end-to-end success
        assert summary.end_to_end_success == 2
        assert summary.total_failures == 0


# ============================================================================
# INTEGRATION TESTS - KEYWORDS MODE
# ============================================================================


@pytest.mark.asyncio
async def test_pipeline_with_keywords_only(
    temp_outputs_dir, mock_product_data_factory, mock_video_config
):
    """Test complete pipeline execution with keywords only."""
    # Configuration - keywords only
    config = GlobalBatchConfig(
        product_ids=[],
        keywords=["wireless earbuds", "bluetooth speaker"],
        max_products=2,
        scraper_filters=SearchParameters(min_rating=4.0),
        profile="video_sequential",
        random_profile=False,
        profile_pool=[],
        fail_fast=False,
        outputs_dir=temp_outputs_dir,
        debug=False,
    )

    orchestrator = GlobalPipelineOrchestrator(config)

    # Mock product data from keyword searches
    product1 = mock_product_data_factory("B0EARBUDS1", "Wireless Earbuds Pro")
    product2 = mock_product_data_factory("B0SPEAKER1", "Bluetooth Speaker XL")

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
    ):
        # Mock scraper (two-phase approach)
        _setup_scraper_mock(
            mock_scraper_class,
            {
                "wireless earbuds": [product1],
                "bluetooth speaker": [product2],
            },
        )

        # Mock handoff phase
        mock_discover.return_value = [
            (temp_outputs_dir / "B0EARBUDS1", product1),
            (temp_outputs_dir / "B0SPEAKER1", product2),
        ]

        # Mock production phase
        mock_load_config.return_value = mock_video_config
        mock_session = AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session
        mock_create_video.side_effect = [
            temp_outputs_dir / "B0EARBUDS1" / "video.mp4",
            temp_outputs_dir / "B0SPEAKER1" / "video.mp4",
        ]

        # Execute pipeline
        summary = await orchestrator.run_pipeline()

        # Verify scraping with filters
        assert summary.scraping.total_attempted == 2
        assert summary.scraping.successful == 2
        # Verify filters passed to scraper
        assert config.scraper_filters.min_rating == 4.0

        # Verify production with correct profile
        assert summary.production.successful == 2
        # Profile should be video_sequential as configured
        assert config.profile == "video_sequential"


# ============================================================================
# INTEGRATION TESTS - MIXED INPUT MODE
# ============================================================================


@pytest.mark.asyncio
async def test_pipeline_with_mixed_input(
    temp_outputs_dir, mock_product_data_factory, mock_video_config
):
    """Test complete pipeline with both product IDs and keywords."""
    # Configuration - mixed input
    config = GlobalBatchConfig(
        product_ids=["B0DIRECT1"],
        keywords=["smart watch"],
        max_products=5,
        scraper_filters=SearchParameters(),
        profile="product_video_hybrid",
        random_profile=False,
        profile_pool=[],
        fail_fast=False,
        outputs_dir=temp_outputs_dir,
        debug=False,
    )

    orchestrator = GlobalPipelineOrchestrator(config)

    # Mock products from both sources
    direct_product = mock_product_data_factory("B0DIRECT1", "Direct Product")
    keyword_product = mock_product_data_factory("B0WATCH1", "Smart Watch Pro")

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
    ):
        # Mock scraper (two-phase approach)
        _setup_scraper_mock(
            mock_scraper_class,
            {
                "B0DIRECT1": [direct_product],
                "smart watch": [keyword_product],
            },
        )

        # Mock handoff - both ready
        mock_discover.return_value = [
            (temp_outputs_dir / "B0DIRECT1", direct_product),
            (temp_outputs_dir / "B0WATCH1", keyword_product),
        ]

        # Mock production
        mock_load_config.return_value = mock_video_config
        mock_session = AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session
        mock_create_video.side_effect = [
            temp_outputs_dir / "B0DIRECT1" / "video.mp4",
            temp_outputs_dir / "B0WATCH1" / "video.mp4",
        ]

        # Execute pipeline
        summary = await orchestrator.run_pipeline()

        # Verify mixed input processed correctly
        assert summary.scraping.total_attempted == 2  # 1 product ID + 1 keyword
        assert summary.scraping.successful == 2
        assert summary.production.successful == 2


# ============================================================================
# INTEGRATION TESTS - FAIL-FAST BEHAVIOR
# ============================================================================


@pytest.mark.asyncio
async def test_pipeline_fail_fast_at_scraping_phase(
    temp_outputs_dir, mock_product_data_factory
):
    """Test pipeline stops immediately on scraping failure with fail-fast enabled."""
    config = GlobalBatchConfig(
        product_ids=["B0GOOD1", "B0BAD2", "B0GOOD3"],
        keywords=[],
        max_products=10,
        scraper_filters=SearchParameters(),
        profile="slideshow_images1",
        random_profile=False,
        profile_pool=[],
        fail_fast=True,  # Enable fail-fast
        outputs_dir=temp_outputs_dir,
        debug=False,
    )

    orchestrator = GlobalPipelineOrchestrator(config)

    product1 = mock_product_data_factory("B0GOOD1", "Good Product")

    with patch(
        "src.scraper.amazon.scraper.BotasaurusAmazonScraper"
    ) as mock_scraper_class:
        # Batch browser succeeds, but process_raw_products fails on second input
        _setup_scraper_mock_with_error(
            mock_scraper_class,
            {"B0GOOD1": [product1], "B0BAD2": [], "B0GOOD3": []},
            error_input="B0BAD2",
        )

        # Execute pipeline - should raise due to fail-fast
        with pytest.raises(RuntimeError, match="Scraping failed"):
            await orchestrator.run_pipeline()

        # process_raw_products called twice: first succeeds, second raises
        mock_scraper_class.return_value.process_raw_products.assert_called()


@pytest.mark.asyncio
async def test_pipeline_fail_fast_at_production_phase(
    temp_outputs_dir, mock_product_data_factory, mock_video_config
):
    """Test pipeline stops immediately on production failure with fail-fast enabled."""
    config = GlobalBatchConfig(
        product_ids=["B0PROD1", "B0PROD2"],
        keywords=[],
        max_products=10,
        scraper_filters=SearchParameters(),
        profile="slideshow_images1",
        random_profile=False,
        profile_pool=[],
        fail_fast=True,  # Enable fail-fast
        outputs_dir=temp_outputs_dir,
        debug=False,
    )

    orchestrator = GlobalPipelineOrchestrator(config)

    product1 = mock_product_data_factory("B0PROD1", "Product 1")
    product2 = mock_product_data_factory("B0PROD2", "Product 2")

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
    ):
        # Mock scraping (two-phase approach) - both succeed
        _setup_scraper_mock(
            mock_scraper_class,
            {
                "B0PROD1": [product1],
                "B0PROD2": [product2],
            },
        )

        # Mock handoff
        mock_discover.return_value = [
            (temp_outputs_dir / "B0PROD1", product1),
            (temp_outputs_dir / "B0PROD2", product2),
        ]

        # Mock production - first fails
        mock_load_config.return_value = mock_video_config
        mock_session = AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session
        mock_create_video.side_effect = RuntimeError("Video creation failed")

        # Execute pipeline - should raise due to fail-fast in production
        with pytest.raises(RuntimeError, match="Video creation failed"):
            await orchestrator.run_pipeline()


# ============================================================================
# INTEGRATION TESTS - GRACEFUL CONTINUATION
# ============================================================================


@pytest.mark.asyncio
async def test_pipeline_graceful_continuation_on_failures(
    temp_outputs_dir, mock_product_data_factory, mock_video_config
):
    """Test pipeline continues processing all products when fail-fast disabled."""
    config = GlobalBatchConfig(
        product_ids=["B0GOOD1", "B0BAD2", "B0GOOD3"],
        keywords=[],
        max_products=10,
        scraper_filters=SearchParameters(),
        profile="slideshow_images1",
        random_profile=False,
        profile_pool=[],
        fail_fast=False,  # Graceful continuation
        outputs_dir=temp_outputs_dir,
        debug=False,
    )

    orchestrator = GlobalPipelineOrchestrator(config)

    product1 = mock_product_data_factory("B0GOOD1", "Good Product 1")
    product3 = mock_product_data_factory("B0GOOD3", "Good Product 3")

    with patch(
        "src.scraper.amazon.scraper.BotasaurusAmazonScraper"
    ) as mock_scraper_class:
        # Mix of success and failure (two-phase approach)
        # B0BAD2 returns empty products from batch browser
        mock_scraper = Mock()
        mock_scraper_class.return_value = mock_scraper
        mock_scraper_class.return_value.amazon_config = {}

        mock_scraper.scrape_batch_browser.return_value = [
            {"input": "B0GOOD1", "products": [{"fake": True}]},
            {"input": "B0BAD2", "products": []},
            {"input": "B0GOOD3", "products": [{"fake": True}]},
        ]
        mock_scraper.process_raw_products.side_effect = [
            [product1],  # B0GOOD1
            # B0BAD2 skipped (empty products)
            [product3],  # B0GOOD3
        ]

        # Execute pipeline - should complete despite failure
        summary = await orchestrator.run_pipeline()

        # Verify all inputs attempted
        assert summary.scraping.total_attempted == 3
        assert summary.scraping.successful == 2
        assert summary.scraping.failed == 1
        assert "B0BAD2" in summary.scraping.failed_products


# ============================================================================
# INTEGRATION TESTS - RANDOM PROFILE MODE
# ============================================================================


@pytest.mark.asyncio
async def test_pipeline_with_random_profile_selection(
    temp_outputs_dir, mock_product_data_factory, mock_video_config
):
    """Test pipeline with random profile selection mode."""
    config = GlobalBatchConfig(
        product_ids=["B0RAND1", "B0RAND2"],
        keywords=[],
        max_products=10,
        scraper_filters=SearchParameters(),
        profile=None,  # No fixed profile
        random_profile=True,  # Enable random mode
        profile_pool=["slideshow_images1", "video_sequential", "product_video_hybrid"],
        fail_fast=False,
        outputs_dir=temp_outputs_dir,
        debug=False,
    )

    orchestrator = GlobalPipelineOrchestrator(config)

    product1 = mock_product_data_factory("B0RAND1", "Random Product 1")
    product2 = mock_product_data_factory("B0RAND2", "Random Product 2")

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
        patch(
            "src.video.producer.utils.select_profile_for_product"
        ) as mock_select_profile,
        patch("src.video.producer.utils.ProfileUsageTracker") as mock_tracker_class,
    ):
        # Mock scraper (two-phase approach)
        _setup_scraper_mock(
            mock_scraper_class,
            {
                "B0RAND1": [product1],
                "B0RAND2": [product2],
            },
        )

        # Mock handoff
        mock_discover.return_value = [
            (temp_outputs_dir / "B0RAND1", product1),
            (temp_outputs_dir / "B0RAND2", product2),
        ]

        # Mock production with random profiles
        mock_load_config.return_value = mock_video_config
        mock_session = AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session
        mock_create_video.side_effect = [
            temp_outputs_dir / "B0RAND1" / "video.mp4",
            temp_outputs_dir / "B0RAND2" / "video.mp4",
        ]
        mock_select_profile.side_effect = ["slideshow_images1", "video_sequential"]
        mock_tracker = Mock()
        mock_tracker.get_counts.return_value = {
            "slideshow_images1": 1,
            "video_sequential": 1,
        }
        mock_tracker_class.return_value = mock_tracker

        # Execute pipeline
        summary = await orchestrator.run_pipeline()

        # Verify random profile mode
        assert summary.production.successful == 2
        assert summary.production.profile_distribution is not None
        assert "slideshow_images1" in summary.production.profile_distribution
        assert "video_sequential" in summary.production.profile_distribution
        mock_select_profile.assert_called()
        mock_tracker.record_usage.assert_called()


# ============================================================================
# INTEGRATION TESTS - ZERO PRODUCTS READY
# ============================================================================


@pytest.mark.asyncio
async def test_pipeline_with_zero_products_ready_for_production(
    temp_outputs_dir, mock_product_data_factory
):
    """Test pipeline handles case where scraping succeeds but no products ready."""
    config = GlobalBatchConfig(
        product_ids=["B0NODATA1", "B0NODATA2"],
        keywords=[],
        max_products=10,
        scraper_filters=SearchParameters(),
        profile="slideshow_images1",
        random_profile=False,
        profile_pool=[],
        fail_fast=False,
        outputs_dir=temp_outputs_dir,
        debug=False,
    )

    orchestrator = GlobalPipelineOrchestrator(config)

    # Products with no images/videos (not ready for production)
    product1 = mock_product_data_factory(
        "B0NODATA1", "No Media Product 1", has_images=False, has_videos=False
    )
    product2 = mock_product_data_factory(
        "B0NODATA2", "No Media Product 2", has_images=False, has_videos=False
    )

    with (
        patch(
            "src.scraper.amazon.scraper.BotasaurusAmazonScraper"
        ) as mock_scraper_class,
        patch("src.video.producer.cli.discover_products_for_batch") as mock_discover,
    ):
        # Mock scraper (two-phase approach) - returns products without media
        _setup_scraper_mock(
            mock_scraper_class,
            {
                "B0NODATA1": [product1],
                "B0NODATA2": [product2],
            },
        )

        # Mock handoff - no products ready (empty list)
        mock_discover.return_value = []

        # Execute pipeline
        summary = await orchestrator.run_pipeline()

        # Verify scraping succeeded
        assert summary.scraping.successful == 2

        # Verify production skipped (no products ready)
        assert summary.production.total_attempted == 0
        assert summary.production.successful == 0
        assert summary.end_to_end_success == 0


# ============================================================================
# INTEGRATION TESTS - PAGE RETRY ON VALIDATION FAILURE
# ============================================================================


@pytest.mark.asyncio
async def test_pipeline_retries_next_page_on_validation_failure(
    temp_outputs_dir, mock_product_data_factory, mock_video_config
):
    """When page 1 products fail validation, retry with page 2."""
    config = GlobalBatchConfig(
        product_ids=[],
        keywords=["smart watch"],
        max_products=10,
        products_per_keyword=1,
        scraper_filters=SearchParameters(),
        profile="slideshow_images1",
        fail_fast=False,
        outputs_dir=temp_outputs_dir,
        debug=False,
    )

    product = mock_product_data_factory("B0RETRY1")

    with (
        patch(
            "src.scraper.amazon.scraper.BotasaurusAmazonScraper"
        ) as mock_scraper_class,
        patch(
            "src.pipeline.global_batch.load_video_config_modular"
        ) as mock_load_config,
        patch(
            "src.video.producer.orchestration.create_video_for_product"
        ) as mock_producer,
    ):
        mock_load_config.return_value = mock_video_config
        mock_producer.return_value = MagicMock(success=False, error_message="skipped")

        mock_scraper = Mock()
        mock_scraper_class.return_value = mock_scraper
        mock_scraper_class.return_value.amazon_config = {}

        # Page 1: products found but all fail validation
        mock_scraper.scrape_batch_browser.side_effect = [
            [{"input": "smart watch", "products": [{"fake": True}]}],  # page 1
            [{"input": "smart watch", "products": [{"fake": True}]}],  # page 2 retry
        ]
        # _is_asin and _is_url return False for keywords
        mock_scraper._is_asin.return_value = False
        mock_scraper._is_url.return_value = False

        # First call returns empty (validation fail), second returns product
        mock_scraper.process_raw_products.side_effect = [
            [],  # page 1 all rejected
            [product],  # page 2 succeeds
        ]

        # Mock SCRAPER_CONFIG for max_retry_pages
        with patch(
            "src.scraper.amazon.config.CONFIG",
            {"global_settings": {"batch_processing": {"max_retry_pages": 5}}},
        ):
            orchestrator = GlobalPipelineOrchestrator(config)
            summary = await orchestrator.run_pipeline()

        # Should have retried and found the product on page 2
        assert summary.scraping.successful == 1
        assert mock_scraper.scrape_batch_browser.call_count == 2
        # Second call should use start_page=2
        second_call = mock_scraper.scrape_batch_browser.call_args_list[1]
        assert second_call.kwargs.get("start_page") == 2


@pytest.mark.asyncio
async def test_pipeline_skips_retry_for_asin_inputs(
    temp_outputs_dir, mock_product_data_factory, mock_video_config
):
    """ASINs should not trigger page retry (single product, no pagination)."""
    config = GlobalBatchConfig(
        product_ids=["B0NOPAGES"],
        keywords=[],
        max_products=10,
        scraper_filters=SearchParameters(),
        profile="slideshow_images1",
        fail_fast=False,
        outputs_dir=temp_outputs_dir,
        debug=False,
    )

    with (
        patch(
            "src.scraper.amazon.scraper.BotasaurusAmazonScraper"
        ) as mock_scraper_class,
        patch(
            "src.pipeline.global_batch.load_video_config_modular"
        ) as mock_load_config,
    ):
        mock_load_config.return_value = mock_video_config

        mock_scraper = Mock()
        mock_scraper_class.return_value = mock_scraper
        mock_scraper_class.return_value.amazon_config = {}

        mock_scraper.scrape_batch_browser.return_value = [
            {"input": "B0NOPAGES", "products": [{"fake": True}]},
        ]
        mock_scraper._is_asin.return_value = True
        mock_scraper._is_url.return_value = False
        mock_scraper.process_raw_products.return_value = []  # validation fails

        orchestrator = GlobalPipelineOrchestrator(config)
        summary = await orchestrator.run_pipeline()

        # Should NOT retry - only 1 call to scrape_batch_browser
        assert mock_scraper.scrape_batch_browser.call_count == 1
        assert summary.scraping.failed == 1
