"""Tests for performance monitoring integration in producer module."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.scraper.amazon.scraper import ProductData
from src.utils.performance import performance_monitor
from src.video.producer import (
    PipelineContext,
    create_video_for_product,
    step_gather_visuals,
)
from src.video.video_config import VideoConfig, VideoProfile


class TestProducerPerformanceIntegration:
    """Test performance monitoring integration in producer."""

    @pytest.mark.asyncio
    async def test_step_gather_visuals_performance_monitoring(self):
        """Test that step_gather_visuals uses performance monitoring."""
        # Create mock context
        mock_product = Mock(spec=ProductData)
        mock_product.downloaded_images = []
        mock_product.downloaded_videos = []
        mock_product.title = "Test Product"

        mock_profile = Mock(spec=VideoProfile)
        mock_profile.description = "test profile"
        mock_profile.use_scraped_images = False
        mock_profile.use_scraped_videos = False
        mock_profile.use_stock_images = False
        mock_profile.use_stock_videos = False
        mock_profile.stock_image_count = 0
        mock_profile.stock_video_count = 0

        mock_config = Mock(spec=VideoConfig)
        mock_config.project_root = Path("/test")

        ctx = PipelineContext(
            product=mock_product,
            profile=mock_profile,
            config=mock_config,
            secrets={},
            session=Mock(),
            run_paths={"gathered_visuals_file": Path("test.json")},
            debug_mode=False,
        )

        # Mock the performance monitor
        with patch.object(performance_monitor, "measure_step") as mock_measure_step:
            mock_measure_step.return_value.__aenter__ = AsyncMock()
            mock_measure_step.return_value.__aexit__ = AsyncMock()

            # Mock the save_visuals_info function
            with patch("src.video.producer.save_visuals_info"):
                await step_gather_visuals(ctx)

            # Verify performance monitoring was called
            mock_measure_step.assert_called_once_with(
                "gather_visuals",
                profile="test profile",
                scraped_images_enabled=False,
                scraped_videos_enabled=False,
                stock_images_count=0,
                stock_videos_count=0,
            )

    @pytest.mark.asyncio
    async def test_create_video_for_product_starts_monitoring(self):
        """Test that create_video_for_product starts performance monitoring."""
        mock_config = Mock(spec=VideoConfig)
        mock_config.get_profile.return_value = Mock(spec=VideoProfile)

        mock_product = Mock(spec=ProductData)
        mock_product.asin = "TEST123"
        mock_product.title = "Test Product"

        # Mock all the dependencies to prevent actual execution
        with (
            patch("src.video.producer.get_video_run_paths") as mock_get_paths,
            patch("src.video.producer.ensure_dirs_exist"),
            patch("src.video.producer._load_pipeline_state"),
            patch.object(performance_monitor, "start_pipeline") as mock_start_pipeline,
            patch.object(
                performance_monitor, "get_pipeline_summary"
            ) as mock_get_summary,
        ):
            mock_get_paths.return_value = {
                "run_root": Path("/test"),
                "final_video_output": Path("/test/output.mp4"),
            }
            mock_get_summary.return_value = {}

            # Mock to raise an exception to exit early
            with patch("src.video.producer.PipelineContext") as mock_context:
                mock_context.side_effect = Exception("Test exit")

                from contextlib import suppress

                with suppress(Exception):
                    await create_video_for_product(
                        config=mock_config,
                        product=mock_product,
                        profile_name="test_profile",
                        secrets={},
                        session=Mock(),
                        debug_mode=False,
                        clean_run=False,
                        debug_step_target=None,
                    )

            # Verify performance monitoring was started
            mock_start_pipeline.assert_called_once()

    def test_performance_monitor_global_instance(self):
        """Test that global performance monitor instance is available."""
        assert performance_monitor is not None
        # Reset any existing metrics for clean test
        performance_monitor.metrics.clear()
        performance_monitor.pipeline_start = None
        performance_monitor.current_step = None
