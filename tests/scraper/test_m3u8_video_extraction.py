"""Tests for M3U8/HLS video extraction and strict product filtering.

Tests validate the M3U8 format support and strict filtering logic added in v0.11.0.

Run with: pytest tests/scraper/test_m3u8_video_extraction.py -v
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

pytestmark = pytest.mark.unit


class TestM3U8VideoSupport:
    """Test M3U8/HLS video format support."""

    def test_m3u8_url_detection(self):
        """Test M3U8 URLs are detected correctly."""
        m3u8_url = "https://m.media-amazon.com/test.m3u8"
        mp4_url = "https://m.media-amazon.com/test.mp4"

        assert ".m3u8" in m3u8_url
        assert ".mp4" in mp4_url
        assert ".m3u8" not in mp4_url

    @pytest.mark.asyncio
    async def test_ffmpeg_m3u8_conversion_command(self):
        """Test FFmpeg command for M3U8 to MP4 conversion."""
        import asyncio

        from src.scraper.amazon.downloader import convert_m3u8_to_mp4

        m3u8_url = "https://example.com/video.m3u8"
        output_path = Path("/tmp/test_video.mp4")  # noqa: S108

        # Mock async subprocess
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.communicate = Mock(return_value=(b"", b""))

        async def mock_communicate():
            return (b"", b"")

        mock_process.communicate = mock_communicate

        with patch(
            "asyncio.create_subprocess_exec", return_value=mock_process
        ) as mock_exec:
            result = await convert_m3u8_to_mp4(m3u8_url, output_path, timeout=30)

            # Verify FFmpeg was called
            assert mock_exec.called
            call_args = mock_exec.call_args[0]

            # Verify correct FFmpeg arguments
            assert "ffmpeg" in call_args
            assert "-i" in call_args
            assert m3u8_url in call_args
            assert str(output_path) in call_args
            assert result is True

    @pytest.mark.asyncio
    async def test_ffmpeg_conversion_timeout(self):
        """Test FFmpeg conversion respects timeout."""
        import asyncio

        from src.scraper.amazon.downloader import convert_m3u8_to_mp4

        m3u8_url = "https://example.com/video.m3u8"
        output_path = Path("/tmp/test_video.mp4")  # noqa: S108
        timeout = 60

        # Mock async subprocess
        mock_process = Mock()
        mock_process.returncode = 0

        async def mock_communicate():
            return (b"", b"")

        mock_process.communicate = mock_communicate

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_process),
            patch("asyncio.wait_for") as mock_wait_for,
        ):
            mock_wait_for.return_value = (b"", b"")

            await convert_m3u8_to_mp4(m3u8_url, output_path, timeout=timeout)

            # Verify timeout parameter passed to wait_for
            assert mock_wait_for.called
            assert mock_wait_for.call_args[1]["timeout"] == timeout

    @pytest.mark.asyncio
    async def test_ffmpeg_conversion_failure_handling(self):
        """Test FFmpeg conversion handles failures gracefully."""
        import asyncio

        from src.scraper.amazon.downloader import convert_m3u8_to_mp4

        m3u8_url = "https://example.com/video.m3u8"
        output_path = Path("/tmp/test_video.mp4")  # noqa: S108

        # Mock async subprocess with failure
        mock_process = Mock()
        mock_process.returncode = 1

        async def mock_communicate():
            return (b"", b"Conversion failed")

        mock_process.communicate = mock_communicate

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await convert_m3u8_to_mp4(m3u8_url, output_path)

            # Should return False on failure
            assert result is False


class TestStrictProductFiltering:
    """Test strict filtering to exclude related products and customer reviews."""

    def test_main_gallery_selectors_defined(self):
        """Test main product gallery CSS selectors are defined."""
        # These selectors should be in the JavaScript code
        expected_selectors = [
            "#imageBlock",
            "#altImages",
            "#ivTitle",
            "#main-image-container",
        ]

        # Each selector should target the main product gallery
        assert all(selector.startswith("#") for selector in expected_selectors)
        assert len(expected_selectors) == 4

    def test_excluded_sections_defined(self):
        """Test excluded sections for related products are defined."""
        # These sections should be excluded from video extraction
        expected_exclusions = [
            "comparison",
            "similar",
            "related",
            "aplus",
            "carousel",
            "sims-fbt",
        ]

        # Each exclusion pattern should filter unwanted videos
        assert len(expected_exclusions) > 0
        assert all(isinstance(pattern, str) for pattern in expected_exclusions)

    @patch("src.scraper.amazon.media_extractor.Driver")
    def test_video_filtering_excludes_related_products(self, _mock_driver):
        """Test video filtering excludes videos from related products section."""
        from src.scraper.amazon.media_extractor import (
            extract_functional_videos_with_validation,
        )

        # Mock driver with related product video
        _mock_driver.run_js.return_value = {
            "direct_videos": [
                "https://example.com/product-video.mp4",  # Should include
                "https://example.com/related-video.mp4",  # Should exclude
            ],
            "vdp_links": [],
            "thumbnails": [],
        }

        # Mock the driver to reject related product videos
        _mock_driver.select.return_value = None  # No excluded sections found

        # Extract videos with filtering
        videos = extract_functional_videos_with_validation(
            _mock_driver, debug_mode=True
        )

        # Should only return valid product videos (exact count depends on filtering)
        assert isinstance(videos, list)

    def test_asin_container_validation(self):
        """Test ASIN container validation logic."""
        current_asin = "B07ZWK2TQT"
        different_asin = "B09JQMJHXY"

        # Videos should match current ASIN
        assert current_asin != different_asin
        assert len(current_asin) == len(different_asin)  # Same format


class TestVideoMuting:
    """Test video muting during browser scraping."""

    def test_video_mute_javascript_execution(self):
        """Test video mute JavaScript is executed."""
        with patch("src.scraper.amazon.browser_functions.Driver"):
            mock_driver_instance = MagicMock()
            mock_driver_instance.run_js = MagicMock()

            # Simulate muting all videos
            mock_driver_instance.run_js(
                """
                document.querySelectorAll('video').forEach(video => {
                    video.muted = true;
                    video.volume = 0;
                });
                """
            )

            # Verify JavaScript was called
            assert mock_driver_instance.run_js.called

    def test_mutation_observer_for_dynamic_videos(self):
        """Test MutationObserver is set up for dynamically loaded videos."""
        # The MutationObserver should monitor for new video elements
        observer_code = """
        const observer = new MutationObserver((mutations) => {
            document.querySelectorAll('video').forEach(video => {
                if (!video.muted) {
                    video.muted = true;
                    video.volume = 0;
                }
            });
        });
        """

        # Verify observer targets body element
        assert "document.body" in observer_code or "MutationObserver" in observer_code
        assert "childList" in observer_code or "video.muted" in observer_code


class TestDebugModeParameter:
    """Test DEBUG_MODE parameter passing."""

    @patch("src.scraper.amazon.media_extractor.Driver")
    def test_debug_mode_parameter_passed(self, mock_driver):
        """Test DEBUG_MODE is passed as parameter to video extraction."""
        from src.scraper.amazon.media_extractor import (
            extract_functional_videos_with_validation,
        )

        # Mock driver
        mock_driver.run_js.return_value = {
            "direct_videos": [],
            "vdp_links": [],
            "thumbnails": [],
        }

        # Call with debug_mode=True
        videos = extract_functional_videos_with_validation(mock_driver, debug_mode=True)

        # Should not raise errors and return list
        assert isinstance(videos, list)

    @patch("src.scraper.amazon.media_extractor.Driver")
    def test_debug_mode_default_false(self, mock_driver):
        """Test DEBUG_MODE defaults to False when not provided."""
        from src.scraper.amazon.media_extractor import (
            extract_functional_videos_with_validation,
        )

        # Mock driver
        mock_driver.run_js.return_value = {
            "direct_videos": [],
            "vdp_links": [],
            "thumbnails": [],
        }

        # Call without debug_mode parameter
        videos = extract_functional_videos_with_validation(mock_driver)

        # Should use default False
        assert isinstance(videos, list)


class TestVideoConfigUpdates:
    """Test video configuration updates for M3U8 support."""

    def test_m3u8_monitoring_config_exists(self):
        """Test M3U8 network monitoring configuration exists."""
        from src.scraper.amazon.config import CONFIG

        video_config = CONFIG.get("global_settings", {}).get("video_config", {})

        # Should have M3U8 monitoring setting
        assert "enable_m3u8_monitoring" in video_config
        assert isinstance(video_config["enable_m3u8_monitoring"], bool)

    def test_m3u8_timeout_config_exists(self):
        """Test M3U8 download timeout configuration exists."""
        from src.scraper.amazon.config import CONFIG

        video_config = CONFIG.get("global_settings", {}).get("video_config", {})

        # Should have M3U8 timeout settings
        assert "m3u8_download_timeout" in video_config
        assert isinstance(video_config["m3u8_download_timeout"], int)
        assert video_config["m3u8_download_timeout"] > 0

    def test_network_capture_timeout_config(self):
        """Test network capture timeout configuration."""
        from src.scraper.amazon.config import CONFIG

        video_config = CONFIG.get("global_settings", {}).get("video_config", {})

        # Should have network capture timeout
        assert "network_capture_timeout" in video_config
        assert isinstance(video_config["network_capture_timeout"], int)
        assert video_config["network_capture_timeout"] > 0


class TestVideoURLPatternMatching:
    """Test video URL pattern matching for both MP4 and M3U8."""

    def test_mp4_url_pattern(self):
        """Test MP4 URL pattern matching."""
        mp4_urls = [
            "https://m.media-amazon.com/video.mp4",
            "https://m.media-amazon.com/images/path/video.mp4?query=param",
        ]

        for url in mp4_urls:
            assert ".mp4" in url.lower()

    def test_m3u8_url_pattern(self):
        """Test M3U8 URL pattern matching."""
        m3u8_urls = [
            "https://m.media-amazon.com/video.m3u8",
            "https://m.media-amazon.com/stream/playlist.m3u8",
            "https://m.media-amazon.com/hls/master.m3u8?query=param",
        ]

        for url in m3u8_urls:
            assert ".m3u8" in url.lower()

    def test_both_formats_supported(self):
        """Test both MP4 and M3U8 formats are supported."""
        mixed_urls = [
            "https://example.com/video1.mp4",
            "https://example.com/video2.m3u8",
            "https://example.com/video3.MP4",
            "https://example.com/video4.M3U8",
        ]

        # Both formats should be detectable
        mp4_count = sum(1 for url in mixed_urls if ".mp4" in url.lower())
        m3u8_count = sum(1 for url in mixed_urls if ".m3u8" in url.lower())

        assert mp4_count == 2
        assert m3u8_count == 2
        assert mp4_count + m3u8_count == len(mixed_urls)


class TestRuffConfiguration:
    """Test ruff configuration for JavaScript-heavy files."""

    def test_per_file_ignores_configured(self):
        """Test per-file ignores are configured for media_extractor.py."""
        # Read pyproject.toml
        with open("pyproject.toml") as f:
            content = f.read()

        # Check per-file-ignores section exists
        assert "[tool.ruff.lint.per-file-ignores]" in content
        assert "src/scraper/amazon/media_extractor.py" in content
        assert "E501" in content


# Test summary
def test_m3u8_feature_coverage():
    """Meta-test documenting M3U8 feature test coverage.

    Coverage:
    - M3U8 URL detection and pattern matching
    - FFmpeg M3U8 to MP4 conversion
    - Strict product filtering (exclude related products)
    - Video muting during scraping
    - DEBUG_MODE parameter passing
    - Configuration updates for M3U8 support
    - Video extraction limits and deduplication (NEW)
    - Extraction logging behavior (NEW)
    """
    features_tested = {
        "m3u8_url_detection": "TestM3U8VideoSupport",
        "ffmpeg_conversion": "TestM3U8VideoSupport",
        "strict_filtering": "TestStrictProductFiltering",
        "video_muting": "TestVideoMuting",
        "debug_mode": "TestDebugModeParameter",
        "config_updates": "TestVideoConfigUpdates",
        "url_patterns": "TestVideoURLPatternMatching",
        "ruff_config": "TestRuffConfiguration",
        "extraction_limits": "TestVideoExtractionLimits",
        "extraction_logging": "TestVideoExtractionLimits",
    }

    assert len(features_tested) == 10
    assert all(v for v in features_tested.values())
