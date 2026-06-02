"""Unit tests for two-part subtitle system."""

import tempfile
from pathlib import Path

import pytest

from src.video.subtitle_positioning import VisualBounds
from src.video.subtitle_utils import create_static_upper_subtitle


@pytest.mark.unit
class TestTwoPartSubtitles:
    """Test suite for two-part subtitle system."""

    @pytest.fixture
    def mock_video_config(self, mock_config):
        """Use the project's `mock_config` from conftest.

        Past iterations used a bare `Mock()`, but `get_style_config` now
        requires a real `VideoConfig` with `style_presets` populated. The
        20 usages of this fixture still pick the same name without churn.
        """
        return mock_config

    @pytest.fixture
    def basic_subtitle_settings(self):
        """Create basic subtitle settings.

        The caller (TwoPartSubtitleHandler.generate_upper_subtitle) folds
        upper-line values into subtitle_settings before passing it to
        create_static_upper_subtitle, so tests pass the already-folded shape:
        anchor/margin/font_size_scale/style_preset reflect the upper line.
        """
        return {
            "subtitle_format": "ass",
            "font_size_percent": 0.03,
            "horizontal_alignment": "center",
            "anchor": "above_content",
            "margin": 0.005,
            "font_size_scale": 0.8,
            "style_preset": "minimal",
            "content_aware": True,
            "randomize_effects": False,
        }

    def test_create_static_upper_subtitle_basic(
        self, mock_video_config, basic_subtitle_settings
    ):
        """Test basic static upper subtitle creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "upper.ass"

            result = create_static_upper_subtitle(
                text="Product: example.com/product",
                output_path=output_path,
                subtitle_settings=basic_subtitle_settings,
                video_config=mock_video_config,
                format_type="ass",
                voiceover_duration=30.0,
            )

            assert result is not None
            assert result.exists()
            assert result == output_path

            # Verify ASS content
            content = output_path.read_text()
            assert "[Script Info]" in content
            assert "[V4+ Styles]" in content
            assert "[Events]" in content
            assert "Product: example.com/product" in content

    def test_create_static_upper_subtitle_with_visual_bounds(
        self, mock_video_config, basic_subtitle_settings
    ):
        """Test upper subtitle positioning with visual bounds."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "upper.ass"

            # Create visual bounds (image at 12% from top, 85% width)
            visual_bounds = VisualBounds(
                x=0.075,  # (1.0 - 0.85) / 2
                y=0.12,  # 12% from top
                width=0.85,
                height=0.8,
            )

            result = create_static_upper_subtitle(
                text="Product: example.com/product",
                output_path=output_path,
                subtitle_settings=basic_subtitle_settings,
                video_config=mock_video_config,
                format_type="ass",
                voiceover_duration=30.0,
                visual_bounds=visual_bounds,
            )

            assert result is not None
            assert result.exists()

            # Verify content-aware positioning
            content = output_path.read_text()
            assert "\\pos(" in content

            # Extract y-position from {\pos(x,y)}
            import re

            pos_match = re.search(r"\\pos\((\d+),(\d+)\)", content)
            assert pos_match is not None

            y_pos = int(pos_match.group(2))

            # ABOVE_CONTENT wants Y = (visual_bounds.y - margin) = 0.115, but
            # that sits inside the 2026 top header, so it clamps up to min_y
            # (0.141 * 1920 = 270px).
            expected_y = int(0.141 * 1920)
            assert (
                abs(y_pos - expected_y) < 10
            ), f"Y position {y_pos} not close to expected {expected_y}"

    def test_create_static_upper_subtitle_without_visual_bounds(
        self, mock_video_config, basic_subtitle_settings
    ):
        """Test upper subtitle defaults to top when visual_bounds not provided."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "upper.ass"

            result = create_static_upper_subtitle(
                text="Product: example.com/product",
                output_path=output_path,
                subtitle_settings=basic_subtitle_settings,
                video_config=mock_video_config,
                format_type="ass",
                voiceover_duration=30.0,
                visual_bounds=None,  # No visual bounds
            )

            assert result is not None
            assert result.exists()

            # Without visual_bounds, should use margin (top of frame)
            content = output_path.read_text()
            import re

            pos_match = re.search(r"\\pos\((\d+),(\d+)\)", content)
            assert pos_match is not None

            y_pos = int(pos_match.group(2))

            # Without visual_bounds, falls back to margin. Safe zone clamps
            # to min_y (0.141 * 1920 = 270px) since margin 0.005 is below it.
            safe_zone_min_y_px = int(0.141 * 1920)
            assert (
                abs(y_pos - safe_zone_min_y_px) < 20
            ), f"Without visual_bounds, y should be near safe zone min {safe_zone_min_y_px}, got {y_pos}"

    def test_upper_settings_are_applied(
        self, mock_video_config, basic_subtitle_settings
    ):
        """Test that folded upper-line settings are honoured by the helper."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "upper.ass"

            # Caller folds upper-line values into the dict; vary them here
            # to confirm the helper reads them via the now-canonical keys.
            settings = basic_subtitle_settings.copy()
            settings["margin"] = 0.01
            settings["font_size_scale"] = 0.9

            result = create_static_upper_subtitle(
                text="Test text",
                output_path=output_path,
                subtitle_settings=settings,
                video_config=mock_video_config,
                format_type="ass",
                voiceover_duration=10.0,
            )

            assert result is not None
            # Settings should be applied (tested indirectly through successful creation)

    def test_url_text_in_subtitle(self, mock_video_config, basic_subtitle_settings):
        """Test that URL text appears in upper subtitle."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "upper.ass"

            # URL text
            result = create_static_upper_subtitle(
                text="https://stte.psee.io/abc123",
                output_path=output_path,
                subtitle_settings=basic_subtitle_settings,
                video_config=mock_video_config,
                format_type="ass",
                voiceover_duration=30.0,
            )

            assert result is not None
            content = output_path.read_text()

            # Shortened URL should appear in subtitle
            assert "stte.psee.io/abc123" in content

    def test_srt_format_support(self, mock_video_config, basic_subtitle_settings):
        """Test that SRT format is also supported for upper subtitles."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "upper.srt"

            result = create_static_upper_subtitle(
                text="Product URL",
                output_path=output_path,
                subtitle_settings=basic_subtitle_settings,
                video_config=mock_video_config,
                format_type="srt",
                voiceover_duration=20.0,
            )

            assert result is not None
            assert result.exists()

            # Verify SRT format
            content = output_path.read_text()
            assert "1\n" in content  # SRT sequence number
            assert "-->" in content  # SRT timing separator
            assert "Product URL" in content

    def test_product_id_randomization_disabled(
        self, mock_video_config, basic_subtitle_settings
    ):
        """Test that randomization is properly disabled for static subtitle."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "upper.ass"

            # Set randomize_effects to False explicitly
            settings = basic_subtitle_settings.copy()
            settings["randomize_effects"] = False

            result = create_static_upper_subtitle(
                text="Static text",
                output_path=output_path,
                subtitle_settings=settings,
                video_config=mock_video_config,
                format_type="ass",
                voiceover_duration=15.0,
                product_id="B0TEST123",  # Provided but shouldn't affect randomization
            )

            assert result is not None
            # Should succeed without randomization

    def test_missing_flat_config_uses_defaults(self, mock_video_config):
        """Test that missing flat config keys use appropriate defaults."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "upper.ass"

            # Minimal settings without two-part config
            minimal_settings = {
                "subtitle_format": "ass",
                "font_size_percent": 0.03,
                "margin": 0.05,
            }

            result = create_static_upper_subtitle(
                text="Test",
                output_path=output_path,
                subtitle_settings=minimal_settings,
                video_config=mock_video_config,
                format_type="ass",
                voiceover_duration=10.0,
            )

            # Should use defaults and not fail
            assert result is not None
            assert result.exists()

    def test_visual_bounds_affects_positioning(
        self, mock_video_config, basic_subtitle_settings
    ):
        """Test that visual_bounds affects positioning differently than None."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # With visual_bounds
            output_with = Path(temp_dir) / "with_bounds.ass"
            visual_bounds = VisualBounds(x=0.1, y=0.2, width=0.8, height=0.6)

            result_with = create_static_upper_subtitle(
                text="Test",
                output_path=output_with,
                subtitle_settings=basic_subtitle_settings,
                video_config=mock_video_config,
                format_type="ass",
                voiceover_duration=10.0,
                visual_bounds=visual_bounds,
            )

            # Without visual_bounds
            output_without = Path(temp_dir) / "without_bounds.ass"

            result_without = create_static_upper_subtitle(
                text="Test",
                output_path=output_without,
                subtitle_settings=basic_subtitle_settings,
                video_config=mock_video_config,
                format_type="ass",
                voiceover_duration=10.0,
                visual_bounds=None,
            )

            # Both should succeed
            assert result_with is not None
            assert result_without is not None

            # Extract positions - they should be different
            import re

            content_with = output_with.read_text()
            content_without = output_without.read_text()

            pos_with = re.search(r"\\pos\((\d+),(\d+)\)", content_with)
            pos_without = re.search(r"\\pos\((\d+),(\d+)\)", content_without)

            assert pos_with is not None
            assert pos_without is not None

            y_with = int(pos_with.group(2))
            y_without = int(pos_without.group(2))

            # WITH visual_bounds: Y = (visual_bounds.y - margin) * height
            # = (0.2 - 0.005) * 1920 = 0.195 * 1920 = 374.4 pixels
            expected_y_with = int((0.2 - 0.005) * 1920)
            assert (
                abs(y_with - expected_y_with) < 10
            ), f"Y with bounds should be above content: {y_with} vs {expected_y_with}"

            # WITHOUT visual_bounds: falls back to margin, clamped by safe zone
            # min_y (0.141 * 1920 = 270px)
            expected_y_without = int(0.141 * 1920)
            assert (
                abs(y_without - expected_y_without) < 20
            ), f"Y without bounds should be at safe zone min: {y_without} vs {expected_y_without}"
