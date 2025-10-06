"""Test suite for UnifiedSubtitleGenerator."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.video.subtitle_positioning import (
    PositionAnchor,
    StylePreset,
    UnifiedSubtitleConfig,
    VisualBounds,
)
from src.video.unified_subtitle_generator import UnifiedSubtitleGenerator


class TestUnifiedSubtitleGenerator:
    """Test cases for UnifiedSubtitleGenerator class."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample unified subtitle configuration."""
        return UnifiedSubtitleConfig(
            anchor=PositionAnchor.BOTTOM,
            margin=0.1,
            content_aware=False,
            style_preset=StylePreset.MODERN,
            max_line_length=30,
            max_words_per_line=3,
            max_subtitle_width_fraction=0.67,
            min_duration=1.0,
            max_duration=6.0,
            randomize_colors=False,
            randomize_effects=False,
        )

    @pytest.fixture
    def generator(self, sample_config):
        """Create a UnifiedSubtitleGenerator instance."""
        frame_size = (1920, 1080)
        return UnifiedSubtitleGenerator(sample_config, frame_size)

    @pytest.fixture
    def sample_timings(self):
        """Create sample timing data."""
        return [
            {"word": "Hello", "start_time": 0.0, "end_time": 0.5},
            {"word": "world", "start_time": 0.6, "end_time": 1.0},
            {"word": "test", "start_time": 1.1, "end_time": 1.5},
        ]

    def test_init(self, sample_config):
        """Test UnifiedSubtitleGenerator initialization."""
        frame_size = (1920, 1080)
        generator = UnifiedSubtitleGenerator(sample_config, frame_size)

        assert generator.config == sample_config
        assert generator.frame_size == frame_size
        assert generator.style_config is not None
        assert generator._selected_colors is not None

    def test_estimate_text_width_pixels(self, generator):
        """Test text width estimation."""
        text = "Hello world"
        font_size = 24

        width = generator.estimate_text_width_pixels(text, font_size)

        assert isinstance(width, int)
        assert width > 0

    def test_estimate_text_width_pixels_narrow_chars(self, generator):
        """Test text width estimation with narrow characters."""
        narrow_text = "iiil"
        normal_text = "MMMM"
        font_size = 24

        narrow_width = generator.estimate_text_width_pixels(narrow_text, font_size)
        normal_width = generator.estimate_text_width_pixels(normal_text, font_size)

        assert narrow_width < normal_width

    def test_format_ass_time(self, generator):
        """Test ASS timestamp formatting."""
        # Test various time values
        assert generator._format_ass_time(0.0) == "0:00:00.00"
        assert generator._format_ass_time(61.5) == "0:01:01.50"
        assert generator._format_ass_time(3661.25) == "1:01:01.25"

    def test_format_ass_time_precision(self, generator):
        """Test ASS timestamp formatting precision."""
        # Test centisecond precision
        assert generator._format_ass_time(1.234) == "0:00:01.23"
        assert generator._format_ass_time(1.789) == "0:00:01.78"

    def test_generate_from_timings_success(self, generator, sample_timings):
        """Test successful subtitle generation from timings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.srt"

            result = generator.generate_from_timings(
                timings=sample_timings,
                output_path=output_path,
                format_type="srt",
                voiceover_duration=2.0,
            )

            assert result.success is True
            assert result.path == output_path
            assert result.format == "srt"
            assert result.segments_created > 0

    def test_generate_from_timings_empty_timings(self, generator):
        """Test generation with empty timings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.srt"

            result = generator.generate_from_timings(
                timings=[],
                output_path=output_path,
                format_type="srt",
                voiceover_duration=2.0,
            )

            assert result.success is False
            assert "No timing data provided" in str(result.errors)

    def test_generate_from_script_success(self, generator):
        """Test successful subtitle generation from script."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.srt"
            script_text = "Hello world. This is a test script."

            result = generator.generate_from_script(
                script_text=script_text,
                output_path=output_path,
                format_type="srt",
                duration=5.0,
            )

            assert result.success is True
            assert result.path == output_path
            assert result.format == "srt"

    def test_generate_from_script_empty_text(self, generator):
        """Test generation with empty script text."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.srt"

            result = generator.generate_from_script(
                script_text="",
                output_path=output_path,
                format_type="srt",
                duration=5.0,
            )

            assert result.success is False

    def test_generate_ass_format(self, generator, sample_timings):
        """Test ASS format generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.ass"

            result = generator.generate_from_timings(
                timings=sample_timings,
                output_path=output_path,
                format_type="ass",
                voiceover_duration=2.0,
            )

            assert result.success is True
            assert result.format == "ass"
            assert output_path.exists()

            # Verify ASS content structure
            content = output_path.read_text()
            assert "[Script Info]" in content
            assert "[V4+ Styles]" in content
            assert "[Events]" in content

    def test_color_selection_consistency(self, sample_config):
        """Test that colors are selected consistently per instance."""
        frame_size = (1920, 1080)
        generator = UnifiedSubtitleGenerator(sample_config, frame_size)

        colors1 = generator._get_colors()
        colors2 = generator._get_colors()

        assert colors1 == colors2

    @patch("src.video.unified_subtitle_generator.get_style_config")
    def test_color_randomization_uses_style_config(self, mock_get_style):
        """Test that colors come from style_config, not legacy randomization.

        This test verifies the fix for the double-randomization bug where
        _select_colors() was overwriting RandomizationEngine colors.
        """
        # Mock get_style_config to return specific randomized colors
        mock_get_style.return_value = {
            "font_name": "Rubik-Bold",
            "font_color": "&H000080FF",  # Orange
            "outline_color": "&H00008000",  # Dark green (WARM pair)
            "bold": True,
            "outline_thickness": 2,
            "shadow": True,
            "effects": [],
            "font_width_to_height_ratio": 0.5,
        }

        config = UnifiedSubtitleConfig(
            anchor=PositionAnchor.BOTTOM,
            margin=0.1,
            content_aware=False,
            style_preset=StylePreset.MODERN,
            randomize_colors=True,
        )
        frame_size = (1920, 1080)

        generator = UnifiedSubtitleGenerator(config, frame_size)
        colors = generator._get_colors()

        # Colors should match what get_style_config returned (not be overwritten)
        assert colors["primary"] == "&H000080FF"  # Orange
        assert colors["outline"] == "&H00008000"  # Dark green
        # NOT black (&H00000000) from old legacy randomization

    def test_select_colors_preserves_style_config(self):
        """Test that _select_colors() uses colors from style_config without overwriting.

        Regression test for bug where _select_colors() had legacy randomization
        that overwrote RandomizationEngine colors with hardcoded black outlines.
        """
        # Create config with specific style colors
        config = UnifiedSubtitleConfig(
            anchor=PositionAnchor.BOTTOM,
            margin=0.1,
            content_aware=False,
            style_preset=StylePreset.BOLD,  # Has specific colors
            randomize_colors=False,  # Disabled to test style_config preservation
        )
        frame_size = (1920, 1080)

        generator = UnifiedSubtitleGenerator(config, frame_size)

        # _select_colors() should return colors from style_config
        # The BOLD preset has specific outline color
        colors = generator._select_colors()

        assert "primary" in colors
        assert "outline" in colors
        # Colors should come from style_config, not be random
        # (Actual values depend on BOLD preset in config)
        assert colors["primary"] == generator.style_config["font_color"]
        assert colors["outline"] == generator.style_config["outline_color"]

    @patch("src.video.unified_subtitle_generator.pysrt")
    def test_generate_srt_error_handling(self, mock_pysrt, generator, sample_timings):
        """Test SRT generation error handling."""
        mock_pysrt.SubRipFile.side_effect = Exception("Test error")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.srt"

            result = generator.generate_from_timings(
                timings=sample_timings,
                output_path=output_path,
                format_type="srt",
                voiceover_duration=2.0,
            )

            assert result.success is False

    def test_create_segments_from_script(self, generator):
        """Test segment creation from script text."""
        script_text = "Hello world. This is a test. Another sentence here."
        duration = 6.0
        visual_bounds = None

        segments = generator._create_script_segments(
            script_text, duration, visual_bounds
        )

        assert len(segments) > 0
        assert all("text" in seg for seg in segments)
        assert all("start" in seg for seg in segments)
        assert all("end" in seg for seg in segments)

        # Check timing progression
        for i in range(1, len(segments)):
            assert segments[i]["start"] >= segments[i - 1]["end"]

    def test_visual_bounds_integration(self, generator, sample_timings):
        """Test generation with visual bounds."""
        visual_bounds = VisualBounds(x=0.1, y=0.1, width=0.8, height=0.6)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.ass"

            result = generator.generate_from_timings(
                timings=sample_timings,
                output_path=output_path,
                format_type="ass",
                voiceover_duration=2.0,
                visual_bounds=visual_bounds,
            )

            assert result.success is True

    def test_debug_mode(self, generator, sample_timings):
        """Test generation with debug mode enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.srt"

            result = generator.generate_from_timings(
                timings=sample_timings,
                output_path=output_path,
                format_type="srt",
                voiceover_duration=2.0,
                debug_mode=True,
            )

            assert result.success is True

    def test_word_count_limit(self):
        """Test subtitle segmentation with word count limit."""
        config = UnifiedSubtitleConfig(
            anchor=PositionAnchor.BOTTOM,
            margin=0.1,
            content_aware=False,
            style_preset=StylePreset.MODERN,
            max_line_length=100,  # High limit to test word count
            max_words_per_line=3,  # Strict word limit
            max_subtitle_width_fraction=0.67,
        )
        generator = UnifiedSubtitleGenerator(config, (1920, 1080))

        # Script with multiple words
        script_text = "This is a long sentence with many words in it"
        segments = generator._create_script_segments(
            script_text, duration=10.0, visual_bounds=None
        )

        # Check that segments respect word limit
        for seg in segments:
            word_count = len(seg["text"].split())
            assert word_count <= 3, f"Segment has {word_count} words: {seg['text']}"

    def test_width_constraint(self):
        """Test subtitle width constraint based on frame width."""
        config = UnifiedSubtitleConfig(
            anchor=PositionAnchor.BOTTOM,
            margin=0.1,
            content_aware=False,
            style_preset=StylePreset.MODERN,
            max_line_length=100,  # High limit
            max_words_per_line=0,  # Disabled
            max_subtitle_width_fraction=0.67,  # 2/3 of frame
        )
        frame_size = (1080, 1920)  # Width, Height
        generator = UnifiedSubtitleGenerator(config, frame_size)

        # Test that width calculation uses frame-based constraint
        max_width = int(frame_size[0] * 0.67)
        assert max_width == 723  # 1080 * 0.67 = 723.6 -> 723

        # Long text that should be broken
        long_text = "A" * 50
        script_text = f"{long_text} {long_text} {long_text}"
        segments = generator._create_script_segments(
            script_text, duration=10.0, visual_bounds=None
        )

        # Verify segments were created (text was broken up)
        assert len(segments) > 1
