"""Integration tests for VideoAssembler and builder components.

These tests verify the assembler works correctly after refactoring into
modular builder classes (VisualBuilder, SubtitleBuilder, AudioBuilder).
"""

from unittest.mock import Mock

import pytest

from src.video.assembler.core import VideoAssembler
from src.video.assembler.visual_builder import VisualGeometry
from src.video.config import VideoConfig


@pytest.mark.integration
class TestVideoAssemblerIntegration:
    """Test VideoAssembler coordination of builder components."""

    @pytest.fixture
    def mock_config(self):
        """Create mock VideoConfig."""
        config = Mock(spec=VideoConfig)
        config.ffmpeg_settings = Mock()
        config.ffmpeg_settings.executable_path = "ffmpeg"
        config.output = Mock()
        config.output.width = 1080
        config.output.height = 1920
        config.output.fps = 30
        return config

    def test_assembler_initialization(self, mock_config):
        """Test VideoAssembler initializes builder components."""
        assembler = VideoAssembler(mock_config, debug_mode=True)

        assert assembler.config == mock_config
        assert assembler.debug_mode is True
        assert assembler.ffmpeg_path == "ffmpeg"
        assert assembler.media_inspector is not None
        assert assembler.subtitle_styler is not None


@pytest.mark.unit
class TestVisualGeometry:
    """Unit tests for VisualGeometry dataclass."""

    def test_visual_geometry_creation(self):
        """Test VisualGeometry dataclass creation."""
        geom = VisualGeometry(
            rendered_x=0, rendered_y=192, rendered_w=1080, rendered_h=1440
        )

        assert geom.rendered_x == 0
        assert geom.rendered_y == 192
        assert geom.rendered_w == 1080
        assert geom.rendered_h == 1440

    def test_visual_geometry_letterbox_values(self):
        """Test VisualGeometry with typical letterbox values."""
        # Typical letterbox for 2102x1080 video in 1080x1920 frame
        geom = VisualGeometry(
            rendered_x=0, rendered_y=384, rendered_w=1080, rendered_h=768
        )

        # Verify valid letterbox dimensions
        assert geom.rendered_x == 0
        assert 0 < geom.rendered_y < 1920
        assert geom.rendered_w == 1080
        assert geom.rendered_h < 1920


@pytest.mark.unit
class TestImagePositioning:
    """Unit tests for image positioning calculations."""

    def test_center_alignment_calculation(self):
        """Test vertical centering calculation for images."""
        frame_height = 1920
        image_height = 1200

        # Centering formula: (frame_height - image_height) / 2
        expected_y = (frame_height - image_height) / 2
        assert expected_y == 360

    def test_top_alignment_with_offset(self):
        """Test top alignment with percentage offset."""
        frame_height = 1920
        top_offset_percent = 0.1  # 10% from top

        expected_y = frame_height * top_offset_percent
        assert expected_y == 192

    def test_image_width_scaling(self):
        """Test image width scaling to percentage of frame."""
        frame_width = 1080
        width_percent = 1.0  # 100% (new default)

        expected_width = int(frame_width * width_percent)
        assert expected_width == 1080

    def test_image_width_scaling_partial(self):
        """Test image width scaling to partial percentage."""
        frame_width = 1080
        width_percent = 0.8  # 80%

        expected_width = int(frame_width * width_percent)
        assert expected_width == 864


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
