"""Tests for unified subtitle positioning system."""

from typing import Any

from src.video.subtitle_positioning import (
    Position,
    PositionAnchor,
    StylePreset,
    UnifiedSubtitleConfig,
    VisualBounds,
    calculate_position,
    convert_legacy_config,
    get_font_size,
    get_style_config,
)


class TestUnifiedSubtitleConfig:
    """Test UnifiedSubtitleConfig model."""

    def test_unified_subtitle_config_defaults(self):
        """Test unified subtitle config with default values."""
        config = UnifiedSubtitleConfig()

        assert config.anchor == PositionAnchor.BOTTOM
        assert config.margin == 0.1
        assert config.content_aware is True
        assert config.style_preset == StylePreset.MODERN
        assert config.font_size_scale == 1.0
        assert config.max_line_length == 38
        assert config.max_duration == 4.5
        assert config.min_duration == 0.4
        assert config.randomize_colors is False
        assert config.randomize_effects is False
        assert config.custom_position is None
        assert config.horizontal_alignment == "center"

    def test_unified_subtitle_config_custom_values(self):
        """Test unified subtitle config with custom values."""
        config = UnifiedSubtitleConfig(
            anchor=PositionAnchor.TOP,
            margin=0.2,
            content_aware=False,
            style_preset=StylePreset.BOLD,
            font_size_scale=1.5,
            max_line_length=50,
            max_duration=5.0,
            min_duration=0.3,
            randomize_colors=True,
            randomize_effects=True,
            custom_position=Position(x=0.5, y=0.8),
            horizontal_alignment="left",
        )

        assert config.anchor == PositionAnchor.TOP
        assert config.margin == 0.2
        assert config.content_aware is False
        assert config.style_preset == StylePreset.BOLD
        assert config.font_size_scale == 1.5
        assert config.max_line_length == 50
        assert config.max_duration == 5.0
        assert config.min_duration == 0.3
        assert config.randomize_colors is True
        assert config.randomize_effects is True
        assert config.custom_position is not None
        assert config.custom_position.x == 0.5
        assert config.custom_position.y == 0.8
        assert config.horizontal_alignment == "left"


class TestPositionCalculation:
    """Test subtitle position calculation."""

    def test_calculate_position_bottom_anchor(self):
        """Test position calculation with bottom anchor."""
        config = UnifiedSubtitleConfig(
            anchor=PositionAnchor.BOTTOM, margin=0.1, content_aware=False
        )
        frame_size = (1080, 1920)

        position = calculate_position(config, frame_size)

        assert position.x == 0.5  # Center horizontally
        assert position.y == 0.9  # 1.0 - 0.1 margin

    def test_calculate_position_top_anchor(self):
        """Test position calculation with top anchor."""
        config = UnifiedSubtitleConfig(
            anchor=PositionAnchor.TOP, margin=0.1, content_aware=False
        )
        frame_size = (1080, 1920)

        position = calculate_position(config, frame_size)

        assert position.x == 0.5  # Center horizontally
        assert position.y == 0.1  # Top with margin

    def test_calculate_position_center_anchor(self):
        """Test position calculation with center anchor."""
        config = UnifiedSubtitleConfig(
            anchor=PositionAnchor.CENTER, margin=0.1, content_aware=False
        )
        frame_size = (1080, 1920)

        position = calculate_position(config, frame_size)

        assert position.x == 0.5  # Center horizontally
        assert position.y == 0.5  # Center vertically

    def test_calculate_position_below_content(self):
        """Test position calculation with below content anchor."""
        config = UnifiedSubtitleConfig(
            anchor=PositionAnchor.BELOW_CONTENT, margin=0.1, content_aware=True
        )
        frame_size = (1080, 1920)
        visual_bounds = VisualBounds(x=0.1, y=0.2, width=0.8, height=0.4)

        position = calculate_position(config, frame_size, visual_bounds)

        assert position.x == 0.5  # Center horizontally
        assert (
            abs(position.y - 0.7) < 0.001
        )  # Below content: 0.2 + 0.4 + 0.1 = 0.7 (allow for float precision)

    def test_calculate_position_above_content(self):
        """Test position calculation with above content anchor."""
        config = UnifiedSubtitleConfig(
            anchor=PositionAnchor.ABOVE_CONTENT, margin=0.1, content_aware=True
        )
        frame_size = (1080, 1920)
        visual_bounds = VisualBounds(x=0.1, y=0.3, width=0.8, height=0.4)

        position = calculate_position(config, frame_size, visual_bounds)

        assert position.x == 0.5  # Center horizontally
        assert (
            abs(position.y - 0.2) < 0.001
        )  # Above content: 0.3 - 0.1 = 0.2 (allow for float precision)

    def test_calculate_position_custom_position_override(self):
        """Test that custom position overrides calculated position."""
        config = UnifiedSubtitleConfig(
            anchor=PositionAnchor.BOTTOM,
            margin=0.1,
            content_aware=False,
            custom_position=Position(x=0.3, y=0.7),
        )
        frame_size = (1080, 1920)

        position = calculate_position(config, frame_size)

        assert position.x == 0.3  # Custom x position
        assert position.y == 0.7  # Custom y position

    def test_calculate_position_horizontal_alignment(self):
        """Test different horizontal alignments."""
        frame_size = (1080, 1920)

        # Left alignment
        config_left = UnifiedSubtitleConfig(horizontal_alignment="left")
        position_left = calculate_position(config_left, frame_size)
        assert position_left.x == 0.1

        # Right alignment
        config_right = UnifiedSubtitleConfig(horizontal_alignment="right")
        position_right = calculate_position(config_right, frame_size)
        assert position_right.x == 0.9

        # Center alignment (default)
        config_center = UnifiedSubtitleConfig(horizontal_alignment="center")
        position_center = calculate_position(config_center, frame_size)
        assert position_center.x == 0.5


class TestFontSizeCalculation:
    """Test font size calculation."""

    def test_get_font_size_default(self):
        """Test default font size calculation."""
        config = UnifiedSubtitleConfig()
        frame_height = 1920

        font_size = get_font_size(config, frame_height)

        expected = int(1920 * 0.04 * 1.0)  # frame_height * base_percent * scale
        assert font_size == expected

    def test_get_font_size_scaled(self):
        """Test scaled font size calculation."""
        config = UnifiedSubtitleConfig(
            font_size_scale=1.2
        )  # Use smaller scale to avoid bounds
        frame_height = 1920

        font_size = get_font_size(config, frame_height)

        expected = int(1920 * 0.04 * 1.2)  # frame_height * base_percent * scale
        # Allow for rounding differences in calculations
        assert abs(font_size - expected) <= 1

    def test_get_font_size_bounds(self):
        """Test font size bounds enforcement."""
        config = UnifiedSubtitleConfig(font_size_scale=0.1)  # Very small
        frame_height = 100  # Small frame

        font_size = get_font_size(config, frame_height)

        assert font_size >= 16  # Minimum bound

        config_large = UnifiedSubtitleConfig(font_size_scale=10.0)  # Very large
        frame_height = 1920

        font_size_large = get_font_size(config_large, frame_height)

        assert font_size_large <= 100  # Maximum bound


class TestStyleConfigs:
    """Test style preset configurations."""

    def test_get_style_config_minimal(self):
        """Test minimal style preset."""
        style = get_style_config(StylePreset.MINIMAL)

        assert style["font_name"] == "Arial"
        assert style["font_color"] == "&H00FFFFFF"
        assert style["bold"] is False
        assert style["effects"] == []

    def test_get_style_config_modern(self):
        """Test modern style preset."""
        style = get_style_config(StylePreset.MODERN)

        assert style["font_name"] == "Montserrat"
        assert style["bold"] is True
        assert "fade" in style["effects"]

    def test_get_style_config_dynamic(self):
        """Test dynamic style preset."""
        style = get_style_config(StylePreset.DYNAMIC)

        assert style["font_name"] == "Impact"
        assert style["bold"] is True
        assert "fade" in style["effects"]
        assert "scale_pulse" in style["effects"]
        assert "karaoke" in style["effects"]

    def test_get_style_config_classic(self):
        """Test classic style preset."""
        style = get_style_config(StylePreset.CLASSIC)

        assert style["font_name"] == "Times New Roman"
        assert style["bold"] is False
        assert style["effects"] == []

    def test_get_style_config_bold(self):
        """Test bold style preset."""
        style = get_style_config(StylePreset.BOLD)

        assert style["font_name"] == "Gabarito"
        assert style["bold"] is True
        assert "fade" in style["effects"]

    def test_get_style_config_invalid_preset(self):
        """Test handling of invalid style preset."""
        # Should fallback to modern preset
        style = get_style_config("invalid_preset")  # type: ignore[arg-type]

        # Should match modern preset
        modern_style = get_style_config(StylePreset.MODERN)
        assert style == modern_style


class TestLegacyConfigConversion:
    """Test legacy configuration conversion."""

    def test_convert_legacy_static_mode(self):
        """Test conversion of legacy static positioning mode."""
        legacy_settings = {
            "positioning_mode": "static",
            "margin": 0.2,
            "font_size_scale": 1.2,
        }

        unified_config = convert_legacy_config(legacy_settings)

        assert unified_config.anchor == PositionAnchor.BOTTOM
        assert unified_config.content_aware is False
        assert unified_config.margin == 0.2
        assert unified_config.font_size_scale == 1.2

    def test_convert_legacy_dynamic_mode(self):
        """Test conversion of legacy dynamic positioning mode."""
        legacy_settings = {
            "positioning_mode": "dynamic",
            "margin": 0.15,
        }

        unified_config = convert_legacy_config(legacy_settings)

        assert unified_config.anchor == PositionAnchor.BELOW_CONTENT
        assert unified_config.content_aware is True
        assert unified_config.margin == 0.15

    def test_convert_legacy_absolute_mode(self):
        """Test conversion of legacy absolute positioning mode."""
        legacy_settings = {
            "positioning_mode": "absolute",
            "absolute_positioning": {"x_pos": "(w-tw)/2", "y_pos": "h*0.8"},
        }

        unified_config = convert_legacy_config(legacy_settings)

        assert unified_config.anchor == PositionAnchor.BOTTOM
        assert unified_config.content_aware is False
        assert unified_config.custom_position is not None

    def test_convert_legacy_with_unified_params(self):
        """Test conversion when unified parameters are already present."""
        legacy_settings = {
            "positioning_mode": "static",  # Legacy param
            "anchor": "below_content",  # New unified param
            "content_aware": True,  # New unified param
            "style_preset": "dynamic",  # New unified param
        }

        unified_config = convert_legacy_config(legacy_settings)

        # Should use unified params over legacy mode
        assert unified_config.anchor == PositionAnchor.BELOW_CONTENT
        assert unified_config.content_aware is True
        assert unified_config.style_preset == StylePreset.DYNAMIC

    def test_convert_legacy_style_detection(self):
        """Test automatic style preset detection from legacy settings."""
        # Test ASS with transforms -> dynamic
        legacy_settings = {"subtitle_format": "ass", "ass_enable_transforms": True}
        config = convert_legacy_config(legacy_settings)
        assert config.style_preset == StylePreset.DYNAMIC

        # Test bold -> bold preset
        legacy_settings = {"bold": True}
        config = convert_legacy_config(legacy_settings)
        assert config.style_preset == StylePreset.BOLD

        # Test SRT -> classic
        legacy_settings = {"subtitle_format": "srt"}
        config = convert_legacy_config(legacy_settings)
        assert config.style_preset == StylePreset.CLASSIC

    def test_convert_legacy_defaults(self):
        """Test conversion with default legacy settings."""
        legacy_settings: dict[str, Any] = {}

        unified_config = convert_legacy_config(legacy_settings)

        # Should use sensible defaults (static mode when no positioning_mode specified)
        assert unified_config.anchor == PositionAnchor.BOTTOM
        assert unified_config.content_aware is False  # Static mode default
        assert (
            unified_config.style_preset == StylePreset.CLASSIC
        )  # Default for legacy without style info
        assert unified_config.margin == 0.1  # Default from UnifiedSubtitleConfig
        assert (
            unified_config.font_size_scale == 1.0
        )  # Default from UnifiedSubtitleConfig


class TestVisualBounds:
    """Test VisualBounds dataclass."""

    def test_visual_bounds_creation(self):
        """Test VisualBounds creation."""
        bounds = VisualBounds(x=0.1, y=0.2, width=0.8, height=0.6)

        assert bounds.x == 0.1
        assert bounds.y == 0.2
        assert bounds.width == 0.8
        assert bounds.height == 0.6

    def test_visual_bounds_content_aware_positioning(self):
        """Test that visual bounds are correctly used in positioning."""
        config = UnifiedSubtitleConfig(
            anchor=PositionAnchor.BELOW_CONTENT, margin=0.05, content_aware=True
        )
        frame_size = (1080, 1920)

        # Content in upper portion of frame
        visual_bounds = VisualBounds(x=0.1, y=0.1, width=0.8, height=0.3)

        position = calculate_position(config, frame_size, visual_bounds)

        # Should be positioned below content: 0.1 + 0.3 + 0.05 = 0.45
        assert position.y == 0.45
        assert position.x == 0.5  # Centered horizontally


class TestPosition:
    """Test Position dataclass."""

    def test_position_creation(self):
        """Test Position creation."""
        pos = Position(x=0.3, y=0.7)

        assert pos.x == 0.3
        assert pos.y == 0.7

    def test_position_boundary_values(self):
        """Test Position with boundary values."""
        # Test 0.0 values
        pos_zero = Position(x=0.0, y=0.0)
        assert pos_zero.x == 0.0
        assert pos_zero.y == 0.0

        # Test 1.0 values
        pos_one = Position(x=1.0, y=1.0)
        assert pos_one.x == 1.0
        assert pos_one.y == 1.0
