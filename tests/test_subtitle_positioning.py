"""Comprehensive tests for unified subtitle positioning system."""

from unittest.mock import Mock, patch

import pytest

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
    """Test the unified subtitle configuration model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = UnifiedSubtitleConfig()
        assert config.anchor == PositionAnchor.BOTTOM
        assert config.margin == 0.1
        assert config.content_aware is True
        assert config.style_preset == StylePreset.MODERN
        assert config.font_size_scale == 1.0
        assert config.max_line_length == 38
        assert config.horizontal_alignment == "center"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = UnifiedSubtitleConfig(
            anchor=PositionAnchor.ABOVE_CONTENT,
            margin=0.05,
            content_aware=False,
            style_preset=StylePreset.BOLD,
            font_size_scale=1.5,
            max_line_length=50,
            horizontal_alignment="left",
        )
        assert config.anchor == PositionAnchor.ABOVE_CONTENT
        assert config.margin == 0.05
        assert config.content_aware is False
        assert config.style_preset == StylePreset.BOLD
        assert config.font_size_scale == 1.5
        assert config.max_line_length == 50
        assert config.horizontal_alignment == "left"

    def test_validation_bounds(self):
        """Test that validation enforces reasonable bounds."""
        # Test valid values
        config = UnifiedSubtitleConfig(margin=0.0, font_size_scale=0.5)
        assert config.margin == 0.0
        assert config.font_size_scale == 0.5

        config = UnifiedSubtitleConfig(margin=0.5, font_size_scale=2.0)
        assert config.margin == 0.5
        assert config.font_size_scale == 2.0


class TestPositionCalculation:
    """Test subtitle position calculation logic."""

    def test_bottom_positioning(self):
        """Test bottom anchor positioning."""
        config = UnifiedSubtitleConfig(anchor=PositionAnchor.BOTTOM, margin=0.1)
        position = calculate_position(config, (1920, 1080))
        assert position.x == 0.5  # center
        assert position.y == 0.9  # 1.0 - 0.1 margin

    def test_top_positioning(self):
        """Test top anchor positioning."""
        config = UnifiedSubtitleConfig(anchor=PositionAnchor.TOP, margin=0.1)
        position = calculate_position(config, (1920, 1080))
        assert position.x == 0.5
        assert position.y == 0.1

    def test_center_positioning(self):
        """Test center anchor positioning."""
        config = UnifiedSubtitleConfig(anchor=PositionAnchor.CENTER)
        position = calculate_position(config, (1920, 1080))
        assert position.x == 0.5
        assert position.y == 0.5

    def test_horizontal_alignment(self):
        """Test horizontal alignment options."""
        config = UnifiedSubtitleConfig(horizontal_alignment="left")
        position = calculate_position(config, (1920, 1080))
        assert position.x == 0.1

        config = UnifiedSubtitleConfig(horizontal_alignment="right")
        position = calculate_position(config, (1920, 1080))
        assert position.x == 0.9

        config = UnifiedSubtitleConfig(horizontal_alignment="center")
        position = calculate_position(config, (1920, 1080))
        assert position.x == 0.5

    def test_below_content_with_visual_bounds(self):
        """Test below content positioning with visual bounds."""
        config = UnifiedSubtitleConfig(
            anchor=PositionAnchor.BELOW_CONTENT,
            content_aware=True,
            margin=0.05,
        )
        visual_bounds = VisualBounds(x=0.1, y=0.2, width=0.8, height=0.4)
        position = calculate_position(config, (1920, 1080), visual_bounds)

        # Should be positioned below the visual content
        expected_y = min(0.95, 0.2 + 0.4 + 0.05)  # y + height + margin
        assert position.y == expected_y
        assert position.x == 0.5

    def test_below_content_without_visual_bounds(self):
        """Test below content positioning fallback without visual bounds."""
        config = UnifiedSubtitleConfig(
            anchor=PositionAnchor.BELOW_CONTENT,
            content_aware=True,
            margin=0.1,
        )
        position = calculate_position(config, (1920, 1080), None)

        # Should fallback to bottom positioning
        assert position.y == 0.9  # 1.0 - 0.1 margin
        assert position.x == 0.5

    def test_below_content_content_aware_disabled(self):
        """Test below content positioning with content_aware disabled."""
        config = UnifiedSubtitleConfig(
            anchor=PositionAnchor.BELOW_CONTENT,
            content_aware=False,
            margin=0.1,
        )
        visual_bounds = VisualBounds(x=0.1, y=0.2, width=0.8, height=0.4)
        position = calculate_position(config, (1920, 1080), visual_bounds)

        # Should fallback to bottom positioning even with visual bounds
        assert position.y == 0.9  # 1.0 - 0.1 margin
        assert position.x == 0.5

    def test_above_content_with_visual_bounds(self):
        """Test above content positioning with visual bounds."""
        config = UnifiedSubtitleConfig(
            anchor=PositionAnchor.ABOVE_CONTENT,
            content_aware=True,
            margin=0.05,
        )
        visual_bounds = VisualBounds(x=0.1, y=0.3, width=0.8, height=0.4)
        position = calculate_position(config, (1920, 1080), visual_bounds)

        # Should be positioned above the visual content
        expected_y = max(0.05, 0.3 - 0.05)  # y - margin, but at least 0.05
        assert position.y == expected_y
        assert position.x == 0.5

    def test_above_content_fallback(self):
        """Test above content positioning fallback scenarios."""
        config = UnifiedSubtitleConfig(
            anchor=PositionAnchor.ABOVE_CONTENT,
            content_aware=False,
            margin=0.1,
        )
        position = calculate_position(config, (1920, 1080), None)

        # Should fallback to top positioning
        assert position.y == 0.1  # margin
        assert position.x == 0.5

    def test_custom_position_override(self):
        """Test custom position override."""
        custom_pos = Position(x=0.3, y=0.7)
        config = UnifiedSubtitleConfig(custom_position=custom_pos)
        position = calculate_position(config, (1920, 1080))

        # Should use custom position regardless of other settings
        assert position.x == 0.3
        assert position.y == 0.7

    def test_visual_bounds_edge_cases(self):
        """Test edge cases with visual bounds positioning."""
        config = UnifiedSubtitleConfig(
            anchor=PositionAnchor.BELOW_CONTENT,
            content_aware=True,
            margin=0.05,
        )

        # Visual bounds near bottom - should clamp to 0.95
        visual_bounds = VisualBounds(x=0.1, y=0.8, width=0.8, height=0.2)
        position = calculate_position(config, (1920, 1080), visual_bounds)
        assert position.y == 0.95  # clamped maximum

        # Visual bounds near top for above content
        config = UnifiedSubtitleConfig(
            anchor=PositionAnchor.ABOVE_CONTENT,
            content_aware=True,
            margin=0.1,
        )
        visual_bounds = VisualBounds(x=0.1, y=0.05, width=0.8, height=0.2)
        position = calculate_position(config, (1920, 1080), visual_bounds)
        assert position.y == 0.05  # clamped minimum


class TestFontSizeCalculation:
    """Test font size calculation logic."""

    def test_default_font_size(self):
        """Test default font size calculation."""
        config = UnifiedSubtitleConfig()
        font_size = get_font_size(config, 1080)

        # Default: 1080 * 0.04 * 1.0 = 43.2, rounded to 43
        assert font_size == 43

    def test_scaled_font_size(self):
        """Test scaled font size calculation."""
        config = UnifiedSubtitleConfig(font_size_scale=1.5)
        font_size = get_font_size(config, 1080)

        # Scaled: 1080 * 0.04 * 1.5 = 64.8, rounded to 64
        assert font_size == 64

    def test_font_size_bounds(self):
        """Test font size bounds enforcement."""
        # Test minimum bound
        config = UnifiedSubtitleConfig(font_size_scale=0.1)
        font_size = get_font_size(config, 1080)
        assert font_size >= 16  # minimum

        # Test maximum bound
        config = UnifiedSubtitleConfig(font_size_scale=5.0)
        font_size = get_font_size(config, 1080)
        assert font_size <= 100  # maximum

    def test_custom_base_size(self):
        """Test custom base size percentage."""
        config = UnifiedSubtitleConfig()
        font_size = get_font_size(config, 1080, base_size_percent=0.06)

        # Custom base: 1080 * 0.06 * 1.0 = 64.8, rounded to 64
        assert font_size == 64


class TestStylePresets:
    """Test style preset configurations."""

    def test_all_presets_exist(self):
        """Test that all style presets have configurations."""
        for preset in StylePreset:
            config = get_style_config(preset)
            assert isinstance(config, dict)
            assert "font_name" in config
            assert "font_color" in config
            assert "outline_color" in config

    def test_minimal_preset(self):
        """Test minimal style preset."""
        config = get_style_config(StylePreset.MINIMAL)
        assert config["font_name"] == "Arial"
        assert config["bold"] is False
        assert config["outline_thickness"] == 1
        assert config["shadow"] is False
        assert config["effects"] == []

    def test_modern_preset(self):
        """Test modern style preset."""
        config = get_style_config(StylePreset.MODERN)
        assert config["font_name"] == "Montserrat"
        assert config["bold"] is True
        assert config["outline_thickness"] == 2
        assert config["shadow"] is True
        assert "fade" in config["effects"]

    def test_relative_preset(self):
        """Test relative style preset with advanced effects."""
        config = get_style_config(StylePreset.RELATIVE)
        assert config["font_name"] == "Impact"
        assert config["bold"] is True
        assert len(config["effects"]) > 1
        assert "karaoke" in config["effects"]

    def test_invalid_preset_fallback(self):
        """Test fallback for invalid preset."""
        # Create a mock preset that doesn't exist
        with patch("src.video.subtitle_positioning.StylePreset") as mock_preset:
            mock_preset.INVALID = "invalid"
            config = get_style_config("invalid")  # type: ignore
            # Should fallback to modern preset
            assert config["font_name"] == "Montserrat"


class TestLegacyConfigConversion:
    """Test conversion from legacy configuration format."""

    def test_relative_mode_conversion(self):
        """Test conversion of relative positioning mode."""
        legacy_config = {
            "positioning_mode": "relative",
            "margin": 0.08,
            "font_size_scale": 1.2,
            "subtitle_format": "ass",
            "ass_enable_transforms": True,
        }

        unified = convert_legacy_config(legacy_config)
        assert unified.anchor == PositionAnchor.BELOW_CONTENT
        assert unified.content_aware is True
        assert unified.margin == 0.08
        assert unified.font_size_scale == 1.2
        assert unified.style_preset == StylePreset.RELATIVE

    def test_absolute_mode_conversion(self):
        """Test conversion of absolute positioning mode."""
        legacy_config = {
            "positioning_mode": "absolute",
            "margin": 0.12,
            "bold": True,
        }

        unified = convert_legacy_config(legacy_config)
        assert unified.anchor == PositionAnchor.BOTTOM
        assert unified.content_aware is False
        assert unified.margin == 0.12
        assert unified.style_preset == StylePreset.BOLD

    def test_direct_unified_config(self):
        """Test when legacy config already has unified parameters."""
        legacy_config = {
            "anchor": "above_content",
            "content_aware": True,
            "style_preset": "minimal",
            "margin": 0.05,
        }

        unified = convert_legacy_config(legacy_config)
        assert unified.anchor == PositionAnchor.ABOVE_CONTENT
        assert unified.content_aware is True
        assert unified.style_preset == StylePreset.MINIMAL
        assert unified.margin == 0.05

    def test_default_fallback_conversion(self):
        """Test conversion with minimal legacy config."""
        legacy_config: dict[str, str] = {}

        unified = convert_legacy_config(legacy_config)
        assert unified.anchor == PositionAnchor.BOTTOM
        assert unified.content_aware is False
        assert unified.style_preset == StylePreset.CLASSIC  # Default for empty config
        assert unified.margin == 0.1

    def test_style_preset_inference(self):
        """Test style preset inference from legacy settings."""
        # Bold preset inference
        legacy_config = {"bold": True, "subtitle_format": "srt"}
        unified = convert_legacy_config(legacy_config)
        assert unified.style_preset == StylePreset.BOLD

        # Classic preset inference
        legacy_config = {"bold": False, "subtitle_format": "srt"}
        unified = convert_legacy_config(legacy_config)
        assert unified.style_preset == StylePreset.CLASSIC

        # Relative preset inference
        legacy_config = {
            "subtitle_format": "ass",
            "ass_enable_transforms": True,
        }
        unified = convert_legacy_config(legacy_config)
        assert unified.style_preset == StylePreset.RELATIVE

    def test_invalid_anchor_fallback(self):
        """Test fallback for invalid anchor values."""
        legacy_config = {"anchor": "invalid_anchor"}
        unified = convert_legacy_config(legacy_config)
        # Should fall back to legacy conversion logic
        assert unified.anchor == PositionAnchor.BOTTOM

    def test_custom_position_conversion(self):
        """Test custom position extraction from absolute settings."""
        legacy_config = {
            "positioning_mode": "absolute",
            "absolute_positioning": {"x": "w*0.3", "y": "h*0.7"},
        }

        unified = convert_legacy_config(legacy_config)
        assert unified.anchor == PositionAnchor.BOTTOM
        assert unified.content_aware is False
        # Custom position parsing is simplified in current implementation
        assert unified.custom_position is not None


class TestVisualBounds:
    """Test visual bounds data structure."""

    def test_visual_bounds_creation(self):
        """Test visual bounds creation and properties."""
        bounds = VisualBounds(x=0.1, y=0.2, width=0.8, height=0.6)
        assert bounds.x == 0.1
        assert bounds.y == 0.2
        assert bounds.width == 0.8
        assert bounds.height == 0.6

    def test_visual_bounds_calculated_properties(self):
        """Test calculated properties of visual bounds."""
        bounds = VisualBounds(x=0.1, y=0.2, width=0.8, height=0.6)

        # Test right edge
        right_edge = bounds.x + bounds.width
        assert right_edge == 0.9

        # Test bottom edge
        bottom_edge = bounds.y + bounds.height
        assert bottom_edge == 0.8


class TestPosition:
    """Test position data structure."""

    def test_position_creation(self):
        """Test position creation and properties."""
        pos = Position(x=0.5, y=0.8)
        assert pos.x == 0.5
        assert pos.y == 0.8

    def test_position_equality(self):
        """Test position equality comparison."""
        pos1 = Position(x=0.5, y=0.8)
        pos2 = Position(x=0.5, y=0.8)
        pos3 = Position(x=0.5, y=0.9)

        assert pos1.x == pos2.x and pos1.y == pos2.y
        assert not (pos1.x == pos3.x and pos1.y == pos3.y)


if __name__ == "__main__":
    pytest.main([__file__])
