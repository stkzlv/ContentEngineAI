import pytest
from pathlib import Path
from unittest.mock import patch, mock_open
from src.video.subtitle_positioning import (
    UnifiedSubtitleConfig,
    PositionAnchor,
    VisualBounds,
    calculate_position,
    get_font_size,
    get_style_config,
    create_unified_config_from_settings,
    StylePreset
)

class TestSubtitlePositioning:
    @pytest.mark.parametrize("anchor,expected_y", [
        (PositionAnchor.TOP, 0.1),
        (PositionAnchor.CENTER, 0.5),
        (PositionAnchor.BOTTOM, 0.9),
    ])
    def test_calculate_position_basic_anchors(self, anchor, expected_y):
        config = UnifiedSubtitleConfig(anchor=anchor, margin=0.1, content_aware=False)
        pos = calculate_position(config, (1080, 1920))
        assert pos.y == pytest.approx(expected_y)
        assert pos.x == 0.5

    def test_calculate_position_above_content(self):
        config = UnifiedSubtitleConfig(anchor=PositionAnchor.ABOVE_CONTENT, margin=0.1, content_aware=True)
        visual_bounds = VisualBounds(x=0.1, y=0.2, width=0.8, height=0.6)
        pos = calculate_position(config, (1080, 1920), visual_bounds)
        assert pos.y == 0.1

    def test_calculate_position_below_content(self):
        config = UnifiedSubtitleConfig(anchor=PositionAnchor.BELOW_CONTENT, margin=0.05, content_aware=True)
        visual_bounds = VisualBounds(x=0.1, y=0.2, width=0.8, height=0.6)
        pos = calculate_position(config, (1080, 1920), visual_bounds)
        assert pos.y == pytest.approx(0.85) # 0.2 + 0.6 + 0.05

    def test_calculate_position_below_content_clamped(self):
        # We need to mock the yaml load to ensure max_safe_y_position is what we expect
        mock_yaml = "text_rendering:\n  max_safe_y_position: 0.95"
        with patch("pathlib.Path.exists", return_value=True), \
             patch("builtins.open", mock_open(read_data=mock_yaml)):
            config = UnifiedSubtitleConfig(anchor=PositionAnchor.BELOW_CONTENT, margin=0.2, content_aware=True)
            visual_bounds = VisualBounds(x=0.1, y=0.2, width=0.8, height=0.7)
            pos = calculate_position(config, (1080, 1920), visual_bounds)
            assert pos.y == pytest.approx(0.95)

    @pytest.mark.parametrize("alignment,expected_x", [
        ("left", 0.1),
        ("center", 0.5),
        ("right", 0.9),
    ])
    def test_horizontal_alignment(self, alignment, expected_x):
        config = UnifiedSubtitleConfig(horizontal_alignment=alignment)
        pos = calculate_position(config, (1080, 1920))
        assert pos.x == pytest.approx(expected_x)

    def test_get_font_size(self):
        # Default base_font_size_percent is 0.04
        # 1000 * 0.04 = 40.
        config = UnifiedSubtitleConfig(font_size_scale=1.0)
        size = get_font_size(config, 1000, base_size_percent=0.04)
        assert size == 40
        
        # 40 * 2.0 = 80. (within default 100 max)
        config_scaled = UnifiedSubtitleConfig(font_size_scale=2.0)
        size_scaled = get_font_size(config_scaled, 1000, base_size_percent=0.04)
        assert size_scaled == 80

    def test_get_style_config_fallback(self):
        # Test fallback when yaml missing
        with patch("pathlib.Path.exists", return_value=False):
            style = get_style_config(StylePreset.MINIMAL)
            assert style["font_name"] == "Arial"
            
            style_modern = get_style_config(StylePreset.MODERN)
            assert "karaoke" in style_modern["effects"]

    def test_get_style_config_yaml(self):
        mock_yaml = "style_presets:\n  minimal:\n    font_name: 'CustomFont'\n    effects: []"
        with patch("pathlib.Path.exists", return_value=True), \
             patch("builtins.open", mock_open(read_data=mock_yaml)):
            style = get_style_config(StylePreset.MINIMAL)
            assert style["font_name"] == "CustomFont"

    def test_get_style_config_random(self):
        config = UnifiedSubtitleConfig(style_preset=StylePreset.RANDOM)
        # Random preset forces randomization
        style = get_style_config(StylePreset.RANDOM, config, "prod123")
        assert len(style["effects"]) == 1
        assert config.randomize_fonts is True

    def test_create_unified_config_from_settings(self):
        settings = {
            "anchor": "top",
            "margin": 0.15,
            "style_preset": "bold",
            "font_size_scale": 1.2
        }
        config = create_unified_config_from_settings(settings)
        assert config.anchor == PositionAnchor.TOP
        assert config.margin == 0.15
        assert config.style_preset == "bold"
        assert config.font_size_scale == 1.2

    def test_create_unified_config_invalid_values(self):
        settings = {
            "anchor": "invalid",
            "style_preset": "garbage"
        }
        config = create_unified_config_from_settings(settings)
        assert config.anchor == PositionAnchor.BOTTOM # Fallback
        assert config.style_preset == "modern" # Fallback