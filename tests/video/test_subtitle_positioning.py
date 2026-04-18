from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from src.video.config.subtitle_models import PlatformSafeZone, SubtitleSettings
from src.video.subtitle_positioning import (
    PositionAnchor,
    StylePreset,
    VisualBounds,
    calculate_position,
    clamp_to_safe_zone,
    get_font_size,
    get_style_config,
)


class TestSubtitlePositioning:
    # Use a permissive safe zone for tests that check raw anchor behavior
    _wide_sz = PlatformSafeZone(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)

    @pytest.mark.parametrize(
        "anchor,expected_y",
        [
            (PositionAnchor.TOP, 0.1),
            (PositionAnchor.CENTER, 0.5),
            (PositionAnchor.BOTTOM, 0.9),
        ],
    )
    def test_calculate_position_basic_anchors(self, anchor, expected_y):
        config = SubtitleSettings(anchor=anchor, margin=0.1, content_aware=False)
        pos = calculate_position(config, (1080, 1920), safe_zone=self._wide_sz)
        assert pos.y == pytest.approx(expected_y)
        assert pos.x == 0.5

    def test_calculate_position_basic_anchors_with_safe_zone(self):
        """Verify safe zone clamps TOP and BOTTOM anchors."""
        sz = PlatformSafeZone()  # default cross-platform safe zone
        top = calculate_position(
            SubtitleSettings(anchor=PositionAnchor.TOP, margin=0.05),
            (1080, 1920),
            safe_zone=sz,
        )
        assert top.y == pytest.approx(sz.min_y)  # margin < min_y, clamped

        bottom = calculate_position(
            SubtitleSettings(anchor=PositionAnchor.BOTTOM, margin=0.05),
            (1080, 1920),
            safe_zone=sz,
        )
        assert bottom.y == pytest.approx(sz.max_y)  # 0.95 > max_y, clamped

    def test_calculate_position_above_content(self):
        config = SubtitleSettings(
            anchor=PositionAnchor.ABOVE_CONTENT, margin=0.1, content_aware=True
        )
        visual_bounds = VisualBounds(x=0.1, y=0.2, width=0.8, height=0.6)
        pos = calculate_position(
            config, (1080, 1920), visual_bounds, safe_zone=self._wide_sz
        )
        assert pos.y == 0.1

    def test_calculate_position_below_content(self):
        config = SubtitleSettings(
            anchor=PositionAnchor.BELOW_CONTENT, margin=0.05, content_aware=True
        )
        visual_bounds = VisualBounds(x=0.1, y=0.2, width=0.8, height=0.6)
        pos = calculate_position(
            config, (1080, 1920), visual_bounds, safe_zone=self._wide_sz
        )
        assert pos.y == pytest.approx(0.85)  # 0.2 + 0.6 + 0.05

    def test_calculate_position_below_content_clamped(self):
        sz = PlatformSafeZone(min_x=0.0, max_x=1.0, min_y=0.0, max_y=0.95)
        config = SubtitleSettings(
            anchor=PositionAnchor.BELOW_CONTENT, margin=0.2, content_aware=True
        )
        visual_bounds = VisualBounds(x=0.1, y=0.2, width=0.8, height=0.7)
        pos = calculate_position(config, (1080, 1920), visual_bounds, safe_zone=sz)
        assert pos.y == pytest.approx(0.95)

    @pytest.mark.parametrize(
        "alignment,expected_x",
        [
            ("left", 0.0),
            ("center", 0.5),
            ("right", 1.0),
        ],
    )
    def test_horizontal_alignment(self, alignment, expected_x):
        config = SubtitleSettings(horizontal_alignment=alignment)
        pos = calculate_position(config, (1080, 1920), safe_zone=self._wide_sz)
        assert pos.x == pytest.approx(expected_x)

    def test_horizontal_alignment_clamped_to_safe_zone(self):
        sz = PlatformSafeZone()
        left = calculate_position(
            SubtitleSettings(horizontal_alignment="left"),
            (1080, 1920),
            safe_zone=sz,
        )
        assert left.x == pytest.approx(sz.min_x)
        right = calculate_position(
            SubtitleSettings(horizontal_alignment="right"),
            (1080, 1920),
            safe_zone=sz,
        )
        assert right.x == pytest.approx(sz.max_x)

    def test_clamp_to_safe_zone(self):
        sz = PlatformSafeZone(min_x=0.05, max_x=0.8, min_y=0.1, max_y=0.75)
        # Within bounds
        assert clamp_to_safe_zone(540, 960, 1080, 1920, sz) == (540, 960)
        # Below min
        assert clamp_to_safe_zone(0, 0, 1080, 1920, sz) == (54, 192)
        # Above max
        assert clamp_to_safe_zone(1080, 1920, 1080, 1920, sz) == (864, 1440)

    def test_custom_position_clamped(self):
        from src.video.subtitle_positioning import Position

        sz = PlatformSafeZone(min_x=0.1, max_x=0.9, min_y=0.1, max_y=0.9)
        config = SubtitleSettings(
            custom_position=Position(x=0.0, y=1.0),
        )
        pos = calculate_position(config, (1080, 1920), safe_zone=sz)
        assert pos.x == pytest.approx(0.1)
        assert pos.y == pytest.approx(0.9)

    def test_get_font_size(self):
        # Default base_font_size_percent is 0.04
        # 1000 * 0.04 = 40.
        config = SubtitleSettings(font_size_scale=1.0)
        size = get_font_size(config, 1000, base_size_percent=0.04)
        assert size == 40

        # 40 * 2.0 = 80. (within default 100 max)
        config_scaled = SubtitleSettings(font_size_scale=2.0)
        size_scaled = get_font_size(config_scaled, 1000, base_size_percent=0.04)
        assert size_scaled == 80

    def test_get_style_config_fallback(self):
        # Test fallback when yaml missing and no video_config
        with patch("pathlib.Path.exists", return_value=False):
            style = get_style_config(StylePreset.MINIMAL)
            # Inline last-resort defaults use Montserrat (modern preset)
            assert style["font_name"] == "Montserrat"

            style_modern = get_style_config(StylePreset.MODERN)
            assert "karaoke" in style_modern["effects"]

    def test_get_style_config_yaml(self):
        mock_yaml = (
            "style_presets:\n  minimal:\n    font_name: 'CustomFont'\n    effects: []"
        )
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=mock_yaml)),
        ):
            style = get_style_config(StylePreset.MINIMAL)
            assert style["font_name"] == "CustomFont"

    def test_get_style_config_random(self):
        config = SubtitleSettings(style_preset=StylePreset.RANDOM)
        # Random preset forces randomization
        style = get_style_config(StylePreset.RANDOM, config, "prod123")
        assert len(style["effects"]) == 1
        assert config.randomize_fonts is True

    def test_from_legacy_dict_basic(self):
        settings = {
            "anchor": "top",
            "margin": 0.15,
            "style_preset": "bold",
            "font_size_scale": 1.2,
        }
        config = SubtitleSettings.from_legacy_dict(settings)
        assert config.anchor == PositionAnchor.TOP
        assert config.margin == 0.15
        assert config.style_preset == StylePreset.BOLD
        assert config.font_size_scale == 1.2

    def test_from_legacy_dict_invalid_values_raise(self):
        from pydantic import ValidationError

        # Strict: invalid Literal/enum values must surface as ValidationError
        # so callers can either fix the YAML or wrap in a fallback. The old
        # wrapper silently fell back; that masked typos in production YAML.
        settings = {"anchor": "invalid", "style_preset": "garbage"}
        with pytest.raises(ValidationError):
            SubtitleSettings.from_legacy_dict(settings)
