from pathlib import Path

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

    def test_default_safe_zone_matches_2026_union(self):
        """Roadmap 1.6 (#140): defaults are the 2026 cross-platform union and
        a render's lowest caption pixel stays above the Reels y=1250 floor.
        """
        sz = PlatformSafeZone()  # 2026 union defaults
        assert sz.min_x == 0.056
        assert sz.max_x == 0.833
        assert sz.min_y == 0.141
        assert sz.max_y == 0.651
        # A y past the floor clamps back above the Reels 35% interactive zone.
        _, clamped_y = clamp_to_safe_zone(540, 1920, 1080, 1920, sz)
        assert clamped_y <= 1250

    def test_clamp_accounts_for_text_height(self):
        """With center-anchored captions (ASS align 5), the clamp keeps the
        whole text box inside the band, not just the center point.
        """
        sz = PlatformSafeZone()  # 2026 union: max_y -> 1249px
        half = 40  # half a ~80px line
        _, center_y = clamp_to_safe_zone(540, 1920, 1080, 1920, sz, half)
        # Center is pulled up so the lowest pixel (center + half) stays <= 1250.
        assert center_y + half <= 1250
        # Degenerate case: text taller than the band centers within it.
        _, mid = clamp_to_safe_zone(540, 1920, 1080, 1920, sz, 2000)
        assert int(1920 * sz.min_y) <= mid <= int(1920 * sz.max_y)

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

    def test_get_style_config_raises_without_video_config(self):
        # The legacy YAML re-read fallback was removed; callers must pass
        # video_config so the typed style_presets path is taken.
        with pytest.raises(ValueError, match="requires video_config"):
            get_style_config(StylePreset.MINIMAL)

    def test_get_style_config_raises_when_preset_missing(self, mock_config):
        # If neither the requested preset nor "modern" exists in
        # style_presets, the function raises with a clear message.
        mock_config.style_presets = {}
        with pytest.raises(ValueError, match="No style preset matches"):
            get_style_config(StylePreset.MINIMAL, video_config=mock_config)

    def test_get_style_config_uses_typed_presets(self, mock_config):
        # Default VideoConfig ships 5 presets (minimal, modern, bold,
        # animated, random); the typed path returns the requested one.
        style = get_style_config(StylePreset.MINIMAL, video_config=mock_config)
        assert "font_name" in style
        assert "effects" in style

    def test_get_style_config_random(self, mock_config):
        config = SubtitleSettings(style_preset=StylePreset.RANDOM)
        # RANDOM preset forces randomization and selects one effect from the
        # preset's effects list.
        style = get_style_config(
            StylePreset.RANDOM, config, "prod123", video_config=mock_config
        )
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
