"""Unit tests for the unified SubtitleSettings / PartialSubtitleSettings models."""

import pytest
from pydantic import ValidationError

from src.video.config import config as video_config
from src.video.config.subtitle_models import (
    PartialSubtitleSettings,
    PlatformSafeZone,
    Position,
    PositionAnchor,
    PycapsSettings,
    StylePreset,
    SubtitleSettings,
    TwoPartSubtitleSettings,
)


@pytest.mark.unit
class TestSubtitleSettingsDefaults:
    """Defaults match the best-practice recipe from docs/subtitle-best-practices.md."""

    def test_duration_defaults_match_recipe(self):
        s = SubtitleSettings()
        assert s.max_duration == 2.5
        assert s.min_duration == 0.6

    def test_text_formatting_defaults(self):
        s = SubtitleSettings()
        assert s.max_words_per_line == 3
        assert s.max_subtitle_width_fraction == 0.80
        assert s.max_line_length == 38

    def test_positioning_defaults(self):
        s = SubtitleSettings()
        assert s.anchor is PositionAnchor.BOTTOM
        assert s.style_preset is StylePreset.MODERN
        assert s.horizontal_alignment == "center"

    def test_safe_zone_nested_default(self):
        s = SubtitleSettings()
        assert isinstance(s.safe_zone, PlatformSafeZone)
        assert s.safe_zone.max_y == 0.75  # TikTok overlay floor

    def test_two_part_nested_default(self):
        s = SubtitleSettings()
        assert isinstance(s.two_part, TwoPartSubtitleSettings)
        assert s.two_part.enabled is False


@pytest.mark.unit
class TestExtraForbidCatchesTypos:
    def test_typo_in_top_level_field_rejected(self):
        with pytest.raises(ValidationError, match="font_nam"):
            SubtitleSettings(font_nam="Montserrat")

    def test_typo_in_partial_override_rejected(self):
        with pytest.raises(ValidationError):
            PartialSubtitleSettings(max_duratio=5.0)


@pytest.mark.unit
class TestFromLegacyDict:
    """from_legacy_dict translates renames and drops dead keys."""

    def test_duration_renames(self):
        s = SubtitleSettings.from_legacy_dict(
            {"max_subtitle_duration": 3.0, "min_subtitle_duration": 0.5}
        )
        assert s.max_duration == 3.0
        assert s.min_duration == 0.5

    def test_two_part_rename(self):
        s = SubtitleSettings.from_legacy_dict({"two_part_subtitles": {"enabled": True}})
        assert s.two_part.enabled is True

    def test_canonical_wins_over_legacy_when_both_present(self):
        s = SubtitleSettings.from_legacy_dict(
            {"max_duration": 2.0, "max_subtitle_duration": 99.0}
        )
        assert s.max_duration == 2.0

    def test_dead_keys_dropped_silently(self):
        s = SubtitleSettings.from_legacy_dict(
            {
                "available_fonts": ["Arial", "Inter"],
                "available_color_combinations": [],
                "margin": 0.15,
            }
        )
        assert s.margin == 0.15

    def test_unknown_key_still_raises(self):
        with pytest.raises(ValidationError):
            SubtitleSettings.from_legacy_dict({"not_a_real_field": "x"})

    def test_merged_subtitle_settings_dump_round_trips(self):
        """A dump from the existing MergedSubtitleSettings code path loads."""
        from src.video.config.visual_models import MergedSubtitleSettings

        merged = MergedSubtitleSettings()
        dump = merged.model_dump()
        s = SubtitleSettings.from_legacy_dict(dump)
        # Canonical names won from the translation
        assert isinstance(s.max_duration, float)
        assert isinstance(s.min_duration, float)


@pytest.mark.unit
class TestAllNineProfilesLoad:
    """Every shipped VideoProfile merges cleanly into SubtitleSettings."""

    def test_all_profiles_load_via_from_legacy_dict(self):
        issues: list[str] = []
        for name in video_config.video_profiles:
            merged = video_config.get_profile_merged_settings(name)
            dump = merged.subtitle_settings.model_dump()
            try:
                s = SubtitleSettings.from_legacy_dict(dump)
            except ValidationError as e:
                issues.append(f"{name}: {e}")
                continue
            # Best-practice invariants from the §10 appendix one-liner
            if s.max_duration != 2.5:
                issues.append(f"{name}: max_duration={s.max_duration}")
            if s.min_duration != 0.6:
                issues.append(f"{name}: min_duration={s.min_duration}")
            if s.max_subtitle_width_fraction < 0.75:
                issues.append(f"{name}: width={s.max_subtitle_width_fraction}")
            if s.max_words_per_line < 3:
                issues.append(f"{name}: wpl={s.max_words_per_line}")
        assert not issues, "Profile round-trip issues: " + " | ".join(issues)


@pytest.mark.unit
class TestPartialSubtitleSettingsMergeInto:
    """merge_into applies only non-None fields; nested models deep-merge."""

    def test_empty_partial_returns_base_unchanged(self):
        base = SubtitleSettings(max_duration=2.5, max_words_per_line=3)
        merged = PartialSubtitleSettings().merge_into(base)
        assert merged.max_duration == 2.5
        assert merged.max_words_per_line == 3

    def test_single_scalar_override_wins(self):
        base = SubtitleSettings(max_duration=2.5)
        partial = PartialSubtitleSettings(max_duration=4.0)
        merged = partial.merge_into(base)
        assert merged.max_duration == 4.0

    def test_enum_override_applies(self):
        base = SubtitleSettings()
        partial = PartialSubtitleSettings(style_preset=StylePreset.BOLD)
        merged = partial.merge_into(base)
        assert merged.style_preset is StylePreset.BOLD

    def test_pycaps_nested_merge_preserves_unspecified(self):
        base = SubtitleSettings(
            pycaps=PycapsSettings(template_name="word-focus", renderer="css")
        )
        partial = PartialSubtitleSettings(pycaps={"template_name": "hype"})
        merged = partial.merge_into(base)
        assert merged.pycaps is not None
        assert merged.pycaps.template_name == "hype"
        # Renderer preserved from base
        assert merged.pycaps.renderer == "css"

    def test_two_part_nested_merge_preserves_unspecified(self):
        base = SubtitleSettings()
        partial = PartialSubtitleSettings(
            two_part={"enabled": True, "upper_line": {"style_preset": "minimal"}}
        )
        merged = partial.merge_into(base)
        assert merged.two_part.enabled is True
        assert merged.two_part.upper_line.style_preset == "minimal"
        # Unspecified upper_line fields preserved from base default
        assert merged.two_part.upper_line.source_field == "shortened_affiliate_link"

    def test_safe_zone_nested_merge(self):
        base = SubtitleSettings()
        partial = PartialSubtitleSettings(safe_zone={"max_y": 0.65})
        merged = partial.merge_into(base)
        assert merged.safe_zone.max_y == 0.65
        # Unspecified coords preserved
        assert merged.safe_zone.min_x == base.safe_zone.min_x

    def test_custom_position_dict_builds_position_model(self):
        base = SubtitleSettings()
        partial = PartialSubtitleSettings(custom_position={"x": 0.5, "y": 0.7})
        merged = partial.merge_into(base)
        assert merged.custom_position == Position(x=0.5, y=0.7)

    def test_chained_overrides_compose(self):
        """Simulates global <- profile <- CLI precedence order."""
        base = SubtitleSettings(max_duration=2.5, max_words_per_line=3)
        profile = PartialSubtitleSettings(max_duration=3.0)
        cli = PartialSubtitleSettings(max_words_per_line=5)
        merged = cli.merge_into(profile.merge_into(base))
        assert merged.max_duration == 3.0
        assert merged.max_words_per_line == 5
