"""Tests for profile-specific settings, CLI override precedence, and Pydantic merge."""

import pytest

from src.video.config import VideoConfig, VideoProfile, load_video_config
from src.video.config.visual_models import (
    MergedProfileSettings,
    MergedSubtitleSettings,
    ProfileInfo,
    VideoSettings,
)


@pytest.mark.unit
class TestProfileSpecificSettings:
    """Test video profile specific setting overrides."""

    def test_profile_image_positioning_overrides(self, mock_config: VideoConfig):
        """Test that profile can override global image positioning settings."""
        profile = VideoProfile(
            description="Test profile with positioning",
            use_scraped_images=True,
            use_scraped_videos=False,
            use_stock_images=False,
            use_stock_videos=False,
            use_dynamic_image_count=False,
            image_width_percent=0.75,
            image_top_position_percent=0.20,
        )

        assert profile.image_width_percent == 0.75
        assert profile.image_top_position_percent == 0.20

    def test_profile_subtitle_positioning_overrides(self, mock_config: VideoConfig):
        """Test that profile can override subtitle positioning settings."""
        profile = VideoProfile(
            description="Test profile with subtitle positioning",
            use_scraped_images=True,
            use_scraped_videos=False,
            use_stock_images=False,
            use_stock_videos=False,
            use_dynamic_image_count=False,
            subtitle_anchor="top",
            subtitle_margin=0.15,
            subtitle_content_aware=False,
        )

        assert profile.subtitle_anchor == "top"
        assert profile.subtitle_margin == 0.15
        assert profile.subtitle_content_aware is False

    def test_profile_vertical_align_field(self):
        """Test video_vertical_align field on VideoProfile."""
        profile = VideoProfile(
            description="Centered profile",
            use_scraped_images=True,
            use_scraped_videos=True,
            video_vertical_align="center",
        )
        assert profile.video_vertical_align == "center"

        # None by default (inherits from global)
        default = VideoProfile(
            description="Default profile",
            use_scraped_images=True,
            use_scraped_videos=False,
        )
        assert default.video_vertical_align is None


@pytest.mark.unit
class TestMergedProfileSettings:
    """Test get_profile_merged_settings returns typed Pydantic models."""

    def test_returns_pydantic_model(self, mock_config: VideoConfig):
        """get_profile_merged_settings returns MergedProfileSettings."""
        merged = mock_config.get_profile_merged_settings("test_profile")

        assert isinstance(merged, MergedProfileSettings)
        assert isinstance(merged.video_settings, VideoSettings)
        assert isinstance(merged.subtitle_settings, MergedSubtitleSettings)
        assert isinstance(merged.profile, ProfileInfo)

    def test_profile_info_populated(self, mock_config: VideoConfig):
        """Profile info contains correct metadata."""
        merged = mock_config.get_profile_merged_settings("test_profile")

        assert merged.profile.name == "test_profile"
        assert merged.profile.description == "Test profile for unit testing"
        assert merged.profile.use_scraped_images is True
        assert merged.profile.use_scraped_videos is False
        assert merged.profile.use_stock_images is True

    def test_global_video_settings_as_base(self, mock_config: VideoConfig):
        """Profile without video overrides inherits global video settings."""
        merged = mock_config.get_profile_merged_settings("test_profile")

        # test_profile doesn't override these, so they come from global
        assert merged.video_settings.resolution == (1080, 1920)
        assert merged.video_settings.frame_rate == 30
        assert merged.video_settings.image_width_percent == 0.8  # global default

    def test_profile_overrides_video_settings(self, mock_config: VideoConfig):
        """Profile video overrides take precedence over global."""
        merged = mock_config.get_profile_merged_settings("product_video_sequential")

        # This profile sets image_width_percent=0.85 (global is 0.8)
        assert merged.video_settings.image_width_percent == 0.85
        assert merged.video_settings.image_top_position_percent == 0.15
        assert merged.video_settings.video_vertical_align == "center"

    def test_profile_overrides_top_align(self, mock_config: VideoConfig):
        """Profile with video_vertical_align=top overrides global center."""
        merged = mock_config.get_profile_merged_settings("product_video_single")

        # product_video_single sets "top", global is "center"
        assert merged.video_settings.video_vertical_align == "top"
        assert merged.video_settings.image_width_percent == 0.75

    def test_subtitle_settings_from_global(self, mock_config: VideoConfig):
        """Subtitle settings merge from global YAML config."""
        merged = mock_config.get_profile_merged_settings("test_profile")

        assert merged.subtitle_settings.anchor == "bottom"
        assert merged.subtitle_settings.margin == 0.05
        # font_directory resolved to absolute path during config loading
        assert merged.subtitle_settings.font_directory.endswith("static/fonts")
        assert merged.subtitle_settings.max_line_length == 38
        assert merged.subtitle_settings.enabled is True

    def test_subtitle_model_dump_roundtrip(self, mock_config: VideoConfig):
        """Subtitle settings can be dumped back to dict for downstream use."""
        merged = mock_config.get_profile_merged_settings("test_profile")
        dumped = merged.subtitle_settings.model_dump()

        assert isinstance(dumped, dict)
        assert dumped["anchor"] == "bottom"
        assert dumped["font_directory"].endswith("static/fonts")

    def test_two_part_subtitle_defaults(self, mock_config: VideoConfig):
        """Two-part subtitle fields have correct defaults when not configured."""
        merged = mock_config.get_profile_merged_settings("test_profile")

        assert merged.subtitle_settings.two_part_subtitles_enabled is False
        assert merged.subtitle_settings.two_part_subtitles_upper_enabled is True
        assert (
            merged.subtitle_settings.two_part_subtitles_lower_anchor == "below_content"
        )


@pytest.mark.unit
class TestCollectOverrides:
    """Test _collect_overrides static method."""

    def test_collects_non_none_fields(self):
        """Only non-None profile fields end up in overrides."""
        profile = VideoProfile(
            description="Test",
            use_scraped_images=True,
            use_scraped_videos=False,
            image_width_percent=0.65,
            # image_top_position_percent is None by default
        )
        field_map = {
            "image_width_percent": "image_width_percent",
            "image_top_position_percent": "image_top_position_percent",
        }
        result = VideoConfig._collect_overrides(profile, field_map)

        assert result == {"image_width_percent": 0.65}

    def test_empty_when_all_none(self):
        """Returns empty dict when no profile fields are set."""
        profile = VideoProfile(
            description="Bare",
            use_scraped_images=True,
            use_scraped_videos=False,
        )
        field_map = {
            "image_width_percent": "image_width_percent",
            "video_vertical_align": "video_vertical_align",
        }
        result = VideoConfig._collect_overrides(profile, field_map)

        assert result == {}

    def test_field_name_remapping(self):
        """Profile field names can be remapped to target field names."""
        profile = VideoProfile(
            description="Remap test",
            use_scraped_images=True,
            use_scraped_videos=False,
            subtitle_anchor="top",
        )
        field_map = {"subtitle_anchor": "anchor"}
        result = VideoConfig._collect_overrides(profile, field_map)

        assert result == {"anchor": "top"}


@pytest.mark.unit
class TestCLIOverridePrecedence:
    """Test CLI override precedence over profile and global settings."""

    def test_cli_overrides_video_setting(self, mock_config: VideoConfig):
        """CLI override for video settings takes highest precedence."""
        cli_overrides = {"video_settings.image_width_percent": 0.65}

        merged = mock_config.get_profile_merged_settings(
            "product_video_sequential", cli_overrides
        )

        # CLI=0.65 beats profile=0.85 beats global=0.8
        assert merged.video_settings.image_width_percent == 0.65

    def test_cli_overrides_subtitle_setting(self, mock_config: VideoConfig):
        """CLI override for subtitle settings takes highest precedence."""
        cli_overrides = {"subtitle_settings.anchor": "top"}

        merged = mock_config.get_profile_merged_settings("test_profile", cli_overrides)

        # CLI="top" beats global="bottom"
        assert merged.subtitle_settings.anchor == "top"

    def test_cli_overrides_multiple(self, mock_config: VideoConfig):
        """Multiple CLI overrides applied simultaneously."""
        cli_overrides = {
            "video_settings.image_width_percent": 0.50,
            "video_settings.image_top_position_percent": 0.30,
            "subtitle_settings.anchor": "center",
            "subtitle_settings.margin": 0.12,
        }

        merged = mock_config.get_profile_merged_settings(
            "product_video_sequential", cli_overrides
        )

        assert merged.video_settings.image_width_percent == 0.50
        assert merged.video_settings.image_top_position_percent == 0.30
        assert merged.subtitle_settings.anchor == "center"
        assert merged.subtitle_settings.margin == 0.12

    def test_precedence_cli_over_profile_over_global(self, mock_config: VideoConfig):
        """Verify CLI > Profile > Global precedence chain."""
        # Without CLI: profile value wins
        merged_no_cli = mock_config.get_profile_merged_settings(
            "product_video_sequential"
        )
        assert merged_no_cli.video_settings.image_width_percent == 0.85

        # With CLI: CLI value wins
        merged_with_cli = mock_config.get_profile_merged_settings(
            "product_video_sequential",
            {"video_settings.image_width_percent": 0.50},
        )
        assert merged_with_cli.video_settings.image_width_percent == 0.50

    def test_unspecified_cli_uses_profile_defaults(self, mock_config: VideoConfig):
        """Settings not in CLI overrides keep profile values."""
        cli_overrides = {"video_settings.image_width_percent": 0.60}

        merged = mock_config.get_profile_merged_settings(
            "product_video_sequential", cli_overrides
        )

        # Overridden
        assert merged.video_settings.image_width_percent == 0.60
        # Not overridden, keeps profile value (0.15)
        assert merged.video_settings.image_top_position_percent == 0.15

    def test_no_cli_overrides_is_noop(self, mock_config: VideoConfig):
        """Passing None or empty cli_overrides changes nothing."""
        merged_none = mock_config.get_profile_merged_settings("test_profile", None)
        merged_empty = mock_config.get_profile_merged_settings("test_profile", {})

        assert (
            merged_none.video_settings.image_width_percent
            == merged_empty.video_settings.image_width_percent
        )
        assert (
            merged_none.subtitle_settings.anchor
            == merged_empty.subtitle_settings.anchor
        )

    def test_malformed_cli_key_ignored(self, mock_config: VideoConfig):
        """CLI keys without a dot separator are silently ignored."""
        cli_overrides = {"bad_key_no_dot": 999}

        merged = mock_config.get_profile_merged_settings("test_profile", cli_overrides)

        # Should still return valid settings
        assert isinstance(merged, MergedProfileSettings)


@pytest.mark.unit
class TestArgparseDefaultBehavior:
    """Test argparse default=None behavior for boolean flags."""

    def test_boolean_flags_default_to_none_when_not_specified(self):
        """Test that argparse boolean flags default to None, not False."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--content-aware",
            action="store_true",
            dest="content_aware",
            default=None,
        )
        parser.add_argument(
            "--no-content-aware",
            action="store_false",
            dest="content_aware",
            default=None,
        )

        args = parser.parse_args([])
        assert args.content_aware is None

    def test_boolean_flags_set_true_when_positive_flag_passed(self):
        """Test that positive flag sets value to True."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--content-aware",
            action="store_true",
            dest="content_aware",
            default=None,
        )
        parser.add_argument(
            "--no-content-aware",
            action="store_false",
            dest="content_aware",
            default=None,
        )

        args = parser.parse_args(["--content-aware"])
        assert args.content_aware is True

    def test_boolean_flags_set_false_when_negative_flag_passed(self):
        """Test that negative flag sets value to False."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--content-aware",
            action="store_true",
            dest="content_aware",
            default=None,
        )
        parser.add_argument(
            "--no-content-aware",
            action="store_false",
            dest="content_aware",
            default=None,
        )

        args = parser.parse_args(["--no-content-aware"])
        assert args.content_aware is False

    def test_randomization_flags_default_to_none(self):
        """Test that randomization flags also default to None."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--randomize-fonts",
            action="store_true",
            dest="randomize_fonts",
            default=None,
        )
        parser.add_argument(
            "--no-randomize-fonts",
            action="store_false",
            dest="randomize_fonts",
            default=None,
        )

        args = parser.parse_args([])
        assert args.randomize_fonts is None
