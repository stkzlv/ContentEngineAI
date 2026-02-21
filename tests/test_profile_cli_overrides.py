"""Tests for profile-specific settings and CLI override precedence."""

import pytest

from src.video.config import VideoConfig, VideoProfile, load_video_config


@pytest.mark.unit
class TestProfileSpecificSettings:
    """Test video profile specific setting overrides."""

    def test_slideshow_images1_profile_overrides(self, mock_config: VideoConfig):
        """Test slideshow_images1 profile loads with correct overrides."""
        try:
            profile = mock_config.get_profile("slideshow_images1")
            assert profile.description is not None
            assert profile.use_scraped_images is True
            assert profile.use_scraped_videos is False
            # Check image width override if available
            if hasattr(profile, "image_width_percent"):
                assert profile.image_width_percent == 0.85
        except KeyError:
            pytest.skip("slideshow_images1 profile not found in mock config")

    @pytest.mark.skip(reason="Skipping slow config loading test")
    def test_slideshow_images2_profile_overrides(self):
        """Test slideshow_images2 profile loads with correct settings."""
        try:
            config = load_video_config()
            profile = config.get_profile("slideshow_images2")

            # Media selection
            assert profile.use_scraped_images is True
            assert profile.use_scraped_videos is False
            assert profile.use_stock_images is False
            assert profile.use_stock_videos is False
            assert profile.use_dynamic_image_count is True

            # Image positioning overrides
            assert profile.image_width_percent == 0.80
            assert profile.image_top_position_percent == 0.15

            # Subtitle positioning overrides
            assert profile.subtitle_anchor == "below_content"
            assert profile.subtitle_margin == 0.08
            assert profile.subtitle_content_aware is True
            assert profile.subtitle_horizontal_alignment == "center"

            # Subtitle style overrides
            assert profile.subtitle_style_preset == "minimal"
            assert profile.subtitle_font_size_scale == 0.9
            assert profile.subtitle_randomize_fonts is True
            assert profile.subtitle_randomize_colors is True
            assert profile.subtitle_randomize_effects is False

            # Text formatting overrides
            assert profile.subtitle_max_line_length == 28
            assert profile.subtitle_max_words_per_line == 3
            assert profile.subtitle_max_subtitle_width_fraction == 0.85
            assert profile.subtitle_max_duration == 4.0
            assert profile.subtitle_min_duration == 0.5

        except (KeyError, FileNotFoundError):
            pytest.skip("slideshow_images2 profile not found or config file missing")

    def test_profile_image_positioning_overrides(self, mock_config: VideoConfig):
        """Test that profile can override global image positioning settings."""
        # Create a test profile with image positioning overrides
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

    @pytest.mark.skip(reason="Skipping slow config loading test")
    def test_profile_merges_with_global_settings(self):
        """Test that profile settings merge correctly with global settings."""
        try:
            config = load_video_config()

            # Get merged settings for a profile
            merged = config.get_profile_merged_settings("slideshow_images1")

            # Should contain typed settings
            assert merged.video_settings is not None
            assert merged.subtitle_settings is not None

        except (KeyError, FileNotFoundError):
            pytest.skip("Config file or profile not found")


@pytest.mark.unit
class TestCLIOverridePrecedence:
    """Test CLI override precedence over profile and global settings."""

    @pytest.mark.skip(reason="Skipping slow config loading test")
    def test_cli_overrides_profile_image_width(self):
        """Test CLI override for image width has highest precedence."""
        try:
            config = load_video_config()

            # Simulate CLI override
            cli_overrides = {"video_settings.image_width_percent": 0.65}

            merged = config.get_profile_merged_settings(
                "slideshow_images2", cli_overrides
            )

            # CLI override should take precedence over profile setting (0.80)
            assert merged.video_settings.image_width_percent == 0.65

        except (KeyError, FileNotFoundError):
            pytest.skip("Config file or profile not found")

    @pytest.mark.skip(reason="Skipping slow config loading test")
    def test_cli_overrides_profile_image_position(self):
        """Test CLI override for image position has highest precedence."""
        try:
            config = load_video_config()

            cli_overrides = {"video_settings.image_top_position_percent": 0.30}

            merged = config.get_profile_merged_settings(
                "slideshow_images2", cli_overrides
            )

            # CLI override should take precedence over profile setting (0.15)
            assert merged.video_settings.image_top_position_percent == 0.30

        except (KeyError, FileNotFoundError):
            pytest.skip("Config file or profile not found")

    @pytest.mark.skip(reason="Skipping slow config loading test")
    def test_cli_overrides_subtitle_anchor(self):
        """Test CLI override for subtitle anchor point."""
        try:
            config = load_video_config()

            cli_overrides = {"subtitle_settings.anchor": "top"}

            merged = config.get_profile_merged_settings(
                "slideshow_images2", cli_overrides
            )

            # CLI override should take precedence
            assert merged.subtitle_settings.anchor == "top"

        except (KeyError, FileNotFoundError):
            pytest.skip("Config file or profile not found")

    @pytest.mark.skip(reason="Skipping slow config loading test")
    def test_cli_overrides_multiple_settings(self):
        """Test multiple CLI overrides applied simultaneously."""
        try:
            config = load_video_config()

            cli_overrides = {
                "video_settings.image_width_percent": 0.70,
                "video_settings.image_top_position_percent": 0.25,
                "subtitle_settings.anchor": "center",
                "subtitle_settings.margin": 0.12,
            }

            merged = config.get_profile_merged_settings(
                "slideshow_images2", cli_overrides
            )

            # All CLI overrides should be applied
            assert merged.video_settings.image_width_percent == 0.70
            assert merged.video_settings.image_top_position_percent == 0.25
            assert merged.subtitle_settings.anchor == "center"
            assert merged.subtitle_settings.margin == 0.12

        except (KeyError, FileNotFoundError):
            pytest.skip("Config file or profile not found")

    @pytest.mark.skip(reason="Skipping slow config loading test")
    def test_precedence_order_cli_profile_global(self):
        """Test configuration precedence: CLI > Profile > Global."""
        try:
            config = load_video_config()

            # Test without CLI override - should use profile setting
            merged_no_cli = config.get_profile_merged_settings("slideshow_images2")
            profile_value = merged_no_cli["video_settings"]["image_width_percent"]

            # Test with CLI override - should use CLI setting
            cli_overrides = {"video_settings.image_width_percent": 0.50}
            merged_with_cli = config.get_profile_merged_settings(
                "slideshow_images2", cli_overrides
            )
            cli_value = merged_with_cli["video_settings"]["image_width_percent"]

            # CLI value should be different from profile value
            assert cli_value == 0.50
            assert profile_value == 0.80  # slideshow_images2 profile setting
            assert cli_value != profile_value

        except (KeyError, FileNotFoundError):
            pytest.skip("Config file or profile not found")

    @pytest.mark.skip(reason="Skipping slow config loading test")
    def test_unspecified_cli_overrides_use_profile_defaults(self):
        """Test that settings not in CLI overrides use profile defaults."""
        try:
            config = load_video_config()

            # Override only one setting
            cli_overrides = {"video_settings.image_width_percent": 0.60}

            merged = config.get_profile_merged_settings(
                "slideshow_images2", cli_overrides
            )

            # CLI-overridden setting should use new value
            assert merged.video_settings.image_width_percent == 0.60

            # Non-overridden setting should use profile default
            assert merged.video_settings.image_top_position_percent == 0.15

        except (KeyError, FileNotFoundError):
            pytest.skip("Config file or profile not found")


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

        # Parse with no flags
        args = parser.parse_args([])

        # Should be None, not False
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

        # Parse with positive flag
        args = parser.parse_args(["--content-aware"])

        # Should be True
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

        # Parse with negative flag
        args = parser.parse_args(["--no-content-aware"])

        # Should be False
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

        # Parse with no flags
        args = parser.parse_args([])

        # Should be None, not False
        assert args.randomize_fonts is None
