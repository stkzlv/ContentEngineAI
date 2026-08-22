"""Tests that an unknown key in a profile block fails at load.

Pydantic drops unknown keys by default, and a dropped key in a profile is
invisible: the render succeeds using the global value, so the profile appears
to work while its override does nothing. `docs/requirements.md` has claimed
strict validation for profiles since before the model had it.
"""

import warnings

import pytest
from pydantic import ValidationError

from src.video.config.visual_models import VideoProfile


def _profile(**extra) -> dict:
    payload = {"description": "A profile"}
    payload.update(extra)
    return payload


@pytest.mark.unit
class TestUnknownKeysFail:
    def test_a_typo_fails_at_load(self):
        with pytest.raises(ValidationError, match="stock_image_cont"):
            VideoProfile(**_profile(stock_image_cont=8))

    def test_a_field_that_belongs_to_another_model_fails(self):
        """`min_images_if_no_video` is a `VideoSettings` field.

        Putting it in a profile block reads as an override and is not one.
        """
        with pytest.raises(ValidationError):
            VideoProfile(**_profile(min_images_if_no_video=3))

    def test_a_known_field_still_loads(self):
        assert VideoProfile(**_profile(stock_image_count=8)).stock_image_count == 8


@pytest.mark.unit
class TestLegacyKeysStillAccepted:
    """The migration validator runs first and removes what it migrates.

    Strictness must not break the flat keys the bundled profiles still use, or
    every profile fails to load.
    """

    def test_a_migrated_flat_key_is_accepted(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            profile = VideoProfile(**_profile(subtitle_anchor="below_content"))
        assert profile.subtitle_settings is not None
        assert profile.subtitle_settings.anchor == "below_content"

    def test_subtitle_format_is_migrated_rather_than_dropped(self):
        """It has no `subtitle_` prefix in its target name, which is how it
        came to be the one flat key missing from the migration map. A profile
        setting it fell back to the global value with no warning.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            profile = VideoProfile(**_profile(subtitle_format="srt"))
        assert profile.subtitle_settings is not None
        assert profile.subtitle_settings.subtitle_format == "srt"

    def test_a_migrated_key_still_warns(self):
        with pytest.warns(DeprecationWarning, match="legacy flat subtitle keys"):
            VideoProfile(**_profile(subtitle_format="ass"))

    def test_a_nested_block_is_accepted(self):
        profile = VideoProfile(**_profile(subtitle_settings={"subtitle_format": "ass"}))
        assert profile.subtitle_settings is not None
        assert profile.subtitle_settings.subtitle_format == "ass"


@pytest.mark.unit
class TestBundledProfilesLoad:
    """Every shipped profile must survive the stricter model.

    A dead key in one of them becomes a config-load failure, which is the
    point, but it has to be found here rather than by a user.
    """

    def test_all_bundled_profiles_parse(self):
        from src.video.config import load_video_config_modular

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config = load_video_config_modular()
        assert len(config.video_profiles) >= 11

    def test_a_profiles_subtitle_format_reaches_the_merged_settings(self):
        """The end of the chain the dropped key never reached."""
        from src.video.config import load_video_config_modular

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config = load_video_config_modular()
        merged = config.get_profile_merged_settings("slideshow_stock")
        assert merged.subtitle_settings.subtitle_format == "ass"
