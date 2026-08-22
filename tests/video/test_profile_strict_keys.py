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

    def test_subtitle_format_is_rejected_rather_than_migrated(self):
        """It is the one flat key deliberately left out of the migration map.

        Honouring it in the merged settings alone would break the render: the
        subtitle file's extension comes from the global value, so a profile
        asking for srt under a global of ass writes SRT text into
        `subtitles.ass` and the assembler hands that to FFmpeg's `ass` filter.
        A load error is the honest answer until the path follows the profile.
        """
        with pytest.raises(ValidationError, match="subtitle_format"):
            VideoProfile(**_profile(subtitle_format="srt"))

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

    def test_no_bundled_profile_sets_a_key_the_model_ignores(self):
        """Stronger than "the config loads", which `extra="forbid"` guarantees.

        A profile could still declare a key that migrates into the nested
        block and is then never read. This asserts the accepted-key set is
        exactly what the model plus the migration maps cover.
        """
        import yaml

        from src.video.config.visual_models import (
            _LEGACY_FLAT_TO_NESTED,
            _LEGACY_PYCAPS_FIELDS,
            _LEGACY_SAFE_ZONE_FIELDS,
        )

        accepted = (
            set(VideoProfile.model_fields)
            | set(_LEGACY_FLAT_TO_NESTED)
            | set(_LEGACY_SAFE_ZONE_FIELDS)
            | set(_LEGACY_PYCAPS_FIELDS)
            | {"two_part_subtitles"}
        )
        with open("config/video_production.yaml", encoding="utf-8") as f:
            profiles = yaml.safe_load(f)["video_profiles"]
        unknown = {
            name: sorted(set(block) - accepted) for name, block in profiles.items()
        }
        assert {n: k for n, k in unknown.items() if k} == {}

    def test_no_bundled_profile_still_sets_subtitle_format(self):
        """Seven of them did, and every one was dead.

        Left in place they would now fail the load outright, so their removal
        is what makes `extra="forbid"` safe to turn on.
        """
        import yaml

        with open("config/video_production.yaml", encoding="utf-8") as f:
            profiles = yaml.safe_load(f)["video_profiles"]
        assert [n for n, b in profiles.items() if "subtitle_format" in b] == []
