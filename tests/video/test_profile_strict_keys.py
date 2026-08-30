"""Tests that an unknown key in a profile block fails at load.

Pydantic drops unknown keys by default, and a dropped key in a profile is
invisible: the render succeeds using the global value, so the profile appears
to work while its override does nothing. `docs/requirements.md` has claimed
strict validation for profiles since before the model had it.
"""

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
class TestLegacyKeysRefused:
    """The flat keys were migrated for one release; that window is closed.

    The bundled profiles are nested now, so the shim's only remaining job was
    to accept config nobody ships. It is replaced by an error that names the
    nested field to move each key to.
    """

    def test_a_flat_key_is_refused_with_its_replacement_named(self):
        """`extra="forbid"` alone would say only "Extra inputs are not
        permitted", leaving the reader to work out that `subtitle_anchor` is
        now `subtitle_settings.anchor`.
        """
        with pytest.raises(ValidationError) as excinfo:
            VideoProfile(**_profile(subtitle_anchor="below_content"))

        message = str(excinfo.value)
        assert "subtitle_anchor -> subtitle_settings.anchor" in message

    def test_every_offending_key_is_listed_at_once(self):
        """One key at a time would mean one config-load cycle per key."""
        with pytest.raises(ValidationError) as excinfo:
            VideoProfile(
                **_profile(
                    subtitle_anchor="below_content",
                    subtitle_safe_zone_min_y=0.1,
                    pycaps_template="hype",
                )
            )

        message = str(excinfo.value)
        assert "subtitle_anchor -> subtitle_settings.anchor" in message
        assert (
            "subtitle_safe_zone_min_y -> subtitle_settings.safe_zone.min_y" in message
        )
        assert "pycaps_template -> subtitle_settings.pycaps.template_name" in message

    def test_a_nested_profile_still_loads(self):
        """The shape the bundled profiles now use."""
        profile = VideoProfile(
            **_profile(subtitle_settings={"anchor": "below_content"})
        )

        assert profile.subtitle_settings is not None
        assert profile.subtitle_settings.anchor == "below_content"

    def test_subtitle_format_is_refused_like_its_siblings(self):
        """It is a normal subtitle key now, in both directions.

        It used to be the one flat key deliberately left out of the map,
        because honouring it in the merged settings alone broke the render:
        the file's extension came from the global value, so a profile asking
        for srt under a global of ass wrote SRT text into `subtitles.ass`.
        The path follows the profile now, so the key is settable per profile
        (see `test_subtitle_format_per_profile.py`) -- in the nested spelling,
        which is the only spelling any subtitle key has.
        """
        with pytest.raises(ValidationError) as excinfo:
            VideoProfile(**_profile(subtitle_format="srt"))

        assert "subtitle_format -> subtitle_settings.subtitle_format" in str(
            excinfo.value
        )

    def test_a_nested_block_is_accepted(self):
        profile = VideoProfile(**_profile(subtitle_settings={"style_preset": "bold"}))
        assert profile.subtitle_settings is not None
        assert profile.subtitle_settings.style_preset == "bold"

    def test_the_nested_route_sets_the_format_too(self):
        """Both spellings reach the same place.

        The nested one was rejected explicitly, because `extra="forbid"` only
        sees the flat key and `PartialSubtitleSettings` declares the field --
        so rejecting only the flat spelling would have left the same render
        failure one line of YAML away. Neither is rejected now.
        """
        profile = VideoProfile(**_profile(subtitle_settings={"subtitle_format": "srt"}))

        assert profile.subtitle_settings is not None
        assert profile.subtitle_settings.subtitle_format == "srt"


@pytest.mark.unit
class TestBundledProfilesLoad:
    """Every shipped profile must survive the stricter model.

    A dead key in one of them becomes a config-load failure, which is the
    point, but it has to be found here rather than by a user.
    """

    def test_all_bundled_profiles_parse(self):
        from src.video.config import load_video_config_modular

        config = load_video_config_modular()

        assert len(config.video_profiles) >= 11

    def test_no_bundled_profile_sets_a_key_the_model_ignores(self):
        """States the accepted-key set, so a reader can see what it is.

        It cannot fail on a config the loader accepts: the set is the model's
        own fields, and `extra="forbid"` rejects everything else. It is here
        as a readable inventory, not as a second gate. It used to add the
        three legacy maps, which the validator popped from; the validator now
        raises on them, so naming them here would advertise 29 keys that fail
        the load.
        """
        import yaml

        accepted = set(VideoProfile.model_fields)
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
