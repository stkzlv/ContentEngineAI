"""Source video audio is not carried into the render, and asking for it fails.

`AudioFilterBuilder.build_audio_filters_with_video_audio` was defined and never
called: the assembler's only audio call site is `build_audio_filters`, whose
signature takes no video-audio arguments. So `video_audio_handling` and
`video_original_volume` reached the merged settings and stopped there, and four
bundled profiles configured a mix they never got.

Wiring it up was the other option. Against it: the setting has never run in any
release, so nothing regresses by removing it; at the configured -30/-35 dB it
sat below the music bed at -24 dB under narration at 0 dB, so the benefit was
inaudible; and both video sources are third-party -- a scraped product video is
a manufacturer's marketing clip with a licensed music bed, stock footage is
whatever the contributor recorded -- so mixing either into a published upload
is an audio-match risk on all three platforms, where a claim or a mute zeroes
the video.

Removing it also makes the keys fail loudly. `VideoProfile` sets
`extra="forbid"`, so a profile naming one now aborts config load rather than
having the value accepted into a merged setting nothing reads.
"""

from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

from src.video.assembler.audio_builder import AudioFilterBuilder
from src.video.config.visual_models import VideoProfile, VideoSettings


class TestTheKeysAreRefused:
    def _profile(self, **extra):
        return VideoProfile(description="d", use_scraped_images=True, **extra)

    @pytest.mark.parametrize(
        "key,value",
        [("video_audio_handling", "mixed"), ("video_original_volume", -30.0)],
    )
    def test_a_profile_naming_one_fails_to_load(self, key, value):
        with pytest.raises(ValidationError) as excinfo:
            self._profile(**{key: value})

        assert "extra_forbidden" in str(excinfo.value)

    @pytest.mark.parametrize("key", ["video_audio_handling", "video_original_volume"])
    def test_neither_model_declares_it(self, key):
        """Declared on `VideoSettings` was how the value looked live."""
        assert key not in VideoProfile.model_fields
        assert key not in VideoSettings.model_fields

    def test_a_profile_without_them_still_loads(self):
        """So the tests above cannot pass by refusing everything."""
        assert self._profile().use_scraped_images is True


class TestTheMixerIsGone:
    def test_the_uncalled_method_no_longer_exists(self):
        assert not hasattr(AudioFilterBuilder, "build_audio_filters_with_video_audio")

    def test_the_surviving_builder_takes_no_video_audio(self):
        """The signature is what made the dead one unreachable."""
        params = inspect.signature(AudioFilterBuilder.build_audio_filters).parameters

        assert not [p for p in params if "video_audio" in p or "original_volume" in p]

    def test_no_bundled_profile_configures_video_audio(self):
        import yaml

        with open("config/video_production.yaml", encoding="utf-8") as handle:
            profiles = yaml.safe_load(handle)["video_profiles"]

        offenders = {
            name: sorted(k for k in (block or {}) if k.startswith("video_audio"))
            for name, block in profiles.items()
            if any(
                k in (block or {})
                for k in ("video_audio_handling", "video_original_volume")
            )
        }
        assert not offenders, f"{offenders} would now abort config load"
