"""A transcription timeout must not discard a render that is already paid for.

The limit comes from `base_timeout_sec + audio_duration * duration_multiplier`,
which knows nothing about how fast the machine transcribes. Measured on the
same 26.3s clip twice in one day: 268.5s on an idle 16-core box and 305.5s
under load, against a 277.7s limit. The second run lost the whole render, and
`generate_subtitles` sits after `generate_script` and `create_voiceover`, so
what was thrown away had already paid for an LLM call and a TTS call.

`make *-lowpri` -- the documented way to run a batch -- slows transcription
deliberately, so the resource rule and the timeout formula pulled against each
other.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.video.config.core_models import WhisperSettings
from src.video.stt_functions import _calculate_timeout, _timeout_schedule


class TestTheSchedule:
    def test_a_retry_gets_a_wider_limit(self):
        settings = WhisperSettings(
            timeout_retry_attempts=1, timeout_retry_multiplier=2.0
        )

        assert _timeout_schedule(300.0, settings) == [300.0, 600.0]

    def test_the_ceiling_is_respected(self):
        settings = WhisperSettings(
            max_timeout_sec=400, timeout_retry_attempts=1, timeout_retry_multiplier=2.0
        )

        assert _timeout_schedule(300.0, settings) == [300.0, 400.0]

    def test_a_limit_already_at_the_ceiling_is_not_retried(self):
        """A retry on the same limit cannot beat the attempt that just failed.

        Repeating it would double the wall clock for a certain second failure,
        on a run that has already spent five minutes discovering the first.
        """
        settings = WhisperSettings(
            max_timeout_sec=300, timeout_retry_attempts=3, timeout_retry_multiplier=2.0
        )

        assert _timeout_schedule(300.0, settings) == [300.0]

    def test_retries_can_be_turned_off(self):
        settings = WhisperSettings(timeout_retry_attempts=0)

        assert _timeout_schedule(300.0, settings) == [300.0]

    def test_a_multiplier_of_one_does_not_loop(self):
        """`* 1.0` never widens, so the guard has to catch it, not the ceiling."""
        settings = WhisperSettings(
            timeout_retry_attempts=5, timeout_retry_multiplier=1.0
        )

        assert _timeout_schedule(300.0, settings) == [300.0]

    def test_more_attempts_widen_further(self):
        settings = WhisperSettings(
            max_timeout_sec=10000,
            timeout_retry_attempts=2,
            timeout_retry_multiplier=2.0,
        )

        assert _timeout_schedule(100.0, settings) == [100.0, 200.0, 400.0]


class TestTheShippedDefaults:
    """The measured numbers are the reason these values are what they are."""

    MEASURED_AUDIO_SEC = 26.3
    MEASURED_SLOWEST_SEC = 305.5

    def test_the_default_limit_clears_the_slowest_measured_run(self):
        """At the old 6.0 multiplier this was 277.7s against a 305.5s run."""
        limit = _calculate_timeout(self.MEASURED_AUDIO_SEC, WhisperSettings())

        assert limit > self.MEASURED_SLOWEST_SEC * 1.5, (
            f"{limit:.0f}s leaves too little margin over the slowest measured "
            f"run ({self.MEASURED_SLOWEST_SEC}s); contention is not linear in "
            "audio length"
        )

    def test_a_full_length_render_is_not_capped(self):
        """A 45s clip is the long end of what the profiles produce.

        `max_timeout_sec` clamping it would put the ceiling back in charge of
        the limit, which is the failure this change removes.
        """
        settings = WhisperSettings()
        limit = _calculate_timeout(45.0, settings)

        assert limit < settings.max_timeout_sec

    def test_the_bundled_config_matches_the_model(self):
        """The YAML restates these, so the two can drift apart."""
        import yaml

        with open("config/ai_services.yaml") as handle:
            section = yaml.safe_load(handle)["whisper_settings"]

        defaults = WhisperSettings()
        assert section["duration_multiplier"] == defaults.duration_multiplier
        assert section["max_timeout_sec"] == defaults.max_timeout_sec
        assert section["timeout_retry_attempts"] == defaults.timeout_retry_attempts


@pytest.mark.asyncio
async def test_a_timed_out_transcription_is_retried(tmp_path):
    """The regression guard: without the loop this returns None.

    The first attempt times out exactly as the lost render did; the second
    succeeds on the widened limit.
    """
    import src.video.stt_functions as stt

    audio = tmp_path / "voiceover.wav"
    audio.write_bytes(b"")

    attempts = {"n": 0}
    result = {
        "segments": [
            {"words": [{"word": " hello", "start": 0.0, "end": 0.5}]},
        ]
    }

    async def flaky(*args, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise TimeoutError
        return result

    settings = WhisperSettings(
        base_timeout_sec=1,
        duration_multiplier=1.0,
        max_timeout_sec=1000,
        timeout_retry_attempts=1,
        enable_resource_monitoring=False,
        enable_resource_cleanup=False,
    )

    with (
        patch.object(stt, "WHISPER_AVAILABLE", True),
        patch.object(stt, "_load_whisper_model", return_value=object()),
        patch.object(stt, "_prepare_transcription_options", return_value={}),
        patch.object(stt, "_log_audio_file_info", return_value=None),
        patch.object(stt, "_get_audio_duration", return_value=10.0),
        patch.object(stt, "_transcribe_with_monitoring", new=flaky),
    ):
        timings = await stt.generate_subtitles_with_whisper(
            audio, tmp_path, settings, debug_mode=False
        )

    assert attempts["n"] == 2, "the first timeout should not have ended the step"
    assert timings, "the retry's transcript should have reached the caller"


@pytest.mark.asyncio
async def test_a_persistent_timeout_still_gives_up(tmp_path):
    """Bounded, so a hung machine cannot hold the pipeline open."""
    import src.video.stt_functions as stt

    audio = tmp_path / "voiceover.wav"
    audio.write_bytes(b"")

    attempts = {"n": 0}

    async def always_slow(*args, **kwargs):
        attempts["n"] += 1
        raise TimeoutError

    settings = WhisperSettings(
        base_timeout_sec=1,
        duration_multiplier=1.0,
        max_timeout_sec=1000,
        timeout_retry_attempts=1,
        enable_resource_monitoring=False,
        enable_resource_cleanup=False,
    )

    with (
        patch.object(stt, "WHISPER_AVAILABLE", True),
        patch.object(stt, "_load_whisper_model", return_value=object()),
        patch.object(stt, "_prepare_transcription_options", return_value={}),
        patch.object(stt, "_log_audio_file_info", return_value=None),
        patch.object(stt, "_get_audio_duration", return_value=10.0),
        patch.object(stt, "_transcribe_with_monitoring", new=always_slow),
    ):
        timings = await stt.generate_subtitles_with_whisper(
            audio, tmp_path, settings, debug_mode=False
        )

    assert attempts["n"] == 2
    assert timings is None


@pytest.mark.asyncio
async def test_a_non_timeout_failure_is_not_retried(tmp_path):
    """A broken model or a corrupt file fails the same way twice."""
    import src.video.stt_functions as stt

    audio = tmp_path / "voiceover.wav"
    audio.write_bytes(b"")

    attempts = {"n": 0}

    async def broken(*args, **kwargs):
        attempts["n"] += 1
        raise RuntimeError("model is broken")

    settings = WhisperSettings(
        timeout_retry_attempts=1,
        enable_resource_monitoring=False,
        enable_resource_cleanup=False,
    )

    with (
        patch.object(stt, "WHISPER_AVAILABLE", True),
        patch.object(stt, "_load_whisper_model", return_value=object()),
        patch.object(stt, "_prepare_transcription_options", return_value={}),
        patch.object(stt, "_log_audio_file_info", return_value=None),
        patch.object(stt, "_get_audio_duration", return_value=10.0),
        patch.object(stt, "_transcribe_with_monitoring", new=broken),
    ):
        timings = await stt.generate_subtitles_with_whisper(
            audio, tmp_path, settings, debug_mode=False
        )

    assert attempts["n"] == 1
    assert timings is None
