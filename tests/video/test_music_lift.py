"""Music rises between voice lines and ducks under them (REQ-CNT-178)."""

from __future__ import annotations

import itertools
import shutil
from pathlib import Path

import pytest

from src.video.assembler.audio_builder import AudioFilterBuilder
from src.video.config.audio_models import MusicLiftSettings
from tests.video.test_audio_loudness_and_ducking import (
    _config,
    _mean_dbfs,
    _run,
    _tone,
    _voice_with_a_gap,
)


@pytest.mark.req("REQ-CNT-178")
def test_the_lift_ships_off() -> None:
    assert MusicLiftSettings().enabled is False
    assert _config().audio_settings.music_lift.enabled is False


@pytest.mark.req("REQ-CNT-178")
def test_off_the_mix_keeps_the_fixed_level() -> None:
    config = _config()
    filters, _ = AudioFilterBuilder(config).build_audio_filters(0, 1, 12.0)
    joined = ";".join(filters)

    assert f"volume={config.audio_settings.music_volume_db}dB" in joined
    assert "sidechaincompress" not in joined


@pytest.mark.req("REQ-CNT-178")
def test_on_the_music_sits_at_the_gap_level_and_ducks_by_the_lift() -> None:
    config = _config()
    config.audio_settings.music_ducking_enabled = False
    config.audio_settings.music_lift = MusicLiftSettings(
        enabled=True, gap_volume_db=-9.0, duck_threshold=0.04, duck_ratio=7.0
    )
    filters, _ = AudioFilterBuilder(config).build_audio_filters(0, 1, 12.0)
    joined = ";".join(filters)

    assert "volume=-9.0dB" in joined
    assert f"volume={config.audio_settings.music_volume_db}dB" not in joined
    assert "sidechaincompress=threshold=0.04:ratio=7.0:" in joined


def _music_leg(tmp_path: Path, lift: bool) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = _config()
    config.audio_settings.loudness_normalization_enabled = False
    config.audio_settings.music_lift.enabled = lift
    voice = _voice_with_a_gap(tmp_path / "voice.wav")
    music = _tone(tmp_path / "music.wav", 600, -12)
    filters, _ = AudioFilterBuilder(config).build_audio_filters(0, 1, 12.0)
    isolated = list(itertools.takewhile(lambda f: "amix" not in f, filters))
    if lift:
        isolated.append("[a_voice_mix]anullsink")
        label = "[a_music_ducked]"
    else:
        isolated.append("[a_voice_proc]anullsink")
        label = "[a_music_proc]"
    out = tmp_path / f"music_{lift}.wav"
    _run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(voice), "-i", str(music)]
        + ["-filter_complex", ";".join(isolated), "-map", label, str(out)]
    )
    return out


@pytest.mark.req("REQ-CNT-178")
@pytest.mark.integration
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_the_music_rises_in_the_gap_and_drops_under_the_voice(tmp_path: Path) -> None:
    fixed = _music_leg(tmp_path / "off", lift=False)
    lifted = _music_leg(tmp_path / "on", lift=True)

    gap_rise = _mean_dbfs(lifted, 5.0, 7.5) - _mean_dbfs(fixed, 5.0, 7.5)
    duck = _mean_dbfs(lifted, 5.0, 7.5) - _mean_dbfs(lifted, 2.0, 3.5)

    # The gap level moves from music_volume_db (-24) to gap_volume_db (-10).
    assert gap_rise == pytest.approx(14.0, abs=1.0)
    assert duck > 3.0
