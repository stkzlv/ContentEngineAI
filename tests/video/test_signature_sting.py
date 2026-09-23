"""The signature sting lands where configured, through the same master (#440).

Asserted against real audio: a one-second tone mixed into silence, and the
energy measured in windows before, during and after its configured start.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from src.video.assembler.audio_builder import AudioFilterBuilder
from src.video.config import SignatureSting, load_video_config_modular

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed"
)


def _run(args: list[str]) -> str:
    done = subprocess.run(args, check=True, capture_output=True, text=True)
    return done.stderr


def _source(path: Path, spec: str, duration: float) -> Path:
    _run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", spec,
          "-t", str(duration), str(path)])  # fmt: skip
    return path


def _max_volume(path: Path, start: float, length: float) -> float:
    out = _run(["ffmpeg", "-v", "info", "-ss", str(start), "-t", str(length),
                "-i", str(path), "-af", "volumedetect", "-f", "null", "-"])  # fmt: skip
    match = re.search(r"max_volume: (-?[\d.]+|-inf) dB", out)
    assert match, out
    return float("-inf") if match.group(1) == "-inf" else float(match.group(1))


def _render(tmp_path: Path, sting: SignatureSting | None) -> Path:
    config = load_video_config_modular()
    config.audio_settings.signature_sting = sting
    config.audio_settings.loudness_normalization_enabled = False
    builder = AudioFilterBuilder(config)
    voice = _source(tmp_path / "voice.wav", "anullsrc=r=48000:cl=mono", 6.0)
    parts: list[str] = []
    voice_idx, music_idx = builder.prepare_audio_inputs(parts, voice, None, 0)
    sting_idx = builder.prepare_sting_input(parts, builder.sting_path())
    delay = builder.sting_delay_sec(1.0, 6.0) if builder.sting_path() is not None else 0
    filters, label = builder.build_audio_filters(
        voice_idx, music_idx, 6.0, sting_input_idx=sting_idx, sting_delay_sec=delay
    )
    out = tmp_path / "mix.wav"
    _run(["ffmpeg", "-y", "-v", "error", *parts,
          "-filter_complex", ";".join(filters), "-map", label, str(out)])  # fmt: skip
    return out


@pytest.fixture
def tone(tmp_path: Path) -> Path:
    return _source(tmp_path / "sting.wav", "sine=frequency=1000:sample_rate=48000", 1)


class TestPlacement:
    def test_start_with_an_offset(self, tmp_path: Path, tone: Path) -> None:
        mix = _render(tmp_path, SignatureSting(path=tone, offset_sec=2.0))
        assert _max_volume(mix, 0.0, 1.8) < -60
        assert _max_volume(mix, 2.1, 0.8) > -40
        assert _max_volume(mix, 3.2, 2.5) < -60

    def test_end_finishes_before_the_end(self, tmp_path: Path, tone: Path) -> None:
        mix = _render(
            tmp_path, SignatureSting(path=tone, position="end", offset_sec=0.5)
        )
        assert _max_volume(mix, 0.0, 4.3) < -60
        assert _max_volume(mix, 4.6, 0.8) > -40
        assert _max_volume(mix, 5.6, 0.4) < -60

    def test_the_delay_is_clamped_inside_the_video(self) -> None:
        config = load_video_config_modular()
        config.audio_settings.signature_sting = SignatureSting(
            path=Path("x.wav"), offset_sec=30.0
        )
        assert AudioFilterBuilder(config).sting_delay_sec(1.0, 6.0) == 5.0


class TestOffIsTheOldMix:
    def test_unconfigured_adds_no_input_or_filter(self) -> None:
        config = load_video_config_modular()
        config.audio_settings.signature_sting = None
        builder = AudioFilterBuilder(config)
        parts = ["-i", "v.mp4"]
        assert builder.prepare_sting_input(parts, builder.sting_path()) is None
        assert parts == ["-i", "v.mp4"]
        filters, _ = builder.build_audio_filters(1, 2, 10.0)
        assert not any("a_sting" in f for f in filters)

    def test_a_missing_file_mixes_nothing(self, tmp_path: Path) -> None:
        config = load_video_config_modular()
        config.audio_settings.signature_sting = SignatureSting(
            path=tmp_path / "absent.wav"
        )
        assert AudioFilterBuilder(config).sting_path() is None

    def test_an_unknown_key_is_refused(self) -> None:
        with pytest.raises(ValueError):
            SignatureSting.model_validate({"path": "a.wav", "positon": "end"})
