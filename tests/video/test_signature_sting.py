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


VOICE_SEC = 5.0
VIDEO_SEC = 6.0  # the outro runs past the narration, as on a real render


def _render(tmp_path: Path, sting: SignatureSting | None) -> Path:
    """Mix through `build_mix`, the one entry point the assembler calls."""
    import asyncio

    from src.video.assembler.media_inspector import MediaInspector

    config = load_video_config_modular()
    config.audio_settings.signature_sting = sting
    config.audio_settings.loudness_normalization_enabled = False
    builder = AudioFilterBuilder(config)
    voice = _source(tmp_path / "voice.wav", "anullsrc=r=48000:cl=mono", VOICE_SEC)
    parts: list[str] = []
    filters, label = asyncio.run(
        builder.build_mix(parts, voice, None, VIDEO_SEC, MediaInspector())
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

    def test_end_finishes_before_the_narration_ends(
        self, tmp_path: Path, tone: Path
    ) -> None:
        """Against the narration, not the video: the mix stops when the voice
        does, and a sting placed against the video's end played in the outro
        and was silenced.
        """
        mix = _render(
            tmp_path, SignatureSting(path=tone, position="end", offset_sec=0.5)
        )
        assert _max_volume(mix, 0.0, 3.3) < -60
        assert _max_volume(mix, 3.6, 0.8) > -40
        assert _max_volume(mix, 4.6, 1.3) < -60

    def test_end_with_no_offset_is_heard_whole(
        self, tmp_path: Path, tone: Path
    ) -> None:
        mix = _render(tmp_path, SignatureSting(path=tone, position="end"))
        assert _max_volume(mix, 4.1, 0.8) > -40

    def test_the_delay_is_clamped_inside_the_video(self) -> None:
        config = load_video_config_modular()
        config.audio_settings.signature_sting = SignatureSting(
            path=Path("x.wav"), offset_sec=30.0
        )
        assert AudioFilterBuilder(config).sting_delay_sec(1.0, 6.0) == 5.0

    def test_the_assembler_mixes_through_build_mix(self) -> None:
        """The sting's input, placement and filter are wired in one method;
        an assembler that built the filters itself would drop the sting.
        """
        source = Path("src/video/assembler/core.py").read_text()
        assert "audio_builder.build_mix(" in source
        assert "audio_builder.build_audio_filters(" not in source


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
