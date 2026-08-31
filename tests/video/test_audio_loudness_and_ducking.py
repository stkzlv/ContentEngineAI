"""The finished mix is mastered to a loudness target, and can duck under voice.

Before this, nothing in the pipeline normalized loudness: `amix` ran with
`normalize=0` and the master level was whatever the fixed voiceover and music
gains happened to sum to. Two real renders measured -17.4 and -17.6 LUFS
integrated against a -14 target, with true peaks of -0.1 and -0.2 dBFS.

That pairing is the reason a fixed gain adjustment is the wrong fix: the mix
was quiet on average *and* nearly touching full scale, so raising the level
would clip and limiting alone would leave it quiet. `loudnorm` (EBU R128)
moves both at once, which is what these tests assert against real audio rather
than against the filter string.

The ducking half is off by default. `sidechaincompress` attenuates the music
while narration plays; it never boosts above the base level, so it is one half
of "music breathes in the gaps" and the config comment says so.
"""

from __future__ import annotations

import itertools
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from src.video.assembler.audio_builder import AudioFilterBuilder

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed"
)


def _config():
    from src.video.config import load_video_config_modular

    return load_video_config_modular()


def _run(args: list[str]) -> None:
    subprocess.run(args, check=True, capture_output=True)


def _tone(path: Path, freq: int, gain_db: float, duration: float = 12.0) -> Path:
    """A sine at `gain_db` relative to full scale.

    lavfi's `sine` is about 18 dB below full scale before any gain, so the
    `+18dB` here is what makes `gain_db` mean what it says. Without it a
    fixture asking for -3 dB produced a -21 dBFS peak, and the true-peak test
    below passed against unnormalised audio -- it was measuring a signal that
    was never near the ceiling it claims to defend.
    """
    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={freq}:duration={duration}:sample_rate=44100",
            "-af",
            f"volume=18dB,volume={gain_db}dB",
            str(path),
        ]
    )
    return path


def _voice_with_a_gap(path: Path) -> Path:
    """Narration that stops between t=4 and t=8.

    A duck needs a voice-free stretch to release into.

    Driven near full scale, like `_tone`, because duck depth is a function of
    how far the key sits above the threshold. This fixture originally used
    lavfi's stock sine level and so sat about 21 dB below real narration,
    which made every measured depth roughly half of what the same settings
    give on a real voiceover.
    """
    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=200:duration=12:sample_rate=44100",
            "-af",
            "volume=18dB,volume=-6dB,volume=enable='between(t,4,8)':volume=0.0001",
            str(path),
        ]
    )
    return path


def _render(tmp_path: Path, voice: Path, music: Path, filters, label) -> Path:
    out = tmp_path / "mixed.wav"
    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(voice),
            "-i",
            str(music),
            "-filter_complex",
            ";".join(filters),
            "-map",
            label,
            str(out),
        ]
    )
    return out


def _integrated_lufs(path: Path) -> float:
    proc = subprocess.run(
        [
            "ffmpeg",
            "-nostats",
            "-i",
            str(path),
            "-af",
            "ebur128=peak=true",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    match = re.search(r"Integrated loudness:\s*\n\s*I:\s*(-?[\d.]+)", proc.stderr)
    assert match, proc.stderr[-2000:]
    return float(match.group(1))


def _true_peak_dbfs(path: Path) -> float:
    proc = subprocess.run(
        [
            "ffmpeg",
            "-nostats",
            "-i",
            str(path),
            "-af",
            "ebur128=peak=true",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    match = re.search(r"True peak:\s*\n\s*Peak:\s*(-?[\d.]+)", proc.stderr)
    assert match, proc.stderr[-2000:]
    return float(match.group(1))


def _mean_dbfs(path: Path, start: float, end: float) -> float:
    proc = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "info",
            "-ss",
            str(start),
            "-to",
            str(end),
            "-i",
            str(path),
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    match = re.search(r"mean_volume:\s*(-?[\d.]+)", proc.stderr)
    assert match, proc.stderr[-2000:]
    return float(match.group(1))


@pytest.mark.integration
class TestTheMixIsMasteredToTheTarget:
    def test_the_output_lands_on_the_configured_target(self, tmp_path):
        """The claim the feature exists to make, measured on real audio."""
        config = _config()
        voice = _tone(tmp_path / "voice.wav", 200, -18)
        music = _tone(tmp_path / "music.wav", 600, -12)
        filters, label = AudioFilterBuilder(config).build_audio_filters(0, 1, 12.0)

        measured = _integrated_lufs(_render(tmp_path, voice, music, filters, label))

        assert measured == pytest.approx(
            config.audio_settings.loudness_target_lufs, abs=1.0
        )

    def test_the_true_peak_stays_under_the_ceiling(self, tmp_path):
        """Renders measured -0.1 dBFS before this, above the -1 dBTP guidance.

        The fixture is driven near full scale on purpose. With lavfi's stock
        sine level this test passed on unnormalised audio, because the mix
        peaked at -18 dBFS and was never near the ceiling.
        """
        config = _config()
        voice = _tone(tmp_path / "voice.wav", 200, -3)
        music = _tone(tmp_path / "music.wav", 600, -3)
        filters, label = AudioFilterBuilder(config).build_audio_filters(0, 1, 12.0)

        peak = _true_peak_dbfs(_render(tmp_path, voice, music, filters, label))

        assert peak <= config.audio_settings.loudness_true_peak_db + 0.5

    def test_a_quiet_mix_is_brought_up_not_left_alone(self, tmp_path):
        """Guards against the filter being present but inert.

        A pass-through would leave this near -40 LUFS, and the target
        assertion above would be the only thing failing.
        """
        config = _config()
        voice = _tone(tmp_path / "voice.wav", 200, -35)
        music = _tone(tmp_path / "music.wav", 600, -40)
        filters, label = AudioFilterBuilder(config).build_audio_filters(0, 1, 12.0)

        measured = _integrated_lufs(_render(tmp_path, voice, music, filters, label))

        assert measured > -20.0

    def test_the_rate_is_resampled_back_off_192k(self, tmp_path):
        """`loudnorm` emits at 192 kHz whatever it was handed.

        Without the `aresample` the encoder negotiates a rate nobody chose.
        """
        config = _config()
        voice = _tone(tmp_path / "voice.wav", 200, -18)
        music = _tone(tmp_path / "music.wav", 600, -12)
        filters, label = AudioFilterBuilder(config).build_audio_filters(0, 1, 12.0)
        rendered = _render(tmp_path, voice, music, filters, label)

        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=sample_rate",
                "-of",
                "csv=p=0",
                str(rendered),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        assert probe.stdout.strip() == str(
            config.audio_settings.output_audio_sample_rate
        )

    def test_it_can_be_turned_off(self, tmp_path):
        config = _config()
        config.audio_settings.loudness_normalization_enabled = False
        filters, _ = AudioFilterBuilder(config).build_audio_filters(0, 1, 12.0)

        assert not any("loudnorm" in f for f in filters)

    def test_the_rate_still_applies_with_normalisation_off(self, tmp_path):
        """`output_audio_sample_rate` names an output property.

        It was emitted as a tail of the loudnorm filter, so switching
        normalisation off silently dropped the rate control with it and left
        the render at whatever the voiceover happened to be.
        """
        config = _config()
        config.audio_settings.loudness_normalization_enabled = False
        voice = _tone(tmp_path / "voice.wav", 200, -18)
        music = _tone(tmp_path / "music.wav", 600, -12)
        filters, label = AudioFilterBuilder(config).build_audio_filters(0, 1, 12.0)
        rendered = _render(tmp_path, voice, music, filters, label)

        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=sample_rate",
                "-of",
                "csv=p=0",
                str(rendered),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        assert probe.stdout.strip() == str(
            config.audio_settings.output_audio_sample_rate
        )


@pytest.mark.integration
class TestTheDuck:
    """Off by default, so these drive it explicitly."""

    def _ducked_music_only(self, tmp_path):
        config = _config()
        config.audio_settings.music_ducking_enabled = True
        config.audio_settings.loudness_normalization_enabled = False
        voice = _voice_with_a_gap(tmp_path / "voice.wav")
        music = _tone(tmp_path / "music.wav", 600, -12)
        filters, _ = AudioFilterBuilder(config).build_audio_filters(0, 1, 12.0)

        # Take the music leg alone, so the measured level is the duck and not
        # the narration sitting on top of it. The split's other output has to
        # go somewhere or the graph will not bind.
        #
        # Everything before the mix, rather than a blocklist of filter names:
        # a blocklist dropped `amix` while keeping the `aresample` that reads
        # its output, leaving a dangling label.
        isolated = list(itertools.takewhile(lambda f: "amix" not in f, filters))
        isolated.append("[a_voice_mix]anullsink")
        out = tmp_path / "music_ducked.wav"
        _run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(voice),
                "-i",
                str(music),
                "-filter_complex",
                ";".join(isolated),
                "-map",
                "[a_music_ducked]",
                str(out),
            ]
        )
        return out

    def test_music_is_quieter_under_narration_than_in_the_gap(self, tmp_path):
        """The contract is that a duck happens, not that it is any depth.

        Asserted loosely on purpose, since the depth is four config fields
        away. The figure this fixture produces is not the one in the config
        table and should not be read as it: this compares speech against a
        pause on a synthetic voice and measures about 12 dB, where the table
        reports attenuation in a single window on a real voiceover and gives
        5.3 dB for the same settings.
        """
        ducked = self._ducked_music_only(tmp_path)

        under_voice = _mean_dbfs(ducked, 2.0, 3.5)
        in_the_gap = _mean_dbfs(ducked, 5.0, 7.5)

        assert in_the_gap - under_voice > 3.0

    def test_the_default_config_does_not_duck(self, tmp_path):
        """Enabling it changes the sound of every render, so it is opt-in."""
        filters, _ = AudioFilterBuilder(_config()).build_audio_filters(0, 1, 12.0)

        assert not any("sidechaincompress" in f for f in filters)

    def test_the_ducked_graph_is_accepted_end_to_end(self, tmp_path):
        """The isolation above drops the amix; this renders the real chain."""
        config = _config()
        config.audio_settings.music_ducking_enabled = True
        voice = _voice_with_a_gap(tmp_path / "voice.wav")
        music = _tone(tmp_path / "music.wav", 600, -12)
        filters, label = AudioFilterBuilder(config).build_audio_filters(0, 1, 12.0)

        rendered = _render(tmp_path, voice, music, filters, label)

        assert rendered.exists()

    def test_ducking_needs_both_tracks(self):
        """A voiceover with no music has nothing to duck, and vice versa."""
        config = _config()
        config.audio_settings.music_ducking_enabled = True
        builder = AudioFilterBuilder(config)

        voice_only, _ = builder.build_audio_filters(0, None, 12.0)
        music_only, _ = builder.build_audio_filters(None, 1, 12.0)

        assert not any("sidechaincompress" in f for f in voice_only)
        assert not any("sidechaincompress" in f for f in music_only)


@pytest.mark.unit
class TestTheChainShape:
    def test_normalisation_precedes_the_pad(self):
        """`apad` appends silence to reach the video duration.

        Normalising after it would be measuring the padding as programme.
        """
        filters, _ = AudioFilterBuilder(_config()).build_audio_filters(0, 1, 12.0)
        joined = ";".join(filters)

        assert joined.index("loudnorm") < joined.index("apad")

    def test_a_single_track_is_still_normalised(self):
        """There is no `amix` on this path, so it is easy to miss."""
        filters, label = AudioFilterBuilder(_config()).build_audio_filters(
            0, None, 12.0
        )

        assert any("loudnorm" in f for f in filters)
        assert label == "[a_final]"

    def test_no_audio_at_all_yields_no_label(self):
        filters, label = AudioFilterBuilder(_config()).build_audio_filters(
            None, None, 12.0
        )

        assert label == ""
        assert not any("loudnorm" in f for f in filters)
