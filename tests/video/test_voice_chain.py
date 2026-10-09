"""The optional voice processing chain (design 0004), held off by default."""

from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.video.assembler.audio_builder import AudioFilterBuilder, voice_chain_filters
from src.video.config import config, load_video_config_modular
from src.video.render_choices import choices_from_context


def _settings(**chain):
    cfg = config.model_copy(deep=True)
    for name, value in {"enabled": True, **chain}.items():
        setattr(cfg.audio_settings.voice_chain, name, value)
    return cfg


@pytest.mark.req("REQ-CNT-073")
def test_the_shipped_config_turns_the_chain_on() -> None:
    assert load_video_config_modular().audio_settings.voice_chain.enabled is True


@pytest.mark.req("REQ-CNT-073")
def test_the_chain_follows_the_configured_parameters() -> None:
    cfg = _settings(highpass_hz=90, harsh_cut_hz=3500, compressor_ratio=4)

    filters, _ = AudioFilterBuilder(cfg).build_audio_filters(0, None, 10.0)
    voice = next(f for f in filters if f.startswith("[0:a]"))

    assert voice.startswith(
        "[0:a]highpass=f=90,equalizer=f=3500:t=o:w=1:g=-2,"
        "acompressor=threshold=-18dB:ratio=4:attack=5:release=80,"
        "deesser=i=0.4,highshelf=f=9000:g=1.5,alimiter=limit=0.891:level=0,"
        "volume="
    )


def test_stages_set_to_nothing_drop_out() -> None:
    chain = voice_chain_filters(
        _settings(deess_intensity=0, air_shelf_db=0, limiter=False).audio_settings
    )

    assert "deesser" not in chain
    assert "highshelf" not in chain
    assert "alimiter" not in chain
    assert chain.startswith("highpass=") and chain.endswith(",")


@pytest.mark.req("REQ-CNT-073")
def test_off_is_the_plain_volume_stage() -> None:
    off = _settings(enabled=False)
    filters, _ = AudioFilterBuilder(off).build_audio_filters(0, None, 10.0)

    volume = off.audio_settings.voiceover_volume_db
    assert f"[0:a]volume={volume}dB[a_voice_proc]" in filters


@pytest.mark.req("REQ-CNT-074")
def test_each_render_records_the_chain_beside_the_voice(tmp_path: Path) -> None:
    def row(cfg):
        return choices_from_context(
            SimpleNamespace(
                state={"tts_metadata": {"voice_name": "Charon"}},
                config=cfg,
                profile_name="p",
                profile=SimpleNamespace(video_assembly_mode="sequential"),
                run_paths={"music_info_file": None, "run_root": tmp_path / "B0X"},
                script="x",
            )
        )

    assert row(_settings())["voice_chain"] is True
    assert row(_settings(enabled=False))["voice_chain"] is False
    assert row(config)["voice_name"] == "Charon"


def _integrated(path: Path) -> float:
    log = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128"]
        + ["-f", "null", "-"],
        capture_output=True,
        text=True,
        check=True,
    ).stderr
    return float(re.findall(r"I:\s+(-?[\d.]+) LUFS", log)[-1])


@pytest.mark.req("REQ-CNT-073")
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_the_mastered_mix_keeps_its_loudness_target(tmp_path: Path) -> None:
    from src.video.assembler.media_inspector import MediaInspector

    # Speech-like: noise in syllable-length bursts, with sibilant highs.
    voice = tmp_path / "voice.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i"]
        + [
            "anoisesrc=d=12:c=pink:r=48000:a=0.5,"
            "volume='0.2+0.8*gt(sin(2*PI*3*t),0)':eval=frame",
            str(voice),
        ],
        check=True,
    )
    levels = {}
    for on in (False, True):
        cfg = load_video_config_modular()
        cfg.audio_settings.voice_chain.enabled = on
        parts: list[str] = []
        filters, label = asyncio.run(
            AudioFilterBuilder(cfg).build_mix(
                parts, voice, None, 12.0, MediaInspector()
            )
        )
        out = tmp_path / f"mix_{on}.wav"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", *parts, "-filter_complex"]
            + [";".join(filters), "-map", label, str(out)],
            check=True,
        )
        levels[on] = _integrated(out)

    target = config.audio_settings.loudness_target_lufs
    # The mix lands short of the target either way (the true-peak ceiling);
    # the chain must not push it past the target or further below it.
    assert levels[True] <= target + 0.5
    assert abs(levels[True] - target) <= abs(levels[False] - target) + 0.5


def _levels(path: Path) -> tuple[float, float]:
    """Integrated loudness (LUFS) and true peak (dBTP) of `path`."""
    log = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af"]
        + ["ebur128=peak=true", "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=True,
    ).stderr
    loud = float(re.findall(r"I:\s+(-?[\d.]+) LUFS", log)[-1])
    peak = float(re.findall(r"Peak:\s+(-?[\d.]+) dBFS", log)[-1])
    return loud, peak


def _chain_output(tmp_path: Path, clicks: float) -> tuple[Path, Path]:
    """A speech-like source, and the same through the chain alone.

    A low and a sibilant partial in syllable-length bursts, plus 0.5 ms
    clicks of amplitude `clicks`: shorter than the compressor's attack, so
    only the limiter can hold them. Float samples keep peaks above 0 dBFS.
    """
    voice = tmp_path / f"voice_{clicks}.wav"
    expr = (
        "0.1*(0.7*sin(2*PI*220*t)+0.3*sin(2*PI*5000*t))"
        f"*(0.2+0.8*gt(sin(2*PI*3*t),0))+{clicks}*lt(mod(t,0.5),0.0005)"
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i"]
        + [f"aevalsrc='{expr}':s=48000:d=8", "-c:a", "pcm_f32le", str(voice)],
        check=True,
    )
    chained = tmp_path / f"chained_{clicks}.wav"
    chain = voice_chain_filters(_settings().audio_settings).rstrip(",")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(voice), "-af", chain]
        + ["-c:a", "pcm_f32le", str(chained)],
        check=True,
    )
    return voice, chained


@pytest.mark.req("REQ-CNT-073")
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_the_chain_polishes_without_moving_the_level(tmp_path: Path) -> None:
    """The chain alone, before the mastering pass that would hide a gain change.

    Without clicks the limiter never engages, so the level must hold within
    0.5 LU: an auto-levelling limiter adds a fixed 1 dB. With clicks the
    ceiling must hold.
    """
    plain, plain_out = _chain_output(tmp_path, 0.0)
    clicked, clicked_out = _chain_output(tmp_path, 2.0)

    assert abs(_levels(plain_out)[0] - _levels(plain)[0]) <= 0.5
    assert _levels(clicked)[1] > 0.0
    assert _levels(clicked_out)[1] <= -0.5
