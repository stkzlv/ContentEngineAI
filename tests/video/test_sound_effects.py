"""Sparse sound effects on a few beats (design 0003), held off by default."""

from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.video.assembler.audio_builder import AudioFilterBuilder
from src.video.config import config, load_video_config_modular
from src.video.sound_effects import (
    SoundEvent,
    choose_file,
    plan_events,
    resolve_effects,
    sentence_starts,
)

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed"
)


def _words(*sentences: tuple[float, list[str]]) -> list[dict]:
    """Word timings, 0.3 s per word, each sentence starting at its time."""
    out = []
    for start, words in sentences:
        for i, word in enumerate(words):
            t = start + i * 0.3
            out.append({"word": word, "start_time": t, "end_time": t + 0.25})
    return out


WORDS = _words(
    (0.0, ["This", "lamp", "clips", "on."]),
    (2.0, ["It", "folds", "flat."]),
    (5.0, ["It", "charges", "by", "USB."]),
    (12.0, ["Link", "in", "bio."]),
)


@pytest.mark.req("REQ-VID-012")
def test_the_shipped_config_keeps_effects_off() -> None:
    effects = load_video_config_modular().audio_settings.sound_effects

    assert effects.enabled is False
    assert effects.hook == effects.reveal == effects.cta == []


def test_sentences_start_after_a_closing_mark() -> None:
    assert sentence_starts(WORDS) == [0.0, 2.0, 5.0, 12.0]


@pytest.mark.req("REQ-VID-012")
def test_the_hook_reveal_and_cta_clear_the_word_onset() -> None:
    events = plan_events(WORDS, 15.0, 2)

    # The hook is the first frame, moved past the first word's onset.
    assert events == [
        SoundEvent("hook", 0.1),
        SoundEvent("reveal", 2.1),
        SoundEvent("cta", 12.1),
    ]


@pytest.mark.req("REQ-VID-012")
def test_the_cap_drops_the_hook_first() -> None:
    # One per 10 s: the hook (0.1 s) and the reveal (2.1 s) share a window.
    events = plan_events(WORDS, 15.0, 1)

    assert [e.kind for e in events] == ["reveal", "cta"]


def test_no_word_timings_means_the_hook_only() -> None:
    assert plan_events([], 9.0, 2) == [SoundEvent("hook", 0.0)]


def _pool(tmp_path: Path, kind: str, n: int = 5) -> list[Path]:
    paths = []
    for i in range(n):
        path = tmp_path / f"{kind}_{i}.wav"
        path.write_bytes(b"x")
        paths.append(path)
    return paths


@pytest.mark.req("REQ-VID-012")
def test_two_products_draw_different_variants(tmp_path: Path) -> None:
    pool = _pool(tmp_path, "cta", 8)
    draws = {choose_file(pool, f"B0{i:03}", "cta", 0) for i in range(12)}

    assert len(draws) > 1
    assert choose_file(pool, "B0X", "cta", 0) == choose_file(pool, "B0X", "cta", 0)


def test_a_missing_file_is_skipped(tmp_path: Path, caplog) -> None:
    present = _pool(tmp_path, "reveal", 1)
    pool = [tmp_path / "gone.wav", *present]

    assert choose_file(pool, "B0X", "reveal", 0) == present[0]
    assert "gone.wav" in caplog.text
    assert choose_file([tmp_path / "gone.wav"], "B0X", "reveal", 0) is None


def _settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        hook=_pool(tmp_path, "hook"),
        reveal=_pool(tmp_path, "reveal"),
        cta=_pool(tmp_path, "cta"),
    )


@pytest.mark.req("REQ-VID-012")
def test_three_events_become_three_delayed_inputs_at_the_level(tmp_path) -> None:
    events = [
        SoundEvent("hook", 0.1),
        SoundEvent("reveal", 4.1),
        SoundEvent("cta", 12.1),
    ]
    effects = resolve_effects(_settings(tmp_path), events, "B0X")
    cfg = config.model_copy(deep=True)
    cfg.audio_settings.sound_effects.level_db = -12
    voice = tmp_path / "voice.wav"
    voice.write_bytes(b"x")
    parts: list[str] = []
    inspector = MagicMock(get_media_duration=AsyncMock(return_value=15.0))

    filters, _ = asyncio.run(
        AudioFilterBuilder(cfg).build_mix(
            parts, voice, None, 15.0, inspector, sound_effects=effects
        )
    )

    level = cfg.audio_settings.voiceover_volume_db - 12
    graph = ";".join(filters)
    for n, delay in enumerate((100, 4100, 12100)):
        assert f"volume={level:g}dB,adelay={delay}:all=1[a_sfx_{n}]" in graph
    assert parts.count("-i") == 4
    assert re.search(r"amix=inputs=4", graph)


def test_off_adds_nothing_to_the_command(tmp_path: Path) -> None:
    from src.video.assembler.core import VideoAssembler

    assembler = VideoAssembler(config)
    assert assembler._sound_effects(6.0, WORDS) == []


@pytest.mark.req("REQ-VID-012")
def test_the_assembler_places_effects_from_the_timeline(tmp_path: Path) -> None:
    from src.video.assembler.core import VideoAssembler

    cfg = config.model_copy(deep=True)
    effects = cfg.audio_settings.sound_effects
    effects.enabled = True
    for kind in ("hook", "reveal", "cta"):
        setattr(effects, kind, _pool(tmp_path, kind))
    assembler = VideoAssembler(cfg)
    assembler.product_id = "B0X"

    placed = assembler._sound_effects(15.0, WORDS)

    starts = sorted(start for _, start in placed)
    assert starts == [0.1, 2.1, 12.1]
    assert all(
        sum(1 for t in starts if s <= t < s + 10.0) <= effects.max_per_10_sec
        for s in starts
    )


@pytest.mark.req("REQ-VID-012")
@needs_ffmpeg
def test_a_mixed_effect_sounds_where_it_was_placed(tmp_path: Path) -> None:
    def tone(name: str, spec: str, seconds: float) -> Path:
        path = tmp_path / name
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", spec]
            + ["-t", str(seconds), str(path)],
            check=True,
        )
        return path

    voice = tone("voice.wav", "anullsrc=r=48000:cl=mono", 6.0)
    blip = tone("blip.wav", "sine=f=1500:sample_rate=48000", 0.3)
    cfg = load_video_config_modular()
    cfg.audio_settings.loudness_normalization_enabled = False
    from src.video.assembler.media_inspector import MediaInspector

    parts: list[str] = []
    filters, label = asyncio.run(
        AudioFilterBuilder(cfg).build_mix(
            parts, voice, None, 6.0, MediaInspector(), sound_effects=[(blip, 3.0)]
        )
    )
    out = tmp_path / "mix.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *parts, "-filter_complex"]
        + [";".join(filters), "-map", label, str(out)],
        check=True,
    )

    def peak(start: float, length: float) -> float:
        log = subprocess.run(
            ["ffmpeg", "-v", "info", "-ss", str(start), "-t", str(length)]
            + ["-i", str(out), "-af", "volumedetect", "-f", "null", "-"],
            capture_output=True,
            text=True,
            check=True,
        ).stderr
        value = re.search(r"max_volume: (-?[\d.]+|-inf) dB", log)
        assert value, log
        return float("-inf") if value.group(1) == "-inf" else float(value.group(1))

    assert peak(3.05, 0.2) > -40
    assert peak(1.0, 1.5) < -80


@pytest.mark.req("REQ-VID-012")
def test_the_render_path_feeds_the_mix() -> None:
    """The assembler hands its plan to the mix, and the step its word timings."""
    import inspect

    from src.video.assembler.core import VideoAssembler
    from src.video.producer import steps

    assemble = inspect.getsource(VideoAssembler.assemble_video)
    assert "sound_effects=self._sound_effects(" in assemble
    assert "spoken_words=_spoken_words(ctx)" in inspect.getsource(
        steps.step_assemble_video
    )


@pytest.mark.req("REQ-VID-012")
def test_an_empty_pool_does_not_take_the_caps_places(tmp_path: Path) -> None:
    from src.video.assembler.core import VideoAssembler

    cfg = config.model_copy(deep=True)
    effects = cfg.audio_settings.sound_effects
    effects.enabled = True
    effects.max_per_10_sec = 1
    effects.hook = _pool(tmp_path, "hook")
    effects.reveal = [tmp_path / "missing.wav"]  # listed, but not on disk
    assembler = VideoAssembler(cfg)

    # The reveal outranks the hook in their shared window, but cannot sound.
    placed = assembler._sound_effects(13.0, WORDS)

    assert [start for _, start in placed] == [0.1]


def test_the_step_reads_word_timings_from_the_transcript(tmp_path: Path) -> None:
    import json

    from src.video.producer import steps

    transcript = tmp_path / "whisper_transcript.json"
    transcript.write_text(
        json.dumps(
            {"segments": [{"words": [{"word": " Hi.", "start": 0.2, "end": 0.5}]}]}
        )
    )
    cfg = config.model_copy(deep=True)
    ctx = SimpleNamespace(config=cfg, run_paths={"whisper_transcript_file": transcript})

    assert steps._spoken_words(ctx) is None
    cfg.audio_settings.sound_effects.enabled = True
    assert steps._spoken_words(ctx) == [
        {"word": "Hi.", "start_time": 0.2, "end_time": 0.5}
    ]
    transcript.write_text("{not json")
    assert steps._spoken_words(ctx) is None
