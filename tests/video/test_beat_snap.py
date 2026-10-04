"""Visual cuts snapped to music beats (design 0005), held off by default."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.video.assembler.visual_builder import VisualFilterBuilder
from src.video.beats import detect_beats, snap_durations
from src.video.config import config
from src.video.config.visual_models import BeatSnapSettings

TD = 0.5
# A click every 0.5 s, 120 BPM, offset so the default cuts sit between beats.
BEATS = [0.12 + 0.5 * n for n in range(60)]


def _cuts(durations: list[float]) -> list[float]:
    cuts, offset = [], 0.0
    for d in durations[:-1]:
        offset += d - TD
        cuts.append(offset + TD / 2)
    return cuts


@pytest.mark.req("REQ-VID-013")
def test_the_shipped_config_keeps_beat_snap_off() -> None:
    assert config.video_settings.beat_snap.enabled is False
    assert config.video_settings.beat_snap.window_ms == 150


@pytest.mark.req("REQ-VID-013")
def test_cuts_land_on_beats_and_the_length_holds() -> None:
    durations = [3.1] * 6

    snapped = snap_durations(durations, [False] * 6, TD, BEATS, 0.15, 1.5, 15.0)

    moved = [
        new
        for old, new in zip(_cuts(durations), _cuts(snapped), strict=True)
        if abs(old - new) > 1e-9
    ]
    # A cut moves only when a beat lies inside the window, and then onto it.
    assert len(moved) >= 3
    assert all(min(abs(b - c) for b in BEATS) <= 0.03 for c in moved)
    assert sum(snapped) == pytest.approx(sum(durations))


def test_no_segment_drops_below_the_minimum() -> None:
    durations = [1.6, 1.6, 1.6, 1.6]

    snapped = snap_durations(durations, [False] * 4, TD, BEATS, 0.15, 1.55, 6.0)

    assert min(snapped) >= 1.55


def test_a_video_clip_is_never_lengthened() -> None:
    # Cuts at 2.75 and 5.25 s, each 0.13 s after a beat: inside the window.
    durations = [3.0, 3.0, 3.0]
    is_video = [True, True, True]
    assert snap_durations(durations, [False] * 3, TD, BEATS, 0.15, 1.5, 9.0) != (
        durations
    )

    assert snap_durations(durations, is_video, TD, BEATS, 0.15, 1.5, 9.0) == (durations)


def test_a_cut_never_moves_past_the_end() -> None:
    # The only beat near the cut lies past the render's end.
    snapped = snap_durations([3.0, 3.0], [False] * 2, TD, [2.85], 0.15, 1.0, 2.8)

    assert snapped == [3.0, 3.0]


def test_without_librosa_beats_are_unavailable(tmp_path: Path, caplog) -> None:
    track = tmp_path / "music.mp3"
    track.write_bytes(b"x")

    with patch.dict("sys.modules", {"librosa.beat": None}):
        assert detect_beats(track) is None

    assert "needs librosa" in caplog.text


def test_cached_beats_are_read_for_the_same_file(tmp_path: Path) -> None:
    track = tmp_path / "music.mp3"
    track.write_bytes(b"x")
    stat = track.stat()
    stamp = {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}
    (tmp_path / "music.mp3.beats.json").write_text(
        json.dumps({"beats": [0.5, 1.0], "stamp": stamp})
    )

    assert detect_beats(track) == [0.5, 1.0]
    track.write_bytes(b"changed")
    with patch.dict("sys.modules", {"librosa.beat": None}):
        assert detect_beats(track) is None


async def _durations(beat_snap: BeatSnapSettings, beats) -> list[float]:
    cfg = config.model_copy(deep=True)
    cfg.video_settings.beat_snap = beat_snap
    inspector = MagicMock()
    inspector.is_video.return_value = False
    inspector.get_media_dimensions = AsyncMock(return_value=(1000, 1000))
    builder = VisualFilterBuilder(
        media_inspector=inspector,
        config=cfg,
        strategy_factory=None,
        profile_settings=None,
    )
    builder.beat_times = beats
    stills = [Path(f"{i}.png") for i in range(5)]
    _, _, timed, *_ = await builder.build_visual_chain(
        visual_inputs=stills,
        total_video_duration=12.0,
        is_relative_mode=True,
        video_settings_dict=cfg.video_settings.model_dump(),
    )
    return [duration for _, duration, _ in timed]


@pytest.mark.req("REQ-VID-013")
@pytest.mark.asyncio
async def test_the_builder_snaps_only_when_on() -> None:
    off = await _durations(BeatSnapSettings(enabled=False), BEATS)
    on = await _durations(BeatSnapSettings(enabled=True), BEATS)
    no_beats = await _durations(BeatSnapSettings(enabled=True), None)

    assert len(set(off)) == 1 and off == no_beats
    assert on != off
    assert sum(on) == pytest.approx(sum(off))


@pytest.mark.req("REQ-VID-013")
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_librosa_finds_the_beats_of_a_click_track(tmp_path: Path) -> None:
    pytest.importorskip("librosa.beat")
    track = tmp_path / "clicks.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i"]
        + ["aevalsrc='0.8*sin(2*PI*1000*t)*lt(mod(t,0.5),0.02)':s=22050:d=12"]
        + [str(track)],
        check=True,
    )

    beats = detect_beats(track)

    assert beats is not None and len(beats) >= 15
    # librosa's frame hop (512 samples, 23 ms at 22.05 kHz) bounds the error.
    assert all(min(abs(b - k * 0.5) for k in range(25)) <= 0.035 for b in beats)


def test_a_cut_may_shorten_a_clip_but_never_lengthen_one() -> None:
    # Cuts at 2.75 and 5.25 s; the beats at 2.62 and 5.12 s are in the window.
    durations = [3.0, 3.0, 3.0]

    clip_first = snap_durations(
        durations, [True, False, False], TD, BEATS, 0.15, 1.5, 9.0
    )
    clip_second = snap_durations(
        durations, [False, True, False], TD, BEATS, 0.15, 1.5, 9.0
    )

    # Both cuts move back 0.13 s: each shortens the segment before it.
    assert clip_first == pytest.approx([2.87, 3.0, 3.13])
    # With the clip second, the first cut would lengthen it and stays; the
    # second shortens it and moves.
    assert clip_second == pytest.approx([3.0, 2.87, 3.13])
