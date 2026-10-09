"""How a render ends after its last spoken word (design 0002), held off."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.video.assembler.audio_builder import AudioFilterBuilder
from src.video.assembler.visual_builder import VisualFilterBuilder
from src.video.config import config
from src.video.config.visual_models import StillMotionSettings
from src.video.producer import steps
from src.video.speech_end import parse_speech_end, speech_end_sec

PROFILE = "slideshow_images1"
FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")


def _config(**profile_fields):
    cfg = config.model_copy(deep=True)
    for name, value in profile_fields.items():
        setattr(cfg.video_profiles[PROFILE], name, value)
    return cfg


@pytest.mark.req("REQ-VID-011")
def test_the_shipped_config_ends_on_the_peak() -> None:
    assert config.video_settings.ending == "peak"
    for name in config.video_profiles:
        assert config.get_profile_merged_settings(name).video_settings.ending == (
            "peak"
        ), name


def test_parse_speech_end() -> None:
    trailing = "silence_start: 0.4\nsilence_end: 0.6\nsilence_start: 2.31\n"
    assert parse_speech_end(trailing, 3.0) == 2.31
    closed_at_eof = trailing + "silence_end: 3 | silence_duration: 0.69\n"
    assert parse_speech_end(closed_at_eof, 3.0) == 2.31
    closed = "silence_start: 0.4\nsilence_end: 0.6 | silence_duration: 0.2\n"
    assert parse_speech_end(closed, 3.0) == 3.0
    assert parse_speech_end("", 3.0) == 3.0
    assert parse_speech_end("silence_start: -0.01\n", 3.0) == 0.0


@needs_ffmpeg
@pytest.mark.asyncio
async def test_speech_end_finds_the_trailing_silence(tmp_path: Path) -> None:
    audio = tmp_path / "voice.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=440:d=1.2"]
        + ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-filter_complex"]
        + ["[1]atrim=duration=0.7[s];[0][s]concat=n=2:v=0:a=1", str(audio)],
        check=True,
    )

    end = await speech_end_sec(audio, "ffmpeg", 1.9)

    assert end == pytest.approx(1.2, abs=0.05)


@pytest.mark.asyncio
async def test_a_failed_measurement_keeps_the_whole_voiceover(tmp_path: Path) -> None:
    assert await speech_end_sec(tmp_path / "missing.wav", "ffmpeg", 4.0) == 4.0
    assert await speech_end_sec(tmp_path / "x.wav", "/no/such/ffmpeg", 4.0) == 4.0


def _ctx(cfg, tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        config=cfg,
        profile_name=PROFILE,
        cli_overrides=None,
        voiceover_duration=10.0,
        run_paths={"voiceover_file": tmp_path / "voice.wav"},
    )


@pytest.mark.req("REQ-VID-011")
@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["peak", "loop"])
async def test_peak_ends_a_margin_after_the_last_word(
    tmp_path: Path, ending: str
) -> None:
    cfg = _config(ending=ending, peak_margin_sec=0.3)
    with patch.object(steps, "speech_end_sec", AsyncMock(return_value=9.4)):
        total, fade, speech = await steps._render_duration(_ctx(cfg, tmp_path))

    assert total == pytest.approx(9.7)
    assert fade == pytest.approx(0.3)
    assert speech == pytest.approx(9.4)


@pytest.mark.asyncio
async def test_outro_keeps_the_tail_and_measures_nothing(tmp_path: Path) -> None:
    measure = AsyncMock()
    cfg = _config()
    cfg.video_settings.ending = "outro"
    with patch.object(steps, "speech_end_sec", measure):
        total, fade, speech = await steps._render_duration(_ctx(cfg, tmp_path))

    measure.assert_not_called()
    assert total == pytest.approx(10.0 + config.outro_duration_sec)
    assert fade is None and speech is None


@pytest.mark.req("REQ-VID-011")
def test_the_music_fades_within_the_margin() -> None:
    builder = AudioFilterBuilder(config)

    peak, _ = builder.build_audio_filters(0, 1, 9.7, music_fade_out_sec=0.25)
    outro, _ = builder.build_audio_filters(0, 1, 11.0)

    assert "afade=t=out:st=9.450:d=0.25" in "\n".join(peak)
    fade = config.audio_settings.music_fade_out_duration
    assert f"afade=t=out:st={11.0 - fade:.3f}:d={fade}" in "\n".join(outro)


async def _chain(tmp_path: Path, cfg, stills: list[Path], duration: float = 9.0):
    inspector = MagicMock()
    inspector.is_video.return_value = False
    inspector.get_media_dimensions = AsyncMock(return_value=(1000, 1000))
    settings = cfg.get_profile_merged_settings(PROFILE)
    builder = VisualFilterBuilder(
        media_inspector=inspector,
        config=cfg,
        strategy_factory=None,
        profile_settings=settings,
        product_id="B0X",
    )
    parts, inputs, timed, label, *_ = await builder.build_visual_chain(
        visual_inputs=stills,
        total_video_duration=duration,
        is_relative_mode=True,
        video_settings_dict=settings.video_settings.model_dump(),
    )
    return parts, inputs, timed, label


def _stills(tmp_path: Path, n: int) -> list[Path]:
    paths = []
    for i in range(n):
        path = tmp_path / f"still_{i}.png"
        path.write_bytes(b"")
        paths.append(path)
    return paths


@pytest.mark.req("REQ-VID-011")
@pytest.mark.asyncio
async def test_loop_closes_on_the_first_image(tmp_path: Path) -> None:
    stills = _stills(tmp_path, 3)
    cfg = _config(ending="loop", first_frame_pre_motion=True)

    parts, _, timed, _ = await _chain(tmp_path, cfg, stills)

    assert [path for path, *_ in timed] == [*stills, stills[0]]
    (opener,) = (p for p in parts if p.startswith("[0:v]"))
    (closer,) = (p for p in parts if p.startswith("[3:v]"))
    assert "max(1.0,zoom-" in opener
    assert "if(eq(on,0),1.0,min(1.100,zoom+" in closer


@pytest.mark.asyncio
async def test_loop_reverses_the_first_still_move(tmp_path: Path) -> None:
    cfg = _config(ending="loop", still_motion=StillMotionSettings(enabled=True))

    parts, _, _, _ = await _chain(tmp_path, cfg, _stills(tmp_path, 3))

    (opener,) = (p for p in parts if p.startswith("[0:v]"))
    (closer,) = (p for p in parts if p.startswith("[3:v]"))
    assert (
        opener.split("[fg_0]")[1].split("[fgs_0]")[0]
        != (closer.split("[fg_3]")[1].split("[fgs_3]")[0])
    )
    assert ":exact=1" in closer


@pytest.mark.asyncio
async def test_outro_and_peak_add_no_closer(tmp_path: Path) -> None:
    stills = _stills(tmp_path, 3)
    for ending in ("outro", "peak"):
        _, _, timed, _ = await _chain(tmp_path, _config(ending=ending), stills)
        assert [path for path, *_ in timed] == stills


def test_the_loop_timeline_drops_stills_to_keep_segments_long() -> None:
    stills = [Path(f"{i}.png") for i in range(6)]

    assert VisualFilterBuilder._loop_timeline(stills, 12.0, 0.5, 1.5) == [
        *stills,
        stills[0],
    ]
    short = VisualFilterBuilder._loop_timeline(stills, 4.0, 0.5, 1.5)
    assert short[-1] == stills[0] and len(short) < 7
    assert all((4.0 + (len(short) - 1) * 0.5) / len(short) >= 1.5 for _ in [0])
    assert VisualFilterBuilder._loop_timeline(stills[:1], 4.0, 0.5, 1.5) == [
        stills[0],
        stills[0],
    ]


def _render_loop(tmp_path: Path, cfg, duration: float) -> np.ndarray:
    """The visual track of a loop render of three distinct stills, grey."""
    stills = []
    for i, colour in enumerate([(200, 40, 40), (40, 200, 40), (40, 40, 200)]):
        img = np.zeros((800, 1000, 3), np.uint8)
        img[:] = colour
        img[300:500, 100 + i * 200 : 400 + i * 200] = 255
        path = tmp_path / f"still_{i}.png"
        Image.fromarray(img).save(path)
        stills.append(path)
    settings = cfg.get_profile_merged_settings(PROFILE)
    from src.video.assembler.media_inspector import MediaInspector

    builder = VisualFilterBuilder(
        media_inspector=MediaInspector(cfg),
        config=cfg,
        strategy_factory=None,
        profile_settings=settings,
        product_id="B0X",
    )
    import asyncio

    parts, inputs, *_rest = asyncio.run(
        builder.build_visual_chain(
            visual_inputs=stills,
            total_video_duration=duration,
            is_relative_mode=True,
            video_settings_dict=settings.video_settings.model_dump(),
            temp_dir=tmp_path,
        )
    )
    label = _rest[1]
    out = tmp_path / "loop.mkv"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex"]
        + [";".join(parts) + f";{label}scale=270:480[small]", "-map", "[small]"]
        + ["-t", str(duration)]
        + ["-c:v", "ffv1", str(out)],
        check=True,
    )
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(out), "-f", "rawvideo"]
        + ["-pix_fmt", "gray", "-"],
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, 480, 270).astype(np.float32)


@pytest.mark.req("REQ-VID-011")
@needs_ffmpeg
@pytest.mark.parametrize(
    "motion",
    [
        {"first_frame_pre_motion": True},
        {"still_motion": StillMotionSettings(enabled=True)},
        {},
    ],
)
def test_a_loop_render_ends_on_its_first_frame(tmp_path: Path, motion: dict) -> None:
    frames = _render_loop(tmp_path, _config(ending="loop", **motion), 6.0)

    first, last, middle = frames[0], frames[-1], frames[len(frames) // 2]
    assert np.abs(last - first).mean() < 2.0
    assert np.abs(middle - first).mean() > 10.0


def test_each_render_records_its_ending(tmp_path: Path) -> None:
    from src.video.render_choices import choices_from_context

    ctx = SimpleNamespace(
        state={},
        config=SimpleNamespace(video_settings=SimpleNamespace(ending="outro")),
        product=SimpleNamespace(asin="B0X"),
        profile_name=PROFILE,
        profile=SimpleNamespace(video_assembly_mode="sequential", ending="peak"),
        run_paths={"music_info_file": None, "run_root": tmp_path / "B0X"},
        script="x",
    )

    assert choices_from_context(ctx)["ending"] == "peak"
    ctx.profile.ending = None
    assert choices_from_context(ctx)["ending"] == "outro"


@pytest.mark.req("REQ-VID-011")
@needs_ffmpeg
@pytest.mark.asyncio
async def test_an_end_sting_finishes_at_the_speech_end(tmp_path: Path) -> None:
    from src.video.assembler.media_inspector import MediaInspector
    from src.video.config import SignatureSting

    cfg = config.model_copy(deep=True)
    for name, spec, sec in (
        ("voice.wav", "anullsrc=r=48000:cl=mono", 5.0),
        ("sting.wav", "sine=f=1000:sample_rate=48000", 1.0),
    ):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", spec,
                        "-t", str(sec), str(tmp_path / name)], check=True)  # fmt: skip
    cfg.audio_settings.signature_sting = SignatureSting(
        path=tmp_path / "sting.wav", position="end"
    )
    builder = AudioFilterBuilder(cfg)

    measured, _ = await builder.build_mix(
        [], tmp_path / "voice.wav", None, 3.25, MediaInspector(), 0.25, 3.0
    )
    unmeasured, _ = await builder.build_mix(
        [], tmp_path / "voice.wav", None, 6.0, MediaInspector()
    )

    assert "adelay=2000:" in "\n".join(measured)
    assert "adelay=4000:" in "\n".join(unmeasured)
