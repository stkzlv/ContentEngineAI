"""Every render writes a cover image (REQ-VID-014, design 0011)."""

from __future__ import annotations

import asyncio
import inspect
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.video.cover import cover_paths, cover_time_sec, write_cover

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")


def _video(path: Path) -> Path:
    """2 s at 1080x1920: a white band at the hook's height for the first
    1.5 s (the hook on screen), black after.
    """
    subprocess.run(
        [
            FFMPEG or "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=black:s=1080x1920:r=25:d=2",
            "-vf",
            "drawbox=x=120:y=540:w=840:h=200:color=white:t=fill:enable='lt(t,1.5)'",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


def _mean_luma(image: Path, y: int, h: int) -> float:
    out = subprocess.run(
        [
            FFMPEG or "ffmpeg",
            "-hide_banner",
            "-i",
            str(image),
            "-vf",
            f"crop=840:{h}:120:{y},format=gray,signalstats,metadata=print:key=lavfi.signalstats.YAVG",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    values = [
        float(line.split("=")[1])
        for line in (out.stdout + out.stderr).splitlines()
        if "YAVG" in line
    ]
    return values[-1]


@pytest.mark.req("REQ-VID-014")
def test_the_cover_time_is_halfway_through_the_hook():
    assert cover_time_sec(1.5) == 0.75
    assert cover_time_sec(0) == 0.0


@needs_ffmpeg
@pytest.mark.req("REQ-VID-014")
def test_the_cover_is_a_full_frame_with_the_hook_on_screen(tmp_path):
    video = _video(tmp_path / "video_B0TESTID01_slideshow.mp4")

    out = asyncio.run(write_cover(video, cover_time_sec(1.5), FFMPEG))

    assert out == tmp_path / "video_B0TESTID01_slideshow.cover.jpg"
    probe = subprocess.run(
        [
            shutil.which("ffprobe") or "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout.strip() == "1080,1920"
    # The band sits at the hook's place, inside the centred 3:4 area
    # (y 240-1680); after the hook ends it would be black.
    assert _mean_luma(out, 560, 160) > 200
    record = json.loads(cover_paths(video)[1].read_text())
    assert record == {
        "file": "video_B0TESTID01_slideshow.cover.jpg",
        "video": "video_B0TESTID01_slideshow.mp4",
        "at_ms": 750,
    }


@needs_ffmpeg
@pytest.mark.req("REQ-VID-014")
def test_a_frame_after_the_hook_would_miss_it(tmp_path):
    """The marker test bites: a cover taken after the hook shows no band."""
    video = _video(tmp_path / "video.mp4")

    out = asyncio.run(write_cover(video, 1.8, FFMPEG))

    assert out is not None
    assert _mean_luma(out, 560, 160) < 30


@pytest.mark.req("REQ-VID-014")
def test_each_profile_video_gets_its_own_cover(tmp_path):
    one, other = (
        cover_paths(tmp_path / f"video_B0TESTID01_{p}.mp4")[0] for p in ("a", "b")
    )

    assert one != other
    assert one.parent == other.parent == tmp_path


@pytest.mark.req("REQ-VID-014")
def test_a_failed_write_leaves_no_earlier_cover(tmp_path):
    video = tmp_path / "absent.mp4"
    for path in cover_paths(video):
        path.write_text("from an earlier render")

    out = asyncio.run(write_cover(video, 0.75, FFMPEG or "ffmpeg"))

    assert out is None
    assert not any(path.exists() for path in cover_paths(video))


@pytest.mark.req("REQ-VID-014")
def test_the_producer_writes_it_when_a_run_finishes_the_video():
    from src.video.producer import orchestration

    source = inspect.getsource(orchestration)
    call = source.index("await write_cover(")
    guard = source.rindex("if finishes_video and final_video:", 0, call)
    rule = source.rindex("finishes_video = debug_step_target in (", 0, guard)

    assert call - guard < 400
    assert "resolved_step_order(ctx.profile)[-1]" in source[rule:guard]
