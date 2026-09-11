"""Oversized videos are bounded during format normalization (#429).

The image inputs got their edge bound; videos still entered the
filtergraph at source resolution, and a compliant 4K stock clip skipped
the normalization transcode entirely, so the bound has to join the skip
condition, not just the transcode.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from src.video.assembler.core import VideoAssembler
from src.video.config import config as video_config

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed"
)


def _clip(path: Path, w: int, h: int) -> Path:
    """A compliant clip: h264, 30fps, yuv420p -- the shape that skips."""
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={w}x{h}:duration=1:rate=30",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _dims(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    w, h = out.split(",")
    return int(w), int(h)


class TestACompliantOversizedClipIsBounded:
    """The dangerous shape: already h264/30fps/yuv420p, so it used to skip."""

    @pytest.mark.asyncio
    async def test_a_4k_clip_comes_back_at_the_bound(self, tmp_path):
        src = _clip(tmp_path / "big.mp4", 3840, 2160)
        assembler = VideoAssembler(video_config)

        out = await assembler._normalize_video_format(src, cache_dir=tmp_path)

        assert out != src, "the oversized compliant clip skipped the transcode"
        w, h = _dims(out)
        assert max(w, h) == video_config.video_settings.max_image_input_edge
        # Aspect preserved: 16:9 in, 16:9 out.
        assert (w, h) == (2560, 1440)

    @pytest.mark.asyncio
    async def test_a_small_compliant_clip_still_skips(self, tmp_path):
        src = _clip(tmp_path / "small.mp4", 1280, 720)
        assembler = VideoAssembler(video_config)

        out = await assembler._normalize_video_format(src, cache_dir=tmp_path)
        assert out == src, "a compliant in-bound clip must not be re-encoded"

    @pytest.mark.asyncio
    async def test_a_portrait_clip_keeps_its_orientation(self, tmp_path):
        src = _clip(tmp_path / "tall.mp4", 2160, 3840)
        assembler = VideoAssembler(video_config)

        out = await assembler._normalize_video_format(src, cache_dir=tmp_path)
        assert out != src
        assert _dims(out) == (1440, 2560)


class TestTheCacheKeyCarriesTheSource:
    @pytest.mark.asyncio
    async def test_a_changed_source_misses_the_cache(self, tmp_path):
        """Same lesson as the bounded images: the cache outlives runs while
        sources can be re-downloaded under stable names.
        """
        import time

        src = _clip(tmp_path / "clip.mp4", 3840, 2160)
        assembler = VideoAssembler(video_config)
        first = await assembler._normalize_video_format(src, cache_dir=tmp_path)

        time.sleep(0.01)
        _clip(src, 2880, 2880)
        second = await assembler._normalize_video_format(src, cache_dir=tmp_path)

        assert second != first, "the previous source's normalization was reused"
        assert _dims(second) == (2560, 2560)
