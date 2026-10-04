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

    @pytest.mark.slow
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

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_a_portrait_clip_keeps_its_orientation(self, tmp_path):
        src = _clip(tmp_path / "tall.mp4", 2160, 3840)
        assembler = VideoAssembler(video_config)

        out = await assembler._normalize_video_format(src, cache_dir=tmp_path)
        assert out != src
        assert _dims(out) == (1440, 2560)


class TestTheCacheKeyCarriesTheSource:
    @pytest.mark.slow
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


class TestTheCacheSurvivesKillsAndSweeps:
    """Review findings: atomic entry creation, and no permanent orphans."""

    def test_the_entry_is_created_by_a_rename(self):
        """A kill mid-transcode must not leave a truncated entry the
        exists() reuse would serve forever.
        """
        source = Path("src/video/assembler/core.py").read_text()
        idx = source.index('f"{cache_path.stem}.part{cache_path.suffix}"')
        # To the end of the enclosing method, not a fixed offset: the code
        # between the anchors grows and a fixed window goes red on growth.
        end = source.index("async def", idx)
        window = source[idx:end]
        assert (
            "os.replace(partial_path, cache_path)" in window
        ), "the cache entry must reach its name only via os.replace"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_superseded_entries_are_swept(self, tmp_path):
        """The stat-keyed name can never match again after a re-download,
        so the old entry must not become a permanent multi-MB orphan.
        """
        import time

        src = _clip(tmp_path / "clip.mp4", 3840, 2160)
        assembler = VideoAssembler(video_config)
        first = await assembler._normalize_video_format(src, cache_dir=tmp_path)
        assert first.exists()

        time.sleep(0.01)
        _clip(src, 2880, 2880)
        second = await assembler._normalize_video_format(src, cache_dir=tmp_path)

        assert not first.exists(), "the superseded entry was left behind"
        entries = list(tmp_path.glob("clip_normalized*"))
        assert entries == [second]


@pytest.mark.req("REQ-OPS-038", "REQ-OPS-039")
@pytest.mark.asyncio
async def test_transcodes_wait_for_the_shared_ffmpeg_limit(tmp_path, monkeypatch):
    """The visual builder normalizes every clip at once; the limit holds them."""
    import asyncio

    from src.utils.async_io import ffmpeg_semaphore
    from src.video.assembler import core as assembler_core

    # 25 fps, so each clip needs a transcode rather than skipping it.
    clips = []
    for i in range(3):
        path = tmp_path / f"clip{i}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=320x240:duration=1:rate=25",
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
        clips.append(path)

    real_exec = asyncio.create_subprocess_exec
    active = 0
    peak = 0

    async def counting_exec(*args, **kwargs):
        nonlocal active, peak
        proc = await real_exec(*args, **kwargs)
        if "libx264" not in args:
            return proc
        active += 1
        peak = max(peak, active)
        real_communicate = proc.communicate

        async def communicate(*a, **k):
            nonlocal active
            try:
                await asyncio.sleep(0.05)
                return await real_communicate(*a, **k)
            finally:
                active -= 1

        proc.communicate = communicate  # type: ignore[method-assign]
        return proc

    monkeypatch.setattr(assembler_core.asyncio, "create_subprocess_exec", counting_exec)
    previous = ffmpeg_semaphore.limit
    ffmpeg_semaphore.set_limit(1)
    try:
        assembler = VideoAssembler(video_config)
        outs = await asyncio.gather(
            *[
                assembler._normalize_video_format(c, cache_dir=tmp_path / "cache")
                for c in clips
            ]
        )
    finally:
        ffmpeg_semaphore.set_limit(previous)

    assert all(out != clip for out, clip in zip(outs, clips, strict=True))
    assert peak == 1
