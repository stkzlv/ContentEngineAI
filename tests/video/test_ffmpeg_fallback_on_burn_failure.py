"""`fallback_ffmpeg` degrades to an FFmpeg burn instead of aborting.

The policy's name promised something it delivered only for the
pycaps-*unavailable* case, caught early in `step_generate_subtitles`. A
*render* failure -- pycaps installed, the CSS renderer unable to rasterize
without a display -- happens after the assembler has run, and every policy
except `warn_and_skip` aborted the run there.

That is the case a default install actually hits, which is why it is worth
degrading rather than failing: the transcript and the assembled video both
exist at that point, so captions are still reachable.

Two of the three burn failures deliberately do not degrade. A missing
transcript leaves nothing to build captions from and a missing assembled
video leaves nothing to burn them onto, so those still abort under `raise`
and `fallback_ffmpeg` alike.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.video.producer.steps import _burn_with_ffmpeg_fallback


def _transcript(path: Path, *, words: bool = True) -> Path:
    """A Whisper result dict in the shape the burn step already requires."""
    segments = [
        {
            "start": 0.0,
            "end": 1.6,
            "text": "hello there",
            **(
                {
                    "words": [
                        {"word": "hello", "start": 0.1, "end": 0.7},
                        {"word": "there", "start": 0.8, "end": 1.5},
                    ]
                }
                if words
                else {}
            ),
        }
    ]
    path.write_text(json.dumps({"segments": segments}), encoding="utf-8")
    return path


def _video(path: Path, seconds: float = 2.0) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=640x360:duration={seconds}:rate=10",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _ctx(video_config):
    """The real config, not a mock of it.

    The subtitle generator resolves style presets off `video_config`, so a
    MagicMock makes every caption path raise for a reason that has nothing to
    do with the fallback.
    """
    ctx = MagicMock()
    ctx.product.asin = "B0TESTTEST"
    ctx.config = video_config
    return ctx


@pytest.fixture(scope="module")
def video_config():
    from src.video.config import load_video_config_modular

    return load_video_config_modular()


@pytest.fixture
def settings(video_config):
    merged = video_config.get_profile_merged_settings("slideshow_images1")
    return merged.subtitle_settings


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
class TestTheFallbackProducesACaptionedVideo:
    """The acceptance criterion is a video with captions, not a return code."""

    @pytest.mark.asyncio
    async def test_it_burns_and_reports_success(self, tmp_path, settings, video_config):
        transcript = _transcript(tmp_path / "transcript.json")
        video = _video(tmp_path / "video.mp4")
        before = video.stat().st_size

        burned = await _burn_with_ffmpeg_fallback(
            _ctx(video_config), transcript, video, settings
        )

        assert burned is True
        assert video.exists()
        assert video.stat().st_size != before, "the file was not replaced"

    @pytest.mark.asyncio
    async def test_the_burned_video_is_still_playable(
        self, tmp_path, settings, video_config
    ):
        """A burn that corrupts the file would satisfy the size check above."""
        transcript = _transcript(tmp_path / "transcript.json")
        video = _video(tmp_path / "video.mp4")

        await _burn_with_ffmpeg_fallback(
            _ctx(video_config), transcript, video, settings
        )

        probe = subprocess.run(
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
                str(video),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert probe.stdout.strip() == "640,360"

    @pytest.mark.asyncio
    async def test_the_audio_survives(self, tmp_path, settings, video_config):
        """The burn copies the audio stream rather than re-encoding it."""
        transcript = _transcript(tmp_path / "transcript.json")
        video = _video(tmp_path / "video.mp4")

        await _burn_with_ffmpeg_fallback(
            _ctx(video_config), transcript, video, settings
        )

        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "csv=p=0",
                str(video),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert probe.stdout.strip() == "aac"

    @pytest.mark.asyncio
    async def test_the_pixels_actually_change(self, tmp_path, settings, video_config):
        """Proves captions were drawn, not merely that ffmpeg exited 0.

        An `ass` filter pointed at a file it cannot read still exits 0 and
        produces a video, so a passing burn is not evidence of a caption.
        """
        transcript = _transcript(tmp_path / "transcript.json")
        video = _video(tmp_path / "video.mp4")
        original = tmp_path / "original.mp4"
        shutil.copy(video, original)

        await _burn_with_ffmpeg_fallback(
            _ctx(video_config), transcript, video, settings
        )

        def frame(path: Path, out: Path) -> Path:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-ss",
                    "1.0",
                    "-i",
                    str(path),
                    "-frames:v",
                    "1",
                    str(out),
                ],
                check=True,
                capture_output=True,
            )
            return out

        before = frame(original, tmp_path / "before.png").read_bytes()
        after = frame(video, tmp_path / "after.png").read_bytes()

        assert before != after, "no caption was drawn onto the frame"


@pytest.mark.unit
class TestItRefusesRatherThanShippingSilently:
    """A failed fallback must let the caller abort, not report success."""

    @pytest.mark.asyncio
    async def test_a_missing_transcript_returns_false(
        self, tmp_path, settings, video_config
    ):
        result = await _burn_with_ffmpeg_fallback(
            _ctx(video_config), tmp_path / "absent.json", tmp_path / "v.mp4", settings
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_an_unreadable_transcript_returns_false(
        self, tmp_path, settings, video_config
    ):
        bad = tmp_path / "transcript.json"
        bad.write_text("{not json", encoding="utf-8")

        result = await _burn_with_ffmpeg_fallback(
            _ctx(video_config), bad, tmp_path / "v.mp4", settings
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_a_transcript_with_no_word_timings_returns_false(
        self, tmp_path, settings, video_config
    ):
        """Whisper can return segments without word-level timings."""
        transcript = _transcript(tmp_path / "transcript.json", words=False)

        result = await _burn_with_ffmpeg_fallback(
            _ctx(video_config), transcript, tmp_path / "v.mp4", settings
        )

        assert result is False

    @pytest.mark.asyncio
    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
    async def test_a_missing_video_returns_false(
        self, tmp_path, settings, video_config
    ):
        """Nothing to burn onto; ffmpeg fails and the caller must hear it."""
        transcript = _transcript(tmp_path / "transcript.json")

        result = await _burn_with_ffmpeg_fallback(
            _ctx(video_config), transcript, tmp_path / "absent.mp4", settings
        )

        assert result is False

    @pytest.mark.asyncio
    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
    async def test_a_failed_burn_leaves_no_partial_file(
        self, tmp_path, settings, video_config
    ):
        """The intermediate must not survive to be mistaken for the render."""
        transcript = _transcript(tmp_path / "transcript.json")
        video = tmp_path / "v.mp4"
        video.write_bytes(b"not a video")

        assert (
            await _burn_with_ffmpeg_fallback(
                _ctx(video_config), transcript, video, settings
            )
            is False
        )
        assert not (tmp_path / "v_ffmpeg_burn.mp4").exists()


@pytest.mark.unit
class TestTheOtherTwoFailuresStillAbort:
    """Only the render failure can degrade; the other two have nothing to use."""

    def test_the_handler_still_raises_for_fallback_ffmpeg(self):
        from src.video.producer.context import PipelineError
        from src.video.producer.steps import _handle_pycaps_burn_failure

        with pytest.raises(PipelineError):
            _handle_pycaps_burn_failure("fallback_ffmpeg", "no transcript")

    def test_raise_still_raises(self):
        from src.video.producer.context import PipelineError
        from src.video.producer.steps import _handle_pycaps_burn_failure

        with pytest.raises(PipelineError):
            _handle_pycaps_burn_failure("raise", "boom")

    def test_warn_and_skip_still_keeps_the_video(self):
        from src.video.producer.steps import _handle_pycaps_burn_failure

        # Returns None; the point is that it does not raise, which is what
        # keeps the caption-less video on this policy.
        _handle_pycaps_burn_failure("warn_and_skip", "boom")


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
class TestTheStepReachesTheFallback:
    """Drives `step_burn_pycaps_subtitles`, not the helper it calls.

    The tests above exercise `_burn_with_ffmpeg_fallback` directly, so they
    pass whether or not the step ever calls it. Reverting the wiring left all
    twelve of them green -- the same gap that has bitten this repo before,
    where the function is correct and nothing reaches it.
    """

    @staticmethod
    def _ctx_for_step(tmp_path, video_config, policy, monkeypatch):
        from unittest.mock import MagicMock

        transcript = _transcript(tmp_path / "transcript.json")
        video = _video(tmp_path / "video.mp4")

        merged = video_config.get_profile_merged_settings("slideshow_images1")
        merged.subtitle_settings.subtitle_engine = "pycaps"
        merged.subtitle_settings.pycaps.fallback_policy = policy

        # Patched on the class: a Pydantic instance refuses new attributes,
        # and monkeypatch undoes it so the module-scoped config is not left
        # altered for other tests.
        monkeypatch.setattr(
            type(video_config),
            "get_profile_merged_settings",
            lambda self, *a, **k: merged,
        )

        ctx = MagicMock()
        ctx.product.asin = "B0TESTTEST"
        ctx.config = video_config
        ctx.profile_name = "slideshow_images1"
        ctx.cli_overrides = {}
        ctx.state = {"subtitle_engine_resolved": "pycaps"}
        ctx.run_paths = {
            "whisper_transcript_file": transcript,
            "final_video_output": video,
            "pycaps_burn_marker_file": tmp_path / "burn.json",
            "run_root": tmp_path,
            "temp_dir": tmp_path,
        }
        return ctx, video

    @pytest.mark.asyncio
    async def test_a_render_failure_degrades_instead_of_aborting(
        self, tmp_path, video_config, monkeypatch
    ):
        """The case a default install hits: pycaps present, renderer cannot draw."""
        from src.video.producer import steps

        ctx, video = self._ctx_for_step(
            tmp_path, video_config, "fallback_ffmpeg", monkeypatch
        )
        original = tmp_path / "original.mp4"
        shutil.copy(video, original)

        failed = MagicMock(
            success=False,
            error="Timeout 30000ms exceeded",
            template_used="word-focus",
            renderer_used="css",
        )
        monkeypatch.setattr(
            steps, "_run_pycaps_render", MagicMock(return_value=failed), raising=False
        )

        called = {}
        # Captured before patching: calling through the module attribute
        # would re-enter the spy.
        real_fallback = steps._burn_with_ffmpeg_fallback

        async def _spy(ctx_, transcript, final, settings):
            called["ran"] = True
            return await real_fallback(ctx_, transcript, final, settings)

        monkeypatch.setattr(steps, "_burn_with_ffmpeg_fallback", _spy)

        # The render call is deep inside the step; patch the renderer class
        # the step imports so it returns the failure above.
        import src.video.pycaps_engine.renderer as renderer_module

        monkeypatch.setattr(
            renderer_module.PycapsRenderer,
            "render",
            lambda self, *a, **k: failed,
            raising=False,
        )

        await steps.step_burn_pycaps_subtitles(ctx)

        assert (
            called.get("ran") is True
        ), "the step did not reach the FFmpeg fallback on a render failure"
        assert (
            video.read_bytes() != original.read_bytes()
        ), "the video was not re-burned"
