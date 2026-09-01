"""`fallback_ffmpeg` degrades to an FFmpeg burn instead of aborting.

The policy's name promised something it delivered only for the
pycaps-*unavailable* case, caught early in `step_generate_subtitles`. A
*render* failure -- pycaps installed, the CSS renderer unable to rasterize
without a display -- happens after the assembler has run, and every policy
except `warn_and_skip` aborted the run there.

That is the case a default install actually hits, which is why it is worth
degrading rather than failing: the transcript and the assembled video both
exist at that point, so captions are still reachable.

Two of the four burn failures deliberately do not degrade. A missing
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
    # A real float: the generator holds the last cue to this, and an unset
    # MagicMock attribute makes it emit no lines at all.
    ctx.voiceover_duration = 2.0
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
    async def test_the_file_is_replaced_by_the_burn(
        self, tmp_path, settings, video_config
    ):
        """The output is a new encode of the input, in place.

        Deliberately not claiming this proves a caption was drawn: the burn
        re-encodes, so the frames differ whether or not libass rendered
        anything, and an ASS whose only line is positioned off-frame passes
        it too. What guards the caption is the `Dialogue:` check on the
        generated file, covered in
        `TestACaptionFreeSubtitleFileIsRefused`.

        The earlier version of this docstring justified itself with "an `ass`
        filter pointed at a file it cannot read still exits 0", which is
        false -- that exits 234 and writes nothing.
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

        assert before != after, "the burn did not re-encode the file in place"


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
class TestACaptionFreeSubtitleFileIsRefused:
    """The case `result.success` cannot see, and the burn cannot either.

    A valid ASS with no `Dialogue:` lines burns cleanly and exits 0, drawing
    nothing. Comparing frames before and after does not catch it: the burn
    re-encodes, so a second lossy generation changes the pixels on its own.
    That is why the earlier version of this file's "pixels actually change"
    test passed against a caption-free burn, and why the guard is a check on
    the file rather than an inference from the exit code.
    """

    @pytest.mark.asyncio
    async def test_an_ass_with_no_dialogue_lines_returns_false(
        self, tmp_path, settings, video_config, monkeypatch
    ):
        from src.video.unified_subtitle_generator import UnifiedSubtitleGenerator

        transcript = _transcript(tmp_path / "transcript.json")
        video = _video(tmp_path / "video.mp4")

        def _empty(self, timings, output_path, **kwargs):
            output_path.write_text(
                "[Script Info]\nScriptType: v4.00+\n\n"
                "[V4+ Styles]\nFormat: Name\nStyle: Default\n\n"
                "[Events]\nFormat: Layer, Start, End, Text\n",
                encoding="utf-8",
            )
            return MagicMock(success=True)

        monkeypatch.setattr(UnifiedSubtitleGenerator, "generate_from_timings", _empty)

        result = await _burn_with_ffmpeg_fallback(
            _ctx(video_config), transcript, video, settings
        )

        assert result is False, (
            "a subtitle file with no lines burned cleanly and was reported as "
            "a successful caption burn"
        )


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


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
class TestTheStepReachesTheFallback:
    """Drives `step_burn_pycaps_subtitles`, not the helper it calls.

    The tests above exercise `_burn_with_ffmpeg_fallback` directly, so they
    pass whether or not the step ever calls it. Reverting the wiring left all
    ten of them green -- the same gap that has bitten this repo before,
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
        ctx.voiceover_duration = 2.0
        ctx.profile_name = "slideshow_images1"
        ctx.cli_overrides = {}
        ctx.state = {"subtitle_engine_resolved": "pycaps"}
        ctx.run_paths = {
            "whisper_transcript_file": transcript,
            "final_video_output": video,
            "pycaps_burn_marker_file": tmp_path / "burn.json",
            "pycaps_metadata_file": tmp_path / "pycaps_metadata.json",
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

        async def _spy(*args, **kwargs):
            called["ran"] = True
            return await real_fallback(*args, **kwargs)

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
        # The marker has to say which burn drew the captions, or a later
        # `--step burn_pycaps_subtitles` reports FFmpeg captions as pycaps
        # ones and skips. Deleting `engine=` left the suite green.
        marker = json.loads(
            ctx.run_paths["pycaps_burn_marker_file"].read_text(encoding="utf-8")
        )
        assert marker["engine"] == "ffmpeg_fallback"
        # And the last cue is held to the narration, not cut at the final
        # word: without `voiceover_duration` this ends at 0:00:01.50.
        subtitle = next(video.parent.glob("*_fallback.ass"))
        assert "0:00:02.00" in subtitle.read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_pycaps_vanishing_mid_run_also_degrades(
        self, tmp_path, video_config, monkeypatch
    ):
        """The fourth burn failure, reachable only on a resume.

        A run records `subtitle_engine_resolved: pycaps` and is interrupted;
        the environment is rebuilt without the optional group; the resume
        trusts that state key by design and reaches a renderer that is gone.
        Both preconditions for degrading hold -- transcript and assembled
        video are on disk -- so aborting would fail a render one FFmpeg pass
        from finished.

        The import is *blocked*, not the renderer patched. Patching
        `PycapsRenderer.render` left `from pycaps.ai import LlmProvider`
        reachable, and the bundled config enables AI tagging, so the real
        scenario died with `ModuleNotFoundError` before the handler ran while
        this test passed.
        """
        import sys

        from src.video.producer import steps

        ctx, video = self._ctx_for_step(
            tmp_path, video_config, "fallback_ffmpeg", monkeypatch
        )
        original = tmp_path / "original.mp4"
        shutil.copy(video, original)

        class _Blocker:
            def find_spec(self, name, path=None, target=None):
                if name == "pycaps" or name.startswith("pycaps."):
                    raise ModuleNotFoundError(f"No module named {name!r}")
                return None

        blocker = _Blocker()
        monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])
        for name in [n for n in sys.modules if n.startswith("pycaps")]:
            monkeypatch.delitem(sys.modules, name, raising=False)

        await steps.step_burn_pycaps_subtitles(ctx)

        assert (
            video.read_bytes() != original.read_bytes()
        ), "pycaps vanishing mid-run aborted instead of degrading"

    @pytest.mark.asyncio
    async def test_a_fallback_clears_stale_pycaps_metadata(
        self, tmp_path, video_config, monkeypatch
    ):
        """Metadata from an earlier pycaps burn must not survive a fallback.

        `pycaps_metadata.json` is not rerun-blocking, so a file written by a
        successful burn survives a re-assembly and is recorded as the next
        run's artifact -- naming a template that was never applied. Both
        `_clear_pycaps_metadata` calls could be deleted with a green suite
        before this existed, because the fixture omitted the path and the
        helper was a no-op in every test.
        """
        import json

        from src.video.producer import steps

        ctx, _ = self._ctx_for_step(
            tmp_path, video_config, "fallback_ffmpeg", monkeypatch
        )
        metadata = ctx.run_paths["pycaps_metadata_file"]
        metadata.write_text(
            json.dumps({"engine": "pycaps", "template": "hype"}), encoding="utf-8"
        )
        ctx.state["pycaps_metadata"] = {"engine": "pycaps"}

        import src.video.pycaps_engine.renderer as renderer_module

        failed = MagicMock(
            success=False,
            error="Timeout 30000ms exceeded",
            template_used="hype",
            renderer_used="css",
        )
        monkeypatch.setattr(
            renderer_module.PycapsRenderer,
            "render",
            lambda self, *a, **k: failed,
            raising=False,
        )

        await steps.step_burn_pycaps_subtitles(ctx)

        assert (
            not metadata.exists()
        ), "a fallback burn left metadata naming a template it never applied"
        assert "pycaps_metadata" not in ctx.state
