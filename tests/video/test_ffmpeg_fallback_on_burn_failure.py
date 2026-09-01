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
    def _ctx_for_step(tmp_path, video_config, policy, monkeypatch, secrets=None):
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
        # Empty by default, which turns AI tagging off. A MagicMock
        # returns a truthy key, so the step built a Gemini adapter and
        # imported `pycaps.ai` before reaching the renderer -- on a box
        # without the optional group (which is what CI installs) that
        # raised, and every render-failure test silently exercised the
        # pycaps-unavailable branch instead. The feature this module
        # covers was unguarded by the gate that approves it.
        ctx.secrets = {} if secrets is None else secrets
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
        from src.video.producer import steps

        # A key, so AI tagging is on and the blocked import this test
        # exists for is the one that raises. With no key the step reaches
        # `renderer.render`, which raises the same error by another route
        # and leaves the `pycaps.ai` import untested.
        ctx, video = self._ctx_for_step(
            tmp_path,
            video_config,
            "fallback_ffmpeg",
            monkeypatch,
            secrets={"GEMINI_API_KEY": "test-key"},
        )
        original = tmp_path / "original.mp4"
        shutil.copy(video, original)

        # Cleared before the degrade, not after it: moving the call below
        # this block let a successful degrade return with an earlier burn's
        # metadata intact, and the whole suite stayed green.
        metadata = ctx.run_paths["pycaps_metadata_file"]
        metadata.write_text(
            json.dumps({"engine": "pycaps", "template": "hype"}), encoding="utf-8"
        )
        ctx.state["pycaps_metadata"] = {"engine": "pycaps"}

        # Captured before patching, so the spy can re-enter the real helper.
        real_fallback = steps._burn_with_ffmpeg_fallback
        seen = {}

        async def _spy(ctx_, transcript, final, settings, bounds=None, **kw):
            seen["bounds"] = bounds
            return await real_fallback(ctx_, transcript, final, settings, bounds, **kw)

        monkeypatch.setattr(steps, "_burn_with_ffmpeg_fallback", _spy)

        self._drop_pycaps(ctx, video, tmp_path, monkeypatch)

        # The route has to be asserted, not just selected. `enable_ai_tagging`
        # is config, so flipping it off would skip the import above and let
        # `render` raise the same error -- leaving the wrapper this test
        # exists for unguarded, silently. Reaching the renderer is the
        # failure, and `AssertionError` is not caught by the handler.
        import src.video.pycaps_engine.renderer as renderer_module

        def _unreachable(self, *a, **k):
            raise AssertionError(
                "the burn step reached the pycaps renderer; the `pycaps.ai` "
                "import this test guards was never taken"
            )

        monkeypatch.setattr(renderer_module.PycapsRenderer, "render", _unreachable)

        await steps.step_burn_pycaps_subtitles(ctx)

        assert (
            video.read_bytes() != original.read_bytes()
        ), "pycaps vanishing mid-run aborted instead of degrading"
        # The same assertion its render-failure twin carries. Without it,
        # dropping `engine=` here leaves the marker claiming pycaps drew
        # captions FFmpeg drew, and a later `--step burn_pycaps_subtitles`
        # reports the wrong engine -- 152 tests stayed green on that.
        marker = json.loads(
            ctx.run_paths["pycaps_burn_marker_file"].read_text(encoding="utf-8")
        )
        assert marker["engine"] == "ffmpeg_fallback"
        assert not metadata.exists()
        assert "pycaps_metadata" not in ctx.state
        # Dropping the argument is a silent default, not a TypeError, and
        # sends a `below_content` anchor to the safe-zone floor instead.
        assert seen["bounds"] is not None

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
        )

        await steps.step_burn_pycaps_subtitles(ctx)

        assert (
            not metadata.exists()
        ), "a fallback burn left metadata naming a template it never applied"
        assert "pycaps_metadata" not in ctx.state

    @staticmethod
    def _drop_transcript(ctx, video, tmp_path, monkeypatch):
        ctx.run_paths["whisper_transcript_file"] = tmp_path / "never-written.json"

    @staticmethod
    def _drop_video(ctx, video, tmp_path, monkeypatch):
        video.unlink()

    @staticmethod
    def _drop_pycaps(ctx, video, tmp_path, monkeypatch):
        import sys

        class _Blocker:
            def find_spec(self, name, path=None, target=None):
                if name == "pycaps" or name.startswith("pycaps."):
                    raise ModuleNotFoundError(f"No module named {name!r}")
                return None

        monkeypatch.setattr(sys, "meta_path", [_Blocker(), *sys.meta_path])
        for name in [n for n in sys.modules if n.startswith("pycaps")]:
            monkeypatch.delitem(sys.modules, name, raising=False)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bail_out",
        ["_drop_transcript", "_drop_video", "_drop_pycaps"],
    )
    async def test_every_burn_failure_clears_stale_pycaps_metadata(
        self, bail_out, tmp_path, video_config, monkeypatch
    ):
        """No burn *failure* may leave the previous burn's record behind.

        `pycaps_metadata.json` is not rerun-blocking, so a file written by an
        earlier successful burn survives and `state.py` records it as *this*
        run's `burn_pycaps_subtitles` artifact -- naming a template that was
        never applied to this render. The render-failure exit is covered
        above; these are the other three, and all three `_clear_pycaps_metadata`
        calls could be deleted with a green suite before this existed.

        Four of the step's six burn-skipping exits clear. The two that do not
        are not failures: `engine != "pycaps"` did not attempt a burn and is
        reached by renders of *other* profiles of the same product, whose
        record it would destroy (the file is product-level, the render is
        not); and `_already_burned` leaves a record that genuinely describes
        the burn on that file.

        `warn_and_skip` on every case, so the step returns rather than
        raising and the assertions can read the state it left behind.
        """
        import json

        from src.video.producer import steps

        ctx, video = self._ctx_for_step(
            tmp_path, video_config, "warn_and_skip", monkeypatch
        )
        metadata = ctx.run_paths["pycaps_metadata_file"]
        metadata.write_text(
            json.dumps({"engine": "pycaps", "template": "hype"}), encoding="utf-8"
        )
        ctx.state["pycaps_metadata"] = {"engine": "pycaps"}

        getattr(self, bail_out)(ctx, video, tmp_path, monkeypatch)

        await steps.step_burn_pycaps_subtitles(ctx)

        assert (
            not metadata.exists()
        ), "an exit that never burned left metadata naming an unapplied template"
        assert "pycaps_metadata" not in ctx.state

    @pytest.mark.asyncio
    @pytest.mark.parametrize("policy", ["raise", "fallback_ffmpeg"])
    async def test_pycaps_vanishing_aborts_when_the_fallback_cannot_save_it(
        self, policy, tmp_path, video_config, monkeypatch
    ):
        """The strict policies must abort, not report an uncaptioned burn.

        The pycaps-missing handler resolves all three policies inline rather
        than through `_handle_pycaps_burn_failure`, and only two of its cells
        were pinned: the successful `fallback_ffmpeg` degrade, and
        `warn_and_skip` via the metadata test. Narrowing the abort condition
        to `== "fallback_ffmpeg"` -- which makes a `raise` run log a warning
        and return a caption-less video reported as a completed burn, the
        defect this module exists to prevent -- left 150 tests green.

        `fallback_ffmpeg` is included with the fallback forced to fail,
        because it shares the same abort and reaching it means the degrade
        was already attempted and lost.
        """
        from src.video.producer import steps
        from src.video.producer.context import PipelineError

        ctx, video = self._ctx_for_step(
            tmp_path,
            video_config,
            policy,
            monkeypatch,
            secrets={"GEMINI_API_KEY": "test-key"},
        )
        original = tmp_path / "original.mp4"
        shutil.copy(video, original)

        async def _fallback_fails(*args, **kwargs):
            return False

        monkeypatch.setattr(steps, "_burn_with_ffmpeg_fallback", _fallback_fails)
        self._drop_pycaps(ctx, video, tmp_path, monkeypatch)

        with pytest.raises(PipelineError):
            await steps.step_burn_pycaps_subtitles(ctx)

        # No marker, or a resume reads the uncaptioned video as burned.
        assert not ctx.run_paths["pycaps_burn_marker_file"].exists()
        assert video.read_bytes() == original.read_bytes()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("policy", ["raise", "warn_and_skip"])
    async def test_only_fallback_ffmpeg_degrades_when_pycaps_is_gone(
        self, policy, tmp_path, video_config, monkeypatch
    ):
        """The other two policies must not quietly acquire the degrade.

        Widening the gate to `if True:` passed the whole suite. It would
        make `raise` degrade instead of aborting, and `warn_and_skip` burn
        captions onto a video its own contract says it keeps untouched.
        Under that widened gate two tests reach this block on another
        policy -- today's gate lets neither in -- and neither could see
        it, for different reasons.
        `test_pycaps_vanishing_aborts_when_the_fallback_cannot_save_it`
        asserts the video and the marker, but stubs the fallback to fail,
        so a widened gate enters the block and still lands on the abort.
        `test_every_burn_failure_clears_stale_pycaps_metadata[_drop_pycaps]`
        runs the real fallback, so the widened gate burned its video --
        but it asserts only the metadata clear. Asserting the video on a
        stubbed fallback would not have closed this; the real fallback
        below is what makes the assertions bite.

        Hence a real fallback here, and assertions on both the video and
        the marker: the gap needed one test with both.
        """
        from src.video.producer import steps
        from src.video.producer.context import PipelineError

        ctx, video = self._ctx_for_step(
            tmp_path,
            video_config,
            policy,
            monkeypatch,
            secrets={"GEMINI_API_KEY": "test-key"},
        )
        original = tmp_path / "original.mp4"
        shutil.copy(video, original)
        self._drop_pycaps(ctx, video, tmp_path, monkeypatch)

        if policy == "raise":
            with pytest.raises(PipelineError):
                await steps.step_burn_pycaps_subtitles(ctx)
        else:
            await steps.step_burn_pycaps_subtitles(ctx)

        assert (
            video.read_bytes() == original.read_bytes()
        ), f"{policy} burned captions the gate should have reserved for fallback_ffmpeg"
        assert not ctx.run_paths["pycaps_burn_marker_file"].exists()
