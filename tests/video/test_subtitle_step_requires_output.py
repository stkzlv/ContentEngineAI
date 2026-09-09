"""The subtitle step cannot report success having produced nothing (#396).

A `--step generate_subtitles` run exited 0, logged `Video producer completed
successfully`, and wrote no subtitle file. The step is recorded done whether
or not it produced anything, and the state recorder registers both subtitle
artifacts *conditionally* -- so a step that produced neither was recorded
with an empty artifact set, which verification satisfies vacuously because it
checks recorded paths. The run then reached an assembler with no captions to
burn.
"""

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.video.producer.context import PipelineError
from src.video.producer.steps import _require_subtitle_artifact


def _ctx(tmp_path: Path, **present: bool) -> SimpleNamespace:
    """A context whose run paths are declared; only the named ones exist."""
    run_paths: dict[str, Path] = {
        "subtitle_file": tmp_path / "subtitles.ass",
        "whisper_transcript_file": tmp_path / "whisper_transcript.json",
        "subtitle_upper_file": tmp_path / "subtitles_upper.ass",
    }
    for key, exists in present.items():
        if exists:
            run_paths[key].write_text("x")
    return SimpleNamespace(run_paths=run_paths)


class TestTheStepMustLeaveACaptionSource:
    def test_nothing_produced_fails(self, tmp_path):
        with pytest.raises(PipelineError) as excinfo:
            _require_subtitle_artifact(_ctx(tmp_path))

        message = str(excinfo.value)
        assert "no caption source" in message
        # Named, because the symptom this replaces was the absence of a file
        # nothing mentioned.
        assert "subtitle_file" in message
        assert "whisper_transcript_file" in message

    @pytest.mark.parametrize(
        "produced",
        ["subtitle_file", "whisper_transcript_file", "subtitle_upper_file"],
    )
    def test_any_one_of_them_is_enough(self, tmp_path, produced):
        """The three engines and modes each leave a different artifact."""
        _require_subtitle_artifact(_ctx(tmp_path, **{produced: True}))

    def test_an_undeclared_path_is_not_reported_as_missing(self, tmp_path):
        """A run with no upper-line path declared should not name it."""
        ctx = _ctx(tmp_path)
        del ctx.run_paths["subtitle_upper_file"]

        with pytest.raises(PipelineError) as excinfo:
            _require_subtitle_artifact(ctx)

        assert "subtitle_upper_file" not in str(excinfo.value)


class TestTheStepActuallyCallsIt:
    """A guard nothing calls is the silent failure it was written against.

    The tests above drive `_require_subtitle_artifact` directly, so deleting
    its call from the step leaves them all green. `step_generate_subtitles`
    needs a voiceover, a script and a Whisper model to run, so the call site
    is read instead.
    """

    def test_the_step_calls_the_guard(self):
        import ast

        tree = ast.parse(Path("src/video/producer/steps.py").read_text())
        step = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "step_generate_subtitles"
        )

        calls = [
            node
            for node in ast.walk(step)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_require_subtitle_artifact"
        ]
        assert len(calls) == 1, (
            f"step_generate_subtitles calls the guard {len(calls)} time(s); "
            "it must run once, after both branches"
        )

    def test_the_guard_runs_after_both_branches(self):
        """Inside either branch it would miss the other one."""
        import ast

        tree = ast.parse(Path("src/video/producer/steps.py").read_text())
        step = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "step_generate_subtitles"
        )

        def _is_guard(node):
            return (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "_require_subtitle_artifact"
            )

        # The step's own statements, not any nested block. Walking every body
        # accepts the guard inside the two-part branch, where it would never
        # see a single-line run.
        top_level = list(step.body)
        for stmt in step.body:
            if isinstance(stmt, ast.AsyncWith):
                top_level = list(stmt.body)
                break

        guard_at = [i for i, stmt in enumerate(top_level) if _is_guard(stmt)]
        assert guard_at, (
            "the guard must sit at the step's top level; inside a branch it "
            "cannot see the other one"
        )

        # ...and after the branch, not before it. Membership alone passes a
        # refactor that hoists the call to the top of the step, where every
        # render fails because nothing has been generated yet.
        branch_at = [
            i
            for i, stmt in enumerate(top_level)
            if isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "two_part_enabled"
        ]
        assert branch_at, "could not find the two_part_enabled branch"
        assert guard_at[0] > branch_at[0], (
            "the guard must run after the branch that generates captions, "
            "not before it"
        )


class TestRecordingNothingIsNotSilent:
    """The state recorder's half: an empty artifact set must say so.

    The step refuses to reach it empty-handed, so this is the second net. It
    warns rather than raises, because a state recorder is the wrong place to
    decide a run failed and subtitles can legitimately be disabled.
    """

    @pytest.mark.asyncio
    async def test_an_empty_artifact_set_warns(self, tmp_path, caplog):
        from src.video.producer.state import _update_state_after_step

        ctx = SimpleNamespace(
            run_paths={
                "subtitle_file": tmp_path / "subtitles.ass",
                "whisper_transcript_file": tmp_path / "whisper_transcript.json",
            },
            state={},
            profile=None,
        )

        with caplog.at_level(logging.WARNING):
            await _update_state_after_step(ctx, "generate_subtitles")

        assert any(
            "no caption artifact" in record.message for record in caplog.records
        ), "recording a subtitle step with nothing to show must not be silent"

    @pytest.mark.asyncio
    async def test_a_produced_artifact_is_recorded(self, tmp_path, caplog):
        from src.video.producer.state import _update_state_after_step

        subtitle = tmp_path / "subtitles.ass"
        subtitle.write_text("[Script Info]")
        ctx = SimpleNamespace(
            run_paths={
                "subtitle_file": subtitle,
                "whisper_transcript_file": tmp_path / "whisper_transcript.json",
            },
            state={},
            profile=None,
        )

        with caplog.at_level(logging.WARNING):
            await _update_state_after_step(ctx, "generate_subtitles")

        assert not any("no caption artifact" in r.message for r in caplog.records)
        assert ctx.state["generate_subtitles"]["artifacts"]["subtitle_file"]


class TestThePycapsTranscriptMustBeThisRunsE:
    """Existence is not enough on the pycaps arm (#396).

    The transcript path is stable across runs and `temp/` survives a failed
    one, so a run whose Whisper call timed out would return the *previous*
    run's transcript. The step then reports success and the burn step draws
    captions written for a script that may no longer be the one narrated.
    This is the bundled default engine.
    """

    @staticmethod
    def _call(tmp_path, whisper, prewrite: bool):
        import asyncio
        import json
        from unittest.mock import AsyncMock, patch

        from src.video.config import config as cfg
        from src.video.subtitle_utils import create_unified_subtitles

        temp = tmp_path / "temp"
        temp.mkdir()
        srt_out = tmp_path / "subtitles.ass"
        transcript = srt_out.with_name("whisper_transcript.json")
        if prewrite:
            transcript.write_text(
                json.dumps({"segments": [{"words": [{"word": "OLD"}]}]})
            )
        audio = tmp_path / "vo.wav"
        audio.write_bytes(b"RIFF0000WAVEfmt ")

        async def go():
            with patch(
                "src.video.subtitle_utils.generate_subtitles_with_whisper",
                new=AsyncMock(side_effect=whisper),
            ):
                return await create_unified_subtitles(
                    audio,
                    srt_out,
                    {"subtitle_engine": "pycaps", "enabled": True},
                    cfg.whisper_settings,
                    None,
                    {},
                    "a script",
                    10.0,
                    False,
                    cfg,
                    temp,
                    "topic-x",
                    engine="pycaps",
                )

        return asyncio.run(go()), transcript

    def test_a_previous_runs_transcript_is_refused(self, tmp_path):
        result, transcript = self._call(
            tmp_path, TimeoutError("whisper timed out"), prewrite=True
        )
        assert transcript.exists(), "the fixture must leave the stale file in place"
        assert result is None, "a transcript this run did not write was accepted"

    def test_this_runs_transcript_is_accepted(self, tmp_path):
        import json

        def _writes(*_args, **kwargs):
            out = kwargs.get("transcript_out_path")
            Path(out).write_text(json.dumps({"segments": [{"words": [{"w": "NEW"}]}]}))
            return [{"word": "NEW", "start_time": 0.0, "end_time": 1.0}]

        result, transcript = self._call(tmp_path, _writes, prewrite=False)
        assert result == transcript
