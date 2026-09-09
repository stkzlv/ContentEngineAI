"""Tests for producer pipeline state loading."""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.video.producer.state import _load_pipeline_state


def _make_ctx(state_file: Path) -> MagicMock:
    """Build a minimal PipelineContext mock for state-loader tests."""
    ctx = MagicMock()
    ctx.run_paths = {"state_file": state_file}
    ctx.state = {}
    ctx._state_lock = asyncio.Lock()
    return ctx


class TestLoadPipelineState:
    """Tests for _load_pipeline_state."""

    def test_missing_file_starts_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = _make_ctx(Path(tmp) / "absent.json")
            loaded = asyncio.run(_load_pipeline_state(ctx))
            assert loaded is False
            assert ctx.state == {}

    def test_top_level_scalar_keys_dont_crash(self):
        # Regression for the 0.43.x pillar tagging bug: pillar / script_template
        # land at the top level as bare strings alongside step dicts. The old
        # loader called .get() on every value and crashed on the strings.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            artifact = tmp_path / "script.txt"
            artifact.write_text("hello")
            state_file = tmp_path / "state.json"
            state_file.write_text(
                json.dumps(
                    {
                        "pillar": "utility",
                        "script_template": "before_after",
                        "tts_metadata": {
                            "voice_profile": "charon",
                            "voice_name": "Charon",
                        },
                        "generate_script": {
                            "status": "done",
                            "artifacts": {"script_file": str(artifact)},
                        },
                    }
                )
            )
            ctx = _make_ctx(state_file)
            loaded = asyncio.run(_load_pipeline_state(ctx))
            assert loaded is True
            assert ctx.state["pillar"] == "utility"
            assert ctx.state["script_template"] == "before_after"
            assert ctx.state["generate_script"]["status"] == "done"

    def test_missing_artifact_truncates_to_failed_step(self):
        # When a `done` step's artifact is gone, state truncates back to that
        # step. Top-level scalars should still be skipped during the scan.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            gather_artifact = tmp_path / "visuals.json"
            gather_artifact.write_text("[]")
            state_file = tmp_path / "state.json"
            state_file.write_text(
                json.dumps(
                    {
                        "pillar": "value",
                        "gather_visuals": {
                            "status": "done",
                            "artifacts": {
                                "gathered_visuals_file": str(gather_artifact)
                            },
                        },
                        "generate_script": {
                            "status": "done",
                            "artifacts": {
                                "script_file": str(tmp_path / "missing_script.txt"),
                            },
                        },
                    }
                )
            )
            ctx = _make_ctx(state_file)
            loaded = asyncio.run(_load_pipeline_state(ctx))
            assert loaded is True
            # Truncated to everything before generate_script (gather_visuals).
            # generate_script and any later steps are dropped.
            assert "gather_visuals" in ctx.state
            assert "generate_script" not in ctx.state

    def test_corrupt_json_starts_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            state_file.write_text("{ not valid json")
            ctx = _make_ctx(state_file)
            loaded = asyncio.run(_load_pipeline_state(ctx))
            assert loaded is False
            assert ctx.state == {}

    def test_all_done_no_scalars(self):
        # Regression / back-compat: pre-pillar state files have no top-level
        # scalars. Loader behavior should be unchanged for them.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            artifact = tmp_path / "script.txt"
            artifact.write_text("hello")
            state_file = tmp_path / "state.json"
            state_file.write_text(
                json.dumps(
                    {
                        "generate_script": {
                            "status": "done",
                            "artifacts": {"script_file": str(artifact)},
                        },
                    }
                )
            )
            ctx = _make_ctx(state_file)
            loaded = asyncio.run(_load_pipeline_state(ctx))
            assert loaded is True
            assert ctx.state["generate_script"]["status"] == "done"


class TestASubtitleStepRecordedWithNothing:
    """An empty artifact set must not survive verification (#396).

    Verification compares recorded paths, so a step recorded done with no
    artifacts passes the loop vacuously and is skipped forever. That is
    exactly the state a run leaves behind when the subtitle step reports
    success having produced nothing.
    """

    def _state(self, tmp_path: Path, subtitle_entry: dict) -> Path:
        script = tmp_path / "script.txt"
        script.write_text("hello")
        state_file = tmp_path / "state.json"
        state_file.write_text(
            json.dumps(
                {
                    "generate_script": {
                        "status": "done",
                        "artifacts": {"script_file": str(script)},
                    },
                    "generate_subtitles": subtitle_entry,
                }
            )
        )
        return state_file

    def test_no_artifacts_and_no_reason_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ctx = _make_ctx(self._state(tmp_path, {"status": "done", "artifacts": {}}))
            ctx.profile = None

            asyncio.run(_load_pipeline_state(ctx))

            assert "generate_subtitles" not in ctx.state, (
                "a subtitle step recorded complete with nothing to show must "
                "be dropped, not skipped on the next run"
            )

    def test_the_reason_survives_the_writer(self):
        """The marker must reach the file, not just the in-memory state.

        `_update_state_after_step` replaces the step entry wholesale, so a key
        written into `ctx.state["generate_subtitles"]` by the step is
        discarded before the state is saved. Hand-writing the JSON in the
        test below cannot see that; this drives the writer.
        """
        from src.video.producer.state import _update_state_after_step

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ctx = _make_ctx(tmp_path / "state.json")
            ctx.run_paths = {
                "state_file": tmp_path / "state.json",
                "subtitle_file": tmp_path / "subtitles.ass",
                "whisper_transcript_file": tmp_path / "whisper_transcript.json",
            }
            # What the step sets when subtitles are switched off.
            ctx.state = {"captions": "disabled"}

            asyncio.run(_update_state_after_step(ctx, "generate_subtitles"))

            entry = ctx.state["generate_subtitles"]
            assert entry.get("captions") == "disabled", (
                "the reason must be copied into the step entry, or the state "
                "file records a step done with nothing and no explanation"
            )

    def test_a_recorded_reason_is_respected(self):
        """Subtitles disabled in config legitimately record no artifact."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ctx = _make_ctx(
                self._state(
                    tmp_path,
                    {"status": "done", "artifacts": {}, "captions": "disabled"},
                )
            )
            ctx.profile = None

            asyncio.run(_load_pipeline_state(ctx))

            assert "generate_subtitles" in ctx.state, (
                "a run with subtitles switched off must not be truncated here "
                "on every resume"
            )
