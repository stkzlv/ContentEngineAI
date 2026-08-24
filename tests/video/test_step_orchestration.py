"""Step wiring: the DAG, the resume scan, and the runner table.

Each test names one way the two execution paths used to disagree with each
other or with the saved state.
"""

from types import SimpleNamespace

import pytest

from src.video.producer.orchestration import (
    completed_steps_from_state,
    data_dependencies,
    step_dependencies,
    step_runners,
    transitive_prereqs,
)
from src.video.producer.state import (
    STEP_ASSEMBLE_VIDEO,
    STEP_BURN_PYCAPS_SUBTITLES,
    STEP_CREATE_VOICEOVER,
    STEP_GATHER_VISUALS,
    STEP_GENERATE_DESCRIPTION,
    STEP_GENERATE_SCRIPT,
    VALID_STEPS,
    resolved_step_order,
)


def _profile(*, stock_only: bool) -> SimpleNamespace:
    """A profile stub `draws_visuals_from_script` can read."""
    return SimpleNamespace(
        use_scraped_images=not stock_only,
        use_scraped_videos=not stock_only,
        use_stock_images=stock_only,
        use_stock_videos=False,
    )


class TestCompletedStepsFromState:
    """A state file holds more than step entries."""

    def test_scalar_keys_do_not_hide_the_completed_steps(self):
        # `.get` on a string raises; the caller's broad handler then treats a
        # good state file as corrupt and re-runs the whole pipeline.
        state = {
            STEP_GATHER_VISUALS: {"status": "done"},
            STEP_GENERATE_SCRIPT: {"status": "done"},
            "script_template": "curiosity_hook",
            "hook_headline": "Why your wifi keeps dropping",
            "subtitle_engine_resolved": "pycaps",
        }
        assert completed_steps_from_state(state) == {
            STEP_GATHER_VISUALS,
            STEP_GENERATE_SCRIPT,
        }

    def test_unfinished_steps_are_excluded(self):
        state = {
            STEP_GATHER_VISUALS: {"status": "done"},
            STEP_GENERATE_SCRIPT: {"status": "failed"},
            STEP_CREATE_VOICEOVER: {},
        }
        assert completed_steps_from_state(state) == {STEP_GATHER_VISUALS}


class TestStepRunners:
    """One table drives both the graph and the `--step` path."""

    def test_every_valid_step_has_a_runner(self):
        # A step in VALID_STEPS with no runner is accepted on the command
        # line, executes nothing, and is still recorded as done.
        assert set(step_runners()) == set(VALID_STEPS)

    def test_runners_are_coroutine_functions(self):
        import inspect

        for name, runner in step_runners().items():
            assert inspect.iscoroutinefunction(runner), name


class TestStepDependencies:
    """The DAG both execution paths read."""

    @pytest.mark.parametrize("stock_only", [False, True])
    def test_covers_every_step_in_the_resolved_order(self, stock_only):
        profile = _profile(stock_only=stock_only)
        assert set(step_dependencies(profile)) == set(resolved_step_order(profile))

    def test_scraped_profile_gathers_before_the_script(self):
        deps = step_dependencies(_profile(stock_only=False))
        assert deps[STEP_GENERATE_SCRIPT] == {STEP_GATHER_VISUALS}
        assert deps[STEP_GATHER_VISUALS] == set()

    def test_stock_profile_writes_the_script_first(self):
        deps = step_dependencies(_profile(stock_only=True))
        assert deps[STEP_GATHER_VISUALS] == {STEP_GENERATE_SCRIPT}
        assert deps[STEP_GENERATE_SCRIPT] == set()

    def test_dependencies_are_declared_in_run_order(self):
        # Adding a step before something it depends on would build a graph
        # naming an absent step.
        for profile in (_profile(stock_only=False), _profile(stock_only=True)):
            order = resolved_step_order(profile)
            deps = step_dependencies(profile)
            for index, step in enumerate(order):
                assert deps[step] <= set(order[:index]), step


class TestDataDependencies:
    """Ordering edges must not be mistaken for data edges."""

    def test_the_script_does_not_read_the_visuals(self):
        # The graph orders gathering first so a product with too few images
        # is rejected before an LLM call is paid for. Treating that as data
        # deletes the script when the footage is re-fetched.
        profile = _profile(stock_only=False)
        assert step_dependencies(profile)[STEP_GENERATE_SCRIPT] == {STEP_GATHER_VISUALS}
        assert data_dependencies(profile)[STEP_GENERATE_SCRIPT] == set()

    def test_stock_paid_steps_read_only_the_script(self):
        profile = _profile(stock_only=True)
        assert STEP_GATHER_VISUALS in step_dependencies(profile)[STEP_CREATE_VOICEOVER]
        assert data_dependencies(profile)[STEP_CREATE_VOICEOVER] == {
            STEP_GENERATE_SCRIPT
        }

    def test_the_assembler_really_does_read_the_visuals(self):
        for profile in (_profile(stock_only=False), _profile(stock_only=True)):
            assert (
                STEP_GATHER_VISUALS in data_dependencies(profile)[STEP_ASSEMBLE_VIDEO]
            )

    def test_a_stock_profile_still_writes_the_script_first(self):
        # The reversal is a real data edge: the stock search terms come from
        # the narration.
        profile = _profile(stock_only=True)
        assert data_dependencies(profile)[STEP_GATHER_VISUALS] == {STEP_GENERATE_SCRIPT}

    def test_scheduling_edges_are_left_intact(self):
        # `step_dependencies` is what the executor walks; narrowing it would
        # let a paid step run beside the check meant to precede it.
        profile = _profile(stock_only=True)
        assert step_dependencies(profile)[STEP_GENERATE_DESCRIPTION] == {
            STEP_GENERATE_SCRIPT,
            STEP_GATHER_VISUALS,
        }


class TestTransitivePrereqs:
    """`--step` requires the DAG's ancestors, not the earlier positions."""

    def test_voiceover_does_not_require_the_description(self):
        # The positional walk this replaced blocked `--step create_voiceover`
        # on `generate_description`, which feeds it nothing.
        deps = step_dependencies(_profile(stock_only=False))
        required = transitive_prereqs(deps, STEP_CREATE_VOICEOVER)
        assert STEP_GENERATE_DESCRIPTION not in required
        assert required == {STEP_GATHER_VISUALS, STEP_GENERATE_SCRIPT}

    def test_reaches_indirect_ancestors(self):
        deps = step_dependencies(_profile(stock_only=False))
        required = transitive_prereqs(deps, STEP_BURN_PYCAPS_SUBTITLES)
        assert STEP_ASSEMBLE_VIDEO in required
        assert STEP_CREATE_VOICEOVER in required
        assert STEP_GATHER_VISUALS in required

    def test_excludes_the_target_itself(self):
        deps = step_dependencies(_profile(stock_only=False))
        assert STEP_ASSEMBLE_VIDEO not in transitive_prereqs(deps, STEP_ASSEMBLE_VIDEO)

    def test_a_root_step_requires_nothing(self):
        deps = step_dependencies(_profile(stock_only=True))
        assert transitive_prereqs(deps, STEP_GENERATE_SCRIPT) == set()


class TestBurnMarker:
    """The burn replaces the assembled video, so it must not run twice."""

    def test_no_marker_means_not_burned(self, tmp_path):
        from src.video.producer.steps import _already_burned

        video = tmp_path / "video.mp4"
        video.write_bytes(b"assembled")
        assert not _already_burned(tmp_path / "absent.json", video)

    def test_a_missing_marker_path_is_tolerated(self, tmp_path):
        from src.video.producer.steps import _already_burned

        video = tmp_path / "video.mp4"
        video.write_bytes(b"assembled")
        assert not _already_burned(None, video)

    def test_recorded_burn_is_recognised(self, tmp_path):
        from src.video.producer.steps import _already_burned, _record_burn

        video = tmp_path / "video.mp4"
        video.write_bytes(b"burned")
        marker = tmp_path / "temp" / "pycaps_burned.json"
        _record_burn(marker, video)
        # A second burn would draw new captions over the old ones.
        assert _already_burned(marker, video)

    def test_a_reassembled_video_burns_again(self, tmp_path):
        from src.video.producer.steps import _already_burned, _record_burn

        video = tmp_path / "video.mp4"
        video.write_bytes(b"burned")
        marker = tmp_path / "temp" / "pycaps_burned.json"
        _record_burn(marker, video)
        # assemble_video re-rendered: the captions are gone and must return.
        video.write_bytes(b"reassembled, a different size")
        assert not _already_burned(marker, video)

    def test_a_corrupt_marker_burns_rather_than_skips(self, tmp_path):
        from src.video.producer.steps import _already_burned

        video = tmp_path / "video.mp4"
        video.write_bytes(b"assembled")
        marker = tmp_path / "pycaps_burned.json"
        marker.write_text("{not json")
        # Shipping an uncaptioned video is worse than burning twice.
        assert not _already_burned(marker, video)


class TestDescriptionArtifactRecording:
    """A recorded artifact that was never written invalidates the state."""

    @staticmethod
    async def _artifacts(tmp_path, *, write_description: bool):
        from unittest.mock import MagicMock

        from src.video.producer.state import (
            STEP_GENERATE_DESCRIPTION,
            _update_state_after_step,
        )

        text_dir = tmp_path / "temp"
        text_dir.mkdir()
        description_file = text_dir / "description.txt"
        if write_description:
            description_file.write_text("a description")
        (tmp_path / "metadata.json").write_text("{}")

        ctx = MagicMock()
        ctx.state = {}
        ctx.run_paths = {
            "description_file": description_file,
            "run_root": tmp_path,
        }
        await _update_state_after_step(ctx, STEP_GENERATE_DESCRIPTION)
        return ctx.state[STEP_GENERATE_DESCRIPTION]["artifacts"]

    @pytest.mark.asyncio
    async def test_absent_description_is_not_recorded(self, tmp_path):
        # The configured path writes platform metadata instead. Recording
        # description.txt anyway failed verification on the next run, which
        # dropped this step and every step after it.
        artifacts = await self._artifacts(tmp_path, write_description=False)
        assert "description_file" not in artifacts
        assert "unified_metadata_file" in artifacts

    @pytest.mark.asyncio
    async def test_written_description_is_recorded(self, tmp_path):
        artifacts = await self._artifacts(tmp_path, write_description=True)
        assert "description_file" in artifacts


class TestMusicArtifactRecording:
    """`download_music` completes without a track when nothing is found."""

    @staticmethod
    async def _artifacts(tmp_path, *, found_a_track: bool):
        from unittest.mock import MagicMock

        from src.video.producer.state import (
            STEP_DOWNLOAD_MUSIC,
            _update_state_after_step,
        )

        music_info = tmp_path / "music_choice.json"
        if found_a_track:
            music_info.write_text("{}")
        ctx = MagicMock()
        ctx.state = {}
        ctx.run_paths = {"music_info_file": music_info}
        await _update_state_after_step(ctx, STEP_DOWNLOAD_MUSIC)
        return ctx.state[STEP_DOWNLOAD_MUSIC]["artifacts"]

    @pytest.mark.asyncio
    async def test_no_track_records_no_artifact(self, tmp_path):
        # Recording a file the step never wrote invalidates the state on
        # every later run, so a finished render re-assembles and re-burns.
        assert await self._artifacts(tmp_path, found_a_track=False) == {}

    @pytest.mark.asyncio
    async def test_a_found_track_is_recorded(self, tmp_path):
        artifacts = await self._artifacts(tmp_path, found_a_track=True)
        assert "music_info_file" in artifacts


class TestStateBelongsToThisRun:
    """`pipeline_state.json` is product-level; some artifacts are not."""

    def test_this_runs_artifact_is_accepted(self, tmp_path):
        from unittest.mock import MagicMock

        from src.video.producer.state import _artifact_invalid_reason

        video = tmp_path / "video_A.mp4"
        video.write_bytes(b"rendered")
        ctx = MagicMock()
        ctx.run_paths = {"final_video_output": video}
        assert _artifact_invalid_reason(ctx, "final_video_output", str(video)) is None

    def test_another_profiles_video_is_rejected(self, tmp_path):
        from unittest.mock import MagicMock

        from src.video.producer.state import _artifact_invalid_reason

        # Present on disk and completely wrong: skipping on it renders
        # nothing and reports the path of a video this run never wrote.
        other = tmp_path / "video_profile_a.mp4"
        other.write_bytes(b"another profile's render")
        ctx = MagicMock()
        ctx.run_paths = {"final_video_output": tmp_path / "video_profile_b.mp4"}
        reason = _artifact_invalid_reason(ctx, "final_video_output", str(other))
        assert reason is not None
        assert "another run" in reason

    def test_a_missing_file_is_still_rejected(self, tmp_path):
        from unittest.mock import MagicMock

        from src.video.producer.state import _artifact_invalid_reason

        ctx = MagicMock()
        ctx.run_paths = {}
        reason = _artifact_invalid_reason(ctx, "script_file", str(tmp_path / "gone"))
        assert reason is not None and "not found" in reason

    def test_an_unregistered_key_is_not_compared(self, tmp_path):
        from unittest.mock import MagicMock

        from src.video.producer.state import _artifact_invalid_reason

        # Per-platform metadata keys are generated, not run-path entries.
        meta = tmp_path / "metadata_youtube.json"
        meta.write_text("{}")
        ctx = MagicMock()
        ctx.run_paths = {}
        assert (
            _artifact_invalid_reason(ctx, "platform_metadata_youtube", str(meta))
            is None
        )


class TestDropDependents:
    """Re-running a step invalidates what reads its output."""

    @staticmethod
    def _ctx(state):
        from unittest.mock import MagicMock

        ctx = MagicMock()
        ctx.state = state
        ctx.profile = _profile(stock_only=False)
        return ctx

    def test_dropped_steps_lose_the_files_they_short_circuit_on(self, tmp_path):
        from src.video.producer.state import _drop_dependents

        # `create_voiceover` returns early whenever voiceover.wav is on disk,
        # so a state-only drop would narrate the superseded script.
        voiceover = tmp_path / "voiceover.wav"
        voiceover.write_bytes(b"audio for the old script")
        duration = tmp_path / "voiceover_duration.txt"
        duration.write_text("41.2")
        state = {
            STEP_GENERATE_SCRIPT: {"status": "done", "artifacts": {}},
            STEP_CREATE_VOICEOVER: {
                "status": "done",
                "artifacts": {
                    "voiceover_file": str(voiceover),
                    "voiceover_duration_file": str(duration),
                },
            },
        }
        _drop_dependents(self._ctx(state), STEP_GENERATE_SCRIPT)
        assert STEP_CREATE_VOICEOVER not in state
        assert not voiceover.exists()

    def test_a_file_a_surviving_step_claims_is_kept(self, tmp_path):
        from src.video.producer.state import _drop_dependents

        # `script.txt` blocks a rerun, and both the script step and the
        # voiceover record it. Dropping only the voiceover must not delete
        # the script the surviving step still claims -- the next run would
        # then regenerate narration nobody asked for.
        script = tmp_path / "script.txt"
        script.write_text("the current script")
        state = {
            STEP_GENERATE_SCRIPT: {
                "status": "done",
                "artifacts": {"script_file": str(script)},
            },
            STEP_CREATE_VOICEOVER: {
                "status": "done",
                "artifacts": {"script_file": str(script)},
            },
        }
        _drop_dependents(self._ctx(state), STEP_GENERATE_SCRIPT)
        assert STEP_CREATE_VOICEOVER not in state
        assert script.exists()

    def test_refetching_visuals_keeps_the_script_and_the_voiceover(self, tmp_path):
        from src.video.producer.state import _drop_dependents

        # `--step gather_visuals` re-fetches footage. The script reads none
        # of it, so deleting the script, the narration and the captions --
        # all of them paid for -- would be pure loss.
        script = tmp_path / "script.txt"
        script.write_text("the script")
        voiceover = tmp_path / "voiceover.wav"
        voiceover.write_bytes(b"narration")
        state = {
            STEP_GATHER_VISUALS: {"status": "done", "artifacts": {}},
            STEP_GENERATE_SCRIPT: {
                "status": "done",
                "artifacts": {"script_file": str(script)},
            },
            STEP_CREATE_VOICEOVER: {
                "status": "done",
                "artifacts": {"voiceover_file": str(voiceover)},
            },
            STEP_ASSEMBLE_VIDEO: {"status": "done", "artifacts": {}},
        }
        _drop_dependents(self._ctx(state), STEP_GATHER_VISUALS)
        assert STEP_GENERATE_SCRIPT in state
        assert STEP_CREATE_VOICEOVER in state
        assert script.exists() and voiceover.exists()
        # The assembler does read the visuals, so it has to go.
        assert STEP_ASSEMBLE_VIDEO not in state

    def test_an_unfinished_entry_is_left_alone(self):
        from src.video.producer.state import _drop_dependents

        # `generate_subtitles` records the engine it resolved while it runs.
        # The state-update loop walks every completed step of this run, so an
        # ancestor is processed while that entry exists but is not yet done;
        # dropping it would discard what this run just wrote.
        state = {
            STEP_GATHER_VISUALS: {"status": "done"},
            "generate_subtitles": {"engine": "pycaps"},
        }
        _drop_dependents(self._ctx(state), STEP_GATHER_VISUALS)
        assert state["generate_subtitles"] == {"engine": "pycaps"}

    def test_reassembling_drops_the_recorded_burn(self):
        from src.video.producer.state import _drop_dependents

        state = {
            STEP_ASSEMBLE_VIDEO: {"status": "done"},
            STEP_BURN_PYCAPS_SUBTITLES: {"status": "done"},
        }
        _drop_dependents(self._ctx(state), STEP_ASSEMBLE_VIDEO)
        # The burn's captions were re-rendered away by the new assembly.
        assert STEP_BURN_PYCAPS_SUBTITLES not in state
        assert STEP_ASSEMBLE_VIDEO in state

    def test_upstream_steps_are_kept(self):
        from src.video.producer.state import _drop_dependents

        state = {
            STEP_GATHER_VISUALS: {"status": "done"},
            STEP_GENERATE_SCRIPT: {"status": "done"},
            STEP_CREATE_VOICEOVER: {"status": "done"},
        }
        _drop_dependents(self._ctx(state), STEP_CREATE_VOICEOVER)
        assert set(state) == {
            STEP_GATHER_VISUALS,
            STEP_GENERATE_SCRIPT,
            STEP_CREATE_VOICEOVER,
        }

    def test_indirect_dependents_go_too(self):
        from src.video.producer.state import (
            STEP_DOWNLOAD_MUSIC,
            STEP_GENERATE_SUBTITLES,
            _drop_dependents,
        )

        state = {
            STEP_GENERATE_SCRIPT: {"status": "done"},
            STEP_CREATE_VOICEOVER: {"status": "done"},
            STEP_GENERATE_SUBTITLES: {"status": "done"},
            STEP_DOWNLOAD_MUSIC: {"status": "done"},
            STEP_ASSEMBLE_VIDEO: {"status": "done"},
            STEP_BURN_PYCAPS_SUBTITLES: {"status": "done"},
        }
        _drop_dependents(self._ctx(state), STEP_CREATE_VOICEOVER)
        assert set(state) == {STEP_GENERATE_SCRIPT, STEP_CREATE_VOICEOVER}
