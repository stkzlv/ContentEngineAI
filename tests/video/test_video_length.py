"""Final video length is recorded; long renders with music are flagged.

REQ-VID-159, REQ-VID-160, design 0025.
"""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.video.config import config
from src.video.producer.steps import record_video_duration
from src.video.render_choices import choices_from_context


def _results(duration: object) -> dict:
    return {
        "success": True,
        "details": {"probe_info": {"format": {"duration": duration}}},
    }


def _music(tmp_path: Path, source: str = "Jamendo") -> Path:
    path = tmp_path / "music_info.json"
    path.write_text(json.dumps({"path": "m.mp3", "source": source}))
    return path


@pytest.mark.req("REQ-VID-159")
def test_the_probed_duration_is_recorded() -> None:
    state: dict = {}

    record_video_duration(state, _results("34.567"), None, 60)

    assert state == {"video_duration_sec": 34.57}


@pytest.mark.req("REQ-VID-159")
@pytest.mark.parametrize(
    "results", [{}, {"details": {}}, _results(None), _results("x")]
)
def test_no_probe_records_nothing(results: dict) -> None:
    state: dict = {}

    record_video_duration(state, results, None, 60)

    assert state == {}


@pytest.mark.req("REQ-VID-160")
def test_music_past_the_ceiling_warns_and_is_recorded(tmp_path: Path, caplog) -> None:
    state: dict = {}

    with caplog.at_level(logging.WARNING):
        record_video_duration(state, _results("67.2"), _music(tmp_path), 60)

    assert state["over_music_claim_ceiling"] is True
    assert "67.2s" in caplog.text and "60s" in caplog.text
    assert "Jamendo" in caplog.text


@pytest.mark.req("REQ-VID-160")
@pytest.mark.parametrize(
    ("duration", "music", "ceiling"),
    [
        ("67.2", False, 60),  # no music, no claim to fear
        ("67.2", True, 0),  # the check is off
        ("59.9", True, 60),  # under the ceiling
    ],
)
def test_no_flag_without_the_risk(
    tmp_path: Path, duration: str, music: bool, ceiling: float
) -> None:
    state: dict = {}

    record_video_duration(
        state, _results(duration), _music(tmp_path) if music else None, ceiling
    )

    assert "over_music_claim_ceiling" not in state
    assert "video_duration_sec" in state


@pytest.mark.req("REQ-VID-160")
def test_an_unreadable_music_record_still_warns(tmp_path: Path, caplog) -> None:
    broken = tmp_path / "music_info.json"
    broken.write_text("not json")
    state: dict = {}

    with caplog.at_level(logging.WARNING):
        record_video_duration(state, _results("70"), broken, 60)
        record_video_duration({}, _results("70"), tmp_path / "missing.json", 60)

    assert state["over_music_claim_ceiling"] is True
    assert caplog.text.count("music source: unknown") == 2


@pytest.mark.req("REQ-VID-160")
def test_the_ceiling_ships_at_sixty_seconds() -> None:
    assert config.video_settings.music_claim_ceiling_sec == 60


@pytest.mark.req("REQ-VID-159", "REQ-VID-160")
def test_the_assembly_step_records_with_the_configured_ceiling() -> None:
    from src.video.producer import steps

    source = inspect.getsource(steps.step_assemble_video)
    call = source[source.index("record_video_duration(") :]
    assert "music_info_path if music_path is not None else None" in call
    assert "ctx.config.video_settings.music_claim_ceiling_sec" in call


@pytest.mark.req("REQ-VID-159", "REQ-VID-160")
@pytest.mark.asyncio
async def test_the_step_entry_keeps_the_length(monkeypatch) -> None:
    from src.video.producer import state

    ctx = SimpleNamespace(
        state={"video_duration_sec": 61.5, "over_music_claim_ceiling": True},
        run_paths={"final_video_output": Path("v.mp4")},
    )
    monkeypatch.setattr(state, "_drop_dependents", lambda ctx, step: None)

    await state._update_state_after_step(ctx, state.STEP_ASSEMBLE_VIDEO)

    entry = ctx.state["assemble_video"]
    assert entry["video_duration_sec"] == 61.5
    assert entry["over_music_claim_ceiling"] is True


def _ctx(tmp_path: Path, state: dict) -> SimpleNamespace:
    return SimpleNamespace(
        state=state,
        config=SimpleNamespace(
            video_settings=SimpleNamespace(
                first_frame_pre_motion=False, video_transition_duration=0.3
            )
        ),
        profile_name="p",
        profile=SimpleNamespace(
            video_assembly_mode="sequential",
            first_frame_pre_motion=None,
            video_transition_duration=0.5,
        ),
        run_paths={"music_info_file": None, "run_root": tmp_path / "B0X"},
        script="The script.",
    )


@pytest.mark.req("REQ-VID-159")
@pytest.mark.parametrize(
    "state",
    [
        {"video_duration_sec": 41.2},
        # A resume keeps only the step entries.
        {"assemble_video": {"status": "done", "video_duration_sec": 41.2}},
    ],
)
def test_the_render_row_carries_the_length(tmp_path: Path, state: dict) -> None:
    assert choices_from_context(_ctx(tmp_path, state))["video_duration_sec"] == 41.2


@pytest.mark.req("REQ-VID-159", "REQ-VID-160")
@pytest.mark.parametrize(
    ("results", "expected"),
    [
        (_results("45"), {"video_duration_sec": 45.0}),  # shorter, no flag
        ({"success": False}, {}),  # no probe: no stale length either
    ],
)
def test_a_re_render_drops_the_previous_length_and_flag(
    tmp_path: Path, results: dict, expected: dict
) -> None:
    state = {"video_duration_sec": 72.0, "over_music_claim_ceiling": True}

    record_video_duration(state, results, _music(tmp_path), 60)

    assert state == expected
