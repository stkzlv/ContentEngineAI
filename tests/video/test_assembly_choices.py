"""Each drawn choice of a render reaches its record (design 0006, #659).

Still-motion moves, beat-snap moves and sound-effect files are drawn inside
the assembler. Without a record, renders with a feature on cannot be told
apart from those without it in the variety report or the metrics.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.video.assembler.visual_builder import VisualFilterBuilder, pick_still_move
from src.video.config import config
from src.video.config.visual_models import BeatSnapSettings, StillMotionSettings
from src.video.render_choices import choices_from_context, distribution

PROFILE = "slideshow_images1"
MOVES = ["push_in", "pull_out", "pan_left", "pan_right", "pan_up"]
BEATS = [0.12 + 0.5 * n for n in range(60)]


async def _builder(tmp_path: Path, *, motion: bool, snap: bool):
    cfg = config.model_copy(deep=True)
    cfg.video_profiles[PROFILE].still_motion = StillMotionSettings(enabled=motion)
    cfg.video_profiles[PROFILE].first_frame_pre_motion = False
    cfg.video_settings.beat_snap = BeatSnapSettings(enabled=snap)
    stills = []
    for i in range(5):
        path = tmp_path / f"still_{i}.png"
        path.write_bytes(b"")
        stills.append(path)
    inspector = MagicMock()
    inspector.is_video.return_value = False
    inspector.get_media_dimensions = AsyncMock(return_value=(1000, 1000))
    settings = cfg.get_profile_merged_settings(PROFILE)
    settings.video_settings.beat_snap = BeatSnapSettings(enabled=snap)
    builder = VisualFilterBuilder(
        media_inspector=inspector,
        config=cfg,
        strategy_factory=None,
        profile_settings=settings,
        product_id="B0X",
    )
    builder.beat_times = BEATS
    await builder.build_visual_chain(
        visual_inputs=stills,
        total_video_duration=12.0,
        is_relative_mode=True,
        video_settings_dict=settings.video_settings.model_dump(),
    )
    return builder


@pytest.mark.req("REQ-PUB-083")
@pytest.mark.asyncio
async def test_the_builder_keeps_the_moves_it_drew(tmp_path: Path) -> None:
    on = await _builder(tmp_path, motion=True, snap=False)
    off = await _builder(tmp_path, motion=False, snap=False)

    previous = None
    expected = []
    for i in range(5):
        previous = pick_still_move("B0X", i, MOVES, previous)
        expected.append(previous)
    assert on.still_moves == expected
    assert off.still_moves == []


@pytest.mark.req("REQ-PUB-083")
@pytest.mark.asyncio
async def test_the_builder_counts_the_cuts_it_snapped(tmp_path: Path) -> None:
    on = await _builder(tmp_path, motion=False, snap=True)
    off = await _builder(tmp_path, motion=False, snap=False)

    assert on.beat_snap_moved is not None and on.beat_snap_moved >= 1
    assert off.beat_snap_moved is None


@pytest.mark.req("REQ-PUB-083")
def test_the_assembler_names_each_effect_it_placed(tmp_path: Path) -> None:
    from src.video.assembler.core import VideoAssembler

    cfg = config.model_copy(deep=True)
    effects = cfg.audio_settings.sound_effects
    effects.enabled = True
    for kind in ("hook", "reveal", "cta"):
        path = tmp_path / f"{kind}_0.wav"
        path.write_bytes(b"x")
        setattr(effects, kind, [path])
    assembler = VideoAssembler(cfg)
    words = [
        {"word": w, "start_time": t, "end_time": t + 0.25}
        for w, t in [("Hi.", 0.0), ("It", 2.0), ("folds.", 2.3), ("Bio.", 12.0)]
    ]

    assembler._sound_effects(15.0, words)

    assert assembler.assembly_choices()["sound_effects"] == [
        "hook:hook_0.wav",
        "reveal:reveal_0.wav",
        "cta:cta_0.wav",
    ]
    assert VideoAssembler(cfg).assembly_choices() == {
        "still_moves": None,
        "beat_snap_moved": None,
        "sound_effects": None,
    }


def _ctx(tmp_path: Path, state: dict) -> SimpleNamespace:
    return SimpleNamespace(
        state=state,
        config=SimpleNamespace(
            video_settings=SimpleNamespace(
                first_frame_pre_motion=False, video_transition_duration=0.3
            )
        ),
        profile_name=PROFILE,
        profile=SimpleNamespace(
            video_assembly_mode="sequential",
            first_frame_pre_motion=None,
            video_transition_duration=0.5,
        ),
        run_paths={"music_info_file": None, "run_root": tmp_path / "B0X"},
        script="The script.",
    )


CHOICES = {
    "still_moves": ["push_in", "pan_left"],
    "beat_snap_moved": 2,
    "sound_effects": ["hook:hit_1.wav"],
}


@pytest.mark.req("REQ-PUB-083")
@pytest.mark.parametrize(
    "state",
    [
        {"assembly_choices": CHOICES},
        # A resume keeps only the step entries.
        {"assemble_video": {"status": "done", "assembly_choices": CHOICES}},
    ],
)
def test_the_row_carries_the_assembly_choices(tmp_path: Path, state) -> None:
    row = choices_from_context(_ctx(tmp_path, state))

    assert row["still_moves"] == ["push_in", "pan_left"]
    assert row["beat_snap_moved"] == 2
    assert row["sound_effects"] == ["hook:hit_1.wav"]


@pytest.mark.req("REQ-PUB-083")
def test_the_report_counts_each_move_and_effect() -> None:
    rows = [
        {"still_moves": ["push_in", "pan_left"], "sound_effects": ["hook:a.wav"]},
        {"still_moves": ["push_in"], "beat_snap_moved": 0},
        {},
    ]

    counts = distribution(rows)

    assert counts["still_moves"] == {"push_in": 2, "pan_left": 1}
    assert counts["sound_effects"] == {"hook:a.wav": 1}
    assert counts["beat_snap_moved"] == {"0": 1}


@pytest.mark.asyncio
async def test_the_step_and_the_state_keep_the_choices(monkeypatch) -> None:
    from src.video.producer import state, steps

    assert 'ctx.state["assembly_choices"] = assembler.assembly_choices()' in (
        inspect.getsource(steps.step_assemble_video)
    )
    ctx = SimpleNamespace(
        state={"assembly_choices": CHOICES},
        run_paths={"final_video_output": Path("v.mp4")},
    )
    monkeypatch.setattr(state, "_drop_dependents", lambda ctx, step: None)
    await state._update_state_after_step(ctx, state.STEP_ASSEMBLE_VIDEO)
    assert ctx.state["assemble_video"]["assembly_choices"] == CHOICES
