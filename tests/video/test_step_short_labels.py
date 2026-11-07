"""A step with no menu path shows a short label, not its cut sentence.

REQ-VID-123: the step screen and the recap name the control for a physical
step ("Power + Volume Down"), asked for in the step-list call.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.ai.step_list import (
    LABEL_RULE,
    Step,
    StepList,
    parse_step_list,
    step_list_prompt,
)
from src.video.producer import steps
from src.video.step_visuals import plan_step_visuals

SOURCE = "https://example.com/help"
PHYSICAL = "Press the power and volume down buttons together"


def _answer(**extra: object) -> str:
    step = {"action": PHYSICAL, "ui_path": "", "expected": "", "source": SOURCE}
    step.update(extra)
    return json.dumps({"steps": [step]})


@pytest.mark.req("REQ-VID-123")
class TestTheLabelIsParsed:
    def test_a_short_label_on_a_pathless_step_is_kept(self) -> None:
        parsed = parse_step_list(_answer(label="Power + Volume Down"))
        assert parsed is not None
        assert parsed.steps[0].label == "Power + Volume Down"

    def test_a_label_over_four_words_is_dropped(self) -> None:
        parsed = parse_step_list(_answer(label="the power and volume buttons"))
        assert parsed is not None
        assert parsed.steps[0].label == ""

    def test_a_step_with_a_path_keeps_no_label(self) -> None:
        parsed = parse_step_list(_answer(ui_path="Settings > Sound", label="Sound"))
        assert parsed is not None
        assert parsed.steps[0].label == ""


@pytest.mark.req("REQ-VID-123")
class TestThePromptAsksForIt:
    def test_only_with_labels_on(self) -> None:
        plain = step_list_prompt("t", "d", 6)
        assert LABEL_RULE not in plain
        assert step_list_prompt("t", "d", 6, labels=True).endswith(LABEL_RULE)

    @pytest.mark.parametrize(
        ("visuals", "short", "asked"),
        [(True, True, True), (True, False, False), (False, True, False)],
    )
    async def test_the_render_asks_when_screens_are_drawn(
        self, config, tmp_path: Path, visuals: bool, short: bool, asked: bool
    ) -> None:
        cfg = config.model_copy(deep=True)
        cfg.llm_settings.topic_scripts.step_list.enabled = True
        cfg.video_settings.step_visuals.enabled = visuals
        cfg.video_settings.step_visuals.short_labels = short
        ctx = SimpleNamespace(
            config=cfg,
            product=SimpleNamespace(topic="t", title="How to x", description="d"),
            secrets={cfg.llm_settings.api_key_env_var: "k"},
            run_paths={"script_file": tmp_path / "text" / "script.txt"},
            state={},
        )
        built = AsyncMock(
            return_value=StepList(
                "Settings",
                "iOS",
                False,
                [Step("a", "A", "b", SOURCE)],
                topic_failures=[],
            )
        )
        with patch("src.ai.step_list.build_step_list", built):
            await steps._topic_step_list(ctx)
        assert built.await_args is not None
        assert built.await_args.kwargs["labels"] is asked


@pytest.mark.req("REQ-VID-123")
class TestTheScreenShowsIt:
    STEPS = [
        {"ui_path": "Settings", "action": "Open Settings"},
        {"ui_path": "", "action": PHYSICAL, "label": "Power + Volume Down"},
    ]
    TEXT = (
        "Take a screenshot fast. First open Settings on the phone now. "
        "Then press the power and volume down buttons together. "
        "The screen flashes. That is the whole trick."
    )

    def _plan(self, steps_: list[dict]) -> list:
        spoken = [
            {"word": w, "start_time": i * 0.5, "end_time": i * 0.5 + 0.4}
            for i, w in enumerate(self.TEXT.split())
        ]
        return plan_step_visuals(
            steps_,
            spoken,
            start_screen="Home screen",
            video_end=20.0,
            opening_max_sec=2.0,
            recap_sec=2.5,
        )

    def test_the_step_and_the_recap_use_the_label(self) -> None:
        screens = self._plan(self.STEPS)
        step = [s for s in screens if s.kind == "step"][1]
        recap = [s for s in screens if s.kind == "recap"][0]
        assert step.target == "Power + Volume Down"
        assert recap.rows == ["Settings", "Power + Volume Down"]

    def test_a_step_list_without_labels_keeps_the_action(self) -> None:
        unlabelled = [dict(s, label="") for s in self.STEPS]
        step = [s for s in self._plan(unlabelled) if s.kind == "step"][1]
        assert step.target == PHYSICAL


@pytest.fixture
def config():
    from src.video.config_adapter import load_video_config_modular

    return load_video_config_modular()
