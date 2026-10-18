"""The checklist graphic: a step's common mistake and its fix (REQ-VID-164)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.ai.step_list import (
    CHECKLIST_RULE,
    PROMPT_PATH,
    parse_step_list,
    step_list_prompt,
)
from src.video.assembler.overlay_builder import build_step_card_drawtext
from src.video.config import config
from src.video.config.visual_models import GraphicsSettings
from src.video.step_cards import StepCard, add_checklist, load_step_cards

SOURCE = "https://support.example.com/1"


def _answer(mistake: dict | None) -> str:
    return json.dumps(
        {
            "start_screen": "Settings",
            "steps": [
                {"action": "Tap Apps", "ui_path": "Apps", "source": SOURCE},
                {"action": "Tap Storage", "ui_path": "Storage", "source": SOURCE},
            ],
            "common_mistake": mistake,
            "topic_check": {
                "specific": True,
                "searchable": True,
                "demonstrable": True,
                "non_default": True,
                "advice": "none",
            },
        }
    )


@pytest.mark.unit
@pytest.mark.req("REQ-VID-164")
class TestTheStepList:
    def test_the_pair_is_kept_when_each_fits_a_glance(self) -> None:
        parsed = parse_step_list(
            _answer(
                {
                    "step": 2,
                    "mistake": "Tapping Clear storage wipes your logins",
                    "dont": "Tap Clear storage",
                    "do": "Tap Clear cache",
                }
            )
        )

        assert parsed is not None
        assert (parsed.mistake_dont, parsed.mistake_do) == (
            "Tap Clear storage",
            "Tap Clear cache",
        )

    @pytest.mark.parametrize(
        "pair",
        [
            {"dont": "Tap Clear storage"},
            {"dont": "", "do": "Tap Clear cache"},
            {
                "dont": "Tap Clear storage because it looks like the right one here",
                "do": "Tap Clear cache",
            },
        ],
    )
    def test_a_missing_or_long_line_drops_both(self, pair: dict) -> None:
        parsed = parse_step_list(_answer({"step": 2, "mistake": "x", **pair}))

        assert parsed is not None
        assert (parsed.mistake_dont, parsed.mistake_do) == (None, None)

    def test_the_prompt_asks_only_when_the_checklist_is_on(self) -> None:
        plain = PROMPT_PATH.read_text(encoding="utf-8").format(
            TOPIC_TITLE="t", TOPIC_DETAIL="d", MAX_STEPS=6
        )

        assert step_list_prompt("t", "d", 6) == plain
        assert step_list_prompt("t", "d", 6, checklist=True).endswith(CHECKLIST_RULE)


@pytest.mark.req("REQ-VID-164")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("graphics", "checklist", "asked"),
    [(True, True, True), (True, False, False), (False, True, False)],
)
async def test_the_script_step_asks_for_it_with_the_graphic_on(
    tmp_path: Path, graphics: bool, checklist: bool, asked: bool
) -> None:
    from src.ai.step_list import Step, StepList
    from src.video.producer import steps

    cfg = config.model_copy(deep=True)
    cfg.llm_settings.topic_scripts.step_list.enabled = True
    cfg.video_settings.graphics.enabled = graphics
    cfg.video_settings.graphics.checklist = checklist
    ctx = SimpleNamespace(
        config=cfg,
        product=SimpleNamespace(topic="t", title="How to x", description="d"),
        secrets={cfg.llm_settings.api_key_env_var: "k"},
        run_paths={"script_file": tmp_path / "text" / "script.txt"},
        state={},
    )
    built = AsyncMock(
        return_value=StepList(
            "Settings", "iOS", False, [Step("a", "A", "b", SOURCE)], topic_failures=[]
        )
    )
    with patch("src.ai.step_list.build_step_list", built):
        await steps._topic_step_list(ctx)

    assert built.await_args is not None
    assert built.await_args.kwargs["checklist"] is asked


CARDS = [
    StepCard("Step 1 of 2", "Apps", 2.0, 5.0),
    StepCard("Step 2 of 2", "Storage", 5.0, 11.0),
]


@pytest.mark.unit
@pytest.mark.req("REQ-VID-164")
class TestTheCards:
    def test_the_mistake_step_ends_on_the_checklist(self) -> None:
        cards = add_checklist(
            CARDS,
            mistake_step=2,
            dont="Tap Clear storage",
            do="Tap Clear cache",
            total=2,
            min_sec=1.5,
        )

        assert cards[:2] == [CARDS[0], StepCard("Step 2 of 2", "Storage", 5.0, 8.0)]
        assert cards[2] == StepCard(
            "Tap Clear storage", "Tap Clear cache", 8.0, 11.0, kind="checklist"
        )

    def test_a_short_step_shows_only_the_checklist(self) -> None:
        cards = add_checklist(
            CARDS, mistake_step=1, dont="a", do="b", total=2, min_sec=2.0
        )

        assert cards[0] == StepCard("a", "b", 2.0, 5.0, kind="checklist")
        assert cards[1] == CARDS[1]

    @pytest.mark.parametrize(
        ("step", "dont", "do"), [(None, "a", "b"), (2, None, "b"), (2, "a", None)]
    )
    def test_nothing_to_show_leaves_the_cards(self, step, dont, do) -> None:
        assert (
            add_checklist(
                CARDS, mistake_step=step, dont=dont, do=do, total=2, min_sec=1.5
            )
            == CARDS
        )

    def test_loaded_from_the_recorded_list(self, tmp_path: Path) -> None:
        record = tmp_path / "step_list.json"
        record.write_text(
            json.dumps(
                {
                    "steps": [
                        {"ui_path": "Apps", "action": "Tap Apps"},
                        {"ui_path": "Storage", "action": "Tap Storage"},
                    ],
                    "mistake_step": 2,
                    "mistake_dont": "Tap Clear storage",
                    "mistake_do": "Tap Clear cache",
                }
            ),
            encoding="utf-8",
        )
        text = "Here is how. Tap Apps first and wait. Then tap Storage and pick it now."
        spoken = [
            {"word": w, "start_time": i * 0.5, "end_time": i * 0.5 + 0.4}
            for i, w in enumerate(text.split())
        ]
        options = {"not_before": 0.0, "min_sec": 0.5, "max_sec": 6.0}

        plain = load_step_cards(record, spoken, **options)
        with_list = load_step_cards(record, spoken, checklist=True, **options)

        assert [c.kind for c in plain] == ["step", "step"]
        assert [c.kind for c in with_list] == ["step", "step", "checklist"]


@pytest.mark.unit
@pytest.mark.req("REQ-VID-164")
@pytest.mark.parametrize("layout", ["centred", "left", "inline"])
def test_drawn_as_two_coloured_lines_in_every_layout(
    tmp_path: Path, layout: str
) -> None:
    settings = GraphicsSettings(enabled=True, checklist=True)
    card = StepCard("Tap Clear storage", "Tap Clear cache", 1.0, 2.0, "checklist")

    chain = build_step_card_drawtext(
        [card], settings, 96, 1080, tmp_path, "[in]", "[out]", layout=layout
    )

    lines = chain.split(";")
    assert len(lines) == 2
    assert f"fontcolor={settings.checklist_dont_color}" in lines[0]
    assert f"fontcolor={settings.checklist_do_color}" in lines[1]
    texts = sorted(p.read_text() for p in tmp_path.glob("step_card_line_*.txt"))
    assert texts == ["Do: Tap Clear cache", "Don't: Tap Clear storage"]


@pytest.mark.req("REQ-VID-164")
def test_the_checklist_ships_off() -> None:
    from src.video.config import load_video_config_modular

    assert load_video_config_modular().video_settings.graphics.checklist is False
