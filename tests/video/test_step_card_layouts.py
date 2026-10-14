"""Step-card layout variants, one per video (REQ-VID-153)."""

from __future__ import annotations

import inspect
import shutil
import subprocess
from pathlib import Path

import pytest

from src.video.assembler.overlay_builder import (
    apply_step_card_overlay,
    pick_step_card_layout,
)
from src.video.config.visual_models import GraphicsSettings, StepCardLayout
from src.video.step_cards import StepCard

CARD = StepCard("Step 1 of 2", "Settings > General", 0.5, 1.0)
POOL = ["centred", "left", "inline"]


def _drawtexts(tmp_path: Path, layout: StepCardLayout) -> list[str]:
    settings = GraphicsSettings(enabled=True, layouts=[layout])
    chain = apply_step_card_overlay(
        ["[0:v]copy[v_out]"], settings, [CARD], 96, 1080, tmp_path
    )
    return chain[-2].split(";")


def _texts(tmp_path: Path) -> list[str]:
    return [
        p.read_text(encoding="utf-8")
        for p in sorted(tmp_path.glob("step_card_line_*.txt"))
    ]


@pytest.mark.req("REQ-VID-153")
def test_one_entry_turns_the_rotation_off() -> None:
    assert pick_step_card_layout(["left"], "B0TEST0001") == "left"


@pytest.mark.req("REQ-VID-153")
def test_a_product_draws_the_same_layout_every_time() -> None:
    ids = [f"B0TEST{n:04d}" for n in range(30)]
    first = [pick_step_card_layout(POOL, i) for i in ids]
    assert first == [pick_step_card_layout(POOL, i) for i in ids]
    assert set(first) == set(POOL)


@pytest.mark.req("REQ-VID-153")
def test_centred_stacks_the_counter_over_the_path(tmp_path: Path) -> None:
    drawtexts = _drawtexts(tmp_path, "centred")
    assert len(drawtexts) == 2
    assert all("x=(w-text_w)/2:" in d for d in drawtexts)


@pytest.mark.req("REQ-VID-153")
def test_left_starts_both_lines_at_the_margin(tmp_path: Path) -> None:
    drawtexts = _drawtexts(tmp_path, "left")
    assert len(drawtexts) == 2
    assert all("x=w*0.1100:" in d for d in drawtexts)


@pytest.mark.req("REQ-VID-153")
def test_inline_draws_one_line(tmp_path: Path) -> None:
    drawtexts = _drawtexts(tmp_path, "inline")
    assert len(drawtexts) == 1
    assert "x=(w-text_w)/2:" in drawtexts[0]
    assert _texts(tmp_path) == ["Step 1 of 2: Settings > General"]


@pytest.mark.req("REQ-VID-153")
def test_the_overlay_draws_the_products_layout(tmp_path: Path) -> None:
    settings = GraphicsSettings(enabled=True, layouts=["left", "inline"])
    for product_id in ("B0TEST0001", "B0TEST0002", "B0TEST0003", "B0TEST0004"):
        chain = apply_step_card_overlay(
            ["[0:v]copy[v_out]"],
            settings,
            [CARD],
            96,
            1080,
            tmp_path,
            product_id=product_id,
        )
        lines = len(chain[-2].split(";"))
        expected = pick_step_card_layout(settings.layouts, product_id)
        assert lines == (1 if expected == "inline" else 2), product_id


@pytest.mark.req("REQ-VID-153")
def test_the_assembler_passes_the_product_id() -> None:
    from src.video.assembler import core

    assemble = inspect.getsource(core.VideoAssembler.assemble_video)
    call = assemble[assemble.index("apply_step_card_overlay(") :]
    assert "product_id=self.product_id" in call[: call.index(")\n")]


def _left_edge_of_box(tmp_path: Path, layout: StepCardLayout) -> int:
    """Leftmost dark column on a grey frame while the card is shown."""
    chain = apply_step_card_overlay(
        ["[0:v]copy[v_out]"],
        GraphicsSettings(enabled=True, layouts=[layout]),
        [CARD],
        96,
        1080,
        tmp_path,
    )
    raw = subprocess.run(
        [
            "ffmpeg", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=0x808080:s=1080x1920:r=10:d=1.5",
            "-filter_complex", ";".join([*chain, "[v_out]format=gray[g]"]),
            "-map", "[g]", "-ss", "0.75", "-frames:v", "1",
            "-f", "rawvideo", "-",
        ],
        check=True,
        capture_output=True,
    ).stdout  # fmt: skip
    dark = [i % 1080 for i, v in enumerate(raw) if v < 100]
    return min(dark)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
@pytest.mark.req("REQ-VID-153")
def test_ffmpeg_draws_the_left_layout_at_the_margin(tmp_path: Path) -> None:
    # 11% of 1080 is 119 px; the box border reaches 14 px further left.
    left = _left_edge_of_box(tmp_path, "left")
    assert 100 <= left <= 110
    assert _left_edge_of_box(tmp_path, "centred") > left + 50


LONG = StepCard(
    "Step 3 of 5", "Settings > Notifications > Notification Summary", 0.5, 1.0
)


@pytest.mark.req("REQ-VID-153")
def test_an_inline_line_that_cannot_fit_is_stacked(tmp_path: Path) -> None:
    settings = GraphicsSettings(enabled=True, layouts=["inline"])
    chain = apply_step_card_overlay(
        ["[0:v]copy[v_out]"], settings, [CARD, LONG], 96, 1080, tmp_path
    )
    # One line for the short card, two for the long one.
    assert len(chain[-2].split(";")) == 3


def _white_extent(tmp_path: Path, card: StepCard, layout: StepCardLayout) -> range:
    chain = apply_step_card_overlay(
        ["[0:v]copy[v_out]"],
        GraphicsSettings(enabled=True, layouts=[layout]),
        [card],
        96,
        1080,
        tmp_path,
    )
    raw = subprocess.run(
        [
            "ffmpeg", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=0x808080:s=1080x1920:r=10:d=1.5",
            "-filter_complex", ";".join([*chain, "[v_out]format=gray[g]"]),
            "-map", "[g]", "-ss", "0.75", "-frames:v", "1",
            "-f", "rawvideo", "-",
        ],
        check=True,
        capture_output=True,
    ).stdout  # fmt: skip
    columns = [i % 1080 for i, v in enumerate(raw) if v > 200]
    return range(min(columns), max(columns) + 1)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
@pytest.mark.req("REQ-VID-153")
@pytest.mark.parametrize("layout", ["centred", "left", "inline"])
def test_ffmpeg_keeps_a_long_path_inside_the_frame(
    tmp_path: Path, layout: StepCardLayout
) -> None:
    extent = _white_extent(tmp_path, LONG, layout)
    assert extent.start > 20
    assert extent.stop < 1060


@pytest.mark.req("REQ-VID-153")
def test_the_assembler_records_the_layout_it_drew() -> None:
    from src.video.assembler import core

    assemble = inspect.getsource(core.VideoAssembler.assemble_video)
    record = assemble[assemble.index("self.step_card_layout = (") :]
    assert "pick_step_card_layout(" in record[:300]
    assert "if video_filters != unchanged" in record[:300]
    assert '"step_card_layout": self.step_card_layout' in inspect.getsource(
        core.VideoAssembler.assembly_choices
    )
