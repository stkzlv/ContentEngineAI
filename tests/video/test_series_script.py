"""A series episode's opening line, prefix and claim queue (REQ-CNT-194/195)."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.video.config.series_models import SeriesConfig
from src.video.series import (
    QUEUE_COLUMNS,
    assign_episode,
    mark_rendered,
    opening_line,
    place_opening,
    prefix,
    with_prefix,
)

CLAIM = "A magnet wipes a phone"
PRODUCT = SimpleNamespace(series="myth", claim=CLAIM, verdict="NO")


def _queue(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=QUEUE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in QUEUE_COLUMNS})
    return path


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.mark.req("REQ-CNT-194")
def test_the_opening_line_is_placed_word_for_word():
    series = SeriesConfig(opening_line="Myth check: {claim}.")
    line = opening_line(series, PRODUCT)
    assert line == "Myth check: A magnet wipes a phone."
    assert place_opening("It doesn't. Here's why.", line).startswith(line + " It")
    already = f"{line} It doesn't."
    assert place_opening(already, line) == already
    assert place_opening("It doesn't.", "") == "It doesn't."


@pytest.mark.req("REQ-CNT-194")
def test_the_prefix_leads_and_carries_the_number_only_when_numbered(tmp_path):
    plain = SeriesConfig(title_prefix="Myth check: ")
    numbered = SeriesConfig(
        title_prefix="Myth check: ", numbering="episode", queue_file=tmp_path / "q.csv"
    )
    assert prefix(plain, 7) == "Myth check: "
    assert prefix(numbered, 7) == "Myth check: 7 "
    assert with_prefix("Magnets and phones", "Myth check: ") == (
        "Myth check: Magnets and phones"
    )
    assert with_prefix("Myth check: Magnets", "Myth check: ") == "Myth check: Magnets"
    assert with_prefix(None, "Myth check: ") is None


@pytest.mark.req("REQ-CNT-195")
def test_an_empty_episode_gets_one_more_than_the_highest_and_keeps_it(tmp_path):
    queue = _queue(
        tmp_path / "q.csv",
        [{"claim": "Old claim", "episode": "4"}, {"claim": CLAIM, "status": "queued"}],
    )
    series = SeriesConfig(numbering="episode", queue_file=queue)
    assert assign_episode(series, PRODUCT) == 5
    assert assign_episode(series, PRODUCT) == 5
    assert _read(queue)[1]["episode"] == "5"


@pytest.mark.req("REQ-CNT-195")
def test_no_numbering_leaves_the_queue_alone(tmp_path):
    queue = _queue(tmp_path / "q.csv", [{"claim": CLAIM}])
    assert assign_episode(SeriesConfig(queue_file=queue), PRODUCT) is None
    assert _read(queue)[0]["episode"] == ""


@pytest.mark.req("REQ-CNT-195")
def test_a_render_marks_the_row_rendered_and_leaves_other_statuses(tmp_path):
    queue = _queue(
        tmp_path / "q.csv",
        [{"claim": CLAIM, "status": "picked"}, {"claim": "Other", "status": "queued"}],
    )
    mark_rendered(SeriesConfig(queue_file=queue), PRODUCT, "topic-magnets-123")
    rows = _read(queue)
    assert (rows[0]["status"], rows[0]["output_id"]) == (
        "rendered",
        "topic-magnets-123",
    )
    assert rows[1]["status"] == "queued"


@pytest.mark.req("REQ-CNT-195")
def test_a_claim_with_no_row_is_logged_not_added(tmp_path, caplog):
    queue = _queue(tmp_path / "q.csv", [{"claim": "Other"}])
    series = SeriesConfig(numbering="episode", queue_file=queue)
    assert assign_episode(series, PRODUCT) is None
    assert "no row for claim" in caplog.text
    assert len(_read(queue)) == 1


@pytest.fixture
def step_ctx(tmp_path):
    from src.video.config import config
    from src.video.producer.context import PipelineContext
    from src.video.producer.state import get_video_run_paths

    cfg = config.model_copy(deep=True)
    cfg.global_output_root_path = tmp_path
    queue = _queue(tmp_path / "q.csv", [{"claim": CLAIM, "episode": ""}])
    cfg.series = {
        "myth": SeriesConfig(
            title_prefix="Myth check: ",
            opening_line="Myth check: {claim}.",
            verdicts=["NO"],
            numbering="episode",
            queue_file=queue,
        )
    }
    paths = get_video_run_paths(cfg, "B0TEST0194", "slideshow_images1")
    product = MagicMock(
        asin="B0TEST0194",
        topic=None,
        topic_kind=None,
        title="Magnets",
        keyword="",
        series="myth",
        claim=CLAIM,
        verdict="NO",
    )
    return PipelineContext(
        product=product,
        profile=cfg.video_profiles["slideshow_images1"],
        profile_name="slideshow_images1",
        config=cfg,
        secrets={},
        session=MagicMock(),
        run_paths=paths,
        debug_mode=False,
    )


@pytest.mark.req("REQ-CNT-194")
@pytest.mark.req("REQ-CNT-195")
@pytest.mark.asyncio
async def test_the_step_numbers_the_episode_and_opens_on_the_line(step_ctx):
    from src.video.producer import steps

    with (
        patch.object(
            steps,
            "generate_ai_script",
            AsyncMock(return_value=("It doesn't. Tell me.", "t", "")),
        ),
        patch.object(steps, "_ensure_fact_checked", AsyncMock()),
        patch.object(steps, "_ensure_hook_headline", AsyncMock()),
    ):
        await steps.step_generate_script(step_ctx)

    assert step_ctx.state["series_episode"] == "1"
    assert step_ctx.script.startswith("Myth check: A magnet wipes a phone. It doesn't.")
    assert steps._series_prefix(step_ctx) == "Myth check: 1 "


@pytest.mark.req("REQ-CNT-194")
@pytest.mark.asyncio
async def test_unified_metadata_leads_with_the_prefix(step_ctx):
    import json

    from src.video.producer import steps

    step_ctx.state["series_episode"] = "3"
    Path(step_ctx.run_paths["run_root"]).mkdir(parents=True, exist_ok=True)
    step_ctx.script = "Myth check: x. It doesn't."
    with patch.object(
        steps, "generate_ai_description", AsyncMock(return_value="Magnets are fine.")
    ):
        await steps._generate_unified_metadata(step_ctx)

    data = json.loads(
        (Path(step_ctx.run_paths["run_root"]) / "metadata.json").read_text()
    )
    assert data["title"].startswith("Myth check: 3 ")
    assert data["description"].startswith("Myth check: 3 Magnets are fine.")


@pytest.mark.req("REQ-CNT-195")
def test_a_bom_csv_from_a_spreadsheet_keeps_its_columns(tmp_path):
    queue = tmp_path / "q.csv"
    queue.write_text(
        "﻿" + ",".join(QUEUE_COLUMNS) + "\n" + f"m1,{CLAIM},NO,s,queued,,,\n",
        encoding="utf-8",
    )
    assert (
        assign_episode(SeriesConfig(numbering="episode", queue_file=queue), PRODUCT)
        == 1
    )
    rows = _read(queue)
    assert rows[0]["id"] == "m1" and list(rows[0]) == QUEUE_COLUMNS


@pytest.mark.req("REQ-CNT-195")
def test_a_row_with_extra_cells_is_left_untouched(tmp_path, caplog):
    queue = tmp_path / "q.csv"
    text = (
        ",".join(QUEUE_COLUMNS) + "\n" + f"m1,{CLAIM},NO,http://a,http://b,queued,,,\n"
    )
    queue.write_text(text, encoding="utf-8")
    mark_rendered(SeriesConfig(queue_file=queue), PRODUCT, "out1")
    assert queue.read_text(encoding="utf-8") == text
    assert "extra cells" in caplog.text


@pytest.mark.req("REQ-CNT-195")
def test_padded_episode_numbers_count(tmp_path):
    queue = _queue(
        tmp_path / "q.csv", [{"claim": "Other", "episode": " 5"}, {"claim": CLAIM}]
    )
    assert (
        assign_episode(SeriesConfig(numbering="episode", queue_file=queue), PRODUCT)
        == 6
    )


@pytest.mark.req("REQ-CNT-195")
def test_a_missing_queue_warns_and_never_raises(tmp_path, caplog):
    series = SeriesConfig(queue_file=tmp_path / "absent" / "q.csv")
    mark_rendered(series, PRODUCT, "out1")
    assert "not found" in caplog.text


@pytest.mark.req("REQ-CNT-195")
def test_a_rerender_leaves_a_scheduled_row_alone(tmp_path):
    queue = _queue(
        tmp_path / "q.csv",
        [{"claim": CLAIM, "status": "scheduled", "output_id": "old"}],
    )
    mark_rendered(SeriesConfig(queue_file=queue), PRODUCT, "new")
    assert (_read(queue)[0]["status"], _read(queue)[0]["output_id"]) == (
        "scheduled",
        "old",
    )


@pytest.mark.req("REQ-CNT-194")
@pytest.mark.asyncio
async def test_the_opening_line_survives_the_fact_check(step_ctx):
    from src.video.producer import steps

    async def drop_first_sentence(ctx, pillar):
        ctx.script = ctx.script.split(". ", 1)[1]

    with (
        patch.object(
            steps,
            "generate_ai_script",
            AsyncMock(return_value=("It doesn't. Tell me.", "t", "")),
        ),
        patch.object(steps, "_ensure_fact_checked", drop_first_sentence),
        patch.object(steps, "_ensure_hook_headline", AsyncMock()),
    ):
        await steps.step_generate_script(step_ctx)

    assert step_ctx.script.startswith("Myth check: A magnet wipes a phone.")
    saved = step_ctx.run_paths["script_file"].read_text(encoding="utf-8")
    assert saved == step_ctx.script


@pytest.mark.req("REQ-CNT-194")
@pytest.mark.asyncio
async def test_optimized_metadata_and_instructions_carry_the_prefix(step_ctx):
    import json

    from src.ai.platform_metadata.models import PlatformMetadata
    from src.video.producer import steps

    step_ctx.state["series_episode"] = "2"
    Path(step_ctx.run_paths["run_root"]).mkdir(parents=True, exist_ok=True)
    step_ctx.run_paths["description_file"].parent.mkdir(parents=True, exist_ok=True)
    meta = PlatformMetadata(
        platform="youtube",
        description="Magnets are fine.",
        hashtags=[],
        keywords=[],
        character_counts={},
        generated_at="2026-01-01T00:00:00",
        product_id="B0TEST0194",
        validation_status="valid",
        validation_messages=[],
        title="Magnets and phones",
    )
    step_ctx.config.description_settings.metadata_mode = "optimized"
    with patch(
        "src.ai.platform_metadata.PlatformMetadataFactory.generate_multi_platform",
        AsyncMock(return_value={"youtube": meta}),
    ):
        ok = await steps._generate_optimized_metadata(step_ctx)

    if not ok:
        pytest.skip("platform metadata is off in this config")
    root = Path(step_ctx.run_paths["run_root"])
    data = json.loads((root / "metadata_youtube.json").read_text())
    assert data["title"].startswith("Myth check: 2 Magnets")
    text_dir = step_ctx.run_paths["description_file"].parent
    instructions = (text_dir / "UPLOAD_INSTRUCTIONS.txt").read_text()
    assert "Myth check: 2 Magnets" in instructions


@pytest.mark.req("REQ-CNT-195")
def test_only_a_whole_render_marks_the_row():
    import inspect

    from src.video.producer import orchestration

    source = inspect.getsource(orchestration.create_video_for_product)
    guard = source.index("if debug_step_target is None:")
    assert guard < source.index("mark_rendered(") < guard + 600
