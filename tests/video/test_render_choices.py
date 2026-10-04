"""Each finished render records its choices; the report flags low variety."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.video.render_choices import (
    choices_from_context,
    choices_path,
    dominance_alerts,
    load_recent,
    record_render_choices,
    report,
    similar_scripts,
    warn_if_similar,
)


def _row(**overrides) -> dict:
    row = {
        "product_id": "B0X",
        "profile": "slideshow_images1",
        "script_template": "before_after",
        "voice_profile": "charon",
        "script": "A lamp that lights the desk.",
    }
    row.update(overrides)
    return row


@pytest.mark.req("REQ-PUB-083")
def test_a_recorded_row_is_read_back_and_a_broken_line_is_skipped(
    tmp_path: Path,
) -> None:
    record_render_choices(tmp_path, _row(product_id="B01"))
    with choices_path(tmp_path).open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    record_render_choices(tmp_path, _row(product_id="B02"))

    rows = load_recent(tmp_path, 10)

    assert [r["product_id"] for r in rows] == ["B01", "B02"]
    assert load_recent(tmp_path, 1)[0]["product_id"] == "B02"


def test_no_file_means_no_rows(tmp_path: Path) -> None:
    assert load_recent(tmp_path, 10) == []


def test_an_unwritable_store_does_not_fail_the_render(tmp_path: Path, caplog) -> None:
    (tmp_path / "state").write_text("a file where the directory should be")

    record_render_choices(tmp_path, _row())

    assert "Could not record render choices" in caplog.text


@pytest.mark.req("REQ-PUB-083")
def test_one_value_holding_most_renders_alerts() -> None:
    rows = [_row(script_template="before_after") for _ in range(5)]
    rows.append(_row(script_template="curiosity_hook"))

    alerts = dominance_alerts(rows, 0.6)

    assert [(dim, value) for dim, value, _ in alerts] == [
        ("script_template", "before_after")
    ]


def test_a_fixed_setting_and_a_short_window_do_not_alert() -> None:
    # voice_profile is the same in every row: a pinned setting, not drift.
    assert dominance_alerts([_row() for _ in range(8)], 0.6) == []
    rows = [_row(script_template="a"), _row(script_template="a"), _row()]
    assert dominance_alerts(rows, 0.6) == []


@pytest.mark.req("REQ-PUB-083")
def test_near_identical_scripts_alert_and_different_ones_do_not() -> None:
    rows = [
        _row(product_id="B01", script="This lamp clips to any desk and folds flat."),
        _row(product_id="B02", script="This lamp clips to any desk and folds flat!"),
        _row(product_id="B03", script="A kettle that boils in under two minutes."),
    ]

    pairs = similar_scripts(rows, 0.9)

    assert [(a, b) for a, b, _ in pairs] == [("B01", "B02")]


def test_the_report_names_each_alert() -> None:
    rows = [_row(product_id=f"B{i}", script=f"script {i} " * 10) for i in range(5)]
    rows.append(_row(product_id="B9", script_template="other", script="unique"))

    text = "\n".join(report(rows, 0.6, 0.95))

    assert "ALERT script_template: before_after in 83% of renders" in text


def _ctx(tmp_path: Path, music: Path | None) -> SimpleNamespace:
    return SimpleNamespace(
        state={
            "script_template": "before_after",
            "pillar": "utility",
            "cta": "Follow for more.",
            "hook_headline": "Desk lamp, no clamp",
            "tts_metadata": {"voice_profile": "charon", "voice_name": "Charon"},
            "subtitle_engine_resolved": "pycaps",
            "pycaps_metadata": {"template": "word-focus"},
            "cold_open_variant": "static_title_card",
        },
        config=SimpleNamespace(
            video_settings=SimpleNamespace(
                first_frame_pre_motion=False, video_transition_duration=0.3
            )
        ),
        product=SimpleNamespace(asin="B0CTX"),
        profile_name="slideshow_images1",
        profile=SimpleNamespace(
            video_assembly_mode="sequential",
            first_frame_pre_motion=None,
            video_transition_duration=0.5,
        ),
        run_paths={"music_info_file": music, "run_root": tmp_path / "B0CTX"},
        script="The script.",
    )


@pytest.mark.req("REQ-PUB-083")
def test_the_row_reads_every_dimension_off_the_context(tmp_path: Path) -> None:
    music = tmp_path / "music_choice.json"
    music.write_text(json.dumps({"name": "Calm Lofi", "path": "/x.mp3"}))

    row = choices_from_context(_ctx(tmp_path, music))

    assert row["voice_profile"] == "charon"
    assert row["caption_template"] == "word-focus"
    assert row["music"] == "Calm Lofi"
    assert row["cold_open_variant"] == "static_title_card"
    # The profile's own value wins; an unset one falls back to the global.
    assert row["transition_sec"] == 0.5
    assert row["pre_motion"] is False
    assert row["script"] == "The script."


@pytest.mark.parametrize("content", [None, "{not json", "[1, 2]"])
def test_a_missing_or_unreadable_music_file_leaves_music_empty(
    tmp_path: Path, content: str | None
) -> None:
    music = tmp_path / "music_choice.json"
    if content is not None:
        music.write_text(content)

    assert choices_from_context(_ctx(tmp_path, music))["music"] is None


@pytest.mark.req("REQ-PUB-083")
@pytest.mark.asyncio
async def test_a_finished_render_appends_its_row(tmp_path: Path) -> None:
    import warnings

    from src.scraper.amazon.models import ProductData
    from src.video.producer import orchestration

    async def fake_load(ctx):
        ctx.state = {}

    async def fake_execute(ctx):
        ctx.state["script_template"] = "before_after"
        ctx.script = "A finished script."
        return True, None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from src.video.config import load_video_config_modular

        config = load_video_config_modular()
    config.global_output_root_path = tmp_path
    product = ProductData(
        title="A product",
        price="$10",
        url="https://www.amazon.com/dp/B0REC00001",
        platform=None,
        asin="B0REC00001",
    )

    with (
        patch.object(orchestration, "_load_pipeline_state", fake_load),
        patch.object(orchestration, "execute_pipeline_parallel", fake_execute),
    ):
        await orchestration.create_video_for_product(
            config, product, "slideshow_images1", {}, None, False, False, None
        )

    rows = load_recent(tmp_path, 10)
    assert len(rows) == 1
    assert rows[0]["product_id"] == "B0REC00001"
    assert rows[0]["script_template"] == "before_after"
    assert rows[0]["script"] == "A finished script."


def test_warn_if_similar_names_a_close_recent_script_and_skips_itself(
    tmp_path: Path, caplog
) -> None:
    text = "This lamp clips to any desk and folds flat for travel."
    record_render_choices(tmp_path, _row(product_id="B0OLD", script=text))
    record_render_choices(tmp_path, _row(product_id="B0SELF", script=text))
    record_render_choices(
        tmp_path, _row(product_id="B0FAR", script="A kettle that boils fast.")
    )

    matches = warn_if_similar(tmp_path, "B0SELF", text + "!")

    assert [m[0] for m in matches] == ["B0OLD"]
    assert "B0OLD" in caplog.text


@pytest.mark.req("REQ-PUB-083")
@pytest.mark.asyncio
async def test_the_script_step_warns_on_a_near_repeat(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    from unittest.mock import AsyncMock, MagicMock

    from src.video.config import config
    from src.video.producer import steps
    from src.video.producer.context import PipelineContext
    from src.video.producer.state import get_video_run_paths

    script = "A magnetic mount that holds the phone through every pothole. Link in bio."
    record_render_choices(tmp_path, _row(product_id="B0EARLIER", script=script))
    monkeypatch.setattr(config, "global_output_root_path", tmp_path)
    ctx = PipelineContext(
        product=MagicMock(asin="B0NEW00001", topic=None, title="Mount"),
        profile=config.video_profiles["slideshow_images1"],
        profile_name="slideshow_images1",
        config=config,
        secrets={},
        session=MagicMock(),
        run_paths=get_video_run_paths(config, "B0NEW00001", "slideshow_images1"),
        debug_mode=False,
    )

    with (
        patch.object(
            steps,
            "generate_ai_script",
            AsyncMock(return_value=(script, "classic_promo", "Link in bio.")),
        ),
        patch.object(steps, "_ensure_fact_checked", AsyncMock()),
        patch.object(steps, "_ensure_hook_headline", AsyncMock()),
    ):
        await steps.step_generate_script(ctx)

    assert "B0EARLIER" in caplog.text


def test_a_write_cut_mid_character_does_not_break_reading(tmp_path: Path) -> None:
    """An out-of-memory kill can stop a write inside a multi-byte character."""
    record_render_choices(tmp_path, _row(product_id="B01"))
    with choices_path(tmp_path).open("ab") as fh:
        # No newline: a real torn write stops mid-row.
        fh.write('{"product_id": "B02", "script": "a –'.encode()[:-2])
    record_render_choices(tmp_path, _row(product_id="B03"))

    assert [r["product_id"] for r in load_recent(tmp_path, 10)] == ["B01", "B03"]
    assert warn_if_similar(tmp_path, "B04", "anything") == []


def test_a_rerun_of_one_product_is_not_a_near_duplicate_or_counted_twice() -> None:
    rows = [_row(product_id="B01", script_template="x") for _ in range(2)]
    rows += [
        _row(product_id=f"B0{i}", script=f"distinct script {i}") for i in range(2, 6)
    ]

    text = "\n".join(report(rows, 0.6, 0.5))

    assert "B01 and B01" not in text
    assert "last 5 product(s)" in text
    assert "script_template: before_after 4, x 1" in text


@pytest.mark.req("REQ-PUB-083")
@pytest.mark.asyncio
async def test_a_step_run_records_nothing(tmp_path: Path) -> None:
    import warnings

    from src.scraper.amazon.models import ProductData
    from src.video.producer import orchestration

    async def fake_load(ctx):
        ctx.state = {}

    async def fake_runner(ctx):
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from src.video.config import load_video_config_modular

        config = load_video_config_modular()
    config.global_output_root_path = tmp_path
    product = ProductData(
        title="A product",
        price="$10",
        url="https://www.amazon.com/dp/B0STEP0001",
        platform=None,
        asin="B0STEP0001",
    )
    runners = {name: fake_runner for name in orchestration.step_runners()}

    with (
        patch.object(orchestration, "_load_pipeline_state", fake_load),
        patch.object(orchestration, "step_runners", lambda: runners),
        patch.object(orchestration, "_load_artifacts_from_state", lambda *a: True),
    ):
        await orchestration.create_video_for_product(
            config,
            product,
            "slideshow_images1",
            {},
            None,
            False,
            False,
            "generate_script",
        )

    assert load_recent(tmp_path, 10) == []


def test_rows_without_a_product_id_are_kept_apart() -> None:
    rows = [_row(product_id=None, script_template=t) for t in "abcde"]

    assert "last 5 product(s)" in report(rows, 0.6, 0.5)[0]
