"""The sample and check stages of the content research (design 0023)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.research import sample as sample_mod
from src.research.__main__ import main
from src.research.checks import check_one, run_checks, summarise
from src.research.config import SampleResearch
from src.research.report import render_report
from src.research.sample import is_task, run_sample, sample_one, variant_config
from src.video.config import load_video_config_modular
from src.video.producer.topic_input import TopicSpec, build_topic_product

CTA = "Drop a comment if this worked."


@pytest.mark.req("REQ-OPS-109")
def test_each_variant_changes_only_its_own_key() -> None:
    base = load_video_config_modular()
    # Pinned on, as shipped, so the variants' change is visible.
    base.llm_settings.topic_scripts.step_list.enabled = True
    template = base.llm_settings.script_templates.fixed_template

    shipped = variant_config(base, "shipped", "How to x")
    free = variant_config(base, "free_form", "How to x")
    task = variant_config(base, "task_answer_first", "How to x")
    problem = variant_config(base, "task_answer_first", "Why x")

    assert shipped.model_dump() == base.model_dump()
    assert free.llm_settings.topic_scripts.step_list.enabled is False
    assert task.llm_settings.script_templates.fixed_template == "topic_answer_first"
    assert task.llm_settings.topic_scripts.step_list.enabled is False
    assert problem.llm_settings.script_templates.fixed_template == template
    assert base.llm_settings.topic_scripts.step_list.enabled is True  # untouched
    with pytest.raises(ValueError, match="Unknown variant"):
        variant_config(base, "louder", None)


def test_a_task_is_a_how_to_title() -> None:
    assert is_task("How to clear app cache on Android")
    assert not is_task("Why your wifi keeps dropping")


def _ctx_step(
    script: str, template: str, steps: int | None = None, explainer: bool = False
):
    async def step(ctx) -> None:
        ctx.script = script
        ctx.state.update(
            {"script_template": template, "cta": CTA, "hook_headline": "Clear cache"}
        )
        temp = Path(ctx.run_paths["intermediate_base"])
        temp.mkdir(parents=True, exist_ok=True)
        (temp / "script_fact_check.json").write_text(
            json.dumps({"flagged": [{"claim": "x"}], "revision": {"accepted": True}})
        )
        if steps:
            (temp / "step_list.json").write_text(
                json.dumps({"steps": [{}] * steps, "explainer": explainer})
            )

    return step


@pytest.mark.req("REQ-OPS-109")
def test_a_sample_records_what_the_script_step_wrote(tmp_path: Path) -> None:
    from src.utils.outputs_paths import with_outputs_root

    config = load_video_config_modular(cli_overrides=with_outputs_root({}, tmp_path))
    product = build_topic_product(
        TopicSpec(title="How to clear app cache", description="d")
    )
    with patch(
        "src.video.producer.steps.step_generate_script",
        _ctx_step(f"Clear it. {CTA}", "topic_answer_first", steps=3),
    ):
        record = asyncio.run(sample_one(config, product, "slideshow_stock", {}, None))

    assert record["script"] == f"Clear it. {CTA}"
    assert record["template"] == "topic_answer_first" and record["cta"] == CTA
    assert record["steps"] == 3 and record["fact_check"]["flagged"]
    # Written under the scratch root, not the real outputs tree.
    assert list(tmp_path.rglob("script_fact_check.json"))


@pytest.mark.req("REQ-OPS-109")
def test_a_failed_sample_is_recorded_not_raised(tmp_path: Path) -> None:
    from src.utils.outputs_paths import with_outputs_root
    from src.video.producer.context import PipelineError

    config = load_video_config_modular(cli_overrides=with_outputs_root({}, tmp_path))
    product = build_topic_product(TopicSpec(title="How to x", description="d"))

    async def fail(ctx) -> None:
        raise PipelineError("dropped")

    with patch("src.video.producer.steps.step_generate_script", fail):
        record = asyncio.run(sample_one(config, product, "slideshow_stock", {}, None))

    assert record["error"] == "dropped" and "script" not in record


@pytest.mark.req("REQ-OPS-109")
def test_products_are_sampled_under_shipped_only(tmp_path: Path, monkeypatch) -> None:
    seen: list[tuple[str, bool]] = []

    async def fake_one(config, product, profile, secrets, session):
        seen.append(
            (
                config.llm_settings.script_templates.fixed_template or "-",
                config.llm_settings.topic_scripts.step_list.enabled,
            )
        )
        return {"id": product.asin, "title": product.title, "script": "x"}

    monkeypatch.setattr(sample_mod, "sample_one", fake_one)
    monkeypatch.setattr(sample_mod, "collect_producer_secrets", lambda c: {})
    product = build_topic_product(TopicSpec(title="Phone mount", description="d"))
    product.topic = None

    records = asyncio.run(
        run_sample(
            ["shipped", "free_form", "task_answer_first"],
            [TopicSpec(title="How to x", description="d")],
            [product],
            "slideshow_stock",
            tmp_path,
        )
    )

    kinds = [(r["variant"], r["kind"]) for r in records]
    assert kinds == [
        ("shipped", "topic"),
        ("shipped", "product"),
        ("free_form", "topic"),
        ("task_answer_first", "topic"),
    ]
    assert seen[2][1] is False and seen[3] == ("topic_answer_first", False)


def _record(**over: object) -> dict[str, object]:
    record: dict[str, object] = {
        "variant": "shipped",
        "kind": "topic",
        # The title carries words the first sentence lacks; the search does not.
        "title": "How to free up iPhone storage by offloading unused apps",
        "search": "free up iphone storage",
        "script": f"Free up iPhone storage in Settings. Tap General. {CTA}",
        "template": "topic_answer_first",
        "cta": CTA,
        "steps": None,
        "fact_check": {"flagged": [], "revision": {}},
    }
    record.update(over)
    return record


@pytest.mark.req("REQ-OPS-110")
def test_check_one_measures_each_rule() -> None:
    ok = check_one(_record(), (5, 30))
    assert ok["in_band"] and ok["phrase_in_first_sentence"] and ok["cta_last"]
    assert not ok["template_misfit"] and ok["lint"] is None and ok["flagged"] == 0

    bad = check_one(
        _record(
            script=f"Your phone is full. {CTA} Thanks.",
            template="topic_mistake_fix",
            fact_check={"flagged": [{}, {}], "revision": {"accepted": True}},
        ),
        (20, 30),
    )
    assert not bad["in_band"] and not bad["phrase_in_first_sentence"]
    assert not bad["cta_last"] and bad["template_misfit"]
    assert bad["flagged"] == 2 and bad["rewritten"]


@pytest.mark.req("REQ-OPS-110")
def test_a_step_list_script_is_held_to_its_own_band() -> None:
    from src.ai.step_list import word_range

    checked = check_one(_record(steps=6), (5, 30))
    assert checked["band"] == list(word_range(6))


def test_the_search_phrase_falls_back_to_the_title_and_the_keyword() -> None:
    topic = check_one(_record(search="", title="How to tap general"), (5, 30))
    product = check_one(
        _record(kind="product", keyword="iphone storage", title="A product"), (5, 30)
    )
    assert topic["phrase_in_first_sentence"] is False
    assert product["phrase_in_first_sentence"] is True


@pytest.mark.req("REQ-OPS-110")
def test_the_summary_counts_shares_and_repeated_openings() -> None:
    rows = [check_one(_record(), (5, 30)) for _ in range(3)]
    rows.append(check_one(_record(error="dropped", script=""), (5, 30)))

    summary = summarise(rows)

    assert summary["samples"] == 4 and summary["errors"] == 1
    assert summary["in_band"] == 1.0
    assert summary["repeated_openings"] == {"free up iphone": 3}


@pytest.mark.req("REQ-OPS-110")
def test_the_report_compares_variants(tmp_path: Path) -> None:
    records = [
        _record(),
        _record(variant="free_form", template="topic_answer_first"),
        _record(variant="task_answer_first", error="dropped", script=""),
    ]
    checks = run_checks(records, (5, 30))
    text = render_report(None, checks)

    assert (
        "| Check | `free_form/topic` | `shipped/topic` | `task_answer_first/topic` |"
        in text
    )
    assert "| Failed or dropped | 0 | 0 | 1 |" in text
    assert "failed: dropped" in text

    (tmp_path / "samples.json").write_text(json.dumps(records))
    assert main(["check", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "checks.json").exists()
    assert "## Script samples" in (tmp_path / "report.md").read_text()


def test_the_variants_must_be_known_and_keep_the_baseline() -> None:
    SampleResearch(variants=["shipped", "free_form"])
    with pytest.raises(ValidationError, match="unknown variant"):
        SampleResearch(variants=["shipped", "louder"])
    with pytest.raises(ValidationError, match="must include shipped"):
        SampleResearch(variants=["free_form"])


@pytest.mark.req("REQ-OPS-109")
def test_a_rerun_does_not_reuse_the_last_runs_scripts(
    tmp_path: Path, monkeypatch
) -> None:
    stale = tmp_path / "samples" / "shipped" / "topic-x" / "temp" / "script.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("Last run's script.")
    seen_stale: list[bool] = []

    async def fake_one(config, product, profile, secrets, session):
        seen_stale.append(stale.exists())
        return {"id": product.asin, "title": product.title, "script": "x"}

    monkeypatch.setattr(sample_mod, "sample_one", fake_one)
    monkeypatch.setattr(sample_mod, "collect_producer_secrets", lambda c: {})
    asyncio.run(
        run_sample(
            ["shipped"],
            [TopicSpec(title="How to x", description="d")],
            [],
            "slideshow_stock",
            tmp_path,
        )
    )

    assert seen_stale == [False]


def test_an_unknown_profile_fails_before_any_sample(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[str] = []

    async def fake_one(*args):
        calls.append("called")
        return {}

    monkeypatch.setattr(sample_mod, "sample_one", fake_one)
    with pytest.raises(ValueError, match="names no video profile"):
        asyncio.run(
            run_sample(["shipped"], [TopicSpec(title="t")], [], "no_such", tmp_path)
        )
    assert calls == []


@pytest.mark.req("REQ-OPS-110")
def test_the_lint_exempts_the_products_own_name_as_the_producer_does() -> None:
    script = f"The Seamless Shower Mat stops slips. It grips. {CTA}"
    record = _record(
        kind="product",
        title="Seamless Shower Mat",
        keyword="shower mat",
        script=script,
    )
    assert check_one(record, (5, 30))["lint"] is None
    other = _record(kind="product", title="Bath mat", keyword="mat", script=script)
    assert "Seamless" in (check_one(other, (5, 30))["lint"] or "")


def test_a_title_with_a_pipe_keeps_the_table_whole() -> None:
    checks = run_checks([_record(title="Mount | Magnetic\r\n| 2 pack")], (5, 30))
    row = next(
        line for line in render_report(None, checks).splitlines() if "Mount" in line
    )
    assert "Mount \\| Magnetic \\| 2 pack" in row
    assert row.count(" | ") == 5  # six cells


@pytest.mark.req("REQ-VID-161")
def test_an_explainer_is_held_to_the_explainer_band() -> None:
    checked = check_one(_record(steps=2, explainer=True), (5, 30))
    assert checked["band"] == [100, 160]


@pytest.mark.req("REQ-VID-161")
def test_a_sample_records_whether_the_list_was_an_explainer(tmp_path: Path) -> None:
    from src.utils.outputs_paths import with_outputs_root

    config = load_video_config_modular(cli_overrides=with_outputs_root({}, tmp_path))
    product = build_topic_product(
        TopicSpec(title="Why wifi drops at night", description="d")
    )
    with patch(
        "src.video.producer.steps.step_generate_script",
        _ctx_step(f"It changes channel. {CTA}", "x", steps=2, explainer=True),
    ):
        record = asyncio.run(sample_one(config, product, "slideshow_stock", {}, None))

    assert record["explainer"] is True and record["steps"] == 2
