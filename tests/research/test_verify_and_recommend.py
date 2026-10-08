"""Verification and config recommendations of the content research."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from src.research import verify as verify_mod
from src.research.__main__ import main
from src.research.recommend import (
    demand_recommendations,
    variant_recommendations,
    wrong_or_outdated,
)
from src.research.report import render_report
from src.research.verify import parse_verdicts, run_verify, tally

APPLE = "https://support.apple.com/x"


@pytest.mark.req("REQ-OPS-111")
def test_a_verdict_without_a_source_is_unverified() -> None:
    answer = "Here you go:\n" + json.dumps(
        [
            {"claim": "Tap General", "verdict": "correct", "source": APPLE},
            {"claim": "Tap Storage", "verdict": "wrong", "source": ""},
            {
                "claim": "Open X",
                "verdict": "Outdated",
                "source": APPLE,
                "correction": "Open Y",
            },
            {"claim": "Tap Z", "verdict": "maybe", "source": APPLE},
            {"verdict": "wrong"},
        ]
    )

    verdicts = parse_verdicts(answer)

    assert verdicts is not None
    assert [v["verdict"] for v in verdicts] == [
        "correct",
        "unverified",
        "outdated",
        "unverified",
    ]
    assert verdicts[2]["correction"] == "Open Y"
    assert tally(verdicts) == {"correct": 1, "wrong": 0, "outdated": 1, "unverified": 2}


@pytest.mark.parametrize("answer", [None, "", "no json", "[not json", '{"a": 1}'])
def test_an_unreadable_answer_is_none(answer) -> None:
    assert parse_verdicts(answer) is None


@pytest.mark.req("REQ-OPS-111")
def test_only_topic_scripts_are_verified(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_verify(title, script, **kwargs):
        calls.append(title)
        if title == "broken":
            return {"error": "verification call failed: 503"}
        return {
            "verdicts": [
                {
                    "claim": "c",
                    "verdict": "wrong",
                    "source": APPLE,
                    "quote": "",
                    "correction": "",
                }
            ]
        }

    monkeypatch.setattr(verify_mod, "verify_script", fake_verify)
    records = [
        {
            "variant": "shipped",
            "kind": "topic",
            "id": "t1",
            "title": "ok",
            "script": "s",
        },
        {
            "variant": "shipped",
            "kind": "topic",
            "id": "t2",
            "title": "broken",
            "script": "s",
        },
        {
            "variant": "shipped",
            "kind": "topic",
            "id": "t3",
            "title": "dropped",
            "error": "x",
        },
        {
            "variant": "shipped",
            "kind": "product",
            "id": "p",
            "title": "prod",
            "script": "s",
        },
    ]

    out = asyncio.run(run_verify(records, api_key="k", model="m", timeout=1))

    assert calls == ["ok", "broken"]
    assert out[0]["tally"]["wrong"] == 1 and "tally" not in out[1]


def _summary(in_band=0.5, misfit=0.5, errors=0):
    return {"in_band": in_band, "template_misfit": misfit, "errors": errors}


def _verified(variant, bad_flags):
    return [
        {
            "variant": variant,
            "tally": {"correct": 1, "wrong": int(b), "outdated": 0, "unverified": 0},
        }
        for b in bad_flags
    ]


@pytest.mark.req("REQ-OPS-112")
def test_the_share_of_scripts_with_errors_not_the_total_decides() -> None:
    # Two of four shipped scripts are wrong; one of two step-list scripts is
    # wrong. Same share, so no recommendation, though the total is lower.
    verification = _verified("shipped", [1, 1, 0, 0]) + _verified("step_lists", [1, 0])
    summary = {"shipped/topic": _summary(), "step_lists/topic": _summary(errors=2)}

    assert wrong_or_outdated(verification, "shipped") == 0.5
    (rec,) = variant_recommendations(summary, verification, all_tasks=True)
    assert rec["decision"] == "keep" and rec["evidence"]["dropped"] == [0, 2]


@pytest.mark.req("REQ-OPS-112")
@pytest.mark.parametrize(
    ("variant_stats", "decision"),
    [
        (_summary(), "recommend"),
        (_summary(in_band=0.25), "keep"),
        (_summary(misfit=0.75), "keep"),
    ],
)
def test_a_variant_must_not_worsen_length_or_fit(variant_stats, decision) -> None:
    verification = _verified("shipped", [1, 1]) + _verified("step_lists", [0, 0])
    summary = {"shipped/topic": _summary(), "step_lists/topic": variant_stats}

    (rec,) = variant_recommendations(summary, verification, all_tasks=True)

    assert rec["decision"] == decision
    assert rec["key"] == "llm_settings.topic_scripts.step_list.enabled"
    assert rec["value"] is True


def test_answer_first_maps_to_a_key_only_when_every_topic_is_a_task() -> None:
    verification = _verified("shipped", [1]) + _verified("task_answer_first", [0])
    summary = {"shipped/topic": _summary(), "task_answer_first/topic": _summary()}

    (all_tasks,) = variant_recommendations(summary, verification, all_tasks=True)
    (mixed,) = variant_recommendations(summary, verification, all_tasks=False)

    assert all_tasks["key"] == "llm_settings.script_templates.topic_templates"
    assert all_tasks["value"] == ["topic_answer_first"]
    assert mixed["key"] == "llm_settings.script_templates.topic_routing.enabled"
    assert mixed["value"] is True


def test_no_verification_leaves_the_decision_open() -> None:
    summary = {"shipped/topic": _summary(), "step_lists/topic": _summary()}
    (rec,) = variant_recommendations(summary, [], all_tasks=True)
    assert rec["decision"] == "undecided"


@pytest.mark.req("REQ-OPS-112")
def test_demand_becomes_keyword_and_topic_suggestions() -> None:
    demand = {
        "products": {
            "drop_candidates": ["sunset lamp"],
            "add_candidates": [
                {"query": "ai glasses", "growth": 900.0},
                {"query": "smart ring", "growth": 1200.0},
            ],
        },
        "topics": {"uncovered": [{"suggestion": "how to turn off ai on google"}]},
    }

    recs = demand_recommendations(demand)

    assert [(r["key"], r["change"]) for r in recs] == [
        ("batch.keywords", "drop"),
        ("batch.keywords", "add"),
        ("topics", "add"),
    ]
    assert recs[1]["items"] == ["smart ring", "ai glasses"]  # by growth


def test_the_report_leads_with_the_recommendations(tmp_path: Path) -> None:
    checks = {
        "summary": {
            "shipped/topic": {
                **_summary(),
                "samples": 1,
                "median_words": 80,
                "phrase_in_first_sentence": 1.0,
                "cta_last": 1.0,
                "with_lint_tell": 0.0,
                "flagged_claims": 0,
                "rewritten": 0,
                "repeated_openings": {},
            }
        },
        "samples": [],
    }
    verification = [
        {
            "variant": "shipped",
            "id": "t",
            "title": "How to x",
            "verdicts": [
                {
                    "claim": "Tap Storage",
                    "verdict": "outdated",
                    "source": APPLE,
                    "quote": "",
                    "correction": "Tap Storage & cache",
                }
            ],
            "tally": {"correct": 0, "wrong": 0, "outdated": 1, "unverified": 0},
        },
    ]
    text = render_report(None, checks, verification, [])

    assert text.index("## Recommended changes") < text.index("## Script samples")
    assert "Nothing here has been applied" in text
    assert '"Tap Storage" is outdated. Tap Storage & cache' in text


def test_verify_needs_samples_and_a_key(tmp_path: Path, monkeypatch, capsys) -> None:
    assert main(["verify", "--out", str(tmp_path)]) == 2
    assert "run the sample stage first" in capsys.readouterr().err

    (tmp_path / "samples.json").write_text("[]")
    from src.video.config import load_video_config_modular

    monkeypatch.delenv(
        load_video_config_modular().llm_settings.api_key_env_var, raising=False
    )
    monkeypatch.setattr("src.research.__main__.load_dotenv", lambda *a, **k: None)
    assert main(["verify", "--out", str(tmp_path)]) == 2
    assert "is not set" in capsys.readouterr().err


@pytest.mark.parametrize(
    "answer",
    [
        'Here is [the] list: [{"claim": "Tap A", "verdict": "correct", "source": "'
        + APPLE
        + '"}]',
        '[{"claim": "Tap A", "verdict": "correct", "source": "'
        + APPLE
        + '"}]\nSources: [1] apple',
        '```json\n[{"claim": "Tap [A]", "verdict": "correct", "source": "'
        + APPLE
        + '"}]\n```',
        # A valid array that is not the verdicts comes first.
        'Checked [1, 2] sources: [{"claim": "Tap A", "verdict": "correct", '
        '"source": "' + APPLE + '"}]',
    ],
)
def test_bracketed_prose_around_the_json_is_skipped(answer) -> None:
    verdicts = parse_verdicts(answer)
    assert verdicts and verdicts[0]["verdict"] == "correct"


@pytest.mark.req("REQ-OPS-112")
def test_a_script_with_only_unverified_verdicts_is_not_clean() -> None:
    unjudged_rows = [
        {
            "variant": "step_lists",
            "tally": {"correct": 0, "wrong": 0, "outdated": 0, "unverified": 3},
        }
    ]
    verification = _verified("shipped", [1, 0]) + unjudged_rows
    summary = {"shipped/topic": _summary(), "step_lists/topic": _summary()}

    assert wrong_or_outdated(verification, "step_lists") is None
    (rec,) = variant_recommendations(summary, verification, all_tasks=True)
    assert rec["decision"] == "undecided"
    assert rec["evidence"]["unjudged"] == [0, 1]


def test_the_none_placeholder_shows_only_when_nothing_is_recommended() -> None:
    from src.research.report import render_recommendations

    one = {
        "products": {"drop_candidates": ["foo"], "add_candidates": []},
        "topics": {"uncovered": []},
    }
    empty: dict[str, dict[str, list[str]]] = {
        "products": {"drop_candidates": [], "add_candidates": []},
        "topics": {"uncovered": []},
    }

    assert "- None:" not in render_recommendations(one, None, None, None)
    assert "- None:" in render_recommendations(empty, None, None, None)
