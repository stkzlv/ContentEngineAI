"""The demand stage of the content research (design 0023)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.research import demand as demand_mod
from src.research.__main__ import main, scraper_keywords
from src.research.config import load_research_config
from src.research.demand import (
    drop_candidates,
    measure,
    new_keywords,
    peak_month,
    recent_ratio,
    run_demand,
    topic_term,
    uncovered,
)
from src.research.report import render_report


def weekly(level: float, weeks: int = 52, tail: float | None = None):
    """A 12-month weekly series at `level`, its last quarter at `tail`."""
    days = [f"2025-{1 + (i * 12) // weeks:02d}-01" for i in range(weeks)]
    values = [level] * weeks
    if tail is not None:
        values[-13:] = [tail] * 13
    return list(zip(days, values, strict=True))


class FakeTrends:
    """Interest per term as a level; terms in `failing` fail their request."""

    def __init__(self, levels, failing=(), rising=None, peaks=None):
        self.levels, self.failing = levels, set(failing)
        self.rising_by_seed = rising or {}
        self.peaks = peaks or {}
        self.calls: list[tuple[list[str], str, str]] = []

    def interest(self, terms, geo, timeframe):
        self.calls.append((terms, geo, timeframe))
        if self.failing & set(terms):
            return None
        out = {}
        for t in terms:
            level = self.levels.get((t, geo), self.levels.get(t, 0.0))
            if timeframe == demand_mod.FIVE_YEARS:
                peak = self.peaks.get(t, 12)
                out[t] = [
                    (f"2024-{m:02d}-01", level * (2 if m == peak else 1))
                    for m in range(1, 13)
                ]
            else:
                tail = self.levels.get((t, geo, "tail"), self.levels.get((t, "tail")))
                out[t] = weekly(level, tail=tail)
        return out

    def rising(self, term, geo, timeframe):
        if term in self.failing:
            return None
        return self.rising_by_seed.get(term, [])


@pytest.mark.req("REQ-OPS-107")
def test_shares_are_measured_against_the_anchor_in_every_request() -> None:
    source = FakeTrends({"anchor": 10.0, **{f"k{i}": float(i) for i in range(6)}})

    found, missing = measure(source, [f"k{i}" for i in range(6)], "anchor", "US")

    # Six terms take two requests of four plus the anchor, each with the anchor.
    year_calls = [c for c in source.calls if c[2] == demand_mod.YEAR]
    assert len(year_calls) == 2 and all(c[0][0] == "anchor" for c in year_calls)
    k5, k1, k3 = found["k5"], found["k1"], found["k3"]
    assert k5 and k1 and k3
    assert k5["share"] == 0.5 and k1["share"] == 0.1
    assert k3["peak_month"] == 12
    assert missing == []


@pytest.mark.req("REQ-OPS-107")
def test_a_failed_request_is_missing_not_zero() -> None:
    source = FakeTrends({"anchor": 10.0, "ok": 5.0, "bad": 5.0}, failing={"bad"})

    found, missing = measure(source, ["bad"], "anchor", "GB")

    assert found["bad"] is None
    assert missing == ["Trends GB: bad"]


def test_peak_month_and_recent_ratio() -> None:
    series = [(f"2024-{m:02d}-01", 9.0 if m == 9 else 3.0) for m in range(1, 13)]
    assert peak_month(series) == 9
    assert peak_month([("2024-01-01", 0.0)]) is None
    assert (recent_ratio(weekly(10.0, tail=20.0)) or 0.0) > 1.0
    assert recent_ratio(weekly(10.0)) == pytest.approx(1.0)
    assert recent_ratio([]) is None


@pytest.mark.req("REQ-OPS-108")
def test_drop_candidates_are_low_everywhere_and_not_rising() -> None:
    def r(share, ratio=1.0):
        return {"share": share, "recent_ratio": ratio, "peak_month": None}

    # Most keywords sit near the median, as real pools do.
    typical = {f"typical {i}": {"US": r(1.0), "GB": r(1.0)} for i in range(4)}
    shares = {
        **typical,
        "strong": {"US": r(1.0), "GB": r(1.0)},
        "middle": {"US": r(0.8), "GB": r(0.8)},
        "low": {"US": r(0.1), "GB": r(0.1)},
        "low but rising": {"US": r(0.1, 1.5), "GB": r(0.1)},
        "low in one country": {"US": r(0.1), "GB": r(0.9)},
        "unread in GB": {"US": r(0.1), "GB": None},
    }

    assert drop_candidates(shares, 0.25) == ["low"]


@pytest.mark.req("REQ-OPS-108")
def test_add_candidates_skip_what_is_already_a_keyword() -> None:
    rising = [
        ("tech gadgets", "US", "Smart Plug", 300.0),
        ("tech gadgets", "US", "smart plug mini", 250.0),  # contains a keyword
        ("tech gadgets", "GB", "ai glasses", 900.0),
        ("phone accessories", "US", "AI glasses", 400.0),  # seen already
    ]

    added = new_keywords(rising, ["smart plug", "USB C hub"])

    assert [a["query"] for a in added] == ["ai glasses"]


@pytest.mark.req("REQ-OPS-108")
def test_uncovered_suggestions_are_those_no_pool_title_covers() -> None:
    suggested = {
        "how to stop": {
            "US": ["how to stop spam calls on iphone", "how to stop snoring"]
        },
        "iphone how to": {"GB": None},
    }
    pool = ["How to stop spam calls on iPhone"]

    assert [u["suggestion"] for u in uncovered(suggested, pool)] == [
        "how to stop snoring"
    ]


def test_topic_terms_come_from_the_search_field_or_the_title() -> None:
    assert topic_term("How to free up iPhone storage", "iphone storage") == (
        "iphone storage"
    )
    assert topic_term("How to clear app cache on Android") == (
        "clear app cache on android"
    )


@pytest.mark.req("REQ-OPS-107", "REQ-OPS-108")
def test_run_demand_records_both_sides_and_renders(monkeypatch) -> None:
    config = load_research_config().model_copy(update={"countries": ["US"]})
    config.products.seeds = ["tech gadgets"]
    config.topics.suggest_stems = ["how to stop"]
    source = FakeTrends(
        {
            config.products.anchor: 10.0,
            config.topics.anchor: 10.0,
            "smart plug": 9.0,
            "sunset lamp": 0.5,
            "clear app cache on android": 4.0,
        },
        rising={"tech gadgets": [("ai glasses", 500.0)]},
    )

    async def fake_all(stems, countries):
        return {"how to stop": {"US": ["how to stop snoring"]}}

    monkeypatch.setattr(demand_mod, "_all_suggestions", fake_all)
    record = run_demand(
        config,
        source,
        ["smart plug", "sunset lamp"],
        [("How to clear app cache on Android", "")],
    )

    assert record["products"]["drop_candidates"] == ["sunset lamp"]
    assert record["products"]["add_candidates"][0]["query"] == "ai glasses"
    assert record["topics"]["terms"]["How to clear app cache on Android"]["US"][
        "share"
    ] == pytest.approx(0.4)
    report = render_report(json.loads(json.dumps(record)))
    assert "| smart plug | 0.90 |" in report
    assert "sunset lamp" in report and "how to stop snoring (US)" in report


def test_the_report_names_missing_data(tmp_path: Path) -> None:
    record = {
        "date": "2026-10-06",
        "countries": ["US"],
        "products": {
            "anchor": "a",
            "terms": {"k": {"US": None}},
            "drop_candidates": [],
            "add_candidates": [],
        },
        "topics": {"anchor": "b", "terms": {}, "suggestions": {}, "uncovered": []},
        "missing": ["Trends US: k"],
    }
    (tmp_path / "demand.json").write_text(json.dumps(record))

    assert main(["report", "--out", str(tmp_path)]) == 0

    text = (tmp_path / "report.md").read_text()
    assert "| k | no data |" in text and "- Trends US: k" in text


def test_without_pytrends_the_demand_stage_says_so(tmp_path: Path, capsys) -> None:
    from src.research.sources import TrendsUnavailableError

    with patch(
        "src.research.__main__.PytrendsSource",
        side_effect=TrendsUnavailableError("no module"),
    ):
        assert main(["demand", "--out", str(tmp_path)]) == 2
    assert "poetry install --with research" in capsys.readouterr().err


def test_the_bundled_config_and_keywords_load() -> None:
    config = load_research_config()
    assert config.products.anchor and config.topics.anchor
    assert len(scraper_keywords()) > 10


def test_a_topic_entry_may_carry_its_search(tmp_path: Path) -> None:
    from src.video.producer.topic_input import TopicInputError, load_topics_file

    pool = tmp_path / "topics.yaml"
    pool.write_text('- title: "How to x on y"\n  search: "x y"\n')
    assert load_topics_file(pool)[0].search == "x y"
    pool.write_text('- title: "How to x on y"\n  search: ["x"]\n')
    with pytest.raises(TopicInputError, match="'search' must be a string"):
        load_topics_file(pool)


@pytest.mark.req("REQ-OPS-107")
def test_a_failed_rising_request_is_missing_not_no_candidates(monkeypatch) -> None:
    config = load_research_config().model_copy(update={"countries": ["US"]})
    config.products.seeds = ["tech gadgets"]
    config.topics.suggest_stems = []
    source = FakeTrends({config.products.anchor: 10.0}, failing={"tech gadgets"})

    async def no_suggestions(stems, countries):
        return {}

    monkeypatch.setattr(demand_mod, "_all_suggestions", no_suggestions)
    record = run_demand(config, source, [], [])

    assert "Trends US: rising searches for tech gadgets" in record["missing"]
