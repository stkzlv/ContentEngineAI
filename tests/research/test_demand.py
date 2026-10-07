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
    measured_candidates,
    new_keywords,
    peak_month,
    run_demand,
    strongest,
    topic_term,
    uncovered,
    yoy_ratio,
)
from src.research.report import render_report


def weekly(level: float, weeks: int = 52, tail: float | None = None):
    """A 12-month weekly series at `level`, its last quarter at `tail`."""
    days = [f"2025-{1 + (i * 12) // weeks:02d}-01" for i in range(weeks)]
    values = [level] * weeks
    if tail is not None:
        values[-13:] = [tail] * 13
    return list(zip(days, values, strict=True))


def five_years(level: float, peak: int = 12, tail: float | None = None):
    """Four complete years, four points a month, the `peak` month doubled each
    year; the last 13 points at `tail` for a year-on-year change.
    """
    series = [
        (f"{y}-{m:02d}-{d:02d}", level * (2 if m == peak else 1))
        for y in (2022, 2023, 2024, 2025)
        for m in range(1, 13)
        for d in (1, 8, 15, 22)
    ]
    if tail is not None:
        series[-13:] = [(day, tail) for day, _ in series[-13:]]
    return series


class FakeTrends:
    """Interest per term as a level; terms in `failing` fail their request."""

    def __init__(self, levels, failing=(), rising=None, peaks=None, failing_rising=()):
        self.levels, self.failing = levels, set(failing)
        self.rising_by_seed = rising or {}
        self.peaks = peaks or {}
        self.failing_rising = set(failing_rising)
        self.calls: list[tuple[list[str], str, str]] = []

    def interest(self, terms, geo, timeframe):
        self.calls.append((terms, geo, timeframe))
        if self.failing & set(terms):
            return None
        out = {}
        for t in terms:
            level = self.levels.get((t, geo), self.levels.get(t, 0.0))
            tail = self.levels.get((t, geo, "tail"), self.levels.get((t, "tail")))
            if timeframe == demand_mod.FIVE_YEARS:
                out[t] = five_years(level, self.peaks.get(t, 12), tail)
            else:
                out[t] = weekly(level)
        return out

    def rising(self, term, geo, timeframe):
        if term in self.failing or term in self.failing_rising:
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


def test_a_peak_must_recur_and_one_spike_cannot_make_one() -> None:
    assert peak_month(five_years(3.0, peak=11)) == 11
    # One April spike in one year, as Trends showed across unrelated searches.
    spiked = [
        (day, v * 20 if day.startswith("2025-04") else v)
        for day, v in five_years(3.0, peak=11)
    ]
    assert peak_month(spiked) == 11
    flat_with_spike = [
        (day, 50.0 if day.startswith("2025-04") else 3.0) for day, _ in five_years(3.0)
    ]
    assert peak_month(flat_with_spike) is None
    # A partial year does not vote.
    assert peak_month([("2026-04-01", 9.0), ("2026-05-01", 1.0)]) is None


@pytest.mark.req("REQ-OPS-108")
def test_the_trend_is_year_on_year() -> None:
    assert yoy_ratio(five_years(10.0, tail=20.0)) == 2.0
    assert yoy_ratio(five_years(10.0)) == 1.0
    assert yoy_ratio(five_years(10.0)[:40]) is None  # under a year and a quarter
    assert yoy_ratio(five_years(0.0, tail=5.0)) is None  # nothing a year earlier


@pytest.mark.req("REQ-OPS-108")
def test_drop_candidates_are_low_everywhere_and_not_rising() -> None:
    def r(share, ratio=1.0):
        return {"share": share, "trend": ratio, "peak_month": None}

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
    config.products.related_from = 1
    config.topics.suggest_stems = ["how to stop"]
    source = FakeTrends(
        {
            config.products.anchor: 10.0,
            config.topics.anchor: 10.0,
            "smart plug": 9.0,
            "sunset lamp": 0.5,
            "clear app cache on android": 4.0,
            "ai glasses": 8.0,
            "tech news today": 0.1,
        },
        # Rising next to the strongest keyword; one is a product, one news.
        rising={"smart plug": [("ai glasses", 500.0), ("tech news today", 900.0)]},
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
    # Measured against the anchor: only the one near today's keywords stays.
    assert [a["query"] for a in record["products"]["add_candidates"]] == ["ai glasses"]
    assert record["products"]["add_candidates"][0]["seed"] == "smart plug"
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
    config.products.related_from = 1
    config.topics.suggest_stems = []
    source = FakeTrends(
        {config.products.anchor: 10.0, "smart plug": 5.0},
        failing_rising={"smart plug"},
    )

    async def no_suggestions(stems, countries):
        return {}

    monkeypatch.setattr(demand_mod, "_all_suggestions", no_suggestions)
    record = run_demand(config, source, ["smart plug"], [])

    assert "Trends US: rising searches for smart plug" in record["missing"]


@pytest.mark.req("REQ-OPS-107")
def test_a_failed_five_year_request_is_missing() -> None:
    class YearOnly(FakeTrends):
        def interest(self, terms, geo, timeframe):
            if timeframe == demand_mod.FIVE_YEARS:
                return None
            return super().interest(terms, geo, timeframe)

    found, missing = measure(
        YearOnly({"anchor": 10.0, "a": 5.0}), ["a"], "anchor", "US"
    )

    reading = found["a"]
    assert reading and reading["share"] == 0.5 and reading["peak_month"] is None
    assert missing == ["Trends US (5 years): a"]


def test_the_anchor_as_a_term_reads_one() -> None:
    found, missing = measure(FakeTrends({"anchor": 10.0}), ["anchor"], "anchor", "US")

    reading = found["anchor"]
    assert reading and reading["share"] == 1.0 and missing == []


@pytest.mark.req("REQ-OPS-108")
def test_one_rise_threshold_serves_the_report_and_the_drop_rule() -> None:
    from src.research.report import _cell

    def r(share, ratio):
        return {"share": share, "trend": ratio, "peak_month": None}

    typical = {f"typical {i}": {"US": r(1.0, 1.0)} for i in range(5)}
    shares = {**typical, "slight rise": {"US": r(0.05, 1.05)}}

    # 1.05 is not a rise: unlabelled in the table, so still dropped.
    assert drop_candidates(shares, 0.25) == ["slight rise"]
    assert _cell(r(0.05, 1.05)) == "0.05"
    assert _cell(r(0.05, 1.15)) == "0.05 rising"
    assert _cell(r(0.05, 0.92)) == "0.05"  # above 1/1.1, so no label
    assert _cell(r(0.05, 0.85)) == "0.05 falling"


@pytest.mark.req("REQ-OPS-108")
def test_a_keyword_with_no_trend_anywhere_is_not_judged() -> None:
    def r(share, trend):
        return {"share": share, "trend": trend, "peak_month": None}

    typical = {f"typical {i}": {"US": r(1.0, 1.0)} for i in range(5)}
    shares = {**typical, "low, trend unknown": {"US": r(0.05, None)}}

    assert drop_candidates(shares, 0.25) == []


def test_the_strongest_keywords_lead_by_their_best_country() -> None:
    def r(share):
        return {"share": share, "trend": None, "peak_month": None}

    products = {
        "a": {"US": r(0.2), "GB": r(2.0)},
        "b": {"US": r(1.0), "GB": r(0.1)},
        "c": {"US": None, "GB": None},
    }
    assert strongest(products, ["US", "GB"], 5) == ["a", "b"]


def test_a_candidate_must_reach_the_median_in_some_country() -> None:
    candidates = [{"query": "x"}, {"query": "y"}, {"query": "z"}]
    readings = {
        "x": {"US": {"share": 0.6}, "GB": {"share": 0.1}},
        "y": {"US": {"share": 0.1}, "GB": {"share": 0.1}},
        "z": {"US": None, "GB": None},
    }

    kept = measured_candidates(candidates, readings, {"US": 0.5, "GB": 0.4})

    assert [k["query"] for k in kept] == ["x"]
    assert kept[0]["shares"] == {"US": 0.6, "GB": 0.1}


@pytest.mark.req("REQ-OPS-108")
def test_uncovered_searches_alternate_between_stems() -> None:
    suggested = {
        "how to turn off": {"US": ["how to turn off a", "how to turn off b"]},
        "samsung how to": {"US": ["samsung how to c"]},
        "gmail how to": {"US": ["gmail how to d", "gmail how to e"]},
    }

    order = [u["suggestion"] for u in uncovered(suggested, [])]

    assert order == [
        "how to turn off a",
        "samsung how to c",
        "gmail how to d",
        "how to turn off b",
        "gmail how to e",
    ]


@pytest.mark.req("REQ-OPS-107")
def test_a_spike_week_does_not_inflate_a_share() -> None:
    class Spiky(FakeTrends):
        def interest(self, terms, geo, timeframe):
            out = super().interest(terms, geo, timeframe)
            if out and timeframe == demand_mod.YEAR and "spiky" in out:
                out["spiky"][10] = (out["spiky"][10][0], 500.0)
            return out

    found, _ = measure(Spiky({"anchor": 10.0, "spiky": 5.0}), ["spiky"], "anchor", "US")

    reading = found["spiky"]
    assert reading and reading["share"] == 0.5


@pytest.mark.req("REQ-OPS-108")
def test_a_seasonal_high_is_not_a_rise() -> None:
    # High every October to December: this autumn equals last autumn.
    holiday = [
        (day, 30.0 if int(day[5:7]) >= 10 else 10.0) for day, _ in five_years(10.0)
    ]
    assert yoy_ratio(holiday) == 1.0


@pytest.mark.req("REQ-OPS-108")
def test_a_keyword_with_no_interest_then_or_now_is_dropped() -> None:
    assert yoy_ratio(five_years(0.0)) == 1.0
    source = FakeTrends(
        {"anchor": 10.0, **{f"typical {i}": 5.0 for i in range(5)}, "dead": 0.0}
    )
    found, missing = measure(
        source, [*(f"typical {i}" for i in range(5)), "dead"], "anchor", "US"
    )
    shares = {k: {"US": v} for k, v in found.items()}

    assert drop_candidates(shares, 0.25) == ["dead"]
    assert missing == []


def test_sparse_noise_gets_no_peak_month() -> None:
    import random

    rng = random.Random(7)  # noqa: S311 - a seeded fixture, not security
    hits = 0
    for _ in range(200):
        sparse = [
            (day, 1.0 if rng.random() < 0.1 else 0.0) for day, _ in five_years(0.0)
        ]
        hits += peak_month(sparse) is not None
    assert hits == 0


@pytest.mark.req("REQ-OPS-108")
def test_add_candidates_are_capped_and_measured_for_share_only(monkeypatch) -> None:
    config = load_research_config().model_copy(update={"countries": ["US"]})
    config.products.related_from = 1
    config.topics.suggest_stems = []
    rising = [(f"gadget {i}", float(1000 - i)) for i in range(30)]
    levels = {config.products.anchor: 10.0, "smart plug": 9.0}
    levels.update({f"gadget {i}": 9.0 for i in range(30)})
    source = FakeTrends(levels, rising={"smart plug": rising})

    async def none(stems, countries):
        return {}

    monkeypatch.setattr(demand_mod, "_all_suggestions", none)
    record = run_demand(config, source, ["smart plug"], [])

    kept = [a["query"] for a in record["products"]["add_candidates"]]
    assert len(kept) == demand_mod.MAX_CANDIDATES and kept[0] == "gadget 0"
    measured = [c for c in source.calls if any(t.startswith("gadget") for t in c[0])]
    assert measured and all(c[2] == demand_mod.YEAR for c in measured)
