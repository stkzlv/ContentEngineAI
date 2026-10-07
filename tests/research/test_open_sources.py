"""Wikipedia pageviews and Stack Exchange questions in the demand stage."""

from __future__ import annotations

import asyncio
import json
from datetime import date
from typing import Any

import pytest

from src.research import demand as demand_mod
from src.research import sources
from src.research.config import QuestionSource, StackExchangeResearch
from src.research.demand import last_months, ranked_questions, views_reading
from src.research.report import render_report


class _Response:
    def __init__(self, status: int, body: Any) -> None:
        self.status, self._body = status, body

    async def text(self) -> str:
        return self._body if isinstance(self._body, str) else json.dumps(self._body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        return None


class _Session:
    """Answers by URL substring, in order; records every request."""

    def __init__(self, routes: list[tuple[str, int, Any]]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def get(self, url, params=None, headers=None, timeout=None) -> _Response:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        for i, (part, status, body) in enumerate(self.routes):
            if part in url or part in json.dumps(params or {}):
                self.routes.pop(i)
                return _Response(status, body)
        return _Response(404, "")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        return None


def monthly(values: list[float], first_year: int = 2024) -> dict[str, Any]:
    items = []
    for n, v in enumerate(values):
        year, month = first_year + n // 12, n % 12 + 1
        items.append({"timestamp": f"{year}{month:02d}0100", "views": v})
    return {"items": items}


@pytest.mark.req("REQ-OPS-113")
def test_redirects_are_followed_and_a_missing_title_has_no_article() -> None:
    session = _Session(
        [
            (
                "api.php",
                200,
                {
                    "query": {
                        "normalized": [{"from": "smart plug", "to": "Smart plug"}],
                        "redirects": [{"from": "Power bank", "to": "Battery pack"}],
                        "pages": [
                            {"title": "Smart plug"},
                            {"title": "Battery pack"},
                            {"title": "Nope", "missing": True},
                        ],
                    }
                },
            )
        ]
    )

    found = asyncio.run(
        sources.resolve_articles(session, ["smart plug", "Power bank", "Nope"])
    )

    assert found == {
        "smart plug": "Smart plug",
        "Power bank": "Battery pack",
        "Nope": None,
    }
    # Wikimedia blocks a client that does not name itself.
    assert "github.com" in session.calls[0][2]["User-Agent"]


@pytest.mark.req("REQ-OPS-113")
def test_a_failed_lookup_or_pageview_request_is_missing() -> None:
    assert asyncio.run(sources.resolve_articles(_Session([]), ["x"])) is None
    assert asyncio.run(sources.pageviews(_Session([]), "X", "a", "b")) is None
    bad = _Session([("pageviews", 200, {"items": [{"views": 1}]})])
    assert asyncio.run(sources.pageviews(bad, "X", "a", "b")) is None


@pytest.mark.req("REQ-OPS-113")
def test_the_article_trend_is_relative_to_all_of_wikipedia() -> None:
    # Flat at 100, then the last three months at 80: down 20% on its own.
    article = [("m", 100.0)] * 12 + [("m", 100.0)] * 9 + [("m", 80.0)] * 3
    falling_site = 0.8

    reading = views_reading(article, falling_site)

    assert reading == {"views": 100, "trend": 1.0}
    assert views_reading(article, None)["trend"] is None


def test_the_window_is_complete_months_before_today() -> None:
    assert last_months(date(2026, 10, 7), 24) == ("20241001", "20260930")
    assert last_months(date(2026, 1, 15), 12) == ("20250101", "20251231")


@pytest.mark.req("REQ-OPS-114")
def test_questions_page_until_done_and_honour_backoff() -> None:
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)

    session = _Session(
        [
            (
                '"page": 1',
                200,
                {"items": [{"title": "a"}], "has_more": True, "backoff": 5},
            ),
            ('"page": 2', 200, {"items": [{"title": "b"}], "has_more": False}),
        ]
    )

    found = asyncio.run(
        sources.questions(session, "apple", "iphone", 0, pages=3, sleep=sleep)
    )

    assert [q["title"] for q in found or []] == ["a", "b"]
    assert slept == [5.0]
    assert session.calls[0][1]["tagged"] == "iphone"
    assert len(session.calls) == 2


@pytest.mark.req("REQ-OPS-114")
def test_a_failed_first_page_is_missing_and_a_later_one_ends_the_list() -> None:
    assert asyncio.run(sources.questions(_Session([]), "apple", "", 0, 2)) is None
    session = _Session([('"page": 1', 200, {"items": [{"t": 1}], "has_more": True})])
    assert asyncio.run(sources.questions(session, "apple", "", 0, 2)) == [{"t": 1}]
    assert "tagged" not in session.calls[0][1]


@pytest.mark.req("REQ-OPS-114")
def test_questions_rank_by_views_per_day_since_asked() -> None:
    day = 86400
    now = 100 * day
    found = [
        # Older and more viewed in total, fewer a day.
        {"title": "old", "view_count": 900, "creation_date": now - 90 * day},
        {
            "title": "it&#39;s new",
            "view_count": 300,
            "creation_date": now - 3 * day,
            "link": "https://x/q/1",
            "tags": ["ios"],
            "is_answered": True,
        },
        {"title": "today", "view_count": 5, "creation_date": now},
        {"title": "broken"},
    ]

    ranked = ranked_questions(found, now, top=2)

    assert [q["title"] for q in ranked] == ["it's new", "old"]
    assert ranked[0]["per_day"] == 100.0 and ranked[0]["tags"] == ["ios"]


@pytest.mark.req("REQ-OPS-113")
@pytest.mark.req("REQ-OPS-114")
def test_open_data_records_both_and_names_what_failed(
    monkeypatch, real_open_data
) -> None:
    routes = [
        (
            "api.php",
            200,
            {
                "query": {
                    "redirects": [{"from": "Power bank", "to": "Battery pack"}],
                    "pages": [
                        {"title": "Battery pack"},
                        {"title": "Gone", "missing": True},
                    ],
                }
            },
        ),
        ("aggregate", 200, monthly([100.0] * 24)),
        ("Battery_pack", 200, monthly([10.0] * 21 + [20.0] * 3)),
        ('"site": "apple"', 200, {"items": [], "has_more": False}),
    ]
    monkeypatch.setattr(demand_mod, "WIKI_PAUSE_SEC", 0)
    monkeypatch.setattr(demand_mod.aiohttp, "ClientSession", lambda: _Session(routes))
    stack = StackExchangeResearch(
        sources=[QuestionSource(site="apple"), QuestionSource(site="android")]
    )

    views, asked, missing = asyncio.run(
        real_open_data(
            {"portable charger": "Power bank", "x": "Gone"}, stack, date(2026, 10, 7)
        )
    )

    assert views["portable charger"] == {
        "article": "Battery pack",
        "views": 10,
        "trend": 2.0,
    }
    assert views["x"] is None
    assert asked == [
        {"source": "apple", "questions": []},
        {"source": "android", "questions": None},
    ]
    assert missing == ["Wikipedia: no article 'Gone'", "Stack Exchange: android"]


@pytest.mark.req("REQ-OPS-113")
@pytest.mark.req("REQ-OPS-114")
def test_the_report_shows_views_questions_and_a_disagreeing_signal() -> None:
    record = {
        "date": "2026-10-07",
        "countries": ["US"],
        "products": {
            "anchor": "a",
            "terms": {"smart lock": {"US": None}},
            "drop_candidates": ["smart lock", "sunset lamp"],
            "add_candidates": [],
            "wikipedia": {
                "smart lock": {"article": "Smart lock", "views": 1249, "trend": 1.4},
                "smart plug": None,
            },
        },
        "topics": {
            "anchor": "b",
            "terms": {},
            "suggestions": {},
            "uncovered": [],
            "questions": [
                {
                    "source": "apple/iphone",
                    "questions": [
                        {
                            "title": "Stop | arrival texts",
                            "link": "https://apple.stackexchange.com/q/1",
                            "views": 1515,
                            "per_day": 141.5,
                            "tags": ["iphone", "maps"],
                            "answered": True,
                        }
                    ],
                },
                {"source": "android", "questions": None},
            ],
        },
        "missing": [],
    }

    report = render_report(record)

    assert "smart lock (Wikipedia views rising), sunset lamp." in report
    assert "| smart lock | Smart lock | 1,249 | rising |" in report
    assert "| smart plug | - | no data | - |" in report
    assert (
        "- [Stop \\| arrival texts](https://apple.stackexchange.com/q/1) "
        "(1,515 views, 141.5 a day; iphone, maps)" in report
    )
    assert "CC BY-SA" in report
    assert "*android*\n\n- no data" in report


def test_a_record_from_before_these_sources_still_renders() -> None:
    record = {
        "date": "2026-10-06",
        "countries": ["US"],
        "products": {
            "anchor": "a",
            "terms": {},
            "drop_candidates": ["k"],
            "add_candidates": [],
        },
        "topics": {"anchor": "b", "terms": {}, "suggestions": {}, "uncovered": []},
        "missing": [],
    }
    report = render_report(record)
    assert "Wikipedia" not in report and "Stack Exchange" not in report


@pytest.mark.req("REQ-OPS-113")
def test_only_keywords_still_scraped_are_looked_up(monkeypatch) -> None:
    from src.research.config import WikipediaArticle, load_research_config
    from tests.research.test_demand import FakeTrends

    config = load_research_config().model_copy(update={"countries": ["US"]})
    config.products.related_from = 0
    config.topics.suggest_stems = []
    config.products.wikipedia = [
        WikipediaArticle(keyword="smart plug", article="Smart plug"),
        WikipediaArticle(keyword="retired gadget", article="Gadget"),
    ]
    asked: list[dict[str, str]] = []

    async def fake(articles, stack, today):
        asked.append(articles)
        return (
            {"smart plug": {"article": "Smart plug", "views": 5, "trend": 1.0}},
            [],
            [],
        )

    monkeypatch.setattr(demand_mod, "_open_data", fake)
    record = demand_mod.run_demand(
        config, FakeTrends({config.products.anchor: 1.0}), ["smart plug"], []
    )

    assert asked == [{"smart plug": "Smart plug"}]
    assert record["products"]["wikipedia"]["smart plug"]["views"] == 5


class _BadBytes(_Response):
    async def text(self) -> str:
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")


@pytest.mark.req("REQ-OPS-113")
def test_an_unreadable_body_is_a_failed_request_not_a_crash() -> None:
    not_json = _Session([("pageviews", 200, "<html>captive portal</html>")])
    assert asyncio.run(sources.pageviews(not_json, "X", "a", "b")) is None

    class Session(_Session):
        def get(self, *args, **kwargs) -> _Response:
            return _BadBytes(200, "")

    assert asyncio.run(sources.pageviews(Session([]), "X", "a", "b")) is None


@pytest.mark.req("REQ-OPS-113")
def test_titles_are_looked_up_fifty_at_a_time() -> None:
    titles = [f"T{n}" for n in range(51)]
    pages = [{"title": t} for t in titles]
    session = _Session(
        [
            ("T0|", 200, {"query": {"pages": pages[:50]}}),
            ("T50", 200, {"query": {"pages": pages[50:]}}),
        ]
    )

    found = asyncio.run(sources.resolve_articles(session, titles))

    assert len(session.calls) == 2
    assert all(len(c[1]["titles"].split("|")) <= 50 for c in session.calls)
    assert found == {t: t for t in titles}


@pytest.mark.req("REQ-OPS-113")
def test_views_are_the_median_of_the_last_twelve_months() -> None:
    series = [("m", 1000.0)] * 12 + [("m", 10.0)] * 12
    assert views_reading(series, 1.0)["views"] == 10


@pytest.mark.req("REQ-OPS-113")
@pytest.mark.parametrize(
    ("routes", "expected"),
    [
        # The title lookup failed: every keyword is missing, one line says why.
        (
            [("aggregate", 200, monthly([5.0] * 24))],
            ["Wikipedia: article lookup"],
        ),
        # The site total failed: no trend for any article.
        (
            [
                ("api.php", 200, {"query": {"pages": [{"title": "A"}]}}),
                ("A/monthly", 200, monthly([5.0] * 24)),
            ],
            ["Wikipedia: total views, so no article trend"],
        ),
        # The article's views failed.
        (
            [
                ("api.php", 200, {"query": {"pages": [{"title": "A"}]}}),
                ("aggregate", 200, monthly([5.0] * 24)),
            ],
            ["Wikipedia: views of 'A'"],
        ),
    ],
)
def test_each_failed_wikipedia_request_is_missing(
    monkeypatch, real_open_data, routes, expected
) -> None:
    monkeypatch.setattr(demand_mod, "WIKI_PAUSE_SEC", 0)
    monkeypatch.setattr(demand_mod.aiohttp, "ClientSession", lambda: _Session(routes))

    views, _, missing = asyncio.run(
        real_open_data({"k": "A"}, StackExchangeResearch(), date(2026, 10, 7))
    )

    assert missing == expected
    if expected[0].startswith("Wikipedia: total"):
        assert views["k"]["trend"] is None and views["k"]["views"] == 5
    else:
        assert views["k"] is None


def test_a_question_title_is_literal_link_text() -> None:
    record = {
        "date": "2026-10-07",
        "countries": ["US"],
        "products": {
            "anchor": "a",
            "terms": {},
            "drop_candidates": ["k"],
            "add_candidates": [],
            # Up, but by less than a rise.
            "wikipedia": {"k": {"article": "K", "views": 1, "trend": 1.05}},
        },
        "topics": {
            "anchor": "b",
            "terms": {},
            "suggestions": {},
            "uncovered": [],
            "questions": [
                {
                    "source": "s",
                    "questions": [
                        {
                            "title": "Why [ and <video> and *#06#",
                            "link": "https://x/q",
                            "views": 1,
                            "per_day": 1.0,
                            "tags": [],
                            "answered": False,
                        }
                    ],
                }
            ],
        },
        "missing": [],
    }

    report = render_report(record)

    assert r"- [Why \[ and \<video\> and \*#06#](https://x/q)" in report
    # Up 5% is not a rise, so the drop candidate carries no mark.
    assert "Wikipedia views rising" not in report
