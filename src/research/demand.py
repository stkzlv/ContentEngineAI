"""The demand stage: relative search interest and suggestions (REQ-OPS-107).

The analysis is pure functions over what the sources returned, so a recorded
run can be re-analysed without a request.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
from collections.abc import Iterable
from datetime import date
from typing import Any

import aiohttp

from src.research.config import ResearchConfig
from src.research.sources import TERMS_PER_REQUEST, TrendsSource, suggestions
from src.scraper.base.keyword_pillars import normalize_keyword
from src.video.search_phrase import contains

logger = logging.getLogger(__name__)

YEAR = "today 12-m"
FIVE_YEARS = "today 5-y"
# Weekly points in the last quarter of a 12-month series.
QUARTER_POINTS = 13
# A last quarter this far above the year's mean is a rise, in the report's
# labels and in the drop rule alike.
RISE = 1.1


def batches(terms: list[str], size: int = TERMS_PER_REQUEST) -> list[list[str]]:
    return [terms[i : i + size] for i in range(0, len(terms), size)]


def _mean(series: list[tuple[str, float]]) -> float:
    return statistics.fmean(v for _, v in series) if series else 0.0


def peak_month(series: list[tuple[str, float]]) -> int | None:
    """The calendar month with the highest average interest, 1-12."""
    by_month: dict[int, list[float]] = {}
    for day, value in series:
        by_month.setdefault(int(day[5:7]), []).append(value)
    if not by_month or not any(any(v) for v in by_month.values()):
        return None
    return max(by_month, key=lambda m: statistics.fmean(by_month[m]))


def recent_ratio(series: list[tuple[str, float]]) -> float | None:
    """Last quarter's mean over the year's mean; above 1 is a rise."""
    year = _mean(series)
    if not year or len(series) <= QUARTER_POINTS:
        return None
    return _mean(series[-QUARTER_POINTS:]) / year


def measure(
    source: TrendsSource, terms: list[str], anchor: str, geo: str
) -> tuple[dict[str, dict[str, Any] | None], list[str]]:
    """Each term's share of the anchor, trend and peak month in one country.

    Terms are compared in requests that all carry the anchor, so shares from
    different requests are on one scale. A term whose request failed is None.
    """
    found: dict[str, dict[str, Any] | None] = {}
    missing: list[str] = []
    if anchor in terms:
        # Its own share is 1.0 by definition; it is never a term in a batch.
        found[anchor] = {"share": 1.0, "recent_ratio": None, "peak_month": None}
    for group in batches([t for t in terms if t != anchor]):
        year = source.interest([anchor, *group], geo, YEAR)
        five = source.interest([anchor, *group], geo, FIVE_YEARS)
        base = _mean(year[anchor]) if year and anchor in year else 0.0
        for term in group:
            if not year or term not in year or not base:
                found[term] = None
                missing.append(f"Trends {geo}: {term}")
                continue
            if not five or term not in five:
                missing.append(f"Trends {geo} (5 years): {term}")
            found[term] = {
                "share": round(_mean(year[term]) / base, 3),
                "recent_ratio": recent_ratio(year[term]),
                "peak_month": peak_month(five[term]) if five and term in five else None,
            }
    return found, missing


def drop_candidates(
    shares: dict[str, dict[str, dict[str, Any] | None]], drop_below: float
) -> list[str]:
    """Keywords far below their side's median everywhere, and not rising.

    A keyword with no reading in some country is not judged: a missing
    request is not low demand.
    """
    geos = {g for per_geo in shares.values() for g in per_geo}
    medians = {
        g: statistics.median(
            r["share"] for per_geo in shares.values() if (r := per_geo.get(g))
        )
        for g in geos
        if any(per_geo.get(g) for per_geo in shares.values())
    }
    out = []
    for term, per_geo in shares.items():
        readings = [per_geo.get(g) for g in geos]
        if not readings or any(r is None for r in readings):
            continue
        low = all(r["share"] < drop_below * medians[g] for g, r in per_geo.items() if r)
        rising = any((r["recent_ratio"] or 0) > RISE for r in per_geo.values() if r)
        if low and not rising:
            out.append(term)
    return out


def new_keywords(
    rising: Iterable[tuple[str, str, str, float]], keywords: list[str]
) -> list[dict[str, Any]]:
    """Rising related searches that are not already a configured keyword."""
    known = {normalize_keyword(k) for k in keywords}
    seen: set[str] = set()
    out = []
    for seed, geo, query, growth in rising:
        q = normalize_keyword(query)
        if q in known or q in seen or any(k in q or q in k for k in known):
            continue
        seen.add(q)
        out.append({"query": query, "seed": seed, "geo": geo, "growth": growth})
    return out


def uncovered(
    suggested: dict[str, dict[str, list[str] | None]], pool: list[str]
) -> list[dict[str, str]]:
    """Suggestions that no pool topic's title covers."""
    seen: set[str] = set()
    out = []
    for stem, per_geo in suggested.items():
        for geo, found in per_geo.items():
            for s in found or []:
                if s in seen or any(contains(t, s, question=True) for t in pool):
                    continue
                seen.add(s)
                out.append({"suggestion": s, "stem": stem, "geo": geo})
    return out


def topic_term(title: str, search: str = "") -> str:
    """The search a topic is measured by: its `search`, or its title."""
    if search:
        return search
    lower = title.strip()
    return (lower[7:] if lower.lower().startswith("how to ") else lower).lower()


async def _all_suggestions(
    stems: list[str], countries: list[str]
) -> dict[str, dict[str, list[str] | None]]:
    async with aiohttp.ClientSession() as session:
        out: dict[str, dict[str, list[str] | None]] = {}
        for stem in stems:
            out[stem] = {}
            for geo in countries:
                out[stem][geo] = await suggestions(session, stem, geo)
        return out


def run_demand(
    config: ResearchConfig,
    source: TrendsSource,
    keywords: list[str],
    pool: list[tuple[str, str]],
) -> dict[str, Any]:
    """Measure both sides and return the demand record the report reads.

    `pool` is (title, search) per topic; an empty search means the title.
    """
    missing: list[str] = []
    products: dict[str, dict[str, Any]] = {k: {} for k in keywords}
    pool_titles = [title for title, _ in pool]
    terms = {title: topic_term(title, search) for title, search in pool}
    topics: dict[str, dict[str, Any]] = {t: {"term": terms[t]} for t in pool_titles}
    rising: list[tuple[str, str, str, float]] = []
    for geo in config.countries:
        found, gaps = measure(source, keywords, config.products.anchor, geo)
        missing += gaps
        for k in keywords:
            products[k][geo] = found.get(k)
        found, gaps = measure(source, list(terms.values()), config.topics.anchor, geo)
        missing += gaps
        for t in pool_titles:
            topics[t][geo] = found.get(terms[t])
        for seed in config.products.seeds:
            found_rising = source.rising(seed, geo, YEAR)
            if found_rising is None:
                missing.append(f"Trends {geo}: rising searches for {seed}")
                continue
            rising += [(seed, geo, q, g) for q, g in found_rising]
    suggested = asyncio.run(
        _all_suggestions(config.topics.suggest_stems, config.countries)
    )
    missing += [
        f"Suggest {geo}: {stem}"
        for stem, per_geo in suggested.items()
        for geo, found in per_geo.items()
        if found is None
    ]
    return {
        "date": date.today().isoformat(),
        "countries": config.countries,
        "products": {
            "anchor": config.products.anchor,
            "terms": products,
            "drop_candidates": drop_candidates(products, config.products.drop_below),
            "add_candidates": new_keywords(rising, keywords),
        },
        "topics": {
            "anchor": config.topics.anchor,
            "terms": topics,
            "suggestions": suggested,
            "uncovered": uncovered(suggested, pool_titles),
        },
        "missing": missing,
    }
