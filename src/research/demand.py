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
# Weekly points in a quarter and in a year of a weekly series.
QUARTER_POINTS = 13
YEAR_POINTS = 52
# A last quarter this far above the same quarter a year earlier is a rise,
# in the report's labels and in the drop rule alike.
RISE = 1.1
# A month is a seasonal peak when it is the top month in this many complete
# years: one anomalous spike, which Trends shows across unrelated searches
# at once, cannot make a peak on its own.
PEAK_YEARS = 2
# A year votes only when its top month stands this far above its median
# month; a flat year has no peak, and a tie would elect January.
PEAK_LIFT = 1.2
# Nor does a year whose top month averages under Trends' smallest unit:
# scattered single readings are noise, not a season.
PEAK_FLOOR = 1.0
# Add candidates measured against the anchor, fastest-rising first.
MAX_CANDIDATES = 20


def batches(terms: list[str], size: int = TERMS_PER_REQUEST) -> list[list[str]]:
    return [terms[i : i + size] for i in range(0, len(terms), size)]


def _median(series: list[tuple[str, float]]) -> float:
    """The median, not the mean: a one-week spike barely moves it."""
    return statistics.median(v for _, v in series) if series else 0.0


def peak_month(series: list[tuple[str, float]]) -> int | None:
    """The month that is the year's top month in at least PEAK_YEARS years.

    Counted over complete calendar years only, so a partial first or last
    year cannot vote. None when no month recurs as the peak.
    """
    by_year: dict[str, dict[int, list[float]]] = {}
    for day, value in series:
        by_year.setdefault(day[:4], {}).setdefault(int(day[5:7]), []).append(value)
    votes: dict[int, int] = {}
    for months in by_year.values():
        if len(months) < 12:
            continue
        means = {m: statistics.fmean(v) for m, v in months.items()}
        top = max(means, key=lambda m: means[m])
        typical = statistics.median(means.values())
        # A sparse year (a typical month of zero) votes for whichever month
        # caught a stray reading, so it does not vote at all.
        if typical <= 0 or means[top] < max(PEAK_FLOOR, PEAK_LIFT * typical):
            continue
        votes[top] = votes.get(top, 0) + 1
    if not votes:
        return None
    month, count = max(votes.items(), key=lambda item: item[1])
    return month if count >= PEAK_YEARS else None


def yoy_ratio(series: list[tuple[str, float]]) -> float | None:
    """The last quarter's median over the same quarter a year earlier.

    Year on year, so a holiday peak inside the last twelve months does not
    make every autumn read as a fall. None without two years of data or
    with no interest a year earlier.
    """
    if len(series) < YEAR_POINTS + QUARTER_POINTS:
        return None
    before = _median(series[-(YEAR_POINTS + QUARTER_POINTS) : -YEAR_POINTS])
    now = _median(series[-QUARTER_POINTS:])
    if not before:
        # Nothing then and nothing now is flat, the case the drop rule is
        # for; something now from nothing has no ratio.
        return 1.0 if not now else None
    return round(now / before, 3)


def measure(
    source: TrendsSource,
    terms: list[str],
    anchor: str,
    geo: str,
    five_years: bool = True,
) -> tuple[dict[str, dict[str, Any] | None], list[str]]:
    """Each term's share of the anchor, trend and peak month in one country.

    Terms are compared in requests that all carry the anchor, so shares from
    different requests are on one scale. A term whose request failed is None.
    """
    found: dict[str, dict[str, Any] | None] = {}
    missing: list[str] = []
    if anchor in terms:
        # Its own share is 1.0 by definition; it is never a term in a batch.
        found[anchor] = {"share": 1.0, "trend": None, "peak_month": None}
    for group in batches([t for t in terms if t != anchor]):
        year = source.interest([anchor, *group], geo, YEAR)
        # Share-only callers (add candidates) skip the five-year request.
        five = (
            source.interest([anchor, *group], geo, FIVE_YEARS) if five_years else None
        )
        base = _median(year[anchor]) if year and anchor in year else 0.0
        for term in group:
            if not year or term not in year or not base:
                found[term] = None
                missing.append(f"Trends {geo}: {term}")
                continue
            series = five.get(term) if five else None
            if series is None and five_years:
                missing.append(f"Trends {geo} (5 years): {term}")
            found[term] = {
                "share": round(_median(year[term]) / base, 3),
                "trend": yoy_ratio(series) if series else None,
                "peak_month": peak_month(series) if series else None,
            }
    return found, missing


def drop_candidates(
    shares: dict[str, dict[str, dict[str, Any] | None]], drop_below: float
) -> list[str]:
    """Keywords far below their side's median everywhere, and not rising.

    A keyword with no reading in some country, or no year-on-year trend in
    any, is not judged: a missing request is not low demand, and an unknown
    trend is not a flat one.
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
        trends = [r["trend"] for r in per_geo.values() if r and r["trend"] is not None]
        if not trends:
            continue
        low = all(r["share"] < drop_below * medians[g] for g, r in per_geo.items() if r)
        rising = any(t > RISE for t in trends)
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
    """Suggestions that no pool topic's title covers, alternating by stem.

    Round robin across the stems, so a list cut to its first ten does not
    hold one stem's suggestions only.
    """
    per_stem: list[list[dict[str, str]]] = []
    seen: set[str] = set()
    for stem, per_geo in suggested.items():
        found_here = []
        for geo, found in per_geo.items():
            for s in found or []:
                if s in seen or any(contains(t, s, question=True) for t in pool):
                    continue
                seen.add(s)
                found_here.append({"suggestion": s, "stem": stem, "geo": geo})
        per_stem.append(found_here)
    out = []
    for rank in range(max((len(f) for f in per_stem), default=0)):
        out += [found[rank] for found in per_stem if rank < len(found)]
    return out


def strongest(
    products: dict[str, dict[str, Any]], countries: list[str], count: int
) -> list[str]:
    """The `count` keywords with the highest share in any country."""

    def best(per_geo: dict[str, Any]) -> float:
        return max((r["share"] for g in countries if (r := per_geo.get(g))), default=0)

    ranked = sorted(products, key=lambda k: -best(products[k]))
    return [k for k in ranked if best(products[k]) > 0][:count]


def measured_candidates(
    candidates: list[dict[str, Any]],
    readings: dict[str, dict[str, dict[str, Any] | None]],
    medians: dict[str, float],
) -> list[dict[str, Any]]:
    """Candidates that measure at or above the keywords' median somewhere.

    A rising search can be a one-off ("august 2026 tech gadgets") or not a
    product at all; measured against the same anchor, the ones worth
    scraping stand where today's keywords stand.
    """
    out = []
    for c in candidates:
        per_geo = readings.get(c["query"], {})
        shares = {g: r["share"] for g, r in per_geo.items() if r}
        if any(shares.get(g, 0) >= m for g, m in medians.items() if m):
            out.append({**c, "shares": shares})
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
    # Rising searches next to the strongest keywords are adjacent products;
    # next to a broad seed they were news, finance and one-off headlines.
    for seed in strongest(products, config.countries, config.products.related_from):
        for geo in config.countries:
            found_rising = source.rising(seed, geo, YEAR)
            if found_rising is None:
                missing.append(f"Trends {geo}: rising searches for {seed}")
                continue
            rising += [(seed, geo, q, g) for q, g in found_rising]
    # The fastest-rising first, capped: each candidate costs Trends requests.
    candidates = sorted(new_keywords(rising, keywords), key=lambda c: -c["growth"])[
        :MAX_CANDIDATES
    ]
    readings: dict[str, dict[str, dict[str, Any] | None]] = {
        c["query"]: {} for c in candidates
    }
    for geo in config.countries:
        found, gaps = measure(
            source,
            [c["query"] for c in candidates],
            config.products.anchor,
            geo,
            five_years=False,
        )
        missing += [f"{g} (add candidate)" for g in gaps]
        for c in candidates:
            readings[c["query"]][geo] = found.get(c["query"])
    medians = {
        g: statistics.median(
            r["share"] for per in products.values() if (r := per.get(g))
        )
        for g in config.countries
        if any(per.get(g) for per in products.values())
    }
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
            "add_candidates": measured_candidates(candidates, readings, medians),
        },
        "topics": {
            "anchor": config.topics.anchor,
            "terms": topics,
            "suggestions": suggested,
            "uncovered": uncovered(suggested, pool_titles),
        },
        "missing": missing,
    }
