"""The demand sources: Google Trends and suggest, Wikipedia, Stack Exchange.

Trends and suggest are unofficial endpoints. Trends answers 429 to a fast
client, so every request is paced and retried. Any request that still fails
is recorded as missing, never as zero interest (REQ-OPS-107).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from typing import Any, Protocol
from urllib.parse import quote

import aiohttp

logger = logging.getLogger(__name__)

SUGGEST_URL = "https://suggestqueries.google.com/complete/search"
# Trends compares at most five terms per request, the anchor among them.
TERMS_PER_REQUEST = 4


class TrendsUnavailableError(RuntimeError):
    """pytrends is not installed (`poetry install --with research`)."""


class TrendsSource(Protocol):
    def interest(
        self, terms: list[str], geo: str, timeframe: str
    ) -> dict[str, list[tuple[str, float]]] | None:
        """(month or week, value) per term, or None when the request failed."""

    def rising(
        self, term: str, geo: str, timeframe: str
    ) -> list[tuple[str, float]] | None:
        """Rising related searches as (query, growth); None when it failed."""


def default_errors(
    trends_errors: Any, requests: Any
) -> tuple[type[BaseException], ...]:
    """What a failed, rate-limited or oddly shaped Trends request raises."""
    return (
        trends_errors.ResponseError,  # TooManyRequestsError subclasses it
        requests.RequestException,
        # A payload shaped differently from what pytrends parses.
        KeyError,
        IndexError,
        TypeError,
        ValueError,
    )


class PytrendsSource:
    """The pytrends client, paced and retried."""

    def __init__(
        self,
        pause_sec: float,
        max_retries: int,
        sleep: Callable[[float], None] = time.sleep,
        client: Any = None,
        errors: tuple[type[BaseException], ...] | None = None,
    ) -> None:
        """`client` and `errors` are for tests; by default, pytrends' own."""
        self._pause = pause_sec
        self._retries = max_retries
        self._sleep = sleep
        if client is not None:
            self._client = client
            self._errors = errors or (RuntimeError,)
            return
        try:
            # Optional `research` group; untyped.
            import pytrends.exceptions as trends_errors  # type: ignore[import-untyped, import-not-found, unused-ignore]
            import pytrends.request as trends  # type: ignore[import-untyped, import-not-found, unused-ignore]
            import requests
        except ImportError as e:
            raise TrendsUnavailableError(str(e)) from e
        # What a failed or rate-limited request raises; TooManyRequestsError
        # subclasses ResponseError.
        self._errors = default_errors(trends_errors, requests)
        # No `retries` argument: with urllib3 2 it raises inside pytrends.
        self._client = trends.TrendReq(hl="en-US", tz=0)

    def _call(self, fn: Callable[[], Any], what: str) -> Any:
        for attempt in range(self._retries + 1):
            self._sleep(self._pause * (attempt + 1))
            try:
                return fn()
            except self._errors as e:
                logger.warning(
                    "Trends %s failed (attempt %d): %s", what, attempt + 1, e
                )
        return None

    def interest(
        self, terms: list[str], geo: str, timeframe: str
    ) -> dict[str, list[tuple[str, float]]] | None:
        def fetch() -> Any:
            self._client.build_payload(terms, timeframe=timeframe, geo=geo)
            return self._client.interest_over_time()

        frame = self._call(fetch, f"interest {terms} {geo}")
        if frame is None or frame.empty:
            return None
        return {
            term: [(str(i.date()), float(v)) for i, v in frame[term].items()]
            for term in terms
            if term in frame
        }

    def rising(
        self, term: str, geo: str, timeframe: str
    ) -> list[tuple[str, float]] | None:
        def fetch() -> Any:
            self._client.build_payload([term], timeframe=timeframe, geo=geo)
            return self._client.related_queries().get(term, {}).get("rising")

        failed = object()

        def guarded() -> Any:
            # A term with no rising searches answers None, which is a reading;
            # only a failed request is missing.
            found = fetch()
            return failed if found is None else found

        frame = self._call(guarded, f"rising {term} {geo}")
        if frame is None:
            return None
        if frame is failed:
            return []
        return [(str(q), float(v)) for q, v in frame.values.tolist()]


async def suggestions(
    session: aiohttp.ClientSession, stem: str, geo: str
) -> list[str] | None:
    """Google's autocomplete for a stem in one country; None when it failed."""
    params = {"client": "firefox", "q": stem, "gl": geo.lower(), "hl": "en"}
    try:
        async with session.get(  # type: ignore[attr-defined]
            SUGGEST_URL, params=params, timeout=20
        ) as resp:
            if resp.status != 200:
                logger.warning("Suggest %r %s answered %d", stem, geo, resp.status)
                return None
            body = json.loads(await resp.text())
    except (aiohttp.ClientError, TimeoutError, ValueError) as e:
        logger.warning("Suggest %r %s failed: %s", stem, geo, e)
        return None
    found = body[1] if isinstance(body, list) and len(body) > 1 else None
    if not isinstance(found, list):
        return None
    return [s for s in found if isinstance(s, str) and s.lower() != stem.lower()]


# Wikimedia blocks clients that send no contact in the User-Agent.
USER_AGENT = "ContentEngineAI-research (https://github.com/stkzlv/ContentEngineAI)"
WIKI_API = "https://en.wikipedia.org/w/api.php"
PAGEVIEWS_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia.org/all-access/user/{article}/monthly/{start}/{end}"
)
SITE_VIEWS_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/aggregate/"
    "en.wikipedia.org/all-access/user/monthly/{start}00/{end}00"
)
STACK_URL = "https://api.stackexchange.com/2.3/questions"
# Wikimedia allows 200 requests a minute to a client that names itself.
WIKI_PAUSE_SEC = 0.5
TITLES_PER_QUERY = 50


async def _json(
    session: aiohttp.ClientSession, url: str, params: dict[str, Any], what: str
) -> Any:
    """The decoded body, or None when the request failed."""
    try:
        async with session.get(  # type: ignore[attr-defined]
            url, params=params, headers={"User-Agent": USER_AGENT}, timeout=30
        ) as resp:
            if resp.status != 200:
                logger.warning("%s answered %d", what, resp.status)
                return None
            return json.loads(await resp.text())
    except (aiohttp.ClientError, TimeoutError, ValueError) as e:
        logger.warning("%s failed: %s", what, e)
        return None


async def resolve_articles(
    session: aiohttp.ClientSession, titles: list[str]
) -> dict[str, str | None] | None:
    """Each title's article after redirects; None for a title with no article.

    Views of a redirect are counted under the redirect, not the article, so a
    pageview request must name the article itself. None when the call failed.
    """
    moved: dict[str, str] = {}
    exists: set[str] = set()
    # The API refuses more than TITLES_PER_QUERY titles in one query.
    for i in range(0, len(titles), TITLES_PER_QUERY):
        body = await _json(
            session,
            WIKI_API,
            {
                "action": "query",
                "titles": "|".join(titles[i : i + TITLES_PER_QUERY]),
                "redirects": "1",
                "format": "json",
                "formatversion": "2",
            },
            "Wikipedia title lookup",
        )
        query = body.get("query") if isinstance(body, dict) else None
        if not isinstance(query, dict):
            return None
        for kind in ("normalized", "redirects"):
            for hop in query.get(kind) or []:
                moved[hop["from"]] = hop["to"]
        exists |= {p["title"] for p in query.get("pages") or [] if not p.get("missing")}
    out: dict[str, str | None] = {}
    for title in titles:
        final, seen = title, set()
        while final in moved and final not in seen:
            seen.add(final)
            final = moved[final]
        out[title] = final if final in exists else None
    return out


async def pageviews(
    session: aiohttp.ClientSession, article: str, start: str, end: str
) -> list[tuple[str, float]] | None:
    """Monthly views of one article by people (not bots), as (YYYY-MM-01, views).

    `start` and `end` are YYYYMMDD. None when the request failed.
    """
    url = PAGEVIEWS_URL.format(
        article=quote(article.replace(" ", "_"), safe=""), start=start, end=end
    )
    body = await _json(session, url, {}, f"Wikipedia pageviews {article!r}")
    return _monthly(body)


async def site_views(
    session: aiohttp.ClientSession, start: str, end: str
) -> list[tuple[str, float]] | None:
    """Monthly views of all of English Wikipedia by people; None on failure."""
    url = SITE_VIEWS_URL.format(start=start, end=end)
    body = await _json(session, url, {}, "Wikipedia total views")
    return _monthly(body)


def _monthly(body: Any) -> list[tuple[str, float]] | None:
    items = body.get("items") if isinstance(body, dict) else None
    if not isinstance(items, list):
        return None
    try:
        return [
            (f"{i['timestamp'][:4]}-{i['timestamp'][4:6]}-01", float(i["views"]))
            for i in items
        ]
    except (KeyError, TypeError, ValueError):
        return None


async def questions(
    session: aiohttp.ClientSession,
    site: str,
    tagged: str,
    since: int,
    pages: int,
    sleep: Callable[[float], Any] = asyncio.sleep,
) -> list[dict[str, Any]] | None:
    """Questions asked on a Stack Exchange site since `since` (epoch seconds).

    Newest first, up to `pages` pages of 100. The API's `backoff` asks the
    client to wait before its next request, and is honoured. None when the
    first page failed; a later page that fails ends the list there.
    """
    found: list[dict[str, Any]] = []
    for page in range(1, pages + 1):
        params: dict[str, Any] = {
            "site": site,
            "sort": "creation",
            "order": "desc",
            "fromdate": since,
            "pagesize": 100,
            "page": page,
        }
        if tagged:
            params["tagged"] = tagged
        body = await _json(
            session, STACK_URL, params, f"Stack Exchange {site} {tagged!r}"
        )
        items = body.get("items") if isinstance(body, dict) else None
        if not isinstance(items, list):
            return None if page == 1 else found
        found += [q for q in items if isinstance(q, dict)]
        if body.get("backoff"):
            await sleep(float(body["backoff"]))
        if not body.get("has_more"):
            break
    return found
