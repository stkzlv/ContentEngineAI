"""Google Trends and Google suggest, the two demand sources.

Both are unofficial endpoints. Trends answers 429 to a fast client, so
every request is paced and retried; a request that still fails is recorded
as missing, never as zero interest (REQ-OPS-107).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any, Protocol

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
    except (aiohttp.ClientError, TimeoutError, json.JSONDecodeError) as e:
        logger.warning("Suggest %r %s failed: %s", stem, geo, e)
        return None
    found = body[1] if isinstance(body, list) and len(body) > 1 else None
    if not isinstance(found, list):
        return None
    return [s for s in found if isinstance(s, str) and s.lower() != stem.lower()]
