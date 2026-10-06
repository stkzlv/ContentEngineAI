"""The Trends and suggest sources tell a failed request from no data."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.research.sources import PytrendsSource, suggestions


class RateLimitedError(RuntimeError):
    """Stands in for pytrends' TooManyRequestsError."""


def source(client) -> PytrendsSource:
    return PytrendsSource(
        0, 1, sleep=lambda s: None, client=client, errors=(RateLimitedError,)
    )


@pytest.mark.req("REQ-OPS-107")
def test_a_rate_limited_request_is_missing() -> None:
    client = MagicMock()
    client.build_payload.side_effect = RateLimitedError("429")

    trends = source(client)

    assert trends.interest(["a"], "US", "today 12-m") is None
    assert trends.rising("a", "US", "today 12-m") is None
    assert client.build_payload.call_count == 4  # two calls, one retry each


@pytest.mark.req("REQ-OPS-107")
def test_an_empty_frame_is_missing_and_no_rising_searches_is_a_reading() -> None:
    client = MagicMock()
    client.interest_over_time.return_value = pd.DataFrame()
    client.related_queries.return_value = {"a": {"top": None, "rising": None}}

    trends = source(client)

    assert trends.interest(["a"], "US", "today 12-m") is None
    assert trends.rising("a", "US", "today 12-m") == []


def test_a_reading_comes_back_per_term() -> None:
    client = MagicMock()
    client.interest_over_time.return_value = pd.DataFrame(
        {"a": [10, 20]}, index=pd.to_datetime(["2026-01-04", "2026-01-11"])
    )
    client.related_queries.return_value = {
        "a": {"rising": pd.DataFrame({"query": ["b"], "value": [250]})}
    }

    trends = source(client)

    assert trends.interest(["a"], "US", "today 12-m") == {
        "a": [("2026-01-04", 10.0), ("2026-01-11", 20.0)]
    }
    assert trends.rising("a", "US", "today 12-m") == [("b", 250.0)]


@pytest.mark.req("REQ-OPS-107")
def test_pytrends_rate_limit_and_shape_errors_are_caught() -> None:
    exceptions = pytest.importorskip("pytrends.exceptions")
    import requests

    from src.research.sources import default_errors

    caught = default_errors(exceptions, requests)
    for error in (
        exceptions.TooManyRequestsError,
        requests.ConnectionError,
        IndexError,
        TypeError,
    ):
        assert issubclass(error, caught), error


class _Response:
    def __init__(self, status: int, text: str) -> None:
        self.status, self._text = status, text

    async def text(self) -> str:
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        return None


class _Session:
    def __init__(self, response: _Response) -> None:
        self.response = response

    def get(self, *args, **kwargs) -> _Response:
        return self.response


@pytest.mark.req("REQ-OPS-107")
@pytest.mark.parametrize(
    ("status", "text", "expected"),
    [
        (
            200,
            '["how to stop", ["how to stop", "how to stop spam calls"]]',
            ["how to stop spam calls"],
        ),
        (429, "", None),
        (200, "<html>", None),
        (200, '["how to stop", "not a list"]', None),
    ],
)
def test_suggestions(status: int, text: str, expected) -> None:
    found = asyncio.run(
        suggestions(_Session(_Response(status, text)), "how to stop", "US")
    )
    assert found == expected
