"""Two defects that made Jamendo an unreliable primary music provider.

Neither surfaced as an error.

1. The search sent `duration_between`. The Jamendo v3.0 parameter is
   `durationbetween`, no underscore, and the API ignores unknown parameters
   rather than rejecting them -- so the request succeeded and the filter never
   applied. Measured against the live API with an otherwise identical query:
   the underscored spelling returned 20 of 20 results outside the requested
   10-60s window, up to 1682 seconds; the correct spelling returned 0 of 20
   outside it.

2. The API intermittently answers a working query with zero results, measured
   at roughly one call in three for identical input. An empty list was treated
   as "no tracks", so one unlucky call dropped Jamendo for the whole render and
   the chain fell through to a provider that may only offer preview-quality
   audio -- a silent change to the audio of a published video.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.audio.jamendo_provider import (
    JAMENDO_EMPTY_RETRIES,
    JamendoProvider,
    jamendo_circuit_breaker,
)


def track(track_id: str = "1", duration: float = 30.0) -> dict:
    return {
        "id": track_id,
        "name": f"Track {track_id}",
        "duration": duration,
        "artist_name": "Someone",
        "license_ccurl": "https://creativecommons.org/",
        "shareurl": "https://jamendo.example/1",
    }


class FakeResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status = status

    async def json(self):
        return self._payload

    async def text(self):
        return "body"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    """Answers each `get` from a queued list of payloads, recording params."""

    def __init__(self, payloads: list[dict]):
        self._payloads = list(payloads)
        self.calls: list[dict] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(dict(params or {}))
        payload = self._payloads.pop(0) if self._payloads else {"results": []}
        return FakeResponse(payload)


@pytest.fixture(autouse=True)
def _closed_circuit():
    """The breaker is module state; a prior test's failure would skip search."""
    jamendo_circuit_breaker.record_success()
    yield
    jamendo_circuit_breaker.record_success()


def provider(**settings) -> JamendoProvider:
    """The id comes from `secrets`, keyed by the configured env-var name."""
    base: dict[str, object] = {"search_queries": []}
    base.update(settings)
    return JamendoProvider(
        secrets={"JAMENDO_CLIENT_ID": "test-client-id"}, settings=base
    )


class TestTheDurationFilterIsSent:
    @pytest.mark.asyncio
    async def test_the_parameter_has_no_underscore(self):
        """`duration_between` is silently ignored by the API."""
        session = FakeSession([{"results": [track()]}])

        await provider().search("chill", 10, 60, 20, session)

        params = session.calls[0]
        assert params.get("durationbetween") == "10_60"
        assert "duration_between" not in params, (
            "the underscored spelling is ignored by the API, so the window "
            "never applies and a 28-minute track can come back"
        )

    @pytest.mark.asyncio
    async def test_the_window_comes_from_the_arguments(self):
        session = FakeSession([{"results": [track()]}])

        await provider().search("chill", 45.5, 90.2, 20, session)

        assert session.calls[0]["durationbetween"] == "45_90"


class TestAnEmptyResponseIsRetried:
    @pytest.mark.asyncio
    async def test_an_empty_first_response_does_not_end_the_search(self):
        """The regression guard: one unlucky call used to drop the provider."""
        session = FakeSession([{"results": []}, {"results": [track("7")]}])

        tracks = await provider().search("chill", 10, 60, 20, session)

        assert [t.id for t in tracks] == ["7"]
        assert len(session.calls) == 2

    @pytest.mark.asyncio
    async def test_it_gives_up_after_the_configured_attempts(self):
        session = FakeSession([{"results": []} for _ in range(10)])

        tracks = await provider().search("chill", 10, 60, 20, session)

        assert tracks == []
        assert len(session.calls) == JAMENDO_EMPTY_RETRIES + 1

    @pytest.mark.asyncio
    async def test_a_first_hit_costs_no_extra_call(self):
        """Retrying is for the empty case only; the happy path is unchanged."""
        session = FakeSession([{"results": [track()]}])

        await provider().search("chill", 10, 60, 20, session)

        assert len(session.calls) == 1

    @pytest.mark.asyncio
    async def test_the_query_is_redrawn_per_attempt(self):
        """The emptiness is not query-specific, so a different query is a
        second sample rather than a second guess -- but only a configured pool
        can supply one.
        """
        session = FakeSession([{"results": []} for _ in range(5)])
        p = provider(search_queries=["alpha", "beta", "gamma"])

        await p.search("ignored", 10, 60, 20, session)

        used = [c.get("fuzzytags") for c in session.calls]
        assert all(q in {"alpha", "beta", "gamma"} for q in used)

    @pytest.mark.asyncio
    async def test_a_real_failure_is_not_retried(self):
        """An HTTP or API error is already recorded against the breaker.

        Retrying it would hammer a failing endpoint and record the failure
        several times over, which is what opens the breaker prematurely.
        """
        session = MagicMock()
        session.get = MagicMock(return_value=FakeResponse({}, status=500))

        tracks = await provider().search("chill", 10, 60, 20, session)

        assert tracks == []
        assert session.get.call_count == 1

    @pytest.mark.asyncio
    async def test_an_api_error_status_is_not_retried(self):
        payload = {"headers": {"status": "error", "error_message": "bad"}}
        session = FakeSession([payload, {"results": [track()]}])

        tracks = await provider().search("chill", 10, 60, 20, session)

        assert tracks == []
        assert len(session.calls) == 1


class TestTheFallthroughIsGreppable:
    @pytest.mark.asyncio
    async def test_giving_up_logs_a_warning(self, caplog):
        """A silent downgrade is the thing being fixed.

        The next provider may only offer preview-quality audio, so a published
        video's audio can change with nothing above DEBUG to say why.
        """
        import logging

        session = FakeSession([{"results": []} for _ in range(5)])

        with caplog.at_level(logging.WARNING):
            await provider().search("chill", 10, 60, 20, session)

        assert any(
            "fall through" in r.message or "fall through" in r.getMessage()
            for r in caplog.records
        ), f"no WARNING names the fallthrough: {[r.getMessage() for r in caplog.records]}"

    @pytest.mark.asyncio
    async def test_a_successful_search_logs_no_warning(self, caplog):
        import logging

        session = FakeSession([{"results": [track()]}])

        with caplog.at_level(logging.WARNING):
            await provider().search("chill", 10, 60, 20, session)

        assert not caplog.records


class TestNoSearchWithoutCredentials:
    @pytest.mark.asyncio
    async def test_a_missing_client_id_skips_without_calling(self):
        session = MagicMock()
        session.get = AsyncMock()
        p = JamendoProvider(secrets={}, settings={"search_queries": []})

        assert await p.search("chill", 10, 60, 20, session) == []
        session.get.assert_not_called()
