"""The music step cannot spend minutes on failures.

In the 2026-09-19 batch the step took 208 s of a 304 s render: two failed
OAuth2 refreshes at ERROR for every candidate, then three one-minute
timeouts downloading one preview before the next candidate was tried.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import aiohttp
import pytest
from aioresponses import aioresponses

from src.audio import freesound_client as fc
from src.audio.base import AudioTrack, BaseAudioProvider
from src.audio.freesound_client import FreesoundClient
from src.audio.manager import AudioManager
from src.utils.circuit_breaker import freesound_circuit_breaker

TOKEN_URL = "https://freesound.org/apiv2/oauth2/access_token/"  # noqa: S105


@pytest.fixture(autouse=True)
def _reset_breaker():
    freesound_circuit_breaker.reset()
    yield
    freesound_circuit_breaker.reset()


def _client() -> FreesoundClient:
    return FreesoundClient(
        FREESOUND_API_KEY="k",  # noqa: S106
        FREESOUND_CLIENT_ID="id",
        FREESOUND_CLIENT_SECRET="secret",  # noqa: S106
        FREESOUND_REFRESH_TOKEN="dead",  # noqa: S106
    )


class TestTheRefreshIsAttemptedOncePerRun:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [400, 401])
    async def test_an_invalid_token_logs_one_warning_and_no_error(self, caplog, status):
        """Freesound answers a dead refresh token with 400 (invalid_grant);
        a 401 is the same case for bad client credentials.
        """
        client = _client()
        with aioresponses() as mocked:
            mocked.post(TOKEN_URL, status=status)
            mocked.post(TOKEN_URL, status=status)
            async with aiohttp.ClientSession() as session:
                with caplog.at_level(logging.DEBUG):
                    first = await client._get_valid_oauth2_token(session)
                    second = await client._get_valid_oauth2_token(session)
                    third = await client.download_full_sound_oauth2(
                        1, Path("/nonexistent"), session
                    )
            posts = [
                key for key in mocked.requests if key[0] == "POST"
            ]  # one request key per URL/method; count the calls under it
            calls = sum(len(mocked.requests[key]) for key in posts)

        assert (first, second, third) == (None, None, None)
        assert calls == 1
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "freesound_oauth2_setup" in warnings[0].message
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

    @pytest.mark.asyncio
    async def test_an_unreachable_endpoint_is_one_warning_without_the_tool(
        self, caplog
    ):
        client = _client()
        with aioresponses() as mocked:
            mocked.post(TOKEN_URL, exception=TimeoutError())
            mocked.post(TOKEN_URL, exception=TimeoutError())
            mocked.post(TOKEN_URL, exception=TimeoutError())
            async with aiohttp.ClientSession() as session:
                with caplog.at_level(logging.DEBUG):
                    first = await client._get_valid_oauth2_token(session)
                    second = await client._get_valid_oauth2_token(session)

        assert (first, second) == (None, None)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "unreachable" in warnings[0].message
        assert "freesound_oauth2_setup" not in warnings[0].message
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

    @pytest.mark.asyncio
    async def test_a_working_refresh_is_unaffected(self):
        client = _client()
        with aioresponses() as mocked:
            mocked.post(
                TOKEN_URL,
                payload={"access_token": "at", "expires_in": 3600},
                status=200,
            )
            async with aiohttp.ClientSession() as session:
                assert await client._get_valid_oauth2_token(session) == "at"
        assert client._refresh_failed is False


class TestACandidateIsDownloadedOnce:
    @pytest.mark.asyncio
    async def test_the_preview_download_makes_one_attempt(self, monkeypatch, tmp_path):
        seen: dict[str, object] = {}

        async def fake_download(url, path, session, **kwargs):
            seen.update(kwargs)
            return False

        monkeypatch.setattr(fc, "util_download_file", fake_download)
        client = _client()
        sound = MagicMock()
        sound.id = 5
        sound.name = "s"
        sound.previews = SimpleNamespace(preview_hq_mp3="https://x/p.mp3")

        result = await client.download_sound_preview_with_api_key(
            sound, tmp_path, MagicMock(), timeout_sec=1
        )

        assert result is None
        assert seen.get("retry_attempts") == 1

    @pytest.mark.asyncio
    async def test_a_stalled_oauth_download_costs_one_timeout(self, tmp_path, caplog):
        client = _client()
        client.oauth_access_token = "at"  # noqa: S105
        client.oauth_token_expiry = time.time() + 1000
        with aioresponses() as mocked:
            mocked.get(
                "https://freesound.org/apiv2/sounds/9/download/",
                exception=TimeoutError(),
            )
            mocked.get(
                "https://freesound.org/apiv2/sounds/9/download/",
                exception=TimeoutError(),
            )
            async with aiohttp.ClientSession() as session:
                with caplog.at_level(logging.WARNING):
                    result = await client.download_full_sound_oauth2(
                        9, tmp_path, session, timeout_sec=1
                    )
            calls = sum(len(v) for v in mocked.requests.values())

        assert result is None
        assert calls == 1


class StallingProvider(BaseAudioProvider):
    def __init__(self, name: str, stall_sec: float):
        self._name = name
        self._stall = stall_sec
        self.downloads = 0

    @property
    def provider_name(self) -> str:
        return self._name

    async def search(self, query, min_duration, max_duration, max_results, session):
        return [
            AudioTrack(
                id=str(i),
                name=f"calm {i}",
                duration=120.0,
                author="A",
                license="CC0",
                url="",
                tags=["calm"],
            )
            for i in range(5)
        ]

    async def download(self, track, output_dir, session):
        self.downloads += 1
        await asyncio.sleep(self._stall)
        return None


class RaisingProvider(StallingProvider):
    """A provider whose download times out on its own, with budget left."""

    async def download(self, track, output_dir, session):
        self.downloads += 1
        raise TimeoutError("provider timeout")


class TestTheChainStaysInsideItsBudget:
    @pytest.mark.asyncio
    async def test_a_providers_own_timeout_is_not_the_budget(self, tmp_path):
        """With budget left, a candidate that times out on its own is
        skipped and the chain goes on; the deadline alone ends it.
        """
        local = tmp_path / "local.mp3"
        local.write_bytes(b"audio")
        first = RaisingProvider("a", 0.0)
        second = StallingProvider("b", 0.0)
        manager = AudioManager(
            providers=[first, second], local_paths=[local], budget_sec=100
        )

        result = await manager.find_music(
            "calm", 60, 300, 10, tmp_path / "out", MagicMock()
        )

        assert result is not None and result["source"] == "Local"
        assert first.downloads == 5 and second.downloads == 5

    @pytest.mark.asyncio
    async def test_stalled_downloads_end_at_the_budget(self, tmp_path, caplog):
        local = tmp_path / "local.mp3"
        local.write_bytes(b"audio")
        providers = [StallingProvider("a", 10.0), StallingProvider("b", 10.0)]
        manager = AudioManager(providers=providers, local_paths=[local], budget_sec=0.5)

        start = time.monotonic()
        with caplog.at_level(logging.WARNING):
            result = await manager.find_music(
                "calm", 60, 300, 10, tmp_path / "out", MagicMock()
            )
        elapsed = time.monotonic() - start

        assert result is not None and result["source"] == "Local"
        assert elapsed < 2.0
        assert providers[0].downloads == 1 and providers[1].downloads == 0
        assert any("Music budget" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_no_budget_means_no_deadline(self, tmp_path):
        local = tmp_path / "local.mp3"
        local.write_bytes(b"audio")
        manager = AudioManager(
            providers=[StallingProvider("a", 0.01)], local_paths=[local]
        )

        result = await manager.find_music(
            "calm", 60, 300, 10, tmp_path / "out", MagicMock()
        )

        assert result is not None and result["source"] == "Local"
