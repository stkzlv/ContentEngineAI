"""The Freesound token settings in the shipped config reach the client.

Both were declared in `config/video_production.yaml` and on the audio
settings model for months while the client used its own constants, so
changing either had no effect.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import aiohttp
import pytest
from aioresponses import aioresponses

from src.audio.freesound_client import FreesoundClient
from src.audio.freesound_provider import FreesoundProvider
from src.utils.circuit_breaker import freesound_circuit_breaker
from src.video.config import FREESOUND_TOKEN_EXPIRY_SEC
from src.video.config_adapter import load_video_config_modular

TOKEN_URL = "https://freesound.org/apiv2/oauth2/access_token/"  # noqa: S105
REPO_CONFIG = Path(__file__).resolve().parents[2] / "config" / "video_production.yaml"
SECRETS = {
    "FREESOUND_API_KEY": "k",
    "FREESOUND_CLIENT_ID": "id",
    "FREESOUND_CLIENT_SECRET": "secret",
    "FREESOUND_REFRESH_TOKEN": "refresh",
}


@pytest.fixture(autouse=True)
def _reset_breaker():
    freesound_circuit_breaker.reset()
    yield
    freesound_circuit_breaker.reset()


def _provider(**audio_settings: float) -> FreesoundProvider:
    config = SimpleNamespace(audio_settings=SimpleNamespace(**audio_settings))
    return FreesoundProvider(config=config, secrets=SECRETS)


async def _token_is_reused(client: FreesoundClient, seconds_left: float) -> bool:
    """Whether a cached token with `seconds_left` of life is used as is."""
    client.oauth_access_token = "cached"  # noqa: S105
    client.oauth_token_expiry = time.time() + seconds_left
    with aioresponses() as mocked:
        mocked.post(TOKEN_URL, payload={"access_token": "fresh"})
        async with aiohttp.ClientSession() as session:
            token = await client._get_valid_oauth2_token(session)
    return token == "cached"  # noqa: S105


class TestTheRefreshBufferIsTheConfiguredOne:
    @pytest.mark.asyncio
    async def test_a_token_inside_the_configured_buffer_is_refreshed(self):
        client = _provider(freesound_token_refresh_buffer_sec=600)._client
        assert not await _token_is_reused(client, seconds_left=300)

    @pytest.mark.asyncio
    async def test_a_token_outside_the_configured_buffer_is_reused(self):
        client = _provider(freesound_token_refresh_buffer_sec=60)._client
        assert await _token_is_reused(client, seconds_left=300)


class TestTheExpiryFallbackIsTheConfiguredOne:
    @pytest.mark.asyncio
    async def test_a_response_without_a_lifetime_uses_the_setting(self):
        client = _provider(freesound_token_expiry_sec=1234)._client
        with aioresponses() as mocked:
            mocked.post(TOKEN_URL, payload={"access_token": "fresh"})
            async with aiohttp.ClientSession() as session:
                before = time.time()
                assert await client._get_valid_oauth2_token(session) == "fresh"
        assert client.oauth_token_expiry is not None
        assert client.oauth_token_expiry - before == pytest.approx(1234, abs=5)

    @pytest.mark.asyncio
    async def test_a_response_with_a_lifetime_wins_over_the_setting(self):
        client = _provider(freesound_token_expiry_sec=1234)._client
        with aioresponses() as mocked:
            mocked.post(
                TOKEN_URL, payload={"access_token": "fresh", "expires_in": 86400}
            )
            async with aiohttp.ClientSession() as session:
                before = time.time()
                await client._get_valid_oauth2_token(session)
        assert client.oauth_token_expiry is not None
        assert client.oauth_token_expiry - before == pytest.approx(86400, abs=5)


def test_the_default_lifetime_is_the_24_hours_freesound_grants():
    assert FREESOUND_TOKEN_EXPIRY_SEC == 86400


def test_settings_that_are_not_numbers_fall_back_to_the_defaults():
    """A mocked config must not turn the buffer into a MagicMock."""
    config = MagicMock()
    client = FreesoundProvider(config=config, secrets=SECRETS)._client
    assert client._token_expiry_sec == FREESOUND_TOKEN_EXPIRY_SEC
    assert isinstance(client._token_refresh_buffer_sec, int | float)


def test_the_shipped_config_values_reach_the_client():
    config = load_video_config_modular()
    client = FreesoundProvider(config=config, secrets=SECRETS)._client
    settings = config.audio_settings
    assert client._token_expiry_sec == settings.freesound_token_expiry_sec
    assert (
        client._token_refresh_buffer_sec == settings.freesound_token_refresh_buffer_sec
    )


def test_the_shipped_config_declares_no_unread_retry_blocks():
    text = REPO_CONFIG.read_text()
    assert "freesound_token_refresh:" not in text
    assert "freesound_download:" not in text
