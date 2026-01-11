import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import aiohttp
import pytest
from aioresponses import aioresponses

from src.audio.freesound_client import FreesoundClient
from src.utils.circuit_breaker import CircuitBreakerError, freesound_circuit_breaker


@pytest.fixture(autouse=True)
def reset_breaker():
    freesound_circuit_breaker.reset()
    yield
    freesound_circuit_breaker.reset()


@pytest.fixture
def fs_client():
    return FreesoundClient(
        FREESOUND_API_KEY="test_api_key",  # noqa: S106
        FREESOUND_CLIENT_ID="test_client_id",
        FREESOUND_CLIENT_SECRET="test_client_secret",  # noqa: S106
        FREESOUND_REFRESH_TOKEN="test_refresh_token",  # noqa: S106
    )


@pytest.mark.asyncio
async def test_search_music_success(fs_client, mock_aioresponses):
    with patch.object(fs_client.fs_api_client, "text_search") as mock_search:
        mock_track = MagicMock()
        mock_track.id = 123
        mock_track.name = "Test Track"
        mock_search.return_value = [mock_track]

        results = await fs_client.search_music("test query", timeout_sec=5)
        assert len(results) == 1
        assert results[0].id == 123
        mock_search.assert_called_once()


@pytest.mark.asyncio
async def test_search_music_timeout(fs_client, mock_aioresponses):
    # Mock search to delay longer than timeout
    async def delayed_search(*args, **kwargs):
        await asyncio.sleep(2)
        return []

    with patch("asyncio.to_thread", side_effect=delayed_search):
        results = await fs_client.search_music("test query", timeout_sec=0.1)
        assert results == []


@pytest.mark.asyncio
async def test_oauth2_token_refresh_success(fs_client, mock_aioresponses):
    mock_aioresponses.post(
        "https://freesound.org/apiv2/oauth2/access_token/",
        payload={
            "access_token": "new_access_token",
            "expires_in": 3600,
            "refresh_token": "new_refresh_token",
        },
        status=200,
    )

    async with aiohttp.ClientSession() as session:
        success = await fs_client._refresh_oauth2_token(session)
        assert success is True
        assert fs_client.oauth_access_token == "new_access_token"  # noqa: S105
        assert fs_client.oauth_refresh_token == "new_refresh_token"  # noqa: S105


@pytest.mark.asyncio
async def test_oauth2_token_refresh_failure_401(fs_client, mock_aioresponses):
    mock_aioresponses.post(
        "https://freesound.org/apiv2/oauth2/access_token/", status=401
    )

    async with aiohttp.ClientSession() as session:
        success = await fs_client._refresh_oauth2_token(session)
        assert success is False


@pytest.mark.asyncio
async def test_get_valid_oauth2_token(fs_client, mock_aioresponses):
    # Case 1: Token is valid
    fs_client.oauth_access_token = "valid_at"  # noqa: S105
    fs_client.oauth_token_expiry = time.time() + 1000

    async with aiohttp.ClientSession() as session:
        token = await fs_client._get_valid_oauth2_token(session)
        assert token == "valid_at"  # noqa: S105

    # Case 2: Token expired, refresh succeeds
    fs_client.oauth_token_expiry = time.time() - 10
    mock_aioresponses.post(
        "https://freesound.org/apiv2/oauth2/access_token/",
        payload={"access_token": "refreshed_at", "expires_in": 3600},
        status=200,
    )
    async with aiohttp.ClientSession() as session:
        token = await fs_client._get_valid_oauth2_token(session)
        assert token == "refreshed_at"  # noqa: S105


@pytest.mark.asyncio
async def test_download_full_sound_oauth2_success(
    fs_client, mock_aioresponses, temp_dir
):
    mock_aioresponses.post(
        "https://freesound.org/apiv2/oauth2/access_token/",
        payload={"access_token": "valid_token", "expires_in": 3600},
        status=200,
    )

    sound_id = 456
    mock_aioresponses.get(
        f"https://freesound.org/apiv2/sounds/{sound_id}/download/",
        body=b"audio data",
        headers={"Content-Disposition": 'attachment; filename="test.wav"'},
        status=200,
    )

    with patch.object(fs_client.fs_api_client, "get_sound") as mock_get_sound:
        mock_get_sound.return_value = {
            "name": "Test Sound",
            "username": "user",
            "license": "CC0",
            "url": "http://fs.org/s/456/",
        }

        async with aiohttp.ClientSession() as session:
            result = await fs_client.download_full_sound_oauth2(
                sound_id, temp_dir, session
            )
            assert result is not None
            path, attribution = result
            assert path.name == "test.wav"
            assert attribution["name"] == "Test Sound"
            assert path.exists()


@pytest.mark.asyncio
async def test_download_sound_preview_with_api_key_success(
    fs_client, mock_aioresponses, temp_dir
):
    mock_sound = MagicMock()
    mock_sound.id = 789
    mock_sound.name = "Preview Sound"
    mock_sound.previews.preview_hq_mp3 = "https://fs.org/preview.mp3"

    mock_aioresponses.get(
        "https://fs.org/preview.mp3", body=b"preview data", status=200
    )

    async with aiohttp.ClientSession() as session:
        result = await fs_client.download_sound_preview_with_api_key(
            mock_sound, temp_dir, session
        )
        assert result is not None
        path, attribution = result
        assert path.name == "Preview_Sound.mp3"  # Sanitized name
        assert attribution["source"] == "Freesound"


@pytest.mark.asyncio
async def test_circuit_breaker_opening(fs_client):
    # threshold is 3
    with patch.object(fs_client, "_search_sync", side_effect=OSError("Network down")):
        for _ in range(3):
            with pytest.raises(OSError):
                await fs_client.search_music("query", timeout_sec=1)

        assert freesound_circuit_breaker.is_open

        # Next call should fail fast with CircuitBreakerError
        with pytest.raises(CircuitBreakerError):
            await fs_client.search_music("query", timeout_sec=1)


@pytest.mark.asyncio
async def test_token_persistence_fail(
    fs_client, mock_aioresponses, temp_dir, monkeypatch
):
    from src.audio.freesound_client import update_env_file

    update_env_file("KEY", "VAL")

    mock_aioresponses.post(
        "https://freesound.org/apiv2/oauth2/access_token/",
        payload={"access_token": "at", "expires_in": 3600, "refresh_token": "new_rt"},
        status=200,
    )

    with patch("src.audio.freesound_client.update_env_file") as mock_update:
        async with aiohttp.ClientSession() as session:
            await fs_client._refresh_oauth2_token(session)
            mock_update.assert_called_once_with("FREESOUND_REFRESH_TOKEN", "new_rt")
