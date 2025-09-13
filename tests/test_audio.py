"""Unit tests for the audio component."""

import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from src.audio.freesound_client import FreesoundClient


class TestFreesoundClient:
    """Test the Freesound client functionality."""

    @pytest.fixture
    def mock_freesound_client_class(self):
        """Fixture for patching the FreesoundClient class."""
        with patch(
            "src.audio.freesound_client.freesound.FreesoundClient"
        ) as mock_class:
            yield mock_class

    @pytest.mark.asyncio
    async def test_search_music_success(self, mock_freesound_client_class):
        """Test successful music search."""
        mock_client = MagicMock()
        mock_freesound_client_class.return_value = mock_client

        mock_sound = MagicMock()
        mock_sound.name = "Test Sound 1"
        mock_results = [mock_sound]
        mock_client.text_search.return_value = mock_results

        client = FreesoundClient(FREESOUND_API_KEY="test_key")

        results = await client.search_music(
            query="background music",
            filters="type:wav duration:[5 TO 30]",
            max_results=10,
            sort_order="rating_desc",
        )

        assert len(results) == 1
        assert results[0].name == "Test Sound 1"

    @pytest.mark.asyncio
    async def test_download_sound_preview_success(
        self, temp_dir: Path, mock_aioresponses
    ):
        """Test successful sound preview download."""
        mock_sound = MagicMock()
        mock_sound.name = "Test Sound"
        mock_sound.previews.preview_hq_mp3 = "http://test.com/preview.mp3"

        # Mock the HTTP response
        mock_aioresponses.get(
            "http://test.com/preview.mp3", status=200, body=b"audio content"
        )

        client = FreesoundClient(FREESOUND_API_KEY="test_key")

        # Create a real aiohttp session
        async with aiohttp.ClientSession() as session:
            result = await client.download_sound_preview_with_api_key(
                mock_sound, temp_dir, session
            )

        assert result is not None
        assert result[0].exists()

    @pytest.mark.asyncio
    async def test_download_sound_preview_fallback_url(
        self, temp_dir: Path, mock_aioresponses
    ):
        """Test sound preview download with fallback URL."""
        mock_sound = MagicMock()
        mock_sound.name = "Test Sound"
        mock_sound.previews.preview_hq_mp3 = None
        mock_sound.previews.preview_lq_mp3 = "http://test.com/preview.mp3"

        # Mock the HTTP response
        mock_aioresponses.get(
            "http://test.com/preview.mp3", status=200, body=b"audio content"
        )

        client = FreesoundClient(FREESOUND_API_KEY="test_key")

        # Create a real aiohttp session
        async with aiohttp.ClientSession() as session:
            result = await client.download_sound_preview_with_api_key(
                mock_sound, temp_dir, session
            )

        assert result is not None
        assert result[0].exists()

    @pytest.mark.asyncio
    async def test_download_sound_preview_download_failure(
        self, temp_dir: Path, mock_aioresponses
    ):
        """Test sound preview download with download failure."""
        mock_sound = MagicMock()
        mock_sound.name = "Test Sound"
        mock_sound.previews.preview_hq_mp3 = "http://test.com/preview.mp3"

        # Mock the HTTP response with failure
        mock_aioresponses.get("http://test.com/preview.mp3", status=404)

        client = FreesoundClient(FREESOUND_API_KEY="test_key")

        # Create a real aiohttp session
        async with aiohttp.ClientSession() as session:
            result = await client.download_sound_preview_with_api_key(
                mock_sound, temp_dir, session
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_valid_oauth2_token_existing_valid(self):
        """Test getting valid OAuth2 token when one already exists."""
        client = FreesoundClient(FREESOUND_API_KEY="test_key")
        client.oauth_access_token = "existing_token"  # noqa: S105
        client.oauth_token_expiry = (
            time.time() + 3600
        )  # Use time.time() instead of asyncio.get_event_loop().time()

        mock_session = AsyncMock()
        result = await client._get_valid_oauth2_token(mock_session)

        assert result == "existing_token"

    @pytest.mark.asyncio
    async def test_get_valid_oauth2_token_refresh_needed(self):
        """Test getting valid OAuth2 token when refresh is needed."""
        client = FreesoundClient(
            FREESOUND_API_KEY="test_key",
            FREESOUND_CLIENT_ID="test_client_id",
            FREESOUND_CLIENT_SECRET="test_client_secret",  # noqa: S106
            FREESOUND_REFRESH_TOKEN="test_refresh_token",  # noqa: S106
        )
        client.oauth_access_token = "expired_token"  # noqa: S105
        client.oauth_token_expiry = (
            time.time() - 3600
        )  # Use time.time() instead of asyncio.get_event_loop().time()

        with patch.object(client, "_refresh_oauth2_token", return_value=True):
            client.oauth_access_token = "new_access_token"  # noqa: S105
            mock_session = AsyncMock()
            result = await client._get_valid_oauth2_token(mock_session)

            assert result == "new_access_token"

    @pytest.mark.asyncio
    async def test_download_full_sound_oauth2_success(
        self, temp_dir: Path, mock_aioresponses
    ):
        """Test successful full sound download with OAuth2."""
        client = FreesoundClient(
            FREESOUND_API_KEY="test_key",
            FREESOUND_CLIENT_ID="test_client_id",
            FREESOUND_CLIENT_SECRET="test_client_secret",  # noqa: S106
            FREESOUND_REFRESH_TOKEN="test_refresh_token",  # noqa: S106
        )

        with patch.object(
            client, "_get_valid_oauth2_token", return_value="valid_token"
        ):
            # Mock the HTTP response
            mock_aioresponses.get(
                "https://freesound.org/apiv2/sounds/12345/download/",
                status=200,
                body=b"full audio content",
                headers={"Content-Disposition": 'attachment; filename="test.wav"'},
            )

            with patch.object(client.fs_api_client, "get_sound") as mock_get_sound:
                mock_get_sound.return_value = MagicMock(
                    username="test_user",
                    url="http://test.com",
                    license="CC0",
                    name="test_sound",
                )

                # Create a real aiohttp session
                async with aiohttp.ClientSession() as session:
                    result = await client.download_full_sound_oauth2(
                        12345, temp_dir, session
                    )

                assert result is not None
                assert result[0].exists()
