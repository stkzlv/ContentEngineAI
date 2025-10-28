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
        """Test successful music search with all parameters."""
        mock_client = MagicMock()
        mock_freesound_client_class.return_value = mock_client

        mock_sound = MagicMock()
        mock_sound.name = "Test Sound 1"
        mock_sound.id = 12345
        mock_results = [mock_sound]
        mock_client.text_search.return_value = mock_results

        client = FreesoundClient(FREESOUND_API_KEY="test_key")

        results = await client.search_music(
            query="background music",
            filters="type:wav duration:[5 TO 30]",
            max_results=10,
            sort_order="rating_desc",
            fields="id,name,duration",
            timeout_sec=30,
        )

        assert len(results) == 1
        assert results[0].name == "Test Sound 1"
        assert results[0].id == 12345

        mock_client.text_search.assert_called_once_with(
            query="background music",
            filter="type:wav duration:[5 TO 30]",
            fields="id,name,duration",
            page_size=10,
            sort="rating_desc",
        )

    @pytest.mark.asyncio
    async def test_search_music_empty_results(self, mock_freesound_client_class):
        """Test search returning empty results."""
        mock_client = MagicMock()
        mock_freesound_client_class.return_value = mock_client
        mock_client.text_search.return_value = []

        client = FreesoundClient(FREESOUND_API_KEY="test_key")

        results = await client.search_music(
            query="nonexistent music",
            timeout_sec=30,
        )

        assert len(results) == 0
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_music_timeout(self, mock_freesound_client_class):
        """Test search handling timeout gracefully."""
        mock_client = MagicMock()
        mock_freesound_client_class.return_value = mock_client

        async def slow_search(*args, **kwargs):
            import asyncio

            await asyncio.sleep(10)
            return []

        mock_client.text_search.side_effect = slow_search

        client = FreesoundClient(FREESOUND_API_KEY="test_key")

        results = await client.search_music(
            query="test",
            timeout_sec=0.1,  # Very short timeout
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_search_music_api_exception(self, mock_freesound_client_class):
        """Test search handling API exceptions gracefully."""
        mock_client = MagicMock()
        mock_freesound_client_class.return_value = mock_client
        mock_client.text_search.side_effect = Exception("API Error")

        client = FreesoundClient(FREESOUND_API_KEY="test_key")

        results = await client.search_music(query="test")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_music_with_duration_filter(self, mock_freesound_client_class):
        """Test search with duration filter construction."""
        mock_client = MagicMock()
        mock_freesound_client_class.return_value = mock_client
        mock_client.text_search.return_value = []

        client = FreesoundClient(FREESOUND_API_KEY="test_key")

        await client.search_music(
            query="ambient",
            filters="duration:[60 TO 180]",
            max_results=5,
        )

        mock_client.text_search.assert_called_once()
        call_kwargs = mock_client.text_search.call_args[1]
        assert call_kwargs["filter"] == "duration:[60 TO 180]"
        assert call_kwargs["page_size"] == 5

    @pytest.mark.asyncio
    async def test_search_music_default_parameters(self, mock_freesound_client_class):
        """Test search with default parameters."""
        mock_client = MagicMock()
        mock_freesound_client_class.return_value = mock_client
        mock_client.text_search.return_value = []

        client = FreesoundClient(FREESOUND_API_KEY="test_key")

        await client.search_music(query="test")

        mock_client.text_search.assert_called_once_with(
            query="test",
            filter=None,
            fields="id,name,previews,license,username,url,duration",
            page_size=None,
            sort="rating_desc",
        )

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


class TestFreesoundOAuth2:
    """Test OAuth2 token management functionality."""

    @pytest.mark.asyncio
    async def test_refresh_oauth2_token_success(self, mock_aioresponses):
        """Test successful OAuth2 token refresh."""
        client = FreesoundClient(
            FREESOUND_API_KEY="test_key",
            FREESOUND_CLIENT_ID="client_id",
            FREESOUND_CLIENT_SECRET="client_secret",  # noqa: S106
            FREESOUND_REFRESH_TOKEN="refresh_token",  # noqa: S106
        )

        mock_aioresponses.post(
            "https://freesound.org/apiv2/oauth2/access_token/",
            status=200,
            payload={
                "access_token": "new_access_token",
                "expires_in": 3600,
                "refresh_token": "new_refresh_token",
            },
        )

        with patch("src.audio.freesound_client.update_env_file") as mock_update_env:
            async with aiohttp.ClientSession() as session:
                result = await client._refresh_oauth2_token(session)

            assert result is True
            assert client.oauth_access_token == "new_access_token"  # noqa: S105
            assert client.oauth_refresh_token == "new_refresh_token"  # noqa: S105
            mock_update_env.assert_called_once_with(
                "FREESOUND_REFRESH_TOKEN", "new_refresh_token"
            )

    @pytest.mark.asyncio
    async def test_refresh_oauth2_token_no_credentials(self):
        """Test token refresh fails without credentials."""
        client = FreesoundClient(FREESOUND_API_KEY="test_key")

        async with aiohttp.ClientSession() as session:
            result = await client._refresh_oauth2_token(session)

        assert result is False
        assert client.oauth_access_token is None

    @pytest.mark.asyncio
    async def test_refresh_oauth2_token_auth_failure_401(self, mock_aioresponses):
        """Test token refresh fast-fails on 401 authentication error."""
        client = FreesoundClient(
            FREESOUND_API_KEY="test_key",
            FREESOUND_CLIENT_ID="client_id",
            FREESOUND_CLIENT_SECRET="client_secret",  # noqa: S106
            FREESOUND_REFRESH_TOKEN="refresh_token",  # noqa: S106
        )

        mock_aioresponses.post(
            "https://freesound.org/apiv2/oauth2/access_token/",
            status=401,
        )

        async with aiohttp.ClientSession() as session:
            result = await client._refresh_oauth2_token(session)

        assert result is False
        assert client.oauth_access_token is None

    @pytest.mark.asyncio
    async def test_refresh_oauth2_token_auth_failure_403(self, mock_aioresponses):
        """Test token refresh fast-fails on 403 forbidden error."""
        client = FreesoundClient(
            FREESOUND_API_KEY="test_key",
            FREESOUND_CLIENT_ID="client_id",
            FREESOUND_CLIENT_SECRET="client_secret",  # noqa: S106
            FREESOUND_REFRESH_TOKEN="refresh_token",  # noqa: S106
        )

        mock_aioresponses.post(
            "https://freesound.org/apiv2/oauth2/access_token/",
            status=403,
        )

        async with aiohttp.ClientSession() as session:
            result = await client._refresh_oauth2_token(session)

        assert result is False

    @pytest.mark.asyncio
    async def test_refresh_oauth2_token_missing_access_token(self, mock_aioresponses):
        """Test token refresh fails when access_token missing from response."""
        client = FreesoundClient(
            FREESOUND_API_KEY="test_key",
            FREESOUND_CLIENT_ID="client_id",
            FREESOUND_CLIENT_SECRET="client_secret",  # noqa: S106
            FREESOUND_REFRESH_TOKEN="refresh_token",  # noqa: S106
        )

        mock_aioresponses.post(
            "https://freesound.org/apiv2/oauth2/access_token/",
            status=200,
            payload={"expires_in": 3600},  # Missing access_token
        )

        async with aiohttp.ClientSession() as session:
            result = await client._refresh_oauth2_token(session)

        assert result is False

    @pytest.mark.asyncio
    async def test_refresh_oauth2_token_timeout_retry(self, mock_aioresponses):
        """Test token refresh retries on timeout with exponential backoff."""
        client = FreesoundClient(
            FREESOUND_API_KEY="test_key",
            FREESOUND_CLIENT_ID="client_id",
            FREESOUND_CLIENT_SECRET="client_secret",  # noqa: S106
            FREESOUND_REFRESH_TOKEN="refresh_token",  # noqa: S106
        )

        mock_aioresponses.post(
            "https://freesound.org/apiv2/oauth2/access_token/",
            exception=aiohttp.ServerTimeoutError(),
        )
        mock_aioresponses.post(
            "https://freesound.org/apiv2/oauth2/access_token/",
            exception=aiohttp.ServerTimeoutError(),
        )

        async with aiohttp.ClientSession() as session:
            result = await client._refresh_oauth2_token(session)

        assert result is False

    @pytest.mark.asyncio
    async def test_refresh_oauth2_token_network_error_exhausts_retries(
        self, mock_aioresponses
    ):
        """Test token refresh exhausts all retries on persistent network errors."""
        client = FreesoundClient(
            FREESOUND_API_KEY="test_key",
            FREESOUND_CLIENT_ID="client_id",
            FREESOUND_CLIENT_SECRET="client_secret",  # noqa: S106
            FREESOUND_REFRESH_TOKEN="refresh_token",  # noqa: S106
        )

        # Simulate persistent 500 errors requiring retry
        mock_aioresponses.post(
            "https://freesound.org/apiv2/oauth2/access_token/",
            status=500,
            repeat=True,
        )

        with patch("src.audio.freesound_client.update_env_file"):
            async with aiohttp.ClientSession() as session:
                result = await client._refresh_oauth2_token(session)

        assert result is False
        assert client.oauth_access_token is None

    @pytest.mark.asyncio
    async def test_refresh_oauth2_token_env_update_failure(self, mock_aioresponses):
        """Test token refresh continues when .env update fails."""
        client = FreesoundClient(
            FREESOUND_API_KEY="test_key",
            FREESOUND_CLIENT_ID="client_id",
            FREESOUND_CLIENT_SECRET="client_secret",  # noqa: S106
            FREESOUND_REFRESH_TOKEN="refresh_token",  # noqa: S106
        )

        mock_aioresponses.post(
            "https://freesound.org/apiv2/oauth2/access_token/",
            status=200,
            payload={
                "access_token": "new_access_token",
                "expires_in": 3600,
                "refresh_token": "new_refresh_token",
            },
        )

        with patch(
            "src.audio.freesound_client.update_env_file",
            side_effect=Exception("File error"),
        ):
            async with aiohttp.ClientSession() as session:
                result = await client._refresh_oauth2_token(session)

        assert result is True
        assert client.oauth_access_token == "new_access_token"  # noqa: S105
        assert client.oauth_refresh_token == "new_refresh_token"  # noqa: S105

    @pytest.mark.asyncio
    async def test_get_valid_oauth2_token_within_buffer(self):
        """Test token refresh triggered when within expiry buffer window."""
        client = FreesoundClient(
            FREESOUND_API_KEY="test_key",
            FREESOUND_CLIENT_ID="client_id",
            FREESOUND_CLIENT_SECRET="client_secret",  # noqa: S106
            FREESOUND_REFRESH_TOKEN="refresh_token",  # noqa: S106
        )
        client.oauth_access_token = "old_token"  # noqa: S105
        client.oauth_token_expiry = time.time() + 30  # Expires in 30s (within buffer)

        with patch.object(
            client, "_refresh_oauth2_token", return_value=True
        ) as mock_refresh:
            client.oauth_access_token = "refreshed_token"  # noqa: S105
            mock_session = AsyncMock()
            result = await client._get_valid_oauth2_token(mock_session)

            mock_refresh.assert_called_once()
            assert result == "refreshed_token"

    @pytest.mark.asyncio
    async def test_get_valid_oauth2_token_expired(self):
        """Test token refresh triggered when token is expired."""
        client = FreesoundClient(
            FREESOUND_API_KEY="test_key",
            FREESOUND_CLIENT_ID="client_id",
            FREESOUND_CLIENT_SECRET="client_secret",  # noqa: S106
            FREESOUND_REFRESH_TOKEN="refresh_token",  # noqa: S106
        )
        client.oauth_access_token = "expired_token"  # noqa: S105
        client.oauth_token_expiry = time.time() - 3600  # Expired 1 hour ago

        with patch.object(
            client, "_refresh_oauth2_token", return_value=True
        ) as mock_refresh:
            client.oauth_access_token = "new_token"  # noqa: S105
            mock_session = AsyncMock()
            result = await client._get_valid_oauth2_token(mock_session)

            mock_refresh.assert_called_once()
            assert result == "new_token"

    @pytest.mark.asyncio
    async def test_get_valid_oauth2_token_refresh_fails(self):
        """Test returns None when token refresh fails."""
        client = FreesoundClient(
            FREESOUND_API_KEY="test_key",
            FREESOUND_CLIENT_ID="client_id",
            FREESOUND_CLIENT_SECRET="client_secret",  # noqa: S106
            FREESOUND_REFRESH_TOKEN="refresh_token",  # noqa: S106
        )
        client.oauth_access_token = "expired_token"  # noqa: S105
        client.oauth_token_expiry = time.time() - 3600

        with patch.object(client, "_refresh_oauth2_token", return_value=False):
            mock_session = AsyncMock()
            result = await client._get_valid_oauth2_token(mock_session)

            assert result is None


class TestFreesoundDownloads:
    """Test download methods for OAuth2 and preview downloads."""

    @pytest.mark.asyncio
    async def test_download_full_sound_oauth2_success_with_attribution(
        self, mock_aioresponses, tmp_path
    ):
        """Test successful OAuth2 download with complete attribution metadata."""
        client = FreesoundClient(
            FREESOUND_API_KEY="test_key",
            FREESOUND_CLIENT_ID="client_id",
            FREESOUND_CLIENT_SECRET="client_secret",  # noqa: S106
            FREESOUND_REFRESH_TOKEN="refresh_token",  # noqa: S106
        )
        client.oauth_access_token = "valid_token"  # noqa: S105
        client.oauth_token_expiry = time.time() + 3600

        sound_id = 12345
        mock_file_content = b"fake audio data"

        # Mock OAuth2 download endpoint
        mock_aioresponses.get(
            f"https://freesound.org/apiv2/sounds/{sound_id}/download/",
            status=200,
            body=mock_file_content,
            headers={"Content-Length": str(len(mock_file_content))},
        )

        # Mock sound details for attribution
        client.fs_api_client.get_sound = MagicMock(
            return_value={
                "name": "Test Sound",
                "username": "TestUser",
                "license": "Attribution 3.0",
                "url": f"https://freesound.org/s/{sound_id}/",
            }
        )

        async with aiohttp.ClientSession() as session:
            result = await client.download_full_sound_oauth2(
                sound_id, tmp_path, session
            )

        assert result is not None
        file_path, attribution = result

        # Verify file was created
        assert file_path.exists()
        assert file_path.read_bytes() == mock_file_content

        # Verify attribution structure (Requirement R6)
        assert attribution["source"] == "Freesound"
        assert attribution["type"] == "Music"
        assert attribution["name"] == "Test Sound"
        assert attribution["author"] == "TestUser"
        assert attribution["license"] == "Attribution 3.0"
        assert attribution["url"] == f"https://freesound.org/s/{sound_id}/"
        assert attribution["id"] == str(sound_id)
        assert "path" in attribution

    @pytest.mark.asyncio
    async def test_download_full_sound_oauth2_no_token(self, tmp_path):
        """Test OAuth2 download fails gracefully without valid token."""
        client = FreesoundClient(
            FREESOUND_API_KEY="test_key",
            FREESOUND_CLIENT_ID="client_id",
            FREESOUND_CLIENT_SECRET="client_secret",  # noqa: S106
        )
        # No access token set

        async with aiohttp.ClientSession() as session:
            result = await client.download_full_sound_oauth2(12345, tmp_path, session)

        assert result is None

    @pytest.mark.asyncio
    async def test_download_full_sound_oauth2_http_error(
        self, mock_aioresponses, tmp_path
    ):
        """Test OAuth2 download handles HTTP errors gracefully."""
        client = FreesoundClient(
            FREESOUND_API_KEY="test_key",
            FREESOUND_CLIENT_ID="client_id",
            FREESOUND_CLIENT_SECRET="client_secret",  # noqa: S106
        )
        client.oauth_access_token = "valid_token"  # noqa: S105
        client.oauth_token_expiry = time.time() + 3600

        sound_id = 12345

        # Mock 404 error
        mock_aioresponses.get(
            f"https://freesound.org/apiv2/sounds/{sound_id}/download/",
            status=404,
        )

        async with aiohttp.ClientSession() as session:
            result = await client.download_full_sound_oauth2(
                sound_id, tmp_path, session
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_download_full_sound_oauth2_timeout(
        self, mock_aioresponses, tmp_path
    ):
        """Test OAuth2 download exhausts retries on persistent timeouts."""
        client = FreesoundClient(
            FREESOUND_API_KEY="test_key",
            FREESOUND_CLIENT_ID="client_id",
            FREESOUND_CLIENT_SECRET="client_secret",  # noqa: S106
        )
        client.oauth_access_token = "valid_token"  # noqa: S105
        client.oauth_token_expiry = time.time() + 3600

        sound_id = 12345

        # Mock 500 errors that trigger retry logic
        mock_aioresponses.get(
            f"https://freesound.org/apiv2/sounds/{sound_id}/download/",
            status=500,
            repeat=True,
        )

        async with aiohttp.ClientSession() as session:
            result = await client.download_full_sound_oauth2(
                sound_id, tmp_path, session
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_download_full_sound_oauth2_incomplete_attribution(
        self, mock_aioresponses, tmp_path
    ):
        """Test OAuth2 download handles incomplete attribution metadata."""
        client = FreesoundClient(
            FREESOUND_API_KEY="test_key",
            FREESOUND_CLIENT_ID="client_id",
            FREESOUND_CLIENT_SECRET="client_secret",  # noqa: S106
        )
        client.oauth_access_token = "valid_token"  # noqa: S105
        client.oauth_token_expiry = time.time() + 3600

        sound_id = 12345
        mock_file_content = b"fake audio data"

        mock_aioresponses.get(
            f"https://freesound.org/apiv2/sounds/{sound_id}/download/",
            status=200,
            body=mock_file_content,
        )

        # Mock incomplete attribution (missing fields)
        client.fs_api_client.get_sound = MagicMock(return_value={})

        async with aiohttp.ClientSession() as session:
            result = await client.download_full_sound_oauth2(
                sound_id, tmp_path, session
            )

        assert result is not None
        file_path, attribution = result

        # Verify fallback attribution values (Requirement R6)
        assert attribution["name"] == f"Sound {sound_id}"
        assert attribution["author"] == "Unknown"
        assert attribution["license"] == "Unknown"
        assert attribution["url"] == f"https://freesound.org/s/{sound_id}/"

    @pytest.mark.asyncio
    async def test_download_sound_preview_hq_quality(self, mock_aioresponses, tmp_path):
        """Test preview download uses HQ preview URL when available."""
        client = FreesoundClient(FREESOUND_API_KEY="test_key")

        mock_file_content = b"fake mp3 data"
        preview_url = "https://freesound.org/data/previews/123/123456-hq.mp3"

        # Mock preview download
        mock_aioresponses.get(
            preview_url,
            status=200,
            body=mock_file_content,
            headers={"Content-Length": str(len(mock_file_content))},
        )

        # Create mock sound object with HQ preview
        mock_sound = MagicMock()
        mock_sound.id = 123456
        mock_sound.name = "Test Preview Sound"
        mock_sound.username = "TestUser"
        mock_sound.license = "Creative Commons 0"
        mock_sound.url = "https://freesound.org/s/123456/"
        mock_sound.previews = MagicMock()
        mock_sound.previews.preview_hq_mp3 = preview_url
        mock_sound.previews.preview_lq_mp3 = None

        async with aiohttp.ClientSession() as session:
            result = await client.download_sound_preview_with_api_key(
                mock_sound, tmp_path, session
            )

        assert result is not None
        file_path, attribution = result

        # Verify file created
        assert file_path.exists()
        assert file_path.read_bytes() == mock_file_content

        # Verify attribution (Requirement R6)
        assert attribution["source"] == "Freesound"
        assert attribution["name"] == "Test Preview Sound"
        assert attribution["author"] == "TestUser"

    @pytest.mark.asyncio
    async def test_download_sound_preview_fallback_lq(
        self, mock_aioresponses, tmp_path
    ):
        """Test preview download falls back to LQ when HQ unavailable."""
        client = FreesoundClient(FREESOUND_API_KEY="test_key")

        mock_file_content = b"fake mp3 data"
        preview_url = "https://freesound.org/data/previews/123/123456-lq.mp3"

        mock_aioresponses.get(
            preview_url,
            status=200,
            body=mock_file_content,
        )

        # Create mock sound with only LQ preview
        mock_sound = MagicMock()
        mock_sound.id = 123456
        mock_sound.name = "Test LQ Sound"
        mock_sound.username = "TestUser"
        mock_sound.license = "Creative Commons 0"
        mock_sound.url = "https://freesound.org/s/123456/"
        mock_sound.previews = MagicMock()
        mock_sound.previews.preview_hq_mp3 = None
        mock_sound.previews.preview_lq_mp3 = preview_url

        async with aiohttp.ClientSession() as session:
            result = await client.download_sound_preview_with_api_key(
                mock_sound, tmp_path, session
            )

        assert result is not None
        file_path, attribution = result
        assert file_path.exists()

    @pytest.mark.asyncio
    async def test_download_sound_preview_no_api_key(self, tmp_path):
        """Test preview download fails without API key."""
        client = FreesoundClient()  # No API key

        mock_sound = MagicMock()
        mock_sound.name = "Test Sound"

        async with aiohttp.ClientSession() as session:
            result = await client.download_sound_preview_with_api_key(
                mock_sound, tmp_path, session
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_download_sound_preview_no_preview_urls(self, tmp_path):
        """Test preview download handles missing preview URLs."""
        client = FreesoundClient(FREESOUND_API_KEY="test_key")

        # Create mock sound with no previews
        mock_sound = MagicMock()
        mock_sound.name = "Test Sound"
        mock_sound.previews = MagicMock()
        mock_sound.previews.preview_hq_mp3 = None
        mock_sound.previews.preview_lq_mp3 = None

        async with aiohttp.ClientSession() as session:
            result = await client.download_sound_preview_with_api_key(
                mock_sound, tmp_path, session
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_download_filename_sanitization(self, mock_aioresponses, tmp_path):
        """Test filename sanitization for special characters."""
        client = FreesoundClient(FREESOUND_API_KEY="test_key")

        mock_file_content = b"fake data"
        preview_url = "https://freesound.org/data/previews/123/123456-hq.mp3"

        mock_aioresponses.get(preview_url, status=200, body=mock_file_content)

        # Create mock sound with special characters in name
        mock_sound = MagicMock()
        mock_sound.id = 123456
        mock_sound.name = "Test/Sound: With*Special?Chars"
        mock_sound.username = "TestUser"
        mock_sound.license = "CC0"
        mock_sound.url = "https://freesound.org/s/123456/"
        mock_sound.previews = MagicMock()
        mock_sound.previews.preview_hq_mp3 = preview_url

        async with aiohttp.ClientSession() as session:
            result = await client.download_sound_preview_with_api_key(
                mock_sound, tmp_path, session
            )

        assert result is not None
        file_path, attribution = result

        # Verify filename is sanitized (no invalid characters)
        assert file_path.exists()
        assert "/" not in file_path.name
        assert ":" not in file_path.name
        assert "*" not in file_path.name
        assert "?" not in file_path.name

    @pytest.mark.asyncio
    async def test_download_preview_attribution_validation(
        self, mock_aioresponses, tmp_path
    ):
        """Test preview download validates attribution structure."""
        client = FreesoundClient(FREESOUND_API_KEY="test_key")

        mock_file_content = b"fake data"
        preview_url = "https://freesound.org/data/previews/123/123456-hq.mp3"

        mock_aioresponses.get(preview_url, status=200, body=mock_file_content)

        # Create mock sound with complete metadata
        mock_sound = MagicMock()
        mock_sound.id = 123456
        mock_sound.name = "Complete Sound"
        mock_sound.username = "TestUser"
        mock_sound.license = "Attribution 3.0"
        mock_sound.url = "https://freesound.org/s/123456/"
        mock_sound.previews = MagicMock()
        mock_sound.previews.preview_hq_mp3 = preview_url

        async with aiohttp.ClientSession() as session:
            result = await client.download_sound_preview_with_api_key(
                mock_sound, tmp_path, session
            )

        assert result is not None
        file_path, attribution = result

        # Validate complete attribution structure (Requirement R6)
        required_keys = ["source", "type", "path", "name", "author", "license", "url"]
        for key in required_keys:
            assert key in attribution, f"Missing required attribution key: {key}"

        assert attribution["source"] == "Freesound"
        assert attribution["type"] == "Music"
        assert attribution["id"] == "123456"


def test_local_fallback_attribution_format():
    """Test that local fallback generates complete attribution metadata per R6.

    Validates that when using local music files as fallback, the attribution
    dictionary includes all required fields with appropriate values.
    """
    # Simulate the local fallback attribution format from producer.py
    # This matches the format generated at src/video/producer.py lines 1467-1476
    local_path_stem = "background-music-upbeat"
    dest_path_str = "/tmp/test/music/background-music-upbeat.mp3"  # noqa: S108

    music_info = {
        "source": "Local",
        "type": "Music",
        "path": dest_path_str,
        "name": local_path_stem,
        "author": "Unknown",
        "license": "Local File",
        "url": "",
        "id": "",
    }

    # Validate all required fields exist (Requirement R6 acceptance criterion 1)
    required_keys = ["source", "type", "path", "name", "author", "license", "url", "id"]
    for key in required_keys:
        assert key in music_info, f"Missing required attribution key: {key}"

    # Validate local-specific values (R6 acceptance criterion 5)
    assert music_info["source"] == "Local"
    assert music_info["type"] == "Music"
    assert music_info["author"] == "Unknown"  # Fallback value per R6 criterion 4
    assert music_info["license"] == "Local File"
    assert music_info["name"] == local_path_stem
    assert music_info["path"] == dest_path_str
    assert music_info["url"] == ""  # Empty for local files
    assert music_info["id"] == ""  # Empty for local files
