"""Tests for FreesoundProvider adapter."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from src.audio.freesound_provider import FreesoundProvider


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.audio_settings.freesound_api_timeout_sec = 15
    config.audio_settings.freesound_download_timeout_sec = 60
    config.audio_settings.freesound_filters = "duration:[60 TO 180]"
    return config


@pytest.fixture
def provider(mock_config):
    return FreesoundProvider(
        config=mock_config,
        secrets={"FREESOUND_API_KEY": "test_key"},  # noqa: S106
    )


@pytest.fixture
def provider_no_key():
    return FreesoundProvider(secrets={})


def test_provider_name(provider):
    assert provider.provider_name == "freesound"


@pytest.mark.asyncio
async def test_search_no_api_key(provider_no_key):
    session = MagicMock()
    tracks = await provider_no_key.search("query", 60, 300, 10, session)
    assert tracks == []


@pytest.mark.asyncio
async def test_search_converts_to_audio_tracks(provider):
    mock_sound = MagicMock()
    mock_sound.id = 123
    mock_sound.name = "Test Track"
    mock_sound.duration = 120.0
    mock_sound.username = "artist"
    mock_sound.license = "CC0"
    mock_sound.url = "https://freesound.org/s/123/"

    with patch.object(
        provider._client, "search_music", new_callable=AsyncMock
    ) as mock_search:
        mock_search.return_value = [mock_sound]
        session = MagicMock()
        tracks = await provider.search("query", 60, 300, 10, session)

        assert len(tracks) == 1
        assert tracks[0].id == "123"
        assert tracks[0].name == "Test Track"
        assert tracks[0].duration == 120.0
        assert tracks[0].author == "artist"
        assert tracks[0].provider_data is mock_sound


@pytest.mark.asyncio
async def test_search_falls_back_to_general_filters(provider):
    with patch.object(
        provider._client, "search_music", new_callable=AsyncMock
    ) as mock_search:
        # First call (duration filter) returns nothing, second (general) returns track
        mock_sound = MagicMock()
        mock_sound.id = 456
        mock_sound.name = "Fallback"
        mock_sound.duration = 90.0
        mock_sound.username = "artist2"
        mock_sound.license = "CC-BY"
        mock_sound.url = "https://freesound.org/s/456/"
        mock_search.side_effect = [[], [mock_sound]]

        session = MagicMock()
        tracks = await provider.search("query", 60, 300, 10, session)

        assert len(tracks) == 1
        assert tracks[0].name == "Fallback"
        assert mock_search.call_count == 2


@pytest.mark.asyncio
async def test_download_tries_oauth2_first(provider, temp_dir):
    from src.audio.base import AudioTrack

    track = AudioTrack(
        id="100",
        name="Track",
        duration=60.0,
        author="A",
        license="CC0",
        url="",
        provider_data=MagicMock(),
    )

    with patch.object(
        provider._client, "download_full_sound_oauth2", new_callable=AsyncMock
    ) as mock_oauth:
        mock_path = temp_dir / "test.wav"
        mock_path.write_bytes(b"audio")
        mock_oauth.return_value = (mock_path, {"source": "Freesound"})

        session = MagicMock()
        result = await provider.download(track, temp_dir, session)
        assert result is not None
        assert result[1]["source"] == "Freesound"
        mock_oauth.assert_called_once()


@pytest.mark.asyncio
async def test_download_falls_back_to_preview(provider, temp_dir):
    from src.audio.base import AudioTrack

    track = AudioTrack(
        id="200",
        name="Track",
        duration=60.0,
        author="A",
        license="CC0",
        url="",
        provider_data=MagicMock(),
    )

    with (
        patch.object(
            provider._client, "download_full_sound_oauth2", new_callable=AsyncMock
        ) as mock_oauth,
        patch.object(
            provider._client,
            "download_sound_preview_with_api_key",
            new_callable=AsyncMock,
        ) as mock_preview,
    ):
        mock_oauth.return_value = None  # OAuth2 fails
        mock_path = temp_dir / "preview.mp3"
        mock_path.write_bytes(b"preview")
        mock_preview.return_value = (mock_path, {"source": "Freesound"})

        session = MagicMock()
        result = await provider.download(track, temp_dir, session)
        assert result is not None
        mock_preview.assert_called_once()


@pytest.mark.asyncio
async def test_download_none_provider_data(provider, temp_dir):
    from src.audio.base import AudioTrack

    track = AudioTrack(
        id="300",
        name="Track",
        duration=60.0,
        author="A",
        license="CC0",
        url="",
        provider_data=None,
    )
    session = MagicMock()
    result = await provider.download(track, temp_dir, session)
    assert result is None
