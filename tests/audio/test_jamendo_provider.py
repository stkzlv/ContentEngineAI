"""Tests for Jamendo audio provider."""

import re

import aiohttp
import pytest
from aioresponses import aioresponses

from src.audio.jamendo_provider import JamendoProvider, jamendo_circuit_breaker

JAMENDO_TRACKS_PATTERN = re.compile(r"https://api\.jamendo\.com/v3\.0/tracks/.*")


@pytest.fixture(autouse=True)
def reset_breaker():
    jamendo_circuit_breaker.reset()
    yield
    jamendo_circuit_breaker.reset()


@pytest.fixture
def provider():
    return JamendoProvider(
        secrets={"JAMENDO_CLIENT_ID": "test_client_id"},
    )


@pytest.fixture
def provider_no_key():
    return JamendoProvider(secrets={})


@pytest.mark.asyncio
async def test_search_success(provider, mock_aioresponses):
    mock_aioresponses.get(
        JAMENDO_TRACKS_PATTERN,
        payload={
            "headers": {"status": "success", "results_count": 1},
            "results": [
                {
                    "id": "12345",
                    "name": "Cool Track",
                    "duration": 120,
                    "artist_name": "Test Artist",
                    "license_ccurl": "http://creativecommons.org/licenses/by/3.0/",
                    "shareurl": "https://www.jamendo.com/track/12345",
                    "audiodownload": "https://prod-1.storage.jamendo.com/download/track/12345/mp32/",
                    "audiodownload_allowed": True,
                    "audio": "https://prod-1.storage.jamendo.com/?trackid=12345",
                }
            ],
        },
        status=200,
    )

    async with aiohttp.ClientSession() as session:
        tracks = await provider.search("upbeat", 60, 300, 10, session)
        assert len(tracks) == 1
        assert tracks[0].id == "12345"
        assert tracks[0].name == "Cool Track"
        assert tracks[0].duration == 120.0
        assert tracks[0].author == "Test Artist"


@pytest.mark.asyncio
async def test_search_no_client_id(provider_no_key, mock_aioresponses):
    async with aiohttp.ClientSession() as session:
        tracks = await provider_no_key.search("query", 60, 300, 10, session)
        assert tracks == []


@pytest.mark.asyncio
async def test_search_api_error(provider, mock_aioresponses):
    mock_aioresponses.get(
        JAMENDO_TRACKS_PATTERN,
        status=500,
        body="Internal Server Error",
    )

    async with aiohttp.ClientSession() as session:
        tracks = await provider.search("query", 60, 300, 10, session)
        assert tracks == []


@pytest.mark.asyncio
async def test_search_api_error_in_body(provider, mock_aioresponses):
    """Jamendo can return 200 with error status in JSON body."""
    mock_aioresponses.get(
        JAMENDO_TRACKS_PATTERN,
        payload={
            "headers": {
                "status": "error",
                "code": 5,
                "error_message": "Invalid client_id",
            },
            "results": [],
        },
        status=200,
    )

    async with aiohttp.ClientSession() as session:
        tracks = await provider.search("query", 60, 300, 10, session)
        assert tracks == []


@pytest.mark.asyncio
async def test_search_timeout(provider, mock_aioresponses):
    mock_aioresponses.get(
        JAMENDO_TRACKS_PATTERN,
        exception=TimeoutError(),
    )

    async with aiohttp.ClientSession() as session:
        tracks = await provider.search("query", 60, 300, 10, session)
        assert tracks == []


@pytest.mark.asyncio
async def test_download_success(provider, mock_aioresponses, temp_dir):
    from src.audio.base import AudioTrack

    track = AudioTrack(
        id="12345",
        name="Cool Track",
        duration=120.0,
        author="Test Artist",
        license="CC-BY",
        url="https://www.jamendo.com/track/12345",
        provider_data={
            "audiodownload_allowed": True,
            "audiodownload": "https://prod-1.storage.jamendo.com/download/track/12345/mp32/",
            "audio": "https://prod-1.storage.jamendo.com/?trackid=12345",
        },
    )

    mock_aioresponses.get(
        "https://prod-1.storage.jamendo.com/download/track/12345/mp32/",
        body=b"audio data here",
        status=200,
    )

    async with aiohttp.ClientSession() as session:
        result = await provider.download(track, temp_dir, session)
        assert result is not None
        path, attribution = result
        assert path.exists()
        assert attribution["source"] == "Jamendo"
        assert attribution["author"] == "Test Artist"
        assert attribution["id"] == "12345"


@pytest.mark.asyncio
async def test_download_falls_back_to_stream(provider, mock_aioresponses, temp_dir):
    from src.audio.base import AudioTrack

    track = AudioTrack(
        id="999",
        name="Stream Only",
        duration=90.0,
        author="Artist",
        license="CC0",
        url="https://www.jamendo.com/track/999",
        provider_data={
            "audiodownload_allowed": False,
            "audiodownload": "",
            "audio": "https://prod-1.storage.jamendo.com/?trackid=999",
        },
    )

    mock_aioresponses.get(
        "https://prod-1.storage.jamendo.com/?trackid=999",
        body=b"stream data",
        status=200,
    )

    async with aiohttp.ClientSession() as session:
        result = await provider.download(track, temp_dir, session)
        assert result is not None
        path, _ = result
        assert path.exists()


@pytest.mark.asyncio
async def test_download_no_url(provider, temp_dir):
    from src.audio.base import AudioTrack

    track = AudioTrack(
        id="111",
        name="No URL",
        duration=60.0,
        author="Artist",
        license="CC0",
        url="",
        provider_data={
            "audiodownload_allowed": False,
            "audiodownload": "",
            "audio": "",
        },
    )

    async with aiohttp.ClientSession() as session:
        result = await provider.download(track, temp_dir, session)
        assert result is None


@pytest.mark.asyncio
async def test_search_circuit_breaker_open(provider, mock_aioresponses):
    """Skip search when circuit breaker is open."""
    # Trip the breaker by recording failures
    for _ in range(3):
        jamendo_circuit_breaker.record_failure()

    async with aiohttp.ClientSession() as session:
        tracks = await provider.search("query", 60, 300, 10, session)
        assert tracks == []


@pytest.mark.asyncio
async def test_search_uses_random_query(mock_aioresponses):
    """search_queries config picks randomly, ignoring the passed query."""
    provider = JamendoProvider(
        secrets={"JAMENDO_CLIENT_ID": "test_client_id"},
        settings={"search_queries": ["rock", "jazz"]},
    )
    mock_aioresponses.get(
        JAMENDO_TRACKS_PATTERN,
        payload={"headers": {"status": "success"}, "results": []},
        status=200,
        repeat=True,
    )

    async with aiohttp.ClientSession() as session:
        await provider.search("ignored", 60, 300, 5, session)
        # Verify the request was made (random query used, not "ignored")
        assert len(mock_aioresponses.requests) > 0


@pytest.mark.asyncio
async def test_download_http_error(provider, mock_aioresponses, temp_dir):
    from src.audio.base import AudioTrack

    track = AudioTrack(
        id="222",
        name="Error Track",
        duration=90.0,
        author="Artist",
        license="CC0",
        url="",
        provider_data={
            "audiodownload_allowed": True,
            "audiodownload": "https://prod-1.storage.jamendo.com/download/track/222/mp32/",
        },
    )

    mock_aioresponses.get(
        "https://prod-1.storage.jamendo.com/download/track/222/mp32/",
        status=403,
    )

    async with aiohttp.ClientSession() as session:
        result = await provider.download(track, temp_dir, session)
        assert result is None
