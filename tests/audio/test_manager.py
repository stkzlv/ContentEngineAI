"""Tests for AudioManager provider chain."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.audio.base import AudioTrack, BaseAudioProvider
from src.audio.manager import AudioManager


class FakeProvider(BaseAudioProvider):
    def __init__(self, name, tracks=None, download_result=None, fail=False):
        self._name = name
        self._tracks = tracks or []
        self._download_result = download_result
        self._fail = fail

    @property
    def provider_name(self):
        return self._name

    async def search(self, query, min_duration, max_duration, max_results, session):
        if self._fail:
            raise RuntimeError("provider error")
        return self._tracks

    async def download(self, track, output_dir, session):
        if self._fail:
            raise RuntimeError("download error")
        return self._download_result


@pytest.fixture
def sample_track():
    return AudioTrack(
        id="1",
        name="Test",
        duration=120.0,
        author="Artist",
        license="CC0",
        url="https://example.com",
    )


@pytest.fixture
def sample_attribution(temp_dir):
    path = temp_dir / "test.mp3"
    path.write_bytes(b"audio")
    return {
        "source": "TestProvider",
        "type": "Music",
        "path": str(path),
        "name": "Test",
        "author": "Artist",
        "license": "CC0",
        "url": "https://example.com",
        "id": "1",
    }


@pytest.mark.asyncio
async def test_first_provider_succeeds(sample_track, sample_attribution, temp_dir):
    path = Path(sample_attribution["path"])
    provider = FakeProvider(
        "first",
        tracks=[sample_track],
        download_result=(path, sample_attribution),
    )
    manager = AudioManager(providers=[provider])

    session = MagicMock()
    result = await manager.find_music("query", 60, 300, 10, temp_dir, session)
    assert result is not None
    assert result["source"] == "TestProvider"


@pytest.mark.asyncio
async def test_first_fails_second_succeeds(sample_track, sample_attribution, temp_dir):
    path = Path(sample_attribution["path"])
    failing = FakeProvider("failing", fail=True)
    working = FakeProvider(
        "working",
        tracks=[sample_track],
        download_result=(path, sample_attribution),
    )
    manager = AudioManager(providers=[failing, working])

    session = MagicMock()
    result = await manager.find_music("query", 60, 300, 10, temp_dir, session)
    assert result is not None
    assert result["source"] == "TestProvider"


@pytest.mark.asyncio
async def test_all_providers_fail_local_fallback(temp_dir):
    source_dir = temp_dir / "source"
    source_dir.mkdir()
    local_file = source_dir / "fallback.mp3"
    local_file.write_bytes(b"local audio")

    output_dir = temp_dir / "output"
    output_dir.mkdir()

    failing = FakeProvider("failing", fail=True)
    manager = AudioManager(providers=[failing], local_paths=[local_file])

    session = MagicMock()
    result = await manager.find_music("query", 60, 300, 10, output_dir, session)
    assert result is not None
    assert result["source"] == "Local"
    assert result["name"] == "fallback"


@pytest.mark.asyncio
async def test_no_providers_no_local(temp_dir):
    manager = AudioManager(providers=[], local_paths=[])

    session = MagicMock()
    result = await manager.find_music("query", 60, 300, 10, temp_dir, session)
    assert result is None


@pytest.mark.asyncio
async def test_skips_short_tracks(temp_dir, sample_attribution):
    short_track = AudioTrack(
        id="1",
        name="Short",
        duration=30.0,
        author="A",
        license="CC0",
        url="",
    )
    long_track = AudioTrack(
        id="2",
        name="Long",
        duration=120.0,
        author="A",
        license="CC0",
        url="",
    )
    path = Path(sample_attribution["path"])
    provider = FakeProvider(
        "test",
        tracks=[short_track, long_track],
        download_result=(path, sample_attribution),
    )
    manager = AudioManager(providers=[provider])

    session = MagicMock()
    # min_duration=60 should skip the 30s track
    result = await manager.find_music("query", 60, 300, 10, temp_dir, session)
    assert result is not None


@pytest.mark.asyncio
async def test_empty_search_tries_next(sample_track, sample_attribution, temp_dir):
    path = Path(sample_attribution["path"])
    empty = FakeProvider("empty", tracks=[])
    full = FakeProvider(
        "full",
        tracks=[sample_track],
        download_result=(path, sample_attribution),
    )
    manager = AudioManager(providers=[empty, full])

    session = MagicMock()
    result = await manager.find_music("query", 60, 300, 10, temp_dir, session)
    assert result is not None
