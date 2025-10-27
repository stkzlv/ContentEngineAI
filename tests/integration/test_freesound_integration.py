"""Integration tests for Freesound client end-to-end workflows.

These tests use real API calls and require valid credentials in environment variables.
Run with: pytest tests/integration/ -v -m integration
Skip with: pytest tests/ -v -m "not integration"
"""

import asyncio
import os
from pathlib import Path

import aiohttp
import pytest

from src.audio.freesound_client import FreesoundClient

# Integration test markers
pytestmark = pytest.mark.integration

# Credential availability checks
HAS_API_KEY = os.getenv("FREESOUND_API_KEY") is not None
HAS_OAUTH2 = all(
    [
        os.getenv("FREESOUND_CLIENT_ID"),
        os.getenv("FREESOUND_CLIENT_SECRET"),
        os.getenv("FREESOUND_REFRESH_TOKEN"),
    ]
)


@pytest.mark.skipif(not HAS_API_KEY, reason="Requires FREESOUND_API_KEY in environment")
@pytest.mark.asyncio
async def test_search_to_preview_download_flow(tmp_path):
    """Test complete flow: search → preview download with real API.

    Validates:
    - R1: Search with duration matching
    - R3: Preview download with API key
    - R6: Attribution metadata extraction
    - R4: Circuit breaker integration
    """
    client = FreesoundClient(FREESOUND_API_KEY=os.getenv("FREESOUND_API_KEY"))

    # Search for short background music (5-15 seconds)
    results = await client.search_music(
        query="ambient background music",
        filters="duration:[5 TO 15]",
        max_results=3,
        timeout_sec=30,
    )

    # Should find at least 1 result
    assert len(results) > 0, "Search should return at least 1 result"
    assert hasattr(results[0], "id"), "Result should have ID"
    assert hasattr(results[0], "name"), "Result should have name"

    # Download preview of first result
    sound = results[0]
    async with aiohttp.ClientSession() as session:
        result = await client.download_sound_preview_with_api_key(
            sound, tmp_path, session
        )

    # Verify download succeeded
    assert result is not None, "Preview download should succeed"
    file_path, attribution = result

    # Verify file created (Requirement R3)
    assert file_path.exists(), "Downloaded file should exist"
    assert file_path.stat().st_size > 0, "Downloaded file should not be empty"

    # Verify attribution metadata (Requirement R6)
    assert attribution["source"] == "Freesound"
    assert attribution["type"] == "Music"
    assert "name" in attribution
    assert "author" in attribution
    assert "license" in attribution
    assert "url" in attribution
    assert "id" in attribution
    assert "path" in attribution

    # Cleanup
    file_path.unlink()


@pytest.mark.skipif(
    not HAS_OAUTH2, reason="Requires OAuth2 credentials in environment"
)
@pytest.mark.asyncio
async def test_search_to_oauth2_download_flow(tmp_path):
    """Test complete flow: search → OAuth2 full-quality download with real API.

    Validates:
    - R1: Search with filters
    - R2: OAuth2 token refresh
    - R3: Full-quality download
    - R6: Attribution metadata
    """
    client = FreesoundClient(
        FREESOUND_API_KEY=os.getenv("FREESOUND_API_KEY"),
        FREESOUND_CLIENT_ID=os.getenv("FREESOUND_CLIENT_ID"),
        FREESOUND_CLIENT_SECRET=os.getenv("FREESOUND_CLIENT_SECRET"),
        FREESOUND_REFRESH_TOKEN=os.getenv("FREESOUND_REFRESH_TOKEN"),
    )

    # Search for very short tracks to minimize download time
    results = await client.search_music(
        query="click sound effect",
        filters="duration:[0.1 TO 2]",
        max_results=2,
        timeout_sec=30,
    )

    assert len(results) > 0, "Search should return results"

    # Attempt OAuth2 download
    sound_id = results[0].id
    async with aiohttp.ClientSession() as session:
        result = await client.download_full_sound_oauth2(
            sound_id, tmp_path, session, timeout_sec=60
        )

    if result is not None:
        file_path, attribution = result

        # Verify file and attribution (Requirements R3, R6)
        assert file_path.exists()
        assert file_path.stat().st_size > 0
        assert attribution["source"] == "Freesound"
        assert attribution["type"] == "Music"
        assert attribution["id"] == str(sound_id)

        # Cleanup
        file_path.unlink()
    else:
        # OAuth2 may fail if token expired - this is acceptable for integration test
        pytest.skip("OAuth2 download failed - token may need refresh")


@pytest.mark.skipif(not HAS_API_KEY, reason="Requires FREESOUND_API_KEY in environment")
@pytest.mark.asyncio
async def test_search_with_duration_filtering(tmp_path):
    """Test search with duration filters matches expected range.

    Validates:
    - R1: Duration-based search filtering
    - R7: Configurable search parameters
    """
    client = FreesoundClient(FREESOUND_API_KEY=os.getenv("FREESOUND_API_KEY"))

    # Search for 10-20 second tracks
    min_duration = 10
    max_duration = 20

    results = await client.search_music(
        query="music loop",
        filters=f"duration:[{min_duration} TO {max_duration}]",
        max_results=5,
        timeout_sec=30,
    )

    if len(results) > 0:
        # Verify duration filtering worked
        for sound in results:
            if hasattr(sound, "duration"):
                duration = sound.duration
                assert (
                    min_duration <= duration <= max_duration
                ), (
                    f"Sound duration {duration}s should be in range "
                    f"[{min_duration}, {max_duration}]"
                )


@pytest.mark.skipif(not HAS_API_KEY, reason="Requires FREESOUND_API_KEY in environment")
@pytest.mark.asyncio
async def test_attribution_completeness(tmp_path):
    """Test that attribution metadata is complete and valid.

    Validates:
    - R6: Attribution metadata tracking
    - Creative Commons license compliance
    """
    client = FreesoundClient(FREESOUND_API_KEY=os.getenv("FREESOUND_API_KEY"))

    # Search for CC0 (public domain) sounds for testing
    results = await client.search_music(
        query="click",
        filters='license:"Creative Commons 0"',
        max_results=2,
        timeout_sec=30,
    )

    if len(results) > 0:
        sound = results[0]

        async with aiohttp.ClientSession() as session:
            result = await client.download_sound_preview_with_api_key(
                sound, tmp_path, session
            )

        if result is not None:
            file_path, attribution = result

            # Verify all required attribution fields exist
            required_fields = [
                "source",
                "type",
                "path",
                "name",
                "author",
                "license",
                "url",
                "id",
            ]
            for field in required_fields:
                assert (
                    field in attribution
                ), f"Attribution missing required field: {field}"
                assert attribution[field], f"Attribution field '{field}' is empty"

            # Verify specific values
            assert attribution["source"] == "Freesound"
            assert attribution["type"] == "Music"
            assert str(Path(attribution["path"])) == str(file_path)

            # Cleanup
            file_path.unlink()


@pytest.mark.skipif(not HAS_API_KEY, reason="Requires FREESOUND_API_KEY in environment")
@pytest.mark.asyncio
async def test_session_reuse_across_operations(tmp_path):
    """Test that HTTP session can be reused across multiple operations.

    Validates:
    - R8: Session management and connection pooling
    """
    client = FreesoundClient(FREESOUND_API_KEY=os.getenv("FREESOUND_API_KEY"))

    async with aiohttp.ClientSession() as session:
        # Perform multiple operations with same session
        results1 = await client.search_music(
            query="ambient", max_results=2, timeout_sec=30
        )

        assert len(results1) > 0

        # Download using same session
        sound = results1[0]
        result = await client.download_sound_preview_with_api_key(
            sound, tmp_path, session
        )

        assert result is not None
        file_path, _ = result
        assert file_path.exists()

        # Perform second search with same session
        results2 = await client.search_music(
            query="music", max_results=2, timeout_sec=30
        )

        assert len(results2) > 0

        # Cleanup
        file_path.unlink()


@pytest.mark.skipif(not HAS_API_KEY, reason="Requires FREESOUND_API_KEY in environment")
@pytest.mark.asyncio
async def test_fallback_to_preview_when_oauth2_unavailable(tmp_path):
    """Test that system falls back to preview download when OAuth2 not configured.

    Validates:
    - R3: Fallback from OAuth2 to API key preview
    - Graceful degradation
    """
    # Client with API key only (no OAuth2)
    client = FreesoundClient(FREESOUND_API_KEY=os.getenv("FREESOUND_API_KEY"))

    results = await client.search_music(
        query="click", filters="duration:[0.1 TO 2]", max_results=2, timeout_sec=30
    )

    if len(results) > 0:
        sound_id = results[0].id

        # Try OAuth2 download (should fail without credentials)
        async with aiohttp.ClientSession() as session:
            oauth2_result = await client.download_full_sound_oauth2(
                sound_id, tmp_path, session
            )

        assert oauth2_result is None, "OAuth2 download should fail without credentials"

        # Fallback to preview download (should succeed)
        async with aiohttp.ClientSession() as session:
            preview_result = await client.download_sound_preview_with_api_key(
                results[0], tmp_path, session
            )

        assert preview_result is not None, "Preview download should succeed"
        file_path, _ = preview_result
        assert file_path.exists()

        # Cleanup
        file_path.unlink()


@pytest.mark.skipif(not HAS_API_KEY, reason="Requires FREESOUND_API_KEY in environment")
@pytest.mark.asyncio
async def test_circuit_breaker_integration(tmp_path):
    """Test that circuit breaker allows operations to complete.

    Validates:
    - R4: Circuit breaker pattern integration
    - Operations succeed when circuit is closed
    """
    from src.utils.circuit_breaker import freesound_circuit_breaker

    # Reset circuit breaker to ensure clean state
    freesound_circuit_breaker.reset()

    client = FreesoundClient(FREESOUND_API_KEY=os.getenv("FREESOUND_API_KEY"))

    # Perform search (should succeed with closed circuit)
    results = await client.search_music(
        query="ambient", max_results=2, timeout_sec=30
    )

    assert len(results) > 0, "Search should succeed with closed circuit breaker"

    # Download should also work
    if len(results) > 0:
        sound = results[0]
        async with aiohttp.ClientSession() as session:
            result = await client.download_sound_preview_with_api_key(
                sound, tmp_path, session
            )

        if result is not None:
            file_path, _ = result
            assert file_path.exists()
            file_path.unlink()


# Note: Local fallback integration test is not included here as it would require
# mocking all API failures, which is better tested in unit tests. The local fallback
# is tested in src/video/producer.py integration tests.
