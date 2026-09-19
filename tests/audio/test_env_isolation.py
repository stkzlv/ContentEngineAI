"""No test writes the developer's `.env`.

`test_oauth2_token_refresh_success` drove the real refresh path with a
mocked response carrying the literal `new_refresh_token` and left the env
writer unpatched, so every suite run from 2026-01-12 replaced the working
Freesound refresh token in the repository's `.env` with that literal.
Freesound rotates refresh tokens on use, so the real one could not be
recovered. The writer now resolves its file through `env_file_path`, and
an autouse fixture points it at a temporary file for every test.
"""

from __future__ import annotations

from pathlib import Path

import aiohttp
import pytest

from src.audio import freesound_client
from src.audio.freesound_client import FreesoundClient, update_env_file
from src.utils.outputs_paths import get_project_root

REAL_ENV = get_project_root() / ".env"

REFRESH_URL = "https://freesound.org/apiv2/oauth2/access_token/"


def _snapshot() -> bytes | None:
    return REAL_ENV.read_bytes() if REAL_ENV.exists() else None


def _client() -> FreesoundClient:
    return FreesoundClient(
        api_key="k",
        FREESOUND_CLIENT_ID="cid",
        FREESOUND_CLIENT_SECRET="secret",  # noqa: S106
        FREESOUND_REFRESH_TOKEN="old_refresh_token",  # noqa: S106
    )


class TestTheWriterIsPointedAway:
    def test_every_test_sees_a_temporary_env_file(self, tmp_path: Path):
        assert freesound_client.env_file_path() == tmp_path / ".env"
        assert freesound_client.env_file_path() != REAL_ENV

    def test_a_direct_write_lands_in_the_temporary_file(self, tmp_path: Path):
        before = _snapshot()
        (tmp_path / ".env").write_text("FREESOUND_REFRESH_TOKEN=old\n")

        update_env_file("FREESOUND_REFRESH_TOKEN", "rotated")

        assert (tmp_path / ".env").read_text().strip() == (
            "FREESOUND_REFRESH_TOKEN=rotated"
        )
        assert _snapshot() == before


class TestTheRefreshPathCannotReachTheRealFile:
    """The exact scenario that destroyed the token, replayed."""

    @pytest.mark.asyncio
    async def test_a_rotated_token_goes_to_the_temporary_file_only(
        self, tmp_path: Path, mock_aioresponses
    ):
        before = _snapshot()
        (tmp_path / ".env").write_text("FREESOUND_REFRESH_TOKEN=old_refresh_token\n")
        mock_aioresponses.post(
            REFRESH_URL,
            payload={
                "access_token": "at",
                "expires_in": 3600,
                "refresh_token": "new_refresh_token",
            },
            status=200,
        )

        async with aiohttp.ClientSession() as session:
            assert await _client()._refresh_oauth2_token(session) is True

        assert "new_refresh_token" in (tmp_path / ".env").read_text()
        assert _snapshot() == before, "a test wrote the developer's .env"
