# src/audio/freesound_client.py
import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Any

import aiohttp
import freesound  # type: ignore[import-untyped]
from dotenv import set_key

from src.utils import download_file as util_download_file
from src.utils import ensure_dirs_exist, sanitize_filename
from src.utils.circuit_breaker import freesound_circuit_breaker
from src.video.video_config import (
    FREESOUND_DEFAULT_DOWNLOAD_TIMEOUT_SEC,
    FREESOUND_DOWNLOAD_CHUNK_SIZE,
    FREESOUND_TOKEN_EXPIRY_SEC,
    FREESOUND_TOKEN_REFRESH_BUFFER_SEC,
)

logger = logging.getLogger(__name__)


def update_env_file(key_to_update: str, new_value: str):
    """Safely updates a key in the project's .env file."""
    try:
        project_root = Path(__file__).resolve().parent.parent.parent
        env_path = project_root / ".env"
        if not env_path.is_file():
            logger.warning(
                f".env file not found at {env_path}. "
                f"Cannot update refresh token automatically."
            )
            return

        set_key(env_path, key_to_update, new_value)
        logger.info(f"Successfully updated '{key_to_update}' in {env_path}")
    except Exception as e:
        logger.error(f"Failed to automatically update .env file: {e}", exc_info=True)


class FreesoundClient:
    def __init__(self, **kwargs: str):
        self.fs_api_client = freesound.FreesoundClient()
        api_key = kwargs.get("FREESOUND_API_KEY")
        if api_key:
            self.fs_api_client.set_token(api_key, auth_type="token")
            logger.info("Freesound client configured with API key for search/previews.")
        else:
            logger.warning(
                "Freesound API key not provided; search/preview functionality will be "
                "limited."
            )

        self.oauth_client_id = kwargs.get("FREESOUND_CLIENT_ID")
        self.oauth_client_secret = kwargs.get("FREESOUND_CLIENT_SECRET")
        self.oauth_refresh_token = kwargs.get("FREESOUND_REFRESH_TOKEN")
        self.oauth_access_token: str | None = None
        self.oauth_token_expiry: float | None = None

    @freesound_circuit_breaker
    async def search_music(
        self,
        query: str,
        filters: str | None = None,
        max_results: int = None,
        sort_order: str = "rating_desc",
        fields: str = "id,name,previews,license,username,url,duration",
        timeout_sec: int = None,
    ) -> list:
        """Search for music tracks with timeout support.

        Args:
        ----
            query: Search query string
            filters: Optional filter string
            max_results: Maximum number of results to return
            sort_order: Sort order for results
            fields: Fields to include in response
            timeout_sec: Request timeout in seconds

        Returns:
        -------
            List of track objects, empty list on error

        """
        try:
            # Use asyncio.wait_for to implement timeout for sync operation
            results = await asyncio.wait_for(
                asyncio.to_thread(
                    self._search_sync,
                    query,
                    filters,
                    fields,
                    max_results,
                    sort_order,
                ),
                timeout=timeout_sec,
            )

            tracks = list(results)
            logger.info(
                f"Retrieved {len(tracks)} music tracks from Freesound for query: "
                f"'{query}' (timeout: {timeout_sec}s)"
            )
            return tracks
        except TimeoutError:
            logger.error(
                f"Freesound search timed out after {timeout_sec} seconds "
                f"for query: '{query}'"
            )
            return []
        except Exception as e:
            logger.error(f"Error searching Freesound: {e}", exc_info=True)
            return []

    def _search_sync(
        self,
        query: str,
        filters: str | None,
        fields: str,
        max_results: int,
        sort_order: str,
    ):
        """Synchronous helper method for the actual search."""
        return self.fs_api_client.text_search(
            query=query,
            filter=filters,
            fields=fields,
            page_size=max_results,
            sort=sort_order,
        )

    @freesound_circuit_breaker
    async def download_sound_preview_with_api_key(
        self,
        sound: Any,
        output_dir: Path,
        session: aiohttp.ClientSession,
        timeout_sec: int = None,
    ) -> tuple[Path, dict[str, Any]] | None:
        preview_url = None
        if hasattr(sound, "previews"):
            preview_url = getattr(sound.previews, "preview_hq_mp3", None) or getattr(
                sound.previews, "preview_lq_mp3", None
            )

        if not preview_url:
            logger.warning(f"No suitable MP3 preview available for sound: {sound.name}")
            return None

        file_path = output_dir / f"{sanitize_filename(sound.name)}.mp3"
        ensure_dirs_exist(file_path.parent)

        download_timeout = timeout_sec or FREESOUND_DEFAULT_DOWNLOAD_TIMEOUT_SEC
        if await util_download_file(
            preview_url, file_path, session, timeout_sec=download_timeout
        ):
            logger.info(f"Downloaded Freesound preview to: {file_path}")
            return file_path, {
                "source": "Freesound",
                "type": "Music",
                "author": sound.username,
                "url": sound.url,
                "license": sound.license,
                "name": sound.name,
                "id": str(sound.id),
                "path": str(file_path),
            }
        return None

    async def _refresh_oauth2_token(self, session: aiohttp.ClientSession) -> bool:
        if not all(
            [self.oauth_client_id, self.oauth_client_secret, self.oauth_refresh_token]
        ):
            logger.debug(
                "OAuth2 client credentials for Freesound not configured. "
                "Cannot refresh token."
            )
            return False

        logger.info("Refreshing Freesound OAuth2 access token...")
        payload = {
            "client_id": self.oauth_client_id,
            "client_secret": self.oauth_client_secret,
            "grant_type": "refresh_token",
            "refresh_token": self.oauth_refresh_token,
        }
        # Fast retry logic with exponential backoff
        max_retries = 2  # Reduced from 3 to fail faster
        for attempt in range(max_retries):
            try:
                # Fast timeout for OAuth requests
                timeout = aiohttp.ClientTimeout(total=5)  # 5 seconds max
                async with session.post(
                    "https://freesound.org/apiv2/oauth2/access_token/",
                    data=payload,
                    timeout=timeout,
                ) as response:
                    # Fast-fail on authentication errors
                    if response.status in (401, 403):
                        logger.error(f"OAuth authentication failed: {response.status}")
                        return False  # Don't retry auth failures

                    response.raise_for_status()
                    token_data = await response.json()
                    self.oauth_access_token = token_data["access_token"]
                    new_refresh_token = token_data.get("refresh_token")

                    if (
                        new_refresh_token
                        and new_refresh_token != self.oauth_refresh_token
                    ):
                        logger.info(
                            "New Freesound refresh token received. "
                            "Updating .env file..."
                        )
                        self.oauth_refresh_token = new_refresh_token
                        update_env_file("FREESOUND_REFRESH_TOKEN", new_refresh_token)

                    self.oauth_token_expiry = time.time() + token_data.get(
                        "expires_in", FREESOUND_TOKEN_EXPIRY_SEC
                    )
                    logger.info("Successfully refreshed Freesound OAuth2 access token.")
                    return True
            except (TimeoutError, aiohttp.ServerTimeoutError) as e:
                logger.warning(
                    f"OAuth token refresh timed out (attempt {attempt + 1}): {e}"
                )
                if attempt == max_retries - 1:
                    logger.error(
                        "OAuth token refresh failed due to timeouts - failing fast"
                    )
                    return False
                # Exponential backoff: 0.5s, 1s
                await asyncio.sleep(0.5 * (2**attempt))
            except (
                RuntimeError,
                aiohttp.ClientConnectorError,
                aiohttp.ClientResponseError,
            ) as e:
                if "Session is closed" in str(e) and attempt < max_retries - 1:
                    logger.warning(
                        f"Session closed error on attempt {attempt + 1}, "
                        f"getting new session"
                    )
                    from src.utils.connection_pool import get_http_session

                    session = await get_http_session()
                    continue
                elif attempt == max_retries - 1:
                    logger.error(
                        f"OAuth token refresh failed after {max_retries} attempts: {e}"
                    )
                    return False

        # If all retry attempts failed, return False
        logger.error("OAuth token refresh failed after all attempts")
        return False

    async def _get_valid_oauth2_token(
        self, session: aiohttp.ClientSession
    ) -> str | None:
        if (
            self.oauth_access_token
            and self.oauth_token_expiry
            and time.time()
            < self.oauth_token_expiry - FREESOUND_TOKEN_REFRESH_BUFFER_SEC
        ):
            return self.oauth_access_token
        if await self._refresh_oauth2_token(session):
            return self.oauth_access_token
        return None

    @freesound_circuit_breaker
    async def download_full_sound_oauth2(
        self,
        sound_id: int,
        output_dir: Path,
        session: aiohttp.ClientSession,
        timeout_sec: int = FREESOUND_DEFAULT_DOWNLOAD_TIMEOUT_SEC,
    ) -> tuple[Path, dict[str, Any]] | None:
        access_token = await self._get_valid_oauth2_token(session)
        if not access_token:
            logger.error(
                f"Cannot download full sound ID {sound_id}: No valid OAuth2 token."
            )
            return None

        download_url = f"https://freesound.org/apiv2/sounds/{sound_id}/download/"
        headers = {"Authorization": f"Bearer {access_token}"}

        # Fast retry logic with exponential backoff for downloads
        max_retries = 2  # Reduced from 3 to fail faster
        for attempt in range(max_retries):
            try:
                # Use the provided timeout_sec parameter
                timeout = aiohttp.ClientTimeout(total=timeout_sec)
                async with session.get(
                    download_url,
                    headers=headers,
                    allow_redirects=True,
                    timeout=timeout,
                ) as response:
                    # Fast-fail on client errors (not server errors which might be
                    # transient)
                    if response.status in (401, 403, 404):
                        logger.error(
                            f"Download failed with status {response.status} - "
                            f"not retrying"
                        )
                        return None

                    response.raise_for_status()
                    content_disposition = response.headers.get(
                        "Content-Disposition", ""
                    )
                    fn_match = re.search(r'filename="?([^"]+)"?', content_disposition)
                    filename = (
                        sanitize_filename(fn_match.group(1))
                        if fn_match
                        else f"freesound_{sound_id}.wav"
                    )

                    file_path = output_dir / filename
                    ensure_dirs_exist(file_path.parent)

                    with open(file_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(
                            FREESOUND_DOWNLOAD_CHUNK_SIZE
                        ):
                            f.write(chunk)

                    if file_path.exists() and file_path.stat().st_size > 0:
                        logger.info(
                            f"Successfully downloaded full sound to: {file_path}"
                        )
                        sound_details = self.fs_api_client.get_sound(
                            sound_id, fields="name,username,license,url"
                        )
                        return file_path, {
                            "source": "Freesound",
                            "path": str(file_path),
                            "name": sound_details.get("name", f"Sound {sound_id}"),
                            "author": sound_details.get("username", "Unknown"),
                            "license": sound_details.get("license", "Unknown"),
                            "url": sound_details.get(
                                "url", f"https://freesound.org/s/{sound_id}/"
                            ),
                        }
                    else:
                        logger.error(
                            f"Downloaded file is empty or missing: {file_path}"
                        )
                        return None
            except (TimeoutError, aiohttp.ServerTimeoutError) as e:
                logger.warning(f"Download timed out (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    logger.error(
                        f"Download failed due to timeouts after {max_retries} attempts"
                    )
                    return None
                # Exponential backoff: 1s, 2s
                await asyncio.sleep(1.0 * (2**attempt))
            except (
                RuntimeError,
                aiohttp.ClientConnectorError,
                aiohttp.ClientResponseError,
            ) as e:
                if "Session is closed" in str(e) and attempt < max_retries - 1:
                    logger.warning(
                        f"Session closed error on download attempt {attempt + 1}, "
                        f"getting new session"
                    )
                    from src.utils.connection_pool import get_http_session

                    session = await get_http_session()
                    continue
                elif attempt == max_retries - 1:
                    logger.error(f"Download failed after {max_retries} retries: {e}")
                    return None

        # If all retries failed, return None
        return None
