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
from src.utils.circuit_breaker import (
    _NETWORK_EXCEPTIONS,
    freesound_circuit_breaker,
)
from src.video.config import (
    FREESOUND_DOWNLOAD_CHUNK_SIZE,
    FREESOUND_TOKEN_EXPIRY_SEC,
    FREESOUND_TOKEN_REFRESH_BUFFER_SEC,
)

logger = logging.getLogger(__name__)


def update_env_file(key_to_update: str, new_value: str):
    """Safely updates an existing key in the project's .env file.

    Only overwrites if the key already exists — never adds new lines.
    """
    try:
        from src.utils.outputs_paths import get_project_root

        project_root = get_project_root()
        env_path = project_root / ".env"
        if not env_path.is_file():
            logger.warning(
                ".env file not found at %s. Cannot update refresh token automatically.",
                env_path,
            )
            return

        # Only update if key already exists in .env
        env_content = env_path.read_text()
        if f"{key_to_update}=" not in env_content:
            logger.warning(
                "Key '%s' not found in %s, skipping (won't add new entries)",
                key_to_update,
                env_path,
            )
            return

        set_key(env_path, key_to_update, new_value, quote_mode="never")
        logger.info("Successfully updated '%s' in %s", key_to_update, env_path)
    except Exception as e:
        logger.exception("Failed to automatically update .env file: %s", e)


class FreesoundClient:
    """Async client for Freesound.org API with OAuth2 authentication and resilience.

    Provides search, download, and authentication capabilities for Freesound API
    with circuit breaker protection, automatic token refresh, and graceful fallback
    strategies. Supports both API key authentication (search/preview) and OAuth2
    (full downloads).

    Attributes
    ----------
        fs_api_client : freesound.FreesoundClient
            Underlying Freesound API client wrapper.
        _api_key : str | None
            Freesound API key for search and preview downloads.
        oauth_client_id : str | None
            OAuth2 client ID for full-quality downloads.
        oauth_client_secret : str | None
            OAuth2 client secret for authentication.
        oauth_refresh_token : str | None
            OAuth2 refresh token for token renewal.
        oauth_access_token : str | None
            Current OAuth2 access token (auto-refreshed).
        oauth_token_expiry : float | None
            Unix timestamp when access token expires.

    """

    def __init__(self, config: Any | None = None, **kwargs: str) -> None:
        """Initialize FreesoundClient with API credentials and OAuth2 configuration.

        Args:
        ----
            config: VideoConfig object for accessing retry/backoff configuration
            **kwargs: Credential configuration parameters:
                - FREESOUND_API_KEY: API key for search/preview operations (required for
                  search)
                - FREESOUND_CLIENT_ID: OAuth2 client ID (required for full downloads)
                - FREESOUND_CLIENT_SECRET: OAuth2 client secret (required for full
                  downloads)
                - FREESOUND_REFRESH_TOKEN: OAuth2 refresh token (required for full
                  downloads)

        """
        self.config = config
        self.fs_api_client: freesound.FreesoundClient = freesound.FreesoundClient()
        self._api_key: str | None = kwargs.get("FREESOUND_API_KEY")

        if self._api_key:
            self.fs_api_client.set_token(self._api_key, auth_type="token")
            logger.info("Freesound client configured with API key for search/previews.")
        else:
            logger.warning(
                "Freesound API key not provided; search/preview functionality will be "
                "limited."
            )

        self.oauth_client_id: str | None = kwargs.get("FREESOUND_CLIENT_ID")
        self.oauth_client_secret: str | None = kwargs.get("FREESOUND_CLIENT_SECRET")
        self.oauth_refresh_token: str | None = kwargs.get("FREESOUND_REFRESH_TOKEN")
        self.oauth_access_token: str | None = None
        self.oauth_token_expiry: float | None = None

        oauth_configured = all(
            [self.oauth_client_id, self.oauth_client_secret, self.oauth_refresh_token]
        )
        if oauth_configured:
            logger.debug("OAuth2 credentials detected for full-quality downloads.")
        else:
            logger.debug(
                "OAuth2 credentials not fully configured; full downloads unavailable."
            )

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
        """Search for music tracks with circuit breaker protection and timeout handling.

        Searches Freesound API with configurable parameters, applies circuit breaker
        pattern for resilience, and gracefully degrades on errors by returning empty
        list. Protected by @freesound_circuit_breaker decorator which fast-fails when
        API repeatedly unavailable.

        Args:
        ----
            query: Search query string (e.g., "ambient music", "upbeat background")
            filters: Optional filter string (e.g., "duration:[60 TO 180]")
            max_results: Maximum number of results to return (default: API limit)
            sort_order: Sort order for results (e.g., "rating_desc", "duration_asc")
            fields: Comma-separated fields to include in response
            timeout_sec: Request timeout in seconds (default: 30s from config)

        Returns:
        -------
            List of track objects with requested fields, empty list on any failure
            (timeout, circuit breaker open, API error, network error)

        """
        logger.debug(
            "Searching Freesound: query='%s', filters='%s', max_results=%s, sort='%s', "
            "timeout=%ss",
            query,
            filters,
            max_results,
            sort_order,
            timeout_sec,
        )

        try:
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
                "Freesound search completed: %s tracks found (query='%s', timeout=%ss)",
                len(tracks),
                query,
                timeout_sec,
            )
            return tracks

        except TimeoutError:
            logger.warning(
                "Freesound search timed out after %ss - returning empty list "
                "(query='%s', filters='%s')",
                timeout_sec,
                query,
                filters,
            )
            return []

        except Exception as e:
            error_type = type(e).__name__
            logger.exception(
                "Freesound search failed with %s: %s - returning empty list "
                "(query='%s', filters='%s')",
                error_type,
                e,
                query,
                filters,
            )
            # Reraise exceptions that should trigger the circuit breaker
            if isinstance(e, _NETWORK_EXCEPTIONS):
                raise
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
        """Download preview-quality sound file using API key authentication.

        Downloads lower-quality MP3 preview files from Freesound using API key
        authentication (no OAuth2 required). Prioritizes high-quality preview, falls
        back to low-quality if unavailable. Returns attribution metadata for Creative
        Commons compliance.

        Args:
        ----
            sound: Freesound sound object with previews, metadata attributes
            output_dir: Directory to save downloaded preview file
            session: Active aiohttp ClientSession for HTTP requests
            timeout_sec: Download timeout in seconds (default: 300s from config)

        Returns:
        -------
            Tuple of (file_path, attribution_metadata) on success, None on any failure.
            Attribution metadata includes: source, type, path, name, author, license,
            url, id

        """
        if not self._api_key:
            logger.warning(
                "API key not configured - cannot download preview (sound: %s)",
                getattr(sound, "name", "unknown"),
            )
            return None

        preview_url = None
        quality = None
        if hasattr(sound, "previews"):
            preview_url = getattr(sound.previews, "preview_hq_mp3", None)
            if preview_url:
                quality = "HQ"
            else:
                preview_url = getattr(sound.previews, "preview_lq_mp3", None)
                if preview_url:
                    quality = "LQ"

        if not preview_url:
            sound_name = getattr(sound, "name", "unknown")
            sound_id = getattr(sound, "id", "unknown")
            logger.warning(
                "No MP3 preview available for sound '%s' (id: %s) - missing both HQ "
                "and LQ preview URLs",
                sound_name,
                sound_id,
            )
            return None

        sound_name = getattr(sound, "name", f"sound_{getattr(sound, 'id', 'unknown')}")
        file_path = output_dir / f"{sanitize_filename(sound_name)}.mp3"
        ensure_dirs_exist(file_path.parent)

        download_timeout = timeout_sec or 60  # Default download timeout in seconds
        logger.debug(
            "Downloading %s preview for '%s' (timeout: %ss)",
            quality,
            sound_name,
            download_timeout,
        )

        if await util_download_file(
            preview_url, file_path, session, timeout_sec=download_timeout
        ):
            file_size_mb = file_path.stat().st_size / 1024 / 1024
            logger.info(
                "Preview download complete: %s (%.2f MB, %s quality)",
                file_path.name,
                file_size_mb,
                quality,
            )

            attribution = {
                "source": "Freesound",
                "type": "Music",
                "path": str(file_path),
                "name": sound_name,
                "author": getattr(sound, "username", "Unknown"),
                "license": getattr(sound, "license", "Unknown"),
                "url": getattr(
                    sound, "url", f"https://freesound.org/s/{getattr(sound, 'id', '')}/"
                ),
                "id": str(getattr(sound, "id", "unknown")),
            }

            if (
                attribution["author"] == "Unknown"
                or attribution["license"] == "Unknown"
            ):
                logger.warning(
                    "Attribution metadata incomplete for sound '%s' (id: %s) - using "
                    "defaults",
                    sound_name,
                    attribution["id"],
                )

            return file_path, attribution

        logger.error(
            "Preview download failed for '%s' (id: %s)",
            sound_name,
            getattr(sound, "id", "unknown"),
        )
        return None

    async def _refresh_oauth2_token(self, session: aiohttp.ClientSession) -> bool:
        """Refresh OAuth2 access token with retry logic and exponential backoff.

        Attempts to refresh the access token using the refresh token grant. Implements
        fast-fail on authentication errors (401/403) and retries on transient failures
        (timeouts, network errors) with exponential backoff. Updates .env file when new
        refresh token is received.

        Retry Strategy:
            - Max attempts: 2
            - Backoff schedule: 0.5s, 1s (exponential: 0.5 * 2^attempt)
            - Fast-fail: 401/403 authentication errors (no retry)
            - Timeout: 5s per request

        Args:
        ----
            session: Active aiohttp ClientSession for HTTP requests

        Returns:
        -------
            True if token refresh succeeded, False on any failure

        """
        if not all(
            [self.oauth_client_id, self.oauth_client_secret, self.oauth_refresh_token]
        ):
            logger.debug(
                "OAuth2 credentials not configured - cannot refresh token "
                "(missing client_id, client_secret, or refresh_token)"
            )
            return False

        logger.info("Refreshing Freesound OAuth2 access token...")
        payload = {
            "client_id": self.oauth_client_id,
            "client_secret": self.oauth_client_secret,
            "grant_type": "refresh_token",
            "refresh_token": self.oauth_refresh_token,
        }

        # Token refresh retry defaults
        max_retries = 2
        timeout_sec = 5
        backoff_base = 0.5
        backoff_mult = 2.0

        for attempt in range(max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=timeout_sec)
                async with session.post(
                    "https://freesound.org/apiv2/oauth2/access_token/",
                    data=payload,
                    timeout=timeout,
                ) as response:
                    if response.status in (401, 403):
                        logger.error(
                            "OAuth2 authentication failed with status %s - invalid "
                            "credentials (not retrying)",
                            response.status,
                        )
                        return False

                    response.raise_for_status()
                    token_data = await response.json()

                    if "access_token" not in token_data:
                        logger.error(
                            "OAuth2 token response missing 'access_token' field - "
                            "invalid response structure"
                        )
                        return False

                    self.oauth_access_token = token_data["access_token"]
                    new_refresh_token = token_data.get("refresh_token")

                    if (
                        new_refresh_token
                        and new_refresh_token != self.oauth_refresh_token
                    ):
                        logger.info(
                            "New refresh token received - updating .env file..."
                        )
                        self.oauth_refresh_token = new_refresh_token
                        try:
                            update_env_file(
                                "FREESOUND_REFRESH_TOKEN", new_refresh_token
                            )
                        except Exception as env_error:
                            logger.warning(
                                "Failed to update .env file with new refresh token: %s "
                                "(token still valid in memory)",
                                env_error,
                            )

                    self.oauth_token_expiry = time.time() + token_data.get(
                        "expires_in", FREESOUND_TOKEN_EXPIRY_SEC
                    )
                    expires_in = token_data.get(
                        "expires_in", FREESOUND_TOKEN_EXPIRY_SEC
                    )
                    logger.info(
                        "OAuth2 token refreshed successfully (expires in %ss)",
                        expires_in,
                    )
                    return True

            except (TimeoutError, aiohttp.ServerTimeoutError):
                logger.warning(
                    "OAuth2 token refresh timed out on attempt %s/%s",
                    attempt + 1,
                    max_retries,
                )
                if attempt == max_retries - 1:
                    logger.error(
                        "OAuth2 token refresh failed - all attempts timed out after "
                        "%ss",
                        timeout_sec,
                    )
                    return False
                await asyncio.sleep(backoff_base * (backoff_mult**attempt))

            except aiohttp.ClientResponseError as e:
                logger.error(
                    "OAuth2 token refresh failed with HTTP %s: %s (attempt %s/%s)",
                    e.status,
                    e.message,
                    attempt + 1,
                    max_retries,
                )
                if attempt == max_retries - 1:
                    return False
                await asyncio.sleep(0.5 * (2**attempt))

            except aiohttp.ClientConnectorError as e:
                logger.warning(
                    "OAuth2 network connection failed: %s (attempt %s/%s)",
                    e,
                    attempt + 1,
                    max_retries,
                )
                if attempt == max_retries - 1:
                    logger.error("OAuth2 token refresh failed - network unreachable")
                    return False
                await asyncio.sleep(0.5 * (2**attempt))

            except RuntimeError as e:
                if "Session is closed" in str(e) and attempt < max_retries - 1:
                    logger.warning(
                        "Session closed on attempt %s - acquiring new session",
                        attempt + 1,
                    )
                    from src.utils.connection_pool import get_http_session

                    session = await get_http_session()
                    continue
                else:
                    logger.error(
                        "OAuth2 token refresh failed with runtime error: %s (attempt "
                        "%s/%s)",
                        e,
                        attempt + 1,
                        max_retries,
                    )
                    if attempt == max_retries - 1:
                        return False
                    await asyncio.sleep(0.5 * (2**attempt))

            except KeyError as e:
                logger.error(
                    "OAuth2 token response missing required field %s - invalid "
                    "response structure",
                    e,
                )
                return False

        logger.error("OAuth2 token refresh failed after %s attempts", max_retries)
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
        timeout_sec: int = 60,  # Default download timeout in seconds
    ) -> tuple[Path, dict[str, Any]] | None:
        """Download full-quality sound file using OAuth2 authentication.

        Downloads high-quality audio files from Freesound using OAuth2 bearer token
        authentication. Implements retry logic with exponential backoff for transient
        failures, automatic session recovery, and chunked streaming for large files.
        Returns complete attribution metadata for Creative Commons compliance.

        Args:
        ----
            sound_id: Freesound sound ID to download
            output_dir: Directory to save downloaded file
            session: Active aiohttp ClientSession for HTTP requests
            timeout_sec: Download timeout in seconds (default: 300s from config)

        Returns:
        -------
            Tuple of (file_path, attribution_metadata) on success, None on any failure.
            Attribution metadata includes: source, path, name, author, license, url

        """
        logger.debug(
            "Attempting OAuth2 download for sound ID %s (timeout: %ss)",
            sound_id,
            timeout_sec,
        )

        access_token = await self._get_valid_oauth2_token(session)
        if not access_token:
            logger.error(
                "Cannot download sound %s - OAuth2 token unavailable (credentials not "
                "configured or token refresh failed)",
                sound_id,
            )
            return None

        download_url = f"https://freesound.org/apiv2/sounds/{sound_id}/download/"
        headers = {"Authorization": f"Bearer {access_token}"}

        # Download retry defaults
        max_retries = 2
        backoff_base = 1.0
        backoff_mult = 2.0

        for attempt in range(max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=timeout_sec)
                async with session.get(
                    download_url,
                    headers=headers,
                    allow_redirects=True,
                    timeout=timeout,
                ) as response:
                    if response.status in (401, 403, 404):
                        logger.error(
                            "OAuth2 download failed with status %s for sound %s (not "
                            "retrying)",
                            response.status,
                            sound_id,
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

                    content_length = response.headers.get("Content-Length")
                    if content_length:
                        logger.debug(
                            "Downloading sound %s: %s (%.2f MB)",
                            sound_id,
                            filename,
                            int(content_length) / 1024 / 1024,
                        )

                    bytes_downloaded = 0
                    with open(file_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(
                            FREESOUND_DOWNLOAD_CHUNK_SIZE
                        ):
                            f.write(chunk)
                            bytes_downloaded += len(chunk)

                    if file_path.exists() and file_path.stat().st_size > 0:
                        logger.info(
                            "OAuth2 download complete: %s (%.2f MB)",
                            filename,
                            bytes_downloaded / 1024 / 1024,
                        )

                        try:
                            sound_details = self.fs_api_client.get_sound(
                                sound_id, fields="name,username,license,url"
                            )
                            attribution = {
                                "source": "Freesound",
                                "type": "Music",
                                "path": str(file_path),
                                "name": sound_details.get("name", f"Sound {sound_id}"),
                                "author": sound_details.get("username", "Unknown"),
                                "license": sound_details.get("license", "Unknown"),
                                "url": sound_details.get(
                                    "url", f"https://freesound.org/s/{sound_id}/"
                                ),
                                "id": str(sound_id),
                            }

                            if attribution["name"] == f"Sound {sound_id}":
                                logger.warning(
                                    "Attribution metadata incomplete for sound %s - "
                                    "using defaults",
                                    sound_id,
                                )

                            return file_path, attribution

                        except Exception as metadata_error:
                            logger.warning(
                                "Failed to fetch metadata for sound %s: %s - using "
                                "minimal attribution",
                                sound_id,
                                metadata_error,
                            )
                            return file_path, {
                                "source": "Freesound",
                                "type": "Music",
                                "path": str(file_path),
                                "name": f"Sound {sound_id}",
                                "author": "Unknown",
                                "license": "Unknown",
                                "url": f"https://freesound.org/s/{sound_id}/",
                                "id": str(sound_id),
                            }
                    else:
                        logger.error(
                            "Downloaded file empty or missing: %s (sound_id: %s)",
                            file_path,
                            sound_id,
                        )
                        return None

            except (TimeoutError, aiohttp.ServerTimeoutError):
                logger.warning(
                    "OAuth2 download timed out after %ss (attempt %s/%s, sound_id: %s)",
                    timeout_sec,
                    attempt + 1,
                    max_retries,
                    sound_id,
                )
                if attempt == max_retries - 1:
                    logger.error(
                        "OAuth2 download failed - all attempts timed out (sound_id: "
                        "%s)",
                        sound_id,
                    )
                    return None
                await asyncio.sleep(backoff_base * (backoff_mult**attempt))

            except aiohttp.ClientConnectorError as e:
                logger.warning(
                    "OAuth2 download network error: %s (attempt %s/%s, sound_id: %s)",
                    e,
                    attempt + 1,
                    max_retries,
                    sound_id,
                )
                if attempt == max_retries - 1:
                    logger.error(
                        "OAuth2 download failed - network unreachable (sound_id: %s)",
                        sound_id,
                    )
                    return None
                await asyncio.sleep(backoff_base * (backoff_mult**attempt))

            except aiohttp.ClientResponseError as e:
                logger.error(
                    "OAuth2 download HTTP error %s: %s (attempt %s/%s, sound_id: %s)",
                    e.status,
                    e.message,
                    attempt + 1,
                    max_retries,
                    sound_id,
                )
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(backoff_base * (backoff_mult**attempt))

            except RuntimeError as e:
                if "Session is closed" in str(e) and attempt < max_retries - 1:
                    logger.warning(
                        "Session closed on attempt %s - acquiring new session "
                        "(sound_id: %s)",
                        attempt + 1,
                        sound_id,
                    )
                    from src.utils.connection_pool import get_http_session

                    session = await get_http_session()
                    continue
                else:
                    logger.error(
                        "OAuth2 download runtime error: %s (attempt %s/%s, sound_id: "
                        "%s)",
                        e,
                        attempt + 1,
                        max_retries,
                        sound_id,
                    )
                    if attempt == max_retries - 1:
                        return None
                    await asyncio.sleep(1.0 * (2**attempt))

        logger.error(
            "OAuth2 download failed after %s attempts (sound_id: %s)",
            max_retries,
            sound_id,
        )
        return None
