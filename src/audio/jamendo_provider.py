"""Jamendo Music API provider (https://developer.jamendo.com/v3.0)."""

import asyncio
import logging
import random
from pathlib import Path
from typing import Any

import aiohttp

from src.utils import ensure_dirs_exist, sanitize_filename
from src.utils.circuit_breaker import CircuitBreaker

from .base import AudioTrack, BaseAudioProvider
from .registry import AudioProvider, register_audio_provider

logger = logging.getLogger(__name__)

JAMENDO_API_BASE = "https://api.jamendo.com/v3.0"
JAMENDO_MAX_RESULTS = 200  # Jamendo API hard limit
# Extra attempts after an empty-but-successful response. Measured at
# roughly one call in three coming back empty for identical input, so two
# retries put the odds of three consecutive misses near one in thirty.
JAMENDO_EMPTY_RETRIES = 2

# Separate circuit breaker for Jamendo
jamendo_circuit_breaker = CircuitBreaker(
    name="jamendo",
    failure_threshold=3,
    timeout=60,
)


@register_audio_provider(AudioProvider.JAMENDO)
class JamendoProvider(BaseAudioProvider):
    """Jamendo Music API provider with Creative Commons licensed tracks."""

    def __init__(
        self,
        secrets: dict[str, str] | None = None,
        settings: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        secrets = secrets or {}
        settings = settings or {}
        env_var = settings.get("client_id_env_var", "JAMENDO_CLIENT_ID")
        self._client_id: str | None = secrets.get(env_var)
        self._timeout_sec: int = settings.get("api_timeout_sec", 15)
        self._download_timeout_sec: int = settings.get("download_timeout_sec", 60)
        # Search mode: "fuzzytags" (default, OR match with relevance),
        # "tags" (AND match), or "search" (free text)
        self._search_mode: str = settings.get("search_mode", "fuzzytags")
        # Optional list of queries; one picked randomly per search for variety
        self._search_queries: list[str] = settings.get("search_queries", [])

        if self._client_id:
            logger.debug("Jamendo provider configured with client_id")
        else:
            logger.debug("Jamendo client_id not found in env var '%s'", env_var)

    @property
    def provider_name(self) -> str:
        return "jamendo"

    async def search(
        self,
        query: str,
        min_duration: float,
        max_duration: float,
        max_results: int,
        session: aiohttp.ClientSession,
    ) -> list[AudioTrack]:
        """Search Jamendo, retrying an empty-but-successful response.

        The API intermittently answers a working query with zero results --
        measured at roughly one call in three for identical input. Treating
        that as "no tracks" dropped Jamendo for the whole render and fell
        through to the next provider, which silently changes the audio quality
        of a published video.
        """
        if not self._client_id:
            logger.debug("Jamendo client_id not configured, skipping")
            return []

        if jamendo_circuit_breaker.is_open:
            logger.warning("Jamendo circuit breaker is open, skipping")
            return []

        for attempt in range(1, JAMENDO_EMPTY_RETRIES + 2):
            # Re-drawn per attempt. The emptiness is not query-specific, so a
            # different query is a second sample rather than a second guess.
            attempt_query = query
            if self._search_queries:
                attempt_query = random.choice(self._search_queries)  # noqa: S311
                logger.info("Jamendo query (random): '%s'", attempt_query)

            tracks = await self._search_once(
                attempt_query,
                min_duration,
                max_duration,
                max_results,
                session,
            )
            if tracks is None:
                # A real failure: HTTP error, API error, or transport. Already
                # logged and recorded against the circuit breaker.
                return []
            if tracks:
                return tracks

            logger.info(
                "Jamendo returned no tracks for '%s' (attempt %d/%d)",
                attempt_query,
                attempt,
                JAMENDO_EMPTY_RETRIES + 1,
            )

        # WARNING, not INFO: a configured primary provider yielding nothing
        # means the chain falls through to a provider that may only offer
        # preview-quality audio, and that downgrade should be greppable.
        logger.warning(
            "Jamendo yielded no tracks after %d attempts; the chain will fall "
            "through to the next provider, which may downgrade audio quality",
            JAMENDO_EMPTY_RETRIES + 1,
        )
        return []

    async def _search_once(
        self,
        query: str,
        min_duration: float,
        max_duration: float,
        max_results: int,
        session: aiohttp.ClientSession,
    ) -> list[AudioTrack] | None:
        """One search request.

        Returns the tracks, an empty list when the API answered with none, or
        `None` when the request itself failed -- which the caller must not
        retry, because the circuit breaker has already recorded it.
        """
        # `search` returns before calling this when the id is unset, so the
        # narrowing is real; it just does not survive the split.
        client_id = self._client_id
        if not client_id:
            return None

        # Convert space-separated query to + delimited for tags/fuzzytags
        tag_query = query.replace(" ", "+")

        params: dict[str, str] = {
            "client_id": client_id,
            "format": "json",
            # `durationbetween`, no underscore. The API ignores unknown
            # parameters rather than rejecting them, so the underscored
            # spelling this used to send simply never filtered: an otherwise
            # identical query returned tracks up to 28 minutes long, and every
            # one of 20 results sat outside the requested window.
            "durationbetween": f"{int(min_duration)}_{int(max_duration)}",
            "vocalinstrumental": "instrumental",
            "order": "popularity_month_desc",
            "limit": str(min(max_results, JAMENDO_MAX_RESULTS)),
            "audiodlformat": "mp32",
            "include": "musicinfo",
        }

        if self._search_mode == "tags":
            params["tags"] = tag_query
        elif self._search_mode == "search":
            params["search"] = query
        else:
            params["fuzzytags"] = tag_query

        try:
            timeout = aiohttp.ClientTimeout(total=self._timeout_sec)
            async with session.get(  # type: ignore[attr-defined]
                f"{JAMENDO_API_BASE}/tracks/",
                params=params,
                timeout=timeout,
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning(
                        "Jamendo search returned %d: %s",
                        resp.status,
                        body[:200],
                    )
                    jamendo_circuit_breaker.record_failure(Exception("API error"))
                    return None

                data = await resp.json()

                # Jamendo returns 200 with error status for bad requests
                api_status = data.get("headers", {}).get("status")
                if api_status == "error":
                    error_msg = data.get("headers", {}).get("error_message", "")
                    logger.warning("Jamendo API error: %s", error_msg)
                    jamendo_circuit_breaker.record_failure(Exception(error_msg))
                    return None

                jamendo_circuit_breaker.record_success()

        except (TimeoutError, aiohttp.ClientError) as exc:
            logger.warning("Jamendo search failed: %s", exc)
            jamendo_circuit_breaker.record_failure(Exception("API error"))
            return None

        results = data.get("results", [])
        logger.info("Jamendo search: %d tracks (query='%s')", len(results), query)

        return [
            AudioTrack(
                id=str(r["id"]),
                name=r.get("name", "Unknown"),
                duration=float(r.get("duration", 0)),
                author=r.get("artist_name", "Unknown"),
                license=r.get("license_ccurl", "Creative Commons"),
                url=r.get("shareurl", ""),
                provider_data=r,
            )
            for r in results
        ]

    async def download(
        self,
        track: AudioTrack,
        output_dir: Path,
        session: aiohttp.ClientSession,
    ) -> tuple[Path, dict[str, Any]] | None:
        raw = track.provider_data or {}

        # Prefer audiodownload if allowed, fall back to stream URL
        download_url = None
        if raw.get("audiodownload_allowed", False) and raw.get("audiodownload"):
            download_url = raw["audiodownload"]
        elif raw.get("audio"):
            download_url = raw["audio"]

        if not download_url:
            logger.warning("No download URL for Jamendo track '%s'", track.name)
            return None

        ensure_dirs_exist(output_dir)
        filename = f"{sanitize_filename(track.name)}.mp3"
        file_path = output_dir / filename

        try:
            proc = await asyncio.create_subprocess_exec(
                "curl",
                "-sS",
                "--max-time",
                str(self._download_timeout_sec),
                "-o",
                str(file_path),
                download_url,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            rc = await proc.wait()

            if rc != 0:
                logger.warning(
                    "Jamendo download failed for '%s' (rc=%d)",
                    track.name,
                    rc,
                )
                if file_path.exists():
                    file_path.unlink()
                return None

        except (OSError, ValueError) as exc:
            logger.warning("Jamendo download failed for '%s': %s", track.name, exc)
            if file_path.exists():
                file_path.unlink()
            return None

        if not file_path.exists() or file_path.stat().st_size == 0:
            logger.warning("Jamendo download empty for '%s'", track.name)
            return None

        size_mb = file_path.stat().st_size / 1024 / 1024
        logger.info("Jamendo download: %s (%.2f MB)", filename, size_mb)

        attribution = {
            "source": "Jamendo",
            "type": "Music",
            "path": str(file_path),
            "name": track.name,
            "author": track.author,
            "license": track.license,
            "url": track.url,
            "id": track.id,
        }
        return file_path, attribution
