"""Audio manager that orchestrates provider chain with local file fallback."""

import logging
import random
import shutil
import time
from pathlib import Path
from typing import Any

import aiohttp

from src.utils import ensure_dirs_exist
from src.utils.circuit_breaker import CircuitBreakerError

from .base import BaseAudioProvider

logger = logging.getLogger(__name__)


class AudioManager:
    """Try each configured provider in order, fall back to local files."""

    def __init__(
        self,
        providers: list[BaseAudioProvider],
        local_paths: list[Path] | None = None,
    ) -> None:
        self._providers = providers
        self._local_paths = local_paths or []

    async def find_music(
        self,
        query: str,
        min_duration: float,
        max_duration: float,
        max_results: int,
        output_dir: Path,
        session: aiohttp.ClientSession,
    ) -> dict[str, Any] | None:
        """Search providers in order, download first suitable track.

        Returns attribution dict or None if nothing found.
        """
        t0 = time.monotonic()
        tried_providers: list[str] = []

        for provider in self._providers:
            tried_providers.append(provider.provider_name)
            try:
                result = await self._try_provider(
                    provider,
                    query,
                    min_duration,
                    max_duration,
                    max_results,
                    output_dir,
                    session,
                )
                if result:
                    self._log_summary(
                        result, query, tried_providers, time.monotonic() - t0
                    )
                    return result
            except CircuitBreakerError:
                logger.warning(
                    "Circuit breaker open for %s, skipping",
                    provider.provider_name,
                )
            except (
                RuntimeError,
                OSError,
                TimeoutError,
                aiohttp.ClientError,
            ) as exc:
                logger.warning(
                    "Provider %s failed: %s",
                    provider.provider_name,
                    exc,
                )

        fallback = self._try_local_fallback(output_dir)
        if fallback:
            tried_providers.append("local")
        self._log_summary(fallback, query, tried_providers, time.monotonic() - t0)
        return fallback

    @staticmethod
    def _log_summary(
        result: dict[str, Any] | None,
        query: str,
        tried: list[str],
        elapsed: float,
    ) -> None:
        logger.info("--- AUDIO SUMMARY ---")
        if result:
            logger.info(
                "Provider: %s (query: %s)",
                result.get("source", "unknown"),
                query,
            )
            logger.info(
                "Track: %s by %s",
                result.get("name", "unknown"),
                result.get("author", "unknown"),
            )
        else:
            logger.info("Result: no track found (tried: %s)", ", ".join(tried))
        logger.info("Duration: %.1fs", elapsed)
        logger.info("---")

    async def _try_provider(
        self,
        provider: BaseAudioProvider,
        query: str,
        min_duration: float,
        max_duration: float,
        max_results: int,
        output_dir: Path,
        session: aiohttp.ClientSession,
    ) -> dict[str, Any] | None:
        tracks = await provider.search(
            query,
            min_duration,
            max_duration,
            max_results,
            session,
        )
        if not tracks:
            logger.info(
                "No tracks from %s, trying next provider",
                provider.provider_name,
            )
            return None

        eligible = [t for t in tracks if t.duration >= min_duration]
        if not eligible:
            logger.info(
                "No tracks from %s meet min duration %.0fs",
                provider.provider_name,
                min_duration,
            )
            return None
        random.shuffle(eligible)

        for track in eligible:
            logger.info(
                "Trying track '%s' (%.0fs) from %s",
                track.name,
                track.duration,
                provider.provider_name,
            )
            try:
                result = await provider.download(track, output_dir, session)
                if result:
                    _, attribution = result
                    return attribution
            except (RuntimeError, OSError, TimeoutError) as exc:
                logger.warning(
                    "Download failed for '%s' from %s: %s",
                    track.name,
                    provider.provider_name,
                    exc,
                )

        logger.info(
            "No suitable track downloaded from %s",
            provider.provider_name,
        )
        return None

    def _try_local_fallback(self, output_dir: Path) -> dict[str, Any] | None:
        existing = [p for p in self._local_paths if p.exists()]
        if not existing:
            logger.warning("No background music from any source.")
            return None

        local_path = random.choice(existing)  # noqa: S311
        ensure_dirs_exist(output_dir)
        dest_path = output_dir / local_path.name
        shutil.copy(local_path, dest_path)

        logger.info("Using local fallback: %s", local_path.name)
        return {
            "source": "Local",
            "type": "Music",
            "path": str(dest_path),
            "name": local_path.stem,
            "author": "Unknown",
            "license": "Local File",
            "url": "",
            "id": "",
        }
