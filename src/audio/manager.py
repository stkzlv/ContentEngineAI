"""Audio manager that orchestrates provider chain with local file fallback."""

import logging
import random
import shutil
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
        for provider in self._providers:
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
                    return result
            except CircuitBreakerError:
                logger.warning(
                    "Circuit breaker open for %s, skipping",
                    provider.provider_name,
                )
            except Exception as exc:
                logger.warning(
                    "Provider %s failed: %s",
                    provider.provider_name,
                    exc,
                )

        return self._try_local_fallback(output_dir)

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

        for track in sorted(tracks, key=lambda t: t.duration):
            if track.duration < min_duration:
                continue
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
