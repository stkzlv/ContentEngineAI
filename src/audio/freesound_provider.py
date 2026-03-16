"""Freesound provider adapter wrapping the existing FreesoundClient."""

import logging
from pathlib import Path
from typing import Any

import aiohttp

from .base import AudioTrack, BaseAudioProvider
from .freesound_client import FreesoundClient
from .registry import AudioProvider, register_audio_provider

logger = logging.getLogger(__name__)


@register_audio_provider(AudioProvider.FREESOUND)
class FreesoundProvider(BaseAudioProvider):
    """Adapter wrapping FreesoundClient behind BaseAudioProvider."""

    def __init__(
        self,
        config: Any | None = None,
        secrets: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._config = config
        secrets = secrets or {}
        self._client = FreesoundClient(config=config, **secrets)
        self._audio_settings = (
            config.audio_settings
            if config and hasattr(config, "audio_settings")
            else None
        )

    @property
    def provider_name(self) -> str:
        return "freesound"

    async def search(
        self,
        query: str,
        min_duration: float,
        max_duration: float,
        max_results: int,
        session: aiohttp.ClientSession,
    ) -> list[AudioTrack]:
        if not self._client._api_key:
            logger.debug("Freesound API key not configured, skipping search")
            return []

        timeout = 10
        if self._audio_settings:
            timeout = getattr(self._audio_settings, "freesound_api_timeout_sec", 10)

        duration_filter = f"duration:[{int(min_duration)} TO {int(max_duration)}]"
        tracks = await self._client.search_music(
            query=query,
            filters=duration_filter,
            max_results=max_results,
            timeout_sec=timeout,
        )

        if not tracks and self._audio_settings:
            general_filters = getattr(
                self._audio_settings,
                "freesound_filters",
                None,
            )
            if general_filters:
                logger.info("Freesound duration search empty, trying general")
                tracks = await self._client.search_music(
                    query=query,
                    filters=general_filters,
                    max_results=max_results,
                    timeout_sec=timeout,
                )

        return [
            AudioTrack(
                id=str(getattr(t, "id", "")),
                name=getattr(t, "name", "Unknown"),
                duration=getattr(t, "duration", 0.0),
                author=getattr(t, "username", "Unknown"),
                license=getattr(t, "license", "Unknown"),
                url=getattr(
                    t,
                    "url",
                    f"https://freesound.org/s/{getattr(t, 'id', '')}/",
                ),
                provider_data=t,
            )
            for t in tracks
        ]

    async def download(
        self,
        track: AudioTrack,
        output_dir: Path,
        session: aiohttp.ClientSession,
    ) -> tuple[Path, dict[str, Any]] | None:
        sound = track.provider_data
        if sound is None:
            return None

        timeout = 60
        if self._audio_settings:
            timeout = getattr(
                self._audio_settings,
                "freesound_download_timeout_sec",
                60,
            )

        # Try OAuth2 full download first
        result = await self._client.download_full_sound_oauth2(
            int(track.id),
            output_dir,
            session,
            timeout_sec=timeout,
        )
        if result:
            return result

        # Fall back to API key preview
        result = await self._client.download_sound_preview_with_api_key(
            sound,
            output_dir,
            session,
            timeout_sec=timeout,
        )
        return result
