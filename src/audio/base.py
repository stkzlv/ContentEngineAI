"""Base abstractions for audio provider platform."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import aiohttp


class AudioProvider(str, Enum):
    """Supported audio provider types."""

    FREESOUND = "freesound"
    JAMENDO = "jamendo"


@dataclass
class AudioTrack:
    """Normalized track metadata from any provider.

    Provides a common shape for search results regardless of the upstream API.
    The ``provider_data`` field carries the raw provider-specific object so
    download logic can access API-specific attributes.
    """

    id: str
    name: str
    duration: float
    author: str
    license: str
    url: str
    provider_data: Any = field(default=None, repr=False)


class BaseAudioProvider(ABC):
    """Abstract base for audio providers.

    Implementations must provide ``search`` and ``download`` methods.
    The return contract for ``download`` is a tuple of (file_path, attribution_dict)
    where the dict contains: source, type, path, name, author, license, url, id.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short name used in logs and config (e.g. 'freesound')."""

    @abstractmethod
    async def search(
        self,
        query: str,
        min_duration: float,
        max_duration: float,
        max_results: int,
        session: aiohttp.ClientSession,
    ) -> list[AudioTrack]:
        """Search for tracks matching criteria.

        Returns an empty list on failure (timeout, API error, etc.).
        """

    @abstractmethod
    async def download(
        self,
        track: AudioTrack,
        output_dir: Path,
        session: aiohttp.ClientSession,
    ) -> tuple[Path, dict[str, Any]] | None:
        """Download a track and return (path, attribution_dict) or None."""
