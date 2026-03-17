"""Audio provider platform for background music search and download.

Provides a pluggable provider system for sourcing background music from
various Creative Commons music APIs (Jamendo, Freesound) with local file
fallback.
"""

# Import providers to trigger registration
from . import (
    freesound_provider,  # noqa: F401
    jamendo_provider,  # noqa: F401
)
from .base import AudioProvider, AudioTrack, BaseAudioProvider
from .manager import AudioManager
from .registry import AudioProviderRegistry, create_audio_provider

__all__ = [
    "AudioManager",
    "AudioProvider",
    "AudioProviderRegistry",
    "AudioTrack",
    "BaseAudioProvider",
    "create_audio_provider",
]
