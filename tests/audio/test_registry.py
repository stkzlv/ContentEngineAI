"""Tests for audio provider registry."""

import pytest

from src.audio.base import AudioProvider, BaseAudioProvider
from src.audio.registry import AudioProviderRegistry, create_audio_provider


def test_freesound_registered():
    assert AudioProviderRegistry.get(AudioProvider.FREESOUND) is not None


def test_jamendo_registered():
    assert AudioProviderRegistry.get(AudioProvider.JAMENDO) is not None


def test_list_available():
    available = AudioProviderRegistry.list_available()
    assert AudioProvider.FREESOUND in available
    assert AudioProvider.JAMENDO in available


def test_create_freesound_provider():
    provider = create_audio_provider("freesound", secrets={})
    assert isinstance(provider, BaseAudioProvider)
    assert provider.provider_name == "freesound"


def test_create_jamendo_provider():
    provider = create_audio_provider("jamendo", secrets={})
    assert isinstance(provider, BaseAudioProvider)
    assert provider.provider_name == "jamendo"


def test_create_unknown_provider():
    with pytest.raises(ValueError, match="Unknown audio provider"):
        create_audio_provider("nonexistent")
