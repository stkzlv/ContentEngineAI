"""Tests for audio provider base abstractions."""

from src.audio.base import AudioProvider, AudioTrack


def test_audio_provider_enum_values():
    providers = {AudioProvider.FREESOUND, AudioProvider.JAMENDO}
    assert "freesound" in providers
    assert "jamendo" in providers


def test_audio_provider_from_string():
    assert AudioProvider("freesound") is AudioProvider.FREESOUND
    assert AudioProvider("jamendo") is AudioProvider.JAMENDO


def test_audio_track_creation():
    track = AudioTrack(
        id="123",
        name="Test Track",
        duration=120.0,
        author="Artist",
        license="CC0",
        url="https://example.com/track",
    )
    assert track.id == "123"
    assert track.name == "Test Track"
    assert track.duration == 120.0
    assert track.provider_data is None


def test_audio_track_with_provider_data():
    raw = {"audiodownload": "https://example.com/dl"}
    track = AudioTrack(
        id="456",
        name="Track",
        duration=60.0,
        author="Artist",
        license="CC-BY",
        url="https://example.com",
        provider_data=raw,
    )
    assert track.provider_data == raw
