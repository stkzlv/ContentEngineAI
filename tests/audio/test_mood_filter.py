"""The chosen track matches the requested mood.

The provider chain asked Freesound for `calm ambient instrumental` in
rating order and accepted the first candidate that downloaded:
"Battleground | UK Drill Instrumental" under a calm smartwatch pitch
(2026-09-19). Rating order does not enforce the query's terms.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.audio.base import AudioTrack, BaseAudioProvider
from src.audio.jamendo_provider import _musicinfo_tags
from src.audio.manager import (
    AudioManager,
    matched_terms,
    query_terms,
    rank_by_mood,
)

QUERY = "calm ambient instrumental"


def _track(name: str, tags: list[str], track_id: str = "1") -> AudioTrack:
    return AudioTrack(
        id=track_id,
        name=name,
        duration=120.0,
        author="A",
        license="CC0",
        url="",
        tags=tags,
    )


DRILL = _track("Battleground | UK Drill Instrumental", ["drill", "trap", "uk"], "1")
CALM = _track("Morning Haze", ["ambient", "calm", "pad"], "2")


class RecordingProvider(BaseAudioProvider):
    """Downloads whatever it is asked to and remembers the order."""

    def __init__(self, name: str, tracks: list[AudioTrack], path: Path):
        self._name = name
        self._tracks = tracks
        self._path = path
        self.downloaded: list[str] = []

    @property
    def provider_name(self) -> str:
        return self._name

    async def search(self, query, min_duration, max_duration, max_results, session):
        return self._tracks

    async def download(self, track, output_dir, session):
        self.downloaded.append(track.name)
        return self._path, {
            "source": self._name,
            "type": "Music",
            "path": str(self._path),
            "name": track.name,
            "author": track.author,
            "license": track.license,
            "url": track.url,
            "id": track.id,
        }


class TestTerms:
    def test_query_words_in_order_without_repeats(self):
        assert query_terms("Calm ambient, calm Instrumental") == [
            "calm",
            "ambient",
            "instrumental",
        ]

    def test_a_term_matches_a_word_it_begins_in_title_or_tags(self):
        track = _track("Calming Waves", ["chillout", "ambient-pad"])

        assert matched_terms(track, ["calm", "chill", "ambient", "drill"]) == [
            "calm",
            "chill",
            "ambient",
        ]

    def test_no_terms_means_every_track_passes(self):
        ranked = rank_by_mood([DRILL, CALM], [])

        assert {track.name for track, _ in ranked} == {DRILL.name, CALM.name}


class TestTheChainPicksTheMood:
    @pytest.mark.asyncio
    async def test_an_on_mood_track_second_in_the_list_is_picked(self, tmp_path):
        path = tmp_path / "t.mp3"
        path.write_bytes(b"audio")
        provider = RecordingProvider("test", [DRILL, CALM], path)
        manager = AudioManager(providers=[provider])

        result = await manager.find_music(QUERY, 60, 300, 10, tmp_path, MagicMock())

        assert result is not None and result["name"] == CALM.name
        assert result["matched_terms"] == ["calm", "ambient"]
        assert provider.downloaded == [CALM.name]

    @pytest.mark.asyncio
    async def test_a_partial_match_is_a_fallback_not_a_first_choice(self, tmp_path):
        """The drill track matches `instrumental`; it is tried only after the
        track that matches more of the query.
        """
        path = tmp_path / "t.mp3"
        path.write_bytes(b"audio")
        ranked = rank_by_mood([DRILL, CALM], query_terms(QUERY))

        assert [track.name for track, _ in ranked] == [CALM.name, DRILL.name]
        assert [matched for _, matched in ranked] == [
            ["calm", "ambient"],
            ["instrumental"],
        ]

    @pytest.mark.asyncio
    async def test_no_matching_candidate_moves_to_the_next_provider(
        self, tmp_path, caplog
    ):
        path = tmp_path / "t.mp3"
        path.write_bytes(b"audio")
        off = RecordingProvider("off", [_track("Battle Anthem", ["metal"])], path)
        on = RecordingProvider("on", [CALM], path)
        manager = AudioManager(providers=[off, on])

        with caplog.at_level(logging.INFO):
            result = await manager.find_music(QUERY, 60, 300, 10, tmp_path, MagicMock())

        assert result is not None and result["source"] == "on"
        assert off.downloaded == []
        assert any(
            "No track from off matches the query terms" in r.message
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_the_summary_names_the_matched_terms(self, tmp_path, caplog):
        path = tmp_path / "t.mp3"
        path.write_bytes(b"audio")
        manager = AudioManager(providers=[RecordingProvider("test", [CALM], path)])

        with caplog.at_level(logging.INFO):
            await manager.find_music(QUERY, 60, 300, 10, tmp_path, MagicMock())

        assert any(r.message == "Matched: calm, ambient" for r in caplog.records)


class TestProvidersCarryTags:
    def test_jamendo_musicinfo_tags_are_flattened_and_lowered(self):
        result = {
            "musicinfo": {
                "tags": {
                    "genres": ["Ambient", "chillout"],
                    "instruments": ["synthesizer"],
                    "vartags": ["Calm"],
                }
            }
        }

        assert _musicinfo_tags(result) == [
            "ambient",
            "chillout",
            "synthesizer",
            "calm",
        ]
        assert _musicinfo_tags({}) == []

    @pytest.mark.asyncio
    async def test_freesound_asks_for_relevance_and_tags(self, monkeypatch):
        from unittest.mock import AsyncMock

        from src.audio import freesound_provider

        provider = freesound_provider.FreesoundProvider.__new__(
            freesound_provider.FreesoundProvider
        )
        provider._audio_settings = None
        provider._client = SimpleNamespace(  # type: ignore[assignment]
            _api_key="k",
            search_music=AsyncMock(
                return_value=[
                    SimpleNamespace(
                        id=7, name="Haze", duration=90.0, tags=["Ambient", "Calm"]
                    )
                ]
            ),
        )

        tracks = await provider.search(QUERY, 60, 300, 10, MagicMock())

        call = provider._client.search_music.call_args.kwargs
        assert call["sort_order"] == "score"
        assert "tags" in call["fields"].split(",")
        assert tracks[0].tags == ["ambient", "calm"]

    @pytest.mark.asyncio
    async def test_the_configured_sort_is_honoured(self):
        """`freesound_sort` was declared and shipped but never read; the
        client's own default applied.
        """
        from unittest.mock import AsyncMock

        from src.audio import freesound_provider

        provider = freesound_provider.FreesoundProvider.__new__(
            freesound_provider.FreesoundProvider
        )
        provider._audio_settings = SimpleNamespace(freesound_sort="downloads_desc")
        provider._client = SimpleNamespace(  # type: ignore[assignment]
            _api_key="k", search_music=AsyncMock(return_value=[])
        )

        await provider.search(QUERY, 60, 300, 10, MagicMock())

        call = provider._client.search_music.call_args.kwargs
        assert call["sort_order"] == "downloads_desc"
