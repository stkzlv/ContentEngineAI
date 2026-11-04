"""The music chain takes only tracks whose licence allows commercial use and
derivatives.

Both providers returned NonCommercial and NoDerivatives tracks among their
top results, and the chain never looked at the licence it recorded.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.audio.base import AudioTrack
from src.audio.licence import is_allowed, licence_code
from src.audio.manager import AudioManager
from src.video.config.audio_models import AudioSettings
from tests.audio.test_mood_filter import RecordingProvider

QUERY = "calm ambient"
DEFAULT = AudioSettings.model_fields["allowed_music_licenses"].default_factory()  # type: ignore[misc]


@pytest.fixture(autouse=True)
def _no_shuffle(monkeypatch):
    """Tied tracks are shuffled; without this the skip test is a coin flip."""
    monkeypatch.setattr("src.audio.manager.random.shuffle", lambda seq: None)


def _track(name: str, licence: str) -> AudioTrack:
    return AudioTrack(
        id=name,
        name=name,
        duration=120.0,
        author="A",
        license=licence,
        url="",
        tags=["calm", "ambient"],
    )


@pytest.mark.req("REQ-CNT-165")
@pytest.mark.parametrize(
    ("licence", "code"),
    [
        ("https://creativecommons.org/licenses/by/4.0/", "by"),
        ("http://creativecommons.org/licenses/by-sa/3.0/", "by-sa"),
        ("https://creativecommons.org/licenses/by-nc/4.0/", "by-nc"),
        ("http://creativecommons.org/licenses/by-nc-nd/3.0/", "by-nc-nd"),
        ("http://creativecommons.org/licenses/by-nd/3.0/", "by-nd"),
        ("http://creativecommons.org/publicdomain/zero/1.0/", "cc0"),
        ("http://artlibre.org/licence/lal/en", "fal"),
        ("CC0", "cc0"),
        ("CC-BY", "by"),
        ("Attribution NonCommercial", "by-nc"),
        ("Attribution 3.0", "by"),
        ("Attribution-NonCommercial", "by-nc"),
        ("Attribution NoDerivs", "by-nd"),
        ("Attribution-NoDerivatives 4.0", "by-nd"),
        ("Attribution-NonCommercial-ShareAlike 3.0", "by-nc-sa"),
        ("Attribution ShareAlike", "by-sa"),
        ("Attribution Something Else", None),
        ("http://creativecommons.org/licenses/by-nc/3.0/deed.en", "by-nc"),
        ("https://creativecommons.org/licenses/sampling+/1.0/", "sampling"),
        ("Creative Commons", None),
        ("https://creativecommons.org/", None),
        ("", None),
    ],
)
def test_a_licence_reads_as_its_code(licence, code):
    assert licence_code(licence) == code


@pytest.mark.req("REQ-CNT-165")
def test_the_default_allows_commercial_use_and_derivatives_only():
    assert set(DEFAULT) == {"cc0", "by", "by-sa", "fal"}
    for refused in ("by-nc", "by-nd", "by-nc-sa", "by-nc-nd"):
        assert not is_allowed(
            f"https://creativecommons.org/licenses/{refused}/4.0/", DEFAULT
        )
    assert not is_allowed("Creative Commons", DEFAULT)
    assert is_allowed("anything", None)


@pytest.mark.req("REQ-CNT-165")
@pytest.mark.asyncio
async def test_the_chain_skips_disallowed_tracks(tmp_path):
    path = tmp_path / "t.mp3"
    path.write_bytes(b"audio")
    nc = _track("nc", "https://creativecommons.org/licenses/by-nc/4.0/")
    nd = _track("nd", "http://creativecommons.org/licenses/by-nd/3.0/")
    ok = _track("ok", "https://creativecommons.org/licenses/by/4.0/")
    provider = RecordingProvider("test", [nc, nd, ok], path)
    manager = AudioManager(providers=[provider], allowed_licences=DEFAULT)

    result = await manager.find_music(QUERY, 60, 300, 10, tmp_path, MagicMock())

    assert result is not None and result["name"] == "ok"
    assert provider.downloaded == ["ok"]


@pytest.mark.req("REQ-CNT-165", "REQ-CNT-077")
@pytest.mark.asyncio
async def test_a_provider_with_only_disallowed_tracks_moves_the_chain_on(tmp_path):
    path = tmp_path / "t.mp3"
    path.write_bytes(b"audio")
    first = RecordingProvider(
        "first", [_track("nc", "https://creativecommons.org/licenses/by-nc/4.0/")], path
    )
    second = RecordingProvider(
        "second",
        [_track("cc0", "http://creativecommons.org/publicdomain/zero/1.0/")],
        path,
    )
    manager = AudioManager(providers=[first, second], allowed_licences=DEFAULT)

    result = await manager.find_music(QUERY, 60, 300, 10, tmp_path, MagicMock())

    assert result is not None and result["name"] == "cc0"
    assert first.downloaded == []


@pytest.mark.req("REQ-CNT-165")
def test_the_producer_passes_the_configured_list():
    """The music step builds its manager from `audio_settings`."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path("src/video/producer/steps.py").read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "AudioManager"
    ]
    assert calls, "the music step must build an AudioManager"
    for call in calls:
        kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
        assert kw.get("allowed_licences") == (
            "ctx.config.audio_settings.allowed_music_licenses"
        )


@pytest.mark.req("REQ-CNT-165")
def test_the_bundled_config_ships_the_default_list():
    from src.video.config import load_video_config_modular

    assert load_video_config_modular().audio_settings.allowed_music_licenses == [
        "cc0",
        "by",
        "by-sa",
        "fal",
    ]


@pytest.mark.req("REQ-CNT-165")
@pytest.mark.parametrize(
    ("allowed", "params"),
    [
        (["cc0", "by", "by-sa", "fal"], {"ccnc": "false", "ccnd": "false"}),
        (["cc0", "by", "by-nc"], {"ccnd": "false"}),
        (["by", "by-nd"], {"ccnc": "false"}),
        (["by-nc-nd"], {}),
        (None, {}),
    ],
)
def test_jamendo_filters_at_the_api_what_the_list_excludes(allowed, params):
    from src.audio.licence import jamendo_licence_params

    assert jamendo_licence_params(allowed) == params


@pytest.mark.req("REQ-CNT-165")
@pytest.mark.asyncio
async def test_the_jamendo_request_carries_the_configured_filters(mock_aioresponses):
    """Most of Jamendo's catalogue is NonCommercial; filtering only the
    results left one or two of twenty, so the request filters too.
    """
    import re
    from types import SimpleNamespace

    import aiohttp

    from src.audio.jamendo_provider import JamendoProvider, jamendo_circuit_breaker

    jamendo_circuit_breaker.reset()
    config = SimpleNamespace(
        audio_settings=SimpleNamespace(allowed_music_licenses=DEFAULT)
    )
    provider = JamendoProvider(secrets={"JAMENDO_CLIENT_ID": "id"}, config=config)
    mock_aioresponses.get(
        re.compile(r"https://api\.jamendo\.com/v3\.0/tracks/.*"),
        payload={"headers": {"status": "success"}, "results": []},
        repeat=True,
    )

    async with aiohttp.ClientSession() as session:
        await provider.search("calm", 60, 300, 10, session)

    urls = [str(url) for (_, url) in mock_aioresponses.requests]
    assert urls and all("ccnc=false" in u and "ccnd=false" in u for u in urls)


@pytest.mark.req("REQ-CNT-165")
def test_an_unknown_code_in_the_config_is_refused():
    """A code like `CC-BY` matches nothing, so every track would be refused."""
    from pydantic import ValidationError

    base = AudioSettings.model_validate(
        {
            "music_volume_db": -24,
            "voiceover_volume_db": 0,
            "background_music_paths": [],
            "freesound_api_key_env_var": "K",
            "freesound_search_query": "q",
            "freesound_filters": "",
            "freesound_max_results": 5,
        }
    ).model_dump()
    with pytest.raises(ValidationError, match="unknown licence codes"):
        AudioSettings.model_validate({**base, "allowed_music_licenses": ["CC-BY"]})
    assert (
        AudioSettings.model_validate(
            {**base, "allowed_music_licenses": None}
        ).allowed_music_licenses
        is None
    )
