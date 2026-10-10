"""Numbers, units and model names rewritten for the voice only (design 0014)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.video.config import load_video_config_modular
from src.video.config.audio_models import TTSNormalisationSettings
from src.video.tts import TTSManager, normalise_for_tts

TABLE = TTSNormalisationSettings(
    enabled=True,
    units={
        "mAh": "milliamp hours",
        "W": "watts",
        "Hz": "hertz",
        "GHz": "gigahertz",
        "inch": "inch",
    },
    lexicon={"A2337": "A two three three seven"},
)


@pytest.mark.req("REQ-CNT-075")
@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        ("A 5000mAh battery.", "A 5000 milliamp hours battery."),
        ("A 10,000 mAh pack", "A 10,000 milliamp hours pack"),
        ("Charges at 65W, fast.", "Charges at 65 watts, fast."),
        ("Dual-band 2.4 GHz and 5GHz", "Dual-band 2.4 gigahertz and 5 gigahertz"),
        ("A 1.83-inch screen", "A 1.83 inch screen"),
        ("Model A2337 ships.", "Model A two three three seven ships."),
    ],
)
def test_each_entry_rewrites_its_fixture(written: str, spoken: str) -> None:
    assert normalise_for_tts(written, TABLE) == spoken


@pytest.mark.req("REQ-CNT-075")
@pytest.mark.parametrize(
    "text",
    [
        "Watch the Hz meter.",  # unit letters with no number before them
        "A 65Wh battery",  # a longer unit the table doesn't list
        "Order XA2337 or A2337B",  # the term inside a longer code
        "Room 3W-2 is open",  # a unit followed by a hyphenated tail
        "Order X-A2337 or A2337-B",  # the term inside a hyphenated code
    ],
)
def test_letters_outside_an_entry_stay(text: str) -> None:
    assert normalise_for_tts(text, TABLE) == text


def test_off_leaves_the_text_alone() -> None:
    off = TABLE.model_copy(update={"enabled": False})

    assert normalise_for_tts("A 5000mAh battery", off) == "A 5000mAh battery"


@pytest.mark.req("REQ-CNT-075")
def test_the_shipped_config_is_on_with_empty_tables() -> None:
    settings = load_video_config_modular().tts_config.tts_normalisation

    assert settings.enabled is True
    assert settings.units == {} and settings.lexicon == {}


async def _voice(tmp_path: Path, settings: TTSNormalisationSettings) -> str:
    """The text `generate_speech` hands the provider for one script."""
    tts = load_video_config_modular().tts_config.model_copy(deep=True)
    tts.tts_normalisation = settings
    sent = AsyncMock(return_value=(tmp_path / "voice.wav", "Charon"))
    script = "A 5000mAh battery and model A2337."
    with patch("src.video.tts._generate_gemini_speech", sent):
        await TTSManager(tts, {}, product_id="B0X").generate_speech(
            script, tmp_path / "voice.wav"
        )
    assert script == "A 5000mAh battery and model A2337."
    return str(sent.call_args.args[0])


@pytest.mark.req("REQ-CNT-076")
@pytest.mark.asyncio
async def test_the_voice_gets_the_spoken_form_and_the_script_keeps_its_own(
    tmp_path: Path,
) -> None:
    sent = await _voice(tmp_path, TABLE)

    assert "5000 milliamp hours" in sent
    assert "A two three three seven" in sent


@pytest.mark.asyncio
async def test_off_sends_todays_text(tmp_path: Path) -> None:
    sent = await _voice(tmp_path, TTSNormalisationSettings())

    assert "5000mAh" in sent and "A2337" in sent


@pytest.mark.req("REQ-CNT-076")
def test_captions_built_from_the_script_get_the_spoken_form() -> None:
    from src.video.tts import spoken_script

    tts = load_video_config_modular().tts_config.model_copy(deep=True)
    assert spoken_script("A 5000mAh pack", tts) == "A 5000mAh pack"
    tts.tts_normalisation = TABLE
    assert spoken_script("A 5000mAh pack", tts) == "A 5000 milliamp hours pack"
    assert spoken_script(None, tts) is None


@pytest.mark.req("REQ-CNT-076")
def test_every_caption_call_site_passes_the_spoken_script() -> None:
    """Script-timed captions are the fallback when STT returns no timings."""
    from src.video.producer import steps, two_part_subtitles

    for module in (steps, two_part_subtitles):
        source = Path(module.__file__).read_text(encoding="utf-8")
        calls = source.split("create_unified_subtitles(")[1:]
        assert calls, module.__name__
        for call in calls:
            assert "spoken_script(" in call.split(")\n")[0], module.__name__
