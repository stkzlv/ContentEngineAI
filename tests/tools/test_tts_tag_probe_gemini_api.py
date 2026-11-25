"""The probe's Gemini API mode and the allowlist it sets (REQ-CNT-182)."""

from __future__ import annotations

import pytest

from src.video.config.audio_models import (
    PERFORMED_TAG_TRANSCRIPTIONS,
    PERFORMED_TTS_TAGS,
)
from tools.tts_tag_probe import GEMINI_API_TAGS, own_word_heard

pytestmark = pytest.mark.req("REQ-CNT-182")


@pytest.mark.parametrize(
    ("tag", "extra", "heard"),
    [
        ("<sigh>", ["sigh"], True),
        ("<sigh>", ["sighs"], True),
        ("<throat-clearing>", ["clearing"], True),
        ("<throat-clearing>", ["clears"], True),
        ("<short pause>", ["pausing"], True),
        ("<laugh>", ["laughter"], True),
        ("<sigh>", ["ugh"], False),
        ("<short pause>", [], False),
        ("", ["if"], False),
    ],
)
def test_the_own_word_is_told_from_a_sound_heard_as_a_word(tag, extra, heard):
    assert own_word_heard(tag, extra) is heard


def test_the_allowlist_holds_only_probed_angle_bracket_tags():
    assert set(GEMINI_API_TAGS) >= PERFORMED_TTS_TAGS
    assert all(t.startswith("<") and t.endswith(">") for t in PERFORMED_TTS_TAGS)


def test_every_transcribed_tag_is_on_the_allowlist():
    assert set(PERFORMED_TAG_TRANSCRIPTIONS) <= PERFORMED_TTS_TAGS
