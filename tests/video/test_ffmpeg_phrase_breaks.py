"""FFmpeg-engine caption segments end at phrase boundaries (REQ-VID-157), and
script-timed captions keep every word (REQ-VID-163).
"""

from __future__ import annotations

import random

import pytest

from src.video.config.subtitle_models import SubtitleSettings
from src.video.unified_subtitle_generator import UnifiedSubtitleGenerator


def _generator(mock_config, phrase_breaks: bool) -> UnifiedSubtitleGenerator:
    settings = SubtitleSettings(
        max_words_per_line=3,
        max_line_length=38,
        max_duration=2.5,
        min_duration=0.6,
        phrase_breaks=phrase_breaks,
    )
    return UnifiedSubtitleGenerator(settings, (1080, 1920), video_config=mock_config)


def _timings(text: str, length: float = 0.28) -> list[dict]:
    return [
        {"word": w, "start_time": i * 0.3, "end_time": i * 0.3 + length}
        for i, w in enumerate(text.split())
    ]


def _texts(segments: list[dict]) -> list[str]:
    return [s["text"] for s in segments]


NAME = "Plug the cable into the back of your new iPhone 15 and wait."


@pytest.mark.req("REQ-VID-157")
def test_whisper_segments_end_before_a_phrase_opener(mock_config):
    segments = _generator(mock_config, True)._create_segments(_timings(NAME), None)
    assert "iPhone 15" in _texts(segments)
    assert "and wait." in _texts(segments)
    assert " ".join(_texts(segments)).split() == NAME.split()


@pytest.mark.req("REQ-VID-157")
def test_whisper_segments_keep_the_count_split_when_off(mock_config):
    segments = _generator(mock_config, False)._create_segments(_timings(NAME), None)
    assert "iPhone 15 and" in _texts(segments)


@pytest.mark.req("REQ-VID-157")
def test_carried_word_starts_the_next_caption(mock_config):
    """The caption ends where its successor's first carried word starts.

    Each word overlaps the next by 50 ms, as the timing smoother's lead leaves
    them, so the caption has to be clamped to the carried word, not to the
    word that tripped the limit.
    """
    timings = _timings(NAME, length=0.35)
    segments = _generator(mock_config, True)._create_segments(timings, None)
    index = _texts(segments).index("iPhone 15")
    following = segments[index + 1]
    assert following["start"] == pytest.approx(11 * 0.3)
    assert segments[index]["end"] <= following["start"]


@pytest.mark.req("REQ-VID-157")
def test_a_break_keeps_at_least_two_words(mock_config):
    """Ending after "Done." scores best but would leave a one-word caption."""
    text = "Done. Now open the settings app and tap reset."
    segments = _generator(mock_config, True)._create_segments(_timings(text), None)
    assert all(len(t.split()) >= 2 for t in _texts(segments)[:-1])


@pytest.mark.req("REQ-VID-157")
def test_carried_words_must_fit_beside_the_next_word(mock_config):
    """A carry that would push the next caption past the character limit is
    not taken, even where it scores better.
    """
    generator = _generator(mock_config, True)
    generator.config.max_line_length = 22
    # "because" opens a phrase, but "because troubleshooting" is 23 characters.
    text = "Open Settings because troubleshooting helps."
    segments = generator._create_segments(_timings(text), None)
    assert all(len(t) <= 22 for t in _texts(segments))
    assert " ".join(_texts(segments)).split() == text.split()


@pytest.mark.req("REQ-VID-157")
def test_script_segments_end_before_a_phrase_opener(mock_config):
    segments = _generator(mock_config, True)._create_script_segments(NAME, 10.0)
    assert "iPhone 15" in _texts(segments)
    assert " ".join(_texts(segments)).split() == NAME.split()


@pytest.mark.req("REQ-VID-163")
@pytest.mark.parametrize("phrase_breaks", [False, True])
def test_script_segments_keep_a_sentence_final_word(mock_config, phrase_breaks):
    """A sentence-final word that arrives when the caption is full opens the
    next caption; it used to be dropped.
    """
    script = "Unplug the router now. Wait thirty seconds. Then plug it back in."
    generator = _generator(mock_config, phrase_breaks)
    segments = generator._create_script_segments(script, 10.0)
    assert " ".join(_texts(segments)).split() == script.split()


@pytest.mark.req("REQ-VID-157")
def test_script_carry_must_fit_the_width(mock_config, monkeypatch):
    """On the script-timed path the carried words also pass the width check."""
    generator = _generator(mock_config, True)
    monkeypatch.setattr(
        generator, "fits_within_image_width", lambda text, *a, **k: len(text) <= 22
    )
    text = "Open Settings because troubleshooting helps."
    segments = generator._create_script_segments(text, 10.0)
    assert all(len(t) <= 22 for t in _texts(segments))
    assert " ".join(_texts(segments)).split() == text.split()


@pytest.mark.req("REQ-VID-157")
def test_bundled_profiles_hand_the_generator_phrase_breaks_on():
    """What the subtitle step builds from each profile's merged settings: the
    model dump it passes to `create_unified_subtitles`, read back the way that
    function reads it.
    """
    from src.video.config_adapter import load_video_config_modular

    config = load_video_config_modular()
    for name in config.video_profiles:
        merged = config.get_profile_merged_settings(name).subtitle_settings
        rebuilt = SubtitleSettings.from_legacy_dict(merged.model_dump())
        assert rebuilt.phrase_breaks is True, name


@pytest.mark.req("REQ-VID-163")
def test_script_breaks_do_not_outrun_the_audio_at_a_fast_rate(mock_config):
    """At 4.5 words a second two words last under `min_duration`, so a break
    that kept two pushed later captions past the audio and lost their words.
    Seeded scripts from a how-to vocabulary; without the floor about half of
    them lost words.
    """
    vocab = (
        "the a your phone router plug it in and then open settings because "
        "wait now. tap reset. iPhone 15 cable back of with to for app."
    ).split()
    rnd = random.Random(1)  # noqa: S311 - a reproducible fixture
    generator = _generator(mock_config, True)
    for _ in range(20):
        words = [rnd.choice(vocab) for _ in range(rnd.randint(30, 90))]
        segments = generator._create_script_segments(" ".join(words), len(words) / 4.5)
        assert " ".join(_texts(segments)).split() == words
