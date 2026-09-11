"""Platform captions must not publish LLM markdown (#417).

The unified description path cleans and validates its text; the three
platform parsers extracted title, description and caption with a bare
whitespace strip, so code fences, ``**bold**`` and ``*emphasis*`` published
verbatim on the platform-specific / optimized-metadata path.
"""

import pytest

from src.ai.platform_metadata.utilities import strip_inline_markdown


class TestTheSharedCleaner:
    def test_emphasis_and_bold_keep_their_text(self):
        assert (
            strip_inline_markdown("This is *too* hot and **very** loud")
            == "This is too hot and very loud"
        )

    def test_code_fences_are_dropped(self):
        assert strip_inline_markdown("```\nGreat gadget\n```") == "Great gadget"

    def test_guarded_asterisks_survive(self):
        """Same guards as the unified path: not-emphasis stays."""
        for text in ("2*3 and 4*5", "1080*1920 display", "See the note* here"):
            assert strip_inline_markdown(text) == text


def _parsers():
    from src.ai.platform_metadata.instagram import InstagramMetadataGenerator
    from src.ai.platform_metadata.tiktok import TikTokMetadataGenerator
    from src.ai.platform_metadata.youtube import YouTubeMetadataGenerator

    return [
        (
            YouTubeMetadataGenerator,
            "TITLE: The **best** earbuds\n\n"
            "DESCRIPTION: They get *really* loud\n\nHASHTAGS: #Tech",
            ("The best earbuds", "They get really loud"),
        ),
        (
            TikTokMetadataGenerator,
            "CAPTION: This is *too* good #fyp\n\nHASHTAGS: #fyp",
            ("This is too good #fyp",),
        ),
        (
            InstagramMetadataGenerator,
            "CAPTION: Our **favorite** find\n\nHASHTAGS: #finds",
            ("Our favorite find",),
        ),
    ]


class TestEveryParserCleans:
    """Each platform's extraction, driven for real.

    The generators only need their settings object to parse, and the
    existing generator tests construct them the same way.
    """

    @pytest.mark.parametrize(
        "generator_cls,response,expected",
        _parsers(),
        ids=lambda p: getattr(p, "__name__", ""),
    )
    def test_markdown_is_stripped_from_the_parsed_fields(
        self, generator_cls, response, expected
    ):
        from unittest.mock import MagicMock

        generator = generator_cls(MagicMock())
        parsed = generator._parse_llm_response(response)
        assert parsed is not None
        for i, want in enumerate(expected):
            assert parsed[i] == want, f"field {i}: {parsed[i]!r}"


class TestUnrelatedDoubleStarsDoNotPair:
    """Two non-bold `**` runs in one field must not pair up (#417 review).

    Stripping would merge "2**3" into "23" and mangle censored words --
    corrupted published text with no log line, worse than the markdown it
    replaces.
    """

    def test_the_corruption_cases_survive(self):
        for text in (
            "2**3 and 4**5",
            "f*** and s*** happen #real",
            "Fast charge** and waterproof** ok",
            "2**8 bytes",
            # A censor run's tail pairing with the next run's head, or
            # with a real bold span's closing pair: both terminal classes
            # must exclude the star.
            "s***t and f***k",
            "f***ing awesome and s***ty cables",
        ):
            assert strip_inline_markdown(text) == text

    def test_real_bold_still_strips(self):
        assert strip_inline_markdown("**bold** stays text") == "bold stays text"
        assert strip_inline_markdown("***word*** nested") == "word nested"

    def test_a_censor_run_survives_beside_real_bold(self):
        """The mixed shape: the censor word must survive while the bold
        in the same field strips -- a censor tail opening a match that a
        real bold span closes was the last cell of the pairing matrix.
        """
        assert (
            strip_inline_markdown("This s***t is **amazing**")
            == "This s***t is amazing"
        )
        assert (
            strip_inline_markdown("f***k yes, **50% off** today")
            == "f***k yes, 50% off today"
        )
