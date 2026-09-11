"""Single-asterisk emphasis must not publish into the caption (#409).

A verification render produced "getting *too* hot" in a description; the
prompt forbids markdown, nothing caught the disobedience, and the asterisks
would publish literally on every platform. The `**bold**` check added for
#404 targets the shapes a reasoning monologue produces and deliberately does
not match a mid-sentence `*italic*` span.

The repair is a strip rather than a rejection: a rejection costs a retry
and, if every model in the chain writes emphasis, the render -- for text
whose repaired form is known exactly.
"""

import ast
import re
from pathlib import Path

from src.ai.description_generator import (
    DescriptionValidationConfig,
    strip_single_asterisk_emphasis,
    validate_description_completeness,
)


class TestWhatTheStripperRemoves:
    def test_the_reported_case(self):
        """The render that filed the issue, repaired exactly."""
        got = strip_single_asterisk_emphasis(
            "But if it's getting *too* hot, your case might be trapping it"
        )
        assert got == "But if it's getting too hot, your case might be trapping it"

    def test_multi_word_spans_and_punctuation_edges(self):
        assert (
            strip_single_asterisk_emphasis("Grab it now *while stocks last*!")
            == "Grab it now while stocks last!"
        )
        assert (
            strip_single_asterisk_emphasis("(*required*) field") == "(required) field"
        )
        assert (
            strip_single_asterisk_emphasis("*Really hot* days ahead")
            == "Really hot days ahead"
        )


class TestWhatTheStripperLeavesAlone:
    """An asterisk that is not emphasis must survive.

    These are the cases the issue names as what makes a naive bare-asterisk
    check wrong: multiplication, a footnote marker, spaced math, and the
    bold form the existing validator already owns.
    """

    def test_multiplication(self):
        for text in ("2*3 and 4*5", "1080*1920 resolution"):
            assert strip_single_asterisk_emphasis(text) == text

    def test_footnote_marker(self):
        text = "See the note* for details"
        assert strip_single_asterisk_emphasis(text) == text

    def test_spaced_asterisks(self):
        text = "a * b * c"
        assert strip_single_asterisk_emphasis(text) == text

    def test_bold_is_left_for_the_validator(self):
        """`**bold**` is the validator's rejection, not this stripper's.

        Stripping one star from it would turn a rejectable monologue marker
        into exactly the emphasis form this fix exists to remove.
        """
        text = "1. **Analyze the Request:** first I will think"
        assert strip_single_asterisk_emphasis(text) == text

        limits = DescriptionValidationConfig()
        ok, reason = validate_description_completeness(
            "This looks like a fine long description of a product that "
            "closes properly and also contains **bold markup** in it.",
            limits,
        )
        assert not ok
        assert "bold" in reason

    def test_a_real_published_description_is_untouched(self):
        """Identity on real output, not just on crafted cases."""
        real = (
            "Think closing apps saves your battery? 🤯 Actually, swiping them "
            "away makes them use MORE power next time. Learn the simple trick "
            "to keep your battery lasting longer instead! 🔋✨"
        )
        assert strip_single_asterisk_emphasis(real) == real


class TestTheCleanerIsTheOnePathToValidation:
    """Four attempt paths receive a completion; all four must clean it.

    A cleanup applied at one site and not the others is this module's
    documented failure shape, so the sites are counted rather than trusted.
    """

    def test_every_attempt_path_calls_the_shared_cleaner(self):
        tree = ast.parse(Path("src/ai/description_generator.py").read_text())
        fn = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "generate_description"
        )
        calls = [
            node
            for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_clean_description"
        ]
        assert len(calls) == 4, (
            f"{len(calls)} _clean_description call(s); the primary loop, the "
            "free-model discovery, the fallback provider and its discovery "
            "must all clean before validating"
        )

    def test_no_attempt_path_cleans_inline(self):
        """An inline re.sub at a call site would bypass the emphasis strip."""
        source = Path("src/ai/description_generator.py").read_text()
        fences = re.findall(r're\.sub\(\s*r"```', source)
        assert len(fences) == 1, (
            "the code-fence strip must live only inside _clean_description; "
            f"found {len(fences)} sites"
        )

    def test_a_stripped_description_validates(self):
        """End to end: emphasis in, valid caption out."""
        from src.ai.description_generator import _clean_description

        raw = (
            "Your phone slows down when it gets *too* hot -- that is "
            "thermal throttling protecting the chip, not a fault. Keep it "
            "out of the sun and drop the case while charging to stay fast!"
        )
        cleaned = _clean_description(raw)
        assert "*" not in cleaned
        ok, _ = validate_description_completeness(
            cleaned, DescriptionValidationConfig()
        )
        assert ok


class TestFenceStripKeepsContent:
    """The fence strip removes markers, not words (#423).

    The old word-and-whitespace tail was meant for a language tag but greedily
    crossed newlines, so a fenced description lost its leading words up to
    the first non-word character -- silently, and a long description that
    loses only its first clause can pass validation and publish corrupted.
    """

    def test_leading_words_survive(self):
        from src.ai.description_generator import _clean_description

        raw = "```\nAmazing sound quality, great battery life```"
        assert _clean_description(raw) == "Amazing sound quality, great battery life"

    def test_language_tags_are_still_consumed(self):
        from src.ai.description_generator import _clean_description

        raw = "```text\nGreat gadget for travel.\n```"
        assert _clean_description(raw) == "Great gadget for travel."
