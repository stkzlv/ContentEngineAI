"""Tests for first comment builder."""

import json
from pathlib import Path

import pytest

from src.publisher.first_comment import build_first_comment, extract_closing_line
from src.publisher.models import FirstCommentConfig, Platform, PublishMetadata


@pytest.fixture
def enabled_config():
    """Config with YouTube and Instagram templates."""
    return FirstCommentConfig(
        enabled=True,
        platforms={
            "youtube": "Get it here: {affiliate_link}\n\nSubscribe!",
            "instagram": "{affiliate_link}",
        },
    )


@pytest.fixture
def product_data():
    """Standard product data with affiliate link."""
    return {
        "title": "Wireless Earbuds",
        "affiliate_link": "https://amzn.to/abc123",
        "shortened_affiliate_link": "https://short.link/x",
    }


@pytest.fixture
def outputs_dir(tmp_path, product_data):
    """Create outputs dir with data.json."""
    product_dir = tmp_path / "B0TEST001"
    product_dir.mkdir()
    (product_dir / "data.json").write_text(json.dumps(product_data))
    return tmp_path


class TestBuildFirstComment:
    """Test build_first_comment()."""

    def test_youtube_template_rendered(self, enabled_config, outputs_dir):
        result = build_first_comment(
            enabled_config, "youtube", "B0TEST001", outputs_dir
        )
        assert result == "Get it here: https://short.link/x\n\nSubscribe!"

    def test_instagram_template_rendered(self, enabled_config, outputs_dir):
        result = build_first_comment(
            enabled_config, "instagram", "B0TEST001", outputs_dir
        )
        assert result == "https://short.link/x"

    def test_tiktok_always_skipped(self, outputs_dir):
        config = FirstCommentConfig(
            enabled=True, platforms={"tiktok": "{affiliate_link}"}
        )
        result = build_first_comment(config, "tiktok", "B0TEST001", outputs_dir)
        assert result is None

    def test_disabled_config_returns_none(self, outputs_dir):
        config = FirstCommentConfig(
            enabled=False, platforms={"youtube": "{affiliate_link}"}
        )
        result = build_first_comment(config, "youtube", "B0TEST001", outputs_dir)
        assert result is None

    def test_unconfigured_platform_returns_none(self, enabled_config, outputs_dir):
        result = build_first_comment(
            enabled_config, "facebook", "B0TEST001", outputs_dir
        )
        assert result is None

    def test_missing_data_json_returns_none(self, enabled_config, tmp_path):
        (tmp_path / "B0MISSING").mkdir()
        result = build_first_comment(enabled_config, "youtube", "B0MISSING", tmp_path)
        assert result is None

    def test_no_affiliate_link_returns_none(self, enabled_config, tmp_path):
        product_dir = tmp_path / "B0NOLINK"
        product_dir.mkdir()
        (product_dir / "data.json").write_text(json.dumps({"title": "No link"}))
        result = build_first_comment(enabled_config, "youtube", "B0NOLINK", tmp_path)
        assert result is None

    def test_falls_back_to_affiliate_link_when_no_shortened(
        self, enabled_config, tmp_path
    ):
        product_dir = tmp_path / "B0LONG"
        product_dir.mkdir()
        data = {"title": "Test", "affiliate_link": "https://amzn.to/long"}
        (product_dir / "data.json").write_text(json.dumps(data))

        result = build_first_comment(enabled_config, "youtube", "B0LONG", tmp_path)
        assert result is not None
        assert "https://amzn.to/long" in result

    def test_data_json_as_list(self, enabled_config, tmp_path):
        """data.json can be a list (scraper output), first element used."""
        product_dir = tmp_path / "B0LIST"
        product_dir.mkdir()
        data = [{"title": "First", "affiliate_link": "https://amzn.to/first"}]
        (product_dir / "data.json").write_text(json.dumps(data))

        result = build_first_comment(enabled_config, "youtube", "B0LIST", tmp_path)
        assert result is not None
        assert "https://amzn.to/first" in result

    def test_invalid_json_returns_none(self, enabled_config, tmp_path):
        product_dir = tmp_path / "B0BAD"
        product_dir.mkdir()
        (product_dir / "data.json").write_text("not json{{{")
        result = build_first_comment(enabled_config, "youtube", "B0BAD", tmp_path)
        assert result is None

    def test_product_title_placeholder(self, tmp_path):
        config = FirstCommentConfig(
            enabled=True,
            platforms={"youtube": "Check out {product_title}: {affiliate_link}"},
        )
        product_dir = tmp_path / "B0TITLE"
        product_dir.mkdir()
        data = {"title": "Cool Gadget", "affiliate_link": "https://amzn.to/x"}
        (product_dir / "data.json").write_text(json.dumps(data))

        result = build_first_comment(config, "youtube", "B0TITLE", tmp_path)
        assert result == "Check out Cool Gadget: https://amzn.to/x"


class TestMoveHashtagsToComment:
    """Test hashtag migration to first comment."""

    def test_hashtags_moved_for_instagram(self, tmp_path):
        config = FirstCommentConfig(
            enabled=True,
            move_hashtags_to_comment=True,
            platforms={"instagram": "{affiliate_link} {hashtags}"},
        )
        product_dir = tmp_path / "B0HASH"
        product_dir.mkdir()
        data = {"title": "Test", "affiliate_link": "https://amzn.to/h"}
        (product_dir / "data.json").write_text(json.dumps(data))

        metadata = PublishMetadata(
            platform=Platform.INSTAGRAM,
            title="Test",
            description="desc",
            hashtags=["earbuds", "tech"],
            keywords=[],
            product_id="B0HASH",
        )

        result = build_first_comment(
            config, "instagram", "B0HASH", tmp_path, metadata=metadata
        )
        assert result is not None
        assert "#earbuds" in result
        assert "#tech" in result
        assert "https://amzn.to/h" in result

    def test_hashtags_not_moved_for_youtube(self, tmp_path):
        config = FirstCommentConfig(
            enabled=True,
            move_hashtags_to_comment=True,
            platforms={"youtube": "{affiliate_link} {hashtags}"},
        )
        product_dir = tmp_path / "B0YT"
        product_dir.mkdir()
        data = {"title": "Test", "affiliate_link": "https://amzn.to/y"}
        (product_dir / "data.json").write_text(json.dumps(data))

        metadata = PublishMetadata(
            platform=Platform.YOUTUBE,
            title="Test",
            description="desc",
            hashtags=["tech"],
            keywords=[],
            product_id="B0YT",
        )

        result = build_first_comment(
            config, "youtube", "B0YT", tmp_path, metadata=metadata
        )
        # Hashtags only moved for instagram, so {hashtags} is empty string
        assert result is not None
        assert "#tech" not in result

    def test_hashtags_not_double_prefixed(self, tmp_path):
        """Hashtags already starting with # don't get double-prefixed."""
        config = FirstCommentConfig(
            enabled=True,
            move_hashtags_to_comment=True,
            platforms={"instagram": "{hashtags}"},
        )
        product_dir = tmp_path / "B0DUP"
        product_dir.mkdir()
        data = {"title": "Test", "affiliate_link": "https://amzn.to/d"}
        (product_dir / "data.json").write_text(json.dumps(data))

        metadata = PublishMetadata(
            platform=Platform.INSTAGRAM,
            title="Test",
            description="desc",
            hashtags=["#already", "nohash"],
            keywords=[],
            product_id="B0DUP",
        )

        result = build_first_comment(
            config, "instagram", "B0DUP", tmp_path, metadata=metadata
        )
        assert result is not None
        assert "#already" in result
        assert "#nohash" in result
        assert "##already" not in result

    def test_no_metadata_means_empty_hashtags(self, tmp_path):
        config = FirstCommentConfig(
            enabled=True,
            move_hashtags_to_comment=True,
            platforms={"instagram": "{affiliate_link}{hashtags}"},
        )
        product_dir = tmp_path / "B0NOMETA"
        product_dir.mkdir()
        data = {"title": "Test", "affiliate_link": "https://amzn.to/n"}
        (product_dir / "data.json").write_text(json.dumps(data))

        result = build_first_comment(
            config, "instagram", "B0NOMETA", tmp_path, metadata=None
        )
        assert result == "https://amzn.to/n"


class TestExtractClosingLine:
    """The engagement-bait beat sits before the CTA and is what YouTube gets.

    YouTube renders URLs in Shorts comments as plain text, so that slot carries
    the script's closing beat instead of a dead link.
    """

    def test_comment_fork_question(self):
        script = (
            "So this 3D pen just showed up. It's lighter than I expected. "
            "PLA or ABS - which filament do you prefer? Link in bio if you want one."
        )
        assert extract_closing_line(script) == (
            "PLA or ABS - which filament do you prefer?"
        )

    def test_ignores_rhetorical_questions_earlier_in_the_script(self):
        """Regression: question-led templates open with rhetorical questions.

        Selecting "the last question" reached back past the closing beat and
        returned a mid-script question. The beat is positional, not punctuational.
        Script is verbatim from a real question_driven render.
        """
        script = (
            "Ever lose your keys and spend twenty minutes tearing the place apart? "
            "This little tracker is a lifesaver. "
            "Worried about your luggage on a trip? "
            "It works with both Apple and Android. "
            "Four trackers for the price of one is a solid deal. "
            "Link in bio if you want one."
        )
        assert extract_closing_line(script) == (
            "Four trackers for the price of one is a solid deal."
        )

    def test_a_beat_ending_on_a_quoted_label_does_not_carry_the_cta(self):
        """The stripper reads the last sentence, so where a sentence ends is
        load-bearing here. A splitter demanding the terminator come last
        merged a beat ending on a quoted interface label into the CTA, and
        the CTA itself became the published first comment -- the failure
        `_CTA_MARKERS` exists to prevent. A how-to quotes labels constantly.
        """
        script = (
            "Open it and read the list. "
            'The other ports might just say "USB 3.1." '
            "Save this for the next time it happens."
        )
        assert extract_closing_line(script) == (
            'The other ports might just say "USB 3.1."'
        )

    def test_a_beat_before_a_lowercase_sentence_is_not_swallowed(self):
        """The other half of the same knob. A splitter demanding a capital
        after the space merges two real sentences whenever the second starts
        `iPhone` or `iOS`, and the beat then carries a sentence that is not
        the beat.
        """
        script = (
            "This works on macOS. "
            "iOS handles it differently. "
            "Save this for the next time it happens."
        )
        assert extract_closing_line(script) == "iOS handles it differently."

    def test_an_inline_quotation_does_not_start_a_new_sentence(self):
        """The closing-quote branch exists for a beat *ending* on a quoted
        label. A quotation ending in a terminator mid-sentence looks the same
        to it, and with a lowercase continuation allowed it split there --
        publishing a first comment that begins "and then". That branch
        therefore requires a capital, which a real next sentence has.
        """
        script = (
            "The port looks identical on both sides. "
            'You ask yourself "why is it slow?" and then you check the wrong one. '
            "Save this for the next time it happens."
        )
        assert extract_closing_line(script) == (
            'You ask yourself "why is it slow?" and then you check the wrong one.'
        )

    def test_the_splitter_is_the_shared_one(self):
        """Three modules reason about where a script's sentences end: this
        one, the fact check's containment guard, and `ends_with_cta`. They
        disagreed, so a repair confined correctly by one was refused by
        another for not ending on a CTA. A test that only drives this module
        cannot see that, which is why it reads the identity.
        """
        from src.ai import script_fact_check, script_generator
        from src.publisher import first_comment
        from src.utils.script_sanitizer import split_sentences

        assert first_comment.split_sentences is split_sentences
        assert script_generator.split_sentences is split_sentences
        assert script_fact_check.split_sentences is split_sentences

    def test_falls_back_to_the_debatable_claim(self):
        """Analytical templates close with a claim, not a question."""
        script = (
            "This tracker works with both phones. "
            "Four trackers for the price of one is a solid deal. "
            "Follow for more finds like this."
        )
        assert extract_closing_line(script) == (
            "Four trackers for the price of one is a solid deal."
        )

    @pytest.mark.parametrize(
        "cta",
        [
            "Link in bio if you want one.",
            "Follow for more finds like this.",
            "Drop a comment if you've tried it.",
            "Share with someone who needs this.",
        ],
    )
    def test_strips_every_allowed_cta_shape(self, cta):
        script = f"It is small. Worth it under twenty dollars? {cta}"
        assert extract_closing_line(script) == "Worth it under twenty dollars?"

    def test_handles_a_script_with_no_cta(self):
        script = "It is small. Worth it under twenty dollars?"
        assert extract_closing_line(script) == "Worth it under twenty dollars?"

    def test_empty_and_cta_only(self):
        assert extract_closing_line("") is None
        assert extract_closing_line("   \n ") is None
        assert extract_closing_line("Link in bio if you want one.") is None


class TestClosingLineTemplate:
    """A script-derived template must not require product data."""

    @staticmethod
    def _config():
        return FirstCommentConfig(enabled=True, platforms={"youtube": "{closing_line}"})

    def test_renders_from_the_script(self, tmp_path):
        d = tmp_path / "B0TEST002" / "temp"
        d.mkdir(parents=True)
        (d / "script.txt").write_text(
            "It arrived today. USB-C or Lightning - which annoys you more? "
            "Link in bio if you want one."
        )
        result = build_first_comment(self._config(), "youtube", "B0TEST002", tmp_path)
        assert result == "USB-C or Lightning - which annoys you more?"

    def test_no_affiliate_link_needed(self, tmp_path):
        """No data.json at all: the template doesn't reference the product."""
        d = tmp_path / "B0TEST003" / "temp"
        d.mkdir(parents=True)
        (d / "script.txt").write_text("Small thing. Worth twenty dollars?")
        assert not (tmp_path / "B0TEST003" / "data.json").exists()
        result = build_first_comment(self._config(), "youtube", "B0TEST003", tmp_path)
        assert result == "Worth twenty dollars?"

    def test_missing_script_skips_rather_than_posting_blank(self, tmp_path):
        (tmp_path / "B0TEST004").mkdir()
        assert (
            build_first_comment(self._config(), "youtube", "B0TEST004", tmp_path)
            is None
        )

    def test_no_url_reaches_the_youtube_slot(self, tmp_path):
        d = tmp_path / "B0TEST005" / "temp"
        d.mkdir(parents=True)
        (d / "script.txt").write_text(
            "It works. Worth it? Link in bio if you want one."
        )
        (tmp_path / "B0TEST005" / "data.json").write_text(
            json.dumps({"title": "T", "affiliate_link": "https://amzn.to/x"})
        )
        result = build_first_comment(self._config(), "youtube", "B0TEST005", tmp_path)
        assert result is not None
        assert "http" not in result
