"""A render with no material connection cannot publish a disclosure token.

The caption prompts instruct the model to write `#ad` and demonstrate it in
every worked example, and they are not told whether the render carries an
affiliate link. So the token arrives in the metadata whatever the publisher
decided, and removing it has to be the publisher's job.

It was already being removed on a stock config, but by two accidents: the
trailing-hashtag rule in `load_platform_metadata`, written for legacy
metadata, and the disclosure dedup, which only matched while `disclosure`
still held its `#ad` default at construction time. Each test below is a case
one of those accidents missed.
"""

from __future__ import annotations

import json

import pytest

from src.publisher.metadata import load_platform_metadata
from src.publisher.models import DEFAULT_DISCLOSURE, Platform, PublishMetadata


def _written(tmp_path, product_id, **fields):
    """A metadata file as the producer writes it, loaded as publish does."""
    directory = tmp_path / product_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metadata_tiktok.json").write_text(
        json.dumps({"title": "T", "keywords": [], "product_id": product_id, **fields}),
        encoding="utf-8",
    )
    return load_platform_metadata(product_id, Platform.TIKTOK, tmp_path)


class TestATopicRenderPublishesNoDisclosure:
    def test_a_token_that_is_not_trailing_is_removed(self, tmp_path):
        """The legacy rule only strips hashtags at the very end."""
        metadata = _written(
            tmp_path,
            "topic-mid",
            description="Fix your wifi #ad by changing the channel.",
            hashtags=["WifiFix"],
            carries_affiliate_content=False,
        )

        assert "#ad" not in metadata.format_content()
        assert metadata.description == "Fix your wifi by changing the channel."

    def test_the_hashtag_is_removed(self, tmp_path):
        metadata = _written(
            tmp_path,
            "topic-tag",
            description="Fix your wifi by changing the channel.",
            hashtags=["WifiFix", "ad"],
            carries_affiliate_content=False,
        )

        assert metadata.hashtags == ["WifiFix"]

    @pytest.mark.parametrize("token", ["ad", "AD", "Ad"])
    def test_case_does_not_matter(self, tmp_path, token):
        metadata = _written(
            tmp_path,
            f"topic-case-{token}",
            description=f"Fix your wifi #{token} today.",
            hashtags=["WifiFix", token],
            carries_affiliate_content=False,
        )

        assert metadata.hashtags == ["WifiFix"]
        assert "#" not in metadata.description

    def test_a_configured_disclosure_does_not_stop_the_ad_strip(self):
        """The dedup matched only the configured token.

        `docs/compliance.md` lists localized variants as planned work, and
        the prompts write `#ad` regardless of what the publisher is
        configured to say, so the two cannot be the same check.
        """
        metadata = PublishMetadata(
            platform=Platform.TIKTOK,
            title="T",
            description="Fix your wifi by changing the channel.",
            hashtags=["WifiFix", "ad"],
            disclosure="#publi",
            carries_affiliate_content=False,
        )

        assert metadata.hashtags == ["WifiFix"]


class TestTheAffiliateCaseIsUnchanged:
    def test_a_product_render_still_leads_with_the_disclosure(self, tmp_path):
        metadata = _written(
            tmp_path,
            "B0ABCDEFGH",
            description="Great earbuds under $50.",
            hashtags=["Earbuds"],
            carries_affiliate_content=True,
        )

        assert metadata.format_content().startswith(DEFAULT_DISCLOSURE)

    def test_the_token_is_still_deduped_from_the_hashtags(self, tmp_path):
        """It leads the caption, so it must not also appear at the bottom."""
        metadata = _written(
            tmp_path,
            "B0DEDUPE12",
            description="Great earbuds under $50.",
            hashtags=["Earbuds", "ad"],
            carries_affiliate_content=True,
        )

        assert metadata.hashtags == ["Earbuds"]
        assert metadata.format_content().count("#ad") == 1

    def test_an_absent_flag_still_discloses(self, tmp_path):
        """A file written before the flag existed must not go out silent.

        A missing disclosure misstates a material connection; a needless one
        only asserts a connection that is not there.
        """
        metadata = _written(
            tmp_path,
            "B0LEGACY12",
            description="Great earbuds under $50.",
            hashtags=["Earbuds"],
        )

        assert metadata.carries_affiliate_content is True
        assert metadata.format_content().startswith(DEFAULT_DISCLOSURE)


class TestTheStripIsNotOverEager:
    def test_words_beginning_with_the_token_survive(self):
        metadata = PublishMetadata(
            platform=Platform.TIKTOK,
            title="T",
            description="Router #advice: use an #adapter if the port is dead. #ad",
            hashtags=["advice", "adapter", "ad"],
            carries_affiliate_content=False,
        )

        assert metadata.hashtags == ["advice", "adapter"]
        assert "#advice" in metadata.description
        assert "#adapter" in metadata.description
        assert "#ad " not in metadata.description
        assert not metadata.description.endswith("#ad")


class TestTheConfiguredTokenIsRemovedFromTheBodyToo:
    """The second token in the set was unreachable and untested.

    The loader used to blank `disclosure` when it decided not to disclose, so
    by the time the strip ran the configured token was gone and the set
    collapsed to `{"ad"}`. Replacing the set with a literal left the suite
    green, and the test named for this arm asserted only the hashtags, which
    the `#ad` literal already covered.
    """

    def test_a_language_variant_is_removed_from_the_description(self):
        metadata = PublishMetadata(
            platform=Platform.TIKTOK,
            title="T",
            description="Arregla tu wifi #publi cambiando el canal. #ad",
            hashtags=["WifiFix", "publi"],
            disclosure="#publi",
            carries_affiliate_content=False,
        )

        assert "#publi" not in metadata.description
        assert "#ad" not in metadata.description
        assert metadata.hashtags == ["WifiFix"]

    def test_the_caption_does_not_lead_with_a_disclosure(self):
        """`format_content` prepended the field regardless of the decision.

        So an object built with the flag off but the field left at its
        default published a caption opening with `#ad` while its body had
        just been stripped of one.
        """
        metadata = PublishMetadata(
            platform=Platform.TIKTOK,
            title="T",
            description="Fix your wifi by changing the channel.",
            hashtags=["WifiFix"],
            carries_affiliate_content=False,
        )

        assert not metadata.format_content().startswith("#ad")


class TestTheScheduleAutoPathStripsItToo:
    """`schedule auto` builds its caption from the metadata JSON directly.

    It never constructs a `PublishMetadata`, so a guard living only on that
    object left this path publishing the token -- and it does not get the
    trailing-hashtag rule either. `CLAUDE.md` names both publish paths as
    re-implementing the same logic, which is why the strip is a shared
    function rather than a method.
    """

    def test_a_topic_caption_loses_the_token(self):
        """Drives the builder, not an assertion that a call exists.

        A test reading the call site passes while the call sits behind a dead
        branch, which is how this path came to be unguarded in the first
        place.
        """
        from src.publisher.schedule import caption_from_metadata

        caption = caption_from_metadata(
            {
                "description": "Fix your wifi. Which fix worked? #ad",
                "hashtags": ["WifiFix", "ad"],
                "carries_affiliate_content": False,
            },
            "topic-wifi",
            Platform.TIKTOK,
        )

        assert "#ad" not in caption
        assert caption == "Fix your wifi. Which fix worked?\n\n#WifiFix #topic-wifi"

    def test_an_affiliate_caption_leads_with_it(self):
        """Placement, not presence.

        The old assertion was `"#ad" in caption`, which passes on a token the
        model left at the end -- below the fold the first-line placement
        exists to clear. That is how this path went without a leading
        disclosure unnoticed.
        """
        from src.publisher.schedule import caption_from_metadata

        caption = caption_from_metadata(
            {
                "description": "Great earbuds under $50.",
                "hashtags": ["Earbuds"],
                "carries_affiliate_content": True,
            },
            "B0ABCDEFGH",
            Platform.TIKTOK,
        )

        assert caption.startswith("#ad\n\n")

    def test_auto_schedule_actually_uses_the_builder(self):
        """The behavioural tests above only bind if the caller calls it.

        Both halves are needed: driving the builder catches a broken rule,
        reading the call site catches the builder being bypassed.
        """
        import ast
        from pathlib import Path

        tree = ast.parse(Path("src/publisher/schedule.py").read_text())
        auto = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
            and n.name == "auto_schedule"
        )

        calls = [
            node
            for node in ast.walk(auto)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "caption_from_metadata"
        ]

        # Two branches build a caption: from a metadata file, and from
        # `data.json` when none exists. Both must go through the builder, or
        # one of them publishes without the leading disclosure.
        assert len(calls) == 2, (
            f"auto_schedule has {len(calls)} caption_from_metadata call(s); "
            "both the metadata branch and the data.json fallback need one"
        )

    def test_an_absent_flag_still_leads_with_it(self):
        """Same default as everywhere else: disclose unless told otherwise."""
        from src.publisher.schedule import caption_from_metadata

        caption = caption_from_metadata(
            {"description": "Great earbuds.", "hashtags": ["Earbuds"]},
            "B0ABCDEFGH",
            Platform.TIKTOK,
        )

        assert caption.startswith("#ad\n\n")

    def test_the_model_token_is_not_doubled(self):
        """The leading line and a trailing token would disclose twice."""
        from src.publisher.schedule import caption_from_metadata

        caption = caption_from_metadata(
            {
                "description": "Great earbuds under $50.",
                "hashtags": ["Earbuds", "ad"],
                "carries_affiliate_content": True,
            },
            "B0ABCDEFGH",
            Platform.TIKTOK,
        )

        assert caption.startswith("#ad\n\n")
        assert caption.count("#ad") == 1

    def test_it_removes_the_token_a_prompt_wrote(self):
        from src.publisher.models import strip_disclosure_tokens

        description, hashtags = strip_disclosure_tokens(
            "Fix your wifi by changing the channel. Which fix worked? #ad",
            ["WifiFix", "ad"],
        )

        assert description == "Fix your wifi by changing the channel. Which fix worked?"
        assert hashtags == ["WifiFix"]


class TestACaptionWithNoTokenIsReturnedUntouched:
    """The whitespace repair runs only where a token was removed.

    Applied unconditionally it rewrote every non-affiliate caption: French
    spacing before `!` and `?`, deliberate ellipses and double spaces were
    all collapsed on renders that never contained a disclosure.
    """

    @pytest.mark.parametrize(
        "description",
        [
            "Sentence one.  Sentence two.",
            "Bonjour ! Ca va ? Oui , merci.",
            "Wait for it . . . boom",
        ],
    )
    def test_untouched(self, description):
        from src.publisher.models import strip_disclosure_tokens

        assert strip_disclosure_tokens(description, [])[0] == description


class TestTheFallbackPathIsCompliantToo:
    """A malformed metadata file must not publish a non-compliant caption.

    `caption_from_metadata` falls back when `PublishMetadata` refuses the
    input -- an empty description, or a YouTube entry with no title. Losing
    the whole scheduling run to one bad file would be worse, but the fallback
    has to apply the same two rules, or it ships exactly the pair of defects
    the function exists to close.
    """

    def test_an_affiliate_youtube_entry_with_no_title_still_leads_with_it(self):
        from src.publisher.schedule import caption_from_metadata

        caption = caption_from_metadata(
            {
                "description": "Great earbuds under $50.",
                "carries_affiliate_content": True,
            },
            "B0ABCDEFGH",
            Platform.YOUTUBE,
        )

        assert caption.startswith("#ad\n\n")

    def test_a_topic_youtube_entry_with_no_title_still_loses_the_token(self):
        from src.publisher.schedule import caption_from_metadata

        caption = caption_from_metadata(
            {
                "description": "Fix your wifi #ad by changing the channel.",
                "carries_affiliate_content": False,
            },
            "topic-wifi",
            Platform.YOUTUBE,
        )

        assert "#ad" not in caption
        assert caption == "Fix your wifi by changing the channel."
