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

        `docs/publisher.md` describes overriding the field for language
        variants, and the prompts write `#ad` regardless of what the publisher
        is configured to say, so the two must not be the same check.
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
