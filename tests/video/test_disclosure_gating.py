"""Tests that `#ad` appears only where there is a material connection.

The failure directions are not symmetric. A disclosure that appears where none
is needed costs reach, because platforms down-rank content marked promotional.
A disclosure that is missing where one is needed is a false statement about a
material connection. So every ambiguous case discloses, and only a record that
positively shows there is nothing to disclose suppresses it.
"""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.publisher.metadata import _load_from_json
from src.publisher.models import Platform
from src.scraper.base.models import carries_affiliate_content
from src.video.producer.steps import _extract_hashtags_from_title


@pytest.mark.unit
class TestWhatCountsAsAffiliateContent:
    def test_a_product_with_a_link_discloses(self):
        product = SimpleNamespace(affiliate_link="https://example.com/dp/X", topic=None)
        assert carries_affiliate_content(product)

    def test_a_topic_with_no_link_does_not(self):
        assert not carries_affiliate_content(
            SimpleNamespace(affiliate_link=None, topic="Why wifi drops")
        )

    def test_a_topic_that_somehow_carries_a_link_discloses(self):
        """The link is the material connection, whatever the record is called."""
        assert carries_affiliate_content(
            SimpleNamespace(affiliate_link="https://example.com/x", topic="A topic")
        )

    def test_a_product_whose_link_failed_to_build_still_discloses(self):
        """`build_affiliate_url` falls back to the input URL and warns.

        Reading a missing link as "no material connection" would drop the
        disclosure from exactly the run that already went wrong.
        """
        assert carries_affiliate_content(
            SimpleNamespace(affiliate_link=None, topic=None)
        )

    def test_an_empty_link_string_still_discloses(self):
        assert carries_affiliate_content(SimpleNamespace(affiliate_link="", topic=None))

    def test_a_record_missing_both_fields_discloses(self):
        """Anything the predicate cannot read is treated as disclosing."""
        assert carries_affiliate_content(SimpleNamespace())


@pytest.mark.unit
class TestHashtag:
    def test_the_ad_tag_is_appended_by_default(self):
        assert "ad" in _extract_hashtags_from_title("Retro Handheld Game Console")

    def test_it_is_omitted_when_told_not_to_disclose(self):
        tags = _extract_hashtags_from_title("Why your bread goes stale", disclose=False)
        assert "ad" not in tags

    def test_the_other_tags_are_unaffected(self):
        with_ad = _extract_hashtags_from_title("Retro Handheld Game Console")
        without = _extract_hashtags_from_title(
            "Retro Handheld Game Console", disclose=False
        )
        assert [t for t in with_ad if t != "ad"] == without


@pytest.mark.unit
class TestCaptionDisclosure:
    """The publisher reads the producer's decision rather than re-deriving it.

    Two paths deciding the same thing drift, and a caption that discloses while
    the frame does not is worse than either choice made consistently.
    """

    def _load(self, **extra):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.json"
            path.write_text(
                json.dumps(
                    {"title": "T", "description": "D", "hashtags": ["Foo"], **extra}
                ),
                encoding="utf-8",
            )
            return _load_from_json(path, Platform.YOUTUBE, "X")

    def test_the_recorded_decision_is_honoured(self):
        assert self._load(carries_affiliate_content=False).disclosure == ""

    def test_a_recorded_true_discloses(self):
        assert self._load(carries_affiliate_content=True).disclosure == "#ad"

    def test_metadata_written_before_the_field_existed_discloses(self):
        """An older `metadata.json` has no such key.

        Reading its absence as "no affiliate content" would silently drop the
        disclosure from every product produced before this change.
        """
        assert self._load().disclosure == "#ad"

    def test_the_caption_leads_with_the_disclosure_when_present(self):
        content = self._load(carries_affiliate_content=True).format_content()
        assert content.splitlines()[0] == "#ad"

    def test_the_caption_leads_with_the_description_when_absent(self):
        content = self._load(carries_affiliate_content=False).format_content()
        assert content.splitlines()[0] == "D"


@pytest.mark.unit
class TestOverlayDefault:
    """The assembler discloses unless told otherwise.

    A caller that forgets to set the flag gets the disclosure, which is the
    direction that cannot misstate a material connection.
    """

    def test_a_fresh_assembler_discloses(self):
        import warnings

        from src.video.assembler.core import VideoAssembler
        from src.video.config import load_video_config_modular

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config = load_video_config_modular()
        assert VideoAssembler(config).carries_affiliate_content is True
