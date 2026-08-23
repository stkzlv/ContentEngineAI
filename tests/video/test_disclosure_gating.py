"""Tests that `#ad` appears only where there is a material connection.

The failure directions are not symmetric. A disclosure that appears where none
is needed costs reach, because platforms down-rank content marked promotional.
A disclosure that is missing where one is needed is a false statement about a
material connection. So every ambiguous case discloses, and only a record that
positively shows there is nothing to disclose suppresses it.
"""

import contextlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

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


@pytest.mark.unit
class TestTheWiring:
    """The two lines that connect the predicate to a render.

    Deleting both left the whole suite green, so a refactor could revert the
    change with no signal: topic renders would go back to burning `#ad` on the
    frame, or to recording no decision so the caption discloses.
    """

    def _topic_product(self):
        from src.video.producer.topic_input import TopicSpec, build_topic_product

        return build_topic_product(
            TopicSpec(title="Why wifi drops", description="d", keywords=[])
        )

    def test_the_unified_metadata_records_the_decision(self, tmp_path):
        """The publisher reads this key; without it the caption discloses."""
        import asyncio
        import json
        from unittest.mock import AsyncMock, MagicMock, patch

        from src.video.producer.steps import _generate_unified_metadata

        ctx = MagicMock()
        ctx.product = self._topic_product()
        ctx.script = "A script."
        ctx.description = None
        ctx.run_paths = {"run_root": tmp_path, "description_file": tmp_path / "d.txt"}
        ctx.state = {}

        with patch(
            "src.video.producer.steps.generate_ai_description",
            new=AsyncMock(return_value="A description."),
        ):
            asyncio.run(_generate_unified_metadata(ctx))

        written = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
        assert written["carries_affiliate_content"] is False
        assert "ad" not in written["hashtags"]

    def test_the_platform_metadata_records_the_decision(self, tmp_path):
        """The optimized mode writes these instead of the unified file.

        Left ungated, a topic render's frame carried no overlay while its
        caption still led with `#ad`.
        """
        import json

        from src.ai.platform_metadata.utilities import save_metadata_to_file

        metadata = MagicMock()
        metadata.to_dict.return_value = {
            "platform": "youtube",
            "title": "T",
            "description": "D",
            "hashtags": ["#Shorts", "#ad"],
        }
        metadata.platform = "youtube"
        path = tmp_path / "metadata_youtube.json"
        save_metadata_to_file(metadata, path, disclose=False)

        written = json.loads(path.read_text(encoding="utf-8"))
        assert written["carries_affiliate_content"] is False
        assert written["hashtags"] == ["#Shorts"]

    def test_the_platform_metadata_defaults_to_disclosing(self, tmp_path):
        import json

        from src.ai.platform_metadata.utilities import save_metadata_to_file

        metadata = MagicMock()
        metadata.to_dict.return_value = {"hashtags": ["#ad"]}
        metadata.platform = "youtube"
        path = tmp_path / "metadata_youtube.json"
        save_metadata_to_file(metadata, path)

        written = json.loads(path.read_text(encoding="utf-8"))
        assert written["carries_affiliate_content"] is True
        assert written["hashtags"] == ["#ad"]

    def test_the_assembler_is_told_the_decision(self, tmp_path):
        """The overlay reads this attribute.

        Without the assignment the assembler keeps its default and burns `#ad`
        onto a topic render, which is the defect this change removes. The step
        is driven only as far as the assignment; what it does afterwards has
        its own coverage.
        """
        import asyncio
        from unittest.mock import MagicMock, patch

        from src.video.producer.steps import step_assemble_video

        captured = MagicMock()
        ctx = MagicMock()
        ctx.product = self._topic_product()
        # `run_paths` is a real dict in production, so a MagicMock would let
        # every lookup succeed and hide a missing key. Only the paths the step
        # reads before the assignment need to exist.
        ctx.run_paths = dict.fromkeys(
            (
                "subtitle_file",
                "final_video_output",
                "script_file",
                "music_info_file",
                "voiceover_file",
                "gathered_visuals_file",
                "assets_dir",
                "temp_dir",
                "run_root",
            ),
            tmp_path / "x",
        )

        with (
            patch("src.video.producer.steps.VideoAssembler", return_value=captured),
            contextlib.suppress(Exception),
        ):
            asyncio.run(step_assemble_video(ctx))

        assert captured.carries_affiliate_content is False

    def test_a_stale_metadata_file_is_backfilled(self, tmp_path):
        """A re-render without `--clean` would otherwise split the surfaces."""
        import json
        from unittest.mock import MagicMock

        from src.video.producer.steps import _check_existing_metadata

        (tmp_path / "metadata.json").write_text(
            json.dumps({"description": "old"}), encoding="utf-8"
        )
        ctx = MagicMock()
        ctx.product = self._topic_product()
        # A real dict: the step reads the run's resolved pillar out of it and
        # writes the value into the file, which a MagicMock cannot serialise.
        ctx.state = {}
        ctx.run_paths = {"run_root": tmp_path, "description_file": tmp_path / "d.txt"}

        assert _check_existing_metadata(ctx) is True
        written = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
        assert written["carries_affiliate_content"] is False


@pytest.mark.unit
class TestAffiliatePhraseGate:
    """The program phrase asserts membership, which is the strongest of the
    claims. Suppressing `#ad` without it would leave the phrase leading the
    caption on a render with no affiliate relationship at all.
    """

    def _metadata(self, carries: bool):
        from src.publisher.models import Platform, PublishMetadata

        return PublishMetadata(
            platform=Platform.YOUTUBE,
            title="T",
            description="D",
            hashtags=["Foo"],
            carries_affiliate_content=carries,
        )

    def test_the_phrase_is_applied_when_there_is_a_connection(self):
        metadata = self._metadata(True)
        if metadata.carries_affiliate_content:
            metadata.affiliate_disclosure = "As an Amazon Associate..."
        assert metadata.affiliate_disclosure is not None

    def test_the_phrase_is_withheld_when_there_is_none(self):
        metadata = self._metadata(False)
        if metadata.carries_affiliate_content:
            metadata.affiliate_disclosure = "As an Amazon Associate..."
        assert metadata.affiliate_disclosure is None

    def test_the_field_defaults_to_carrying(self):
        from src.publisher.models import Platform, PublishMetadata

        assert (
            PublishMetadata(
                platform=Platform.YOUTUBE, title="T", description="D"
            ).carries_affiliate_content
            is True
        )
