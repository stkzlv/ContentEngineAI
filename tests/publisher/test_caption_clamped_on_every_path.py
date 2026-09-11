"""Every publish path clamps its caption to the platform it targets (#403).

Four paths build a caption and three of them re-implement the others'
handling, which is where this drifted: `schedule auto` never clamped at all,
and the unified path clamped against the wrong platform. The unified path is
covered in `test_publish_modes.py`; these are the other two.
"""

import ast
from pathlib import Path

from src.publisher.models import Platform


class TestSchedulePathClamps:
    """`schedule auto` builds a `PublishMetadata` and clamps at each use.

    The builder returns the metadata object rather than a caption string
    (#408): each branch clamps with its own target list in scope, so the
    `targets` parameter, the `metas_used` side record and the two
    source-reading tests that pinned them are gone with the defect class.
    """

    def _meta(self, chars: int) -> dict:
        return {
            "title": "A topic video",
            "description": "word " * (chars // 5),
            "hashtags": ["tech", "howto"],
            "keywords": [],
            "carries_affiliate_content": True,
        }

    def test_instagram_caption_fits_its_cap(self):
        from src.publisher.schedule import metadata_from_file

        m = metadata_from_file(self._meta(3000), "B0TEST001", Platform.INSTAGRAM)
        caption = m.clamped_for([Platform.INSTAGRAM]).format_content()

        assert len(caption) <= 2200

    def test_youtube_only_post_keeps_its_larger_budget(self):
        """The platform-specific branch: one post per platform."""
        from src.publisher.schedule import metadata_from_file

        m = metadata_from_file(self._meta(3000), "B0TEST001", Platform.YOUTUBE)
        caption = m.clamped_for([Platform.YOUTUBE]).format_content()

        assert 2200 < len(caption) <= 5000

    def test_unified_caption_is_clamped_for_every_target(self):
        """The unified branch posts one caption to all three platforms."""
        from src.publisher.schedule import metadata_from_file

        m = metadata_from_file(self._meta(3000), "B0TEST001", Platform.YOUTUBE)
        caption = m.clamped_for(
            [Platform.YOUTUBE, Platform.TIKTOK, Platform.INSTAGRAM]
        ).format_content()

        assert len(caption) <= 2200

    def test_one_metadata_serves_both_uses_unmutated(self):
        """The invariant the pure clamp exists for (#408).

        A mutating clamp would hand the second use the first one's narrower
        budget: clamp for all three, then for YouTube alone, and the
        YouTube-only caption would be stuck at 2200.
        """
        from src.publisher.schedule import metadata_from_file

        m = metadata_from_file(self._meta(3000), "B0TEST001", Platform.YOUTUBE)

        unified = m.clamped_for(
            [Platform.YOUTUBE, Platform.TIKTOK, Platform.INSTAGRAM]
        ).format_content()
        youtube_only = m.clamped_for([Platform.YOUTUBE]).format_content()

        assert len(unified) <= 2200
        assert 2200 < len(youtube_only) <= 5000, (
            "the first clamp narrowed the second's budget; clamped_for "
            "must not mutate the original"
        )

    def test_the_clamp_does_not_rewrite_the_originals_counts(self):
        """`replace` passes `character_counts` by reference, and the clamp
        writes into it -- the copy must own its dict, or the original's
        counts describe the clamped copy while its text is unclamped.
        """
        from src.publisher.schedule import metadata_from_file

        m = metadata_from_file(self._meta(3000), "B0TEST001", Platform.YOUTUBE)
        before = dict(m.character_counts)

        clamped = m.clamped_for([Platform.INSTAGRAM])

        assert clamped.character_counts is not m.character_counts
        assert (
            m.character_counts == before
        ), "clamping the copy rewrote the original's character_counts"

    def test_a_refused_record_is_repaired_not_bypassed(self):
        """The builder must yield a metadata object even for a record
        `PublishMetadata` refuses, or the exception path can skip a clamp
        by returning a hand-assembled string -- which it once did.
        """
        from src.publisher.models import PublishMetadata
        from src.publisher.schedule import metadata_from_file

        meta = {
            "title": None,  # refused for YouTube
            "description": "word " * 700,
            "hashtags": ["tech"],
            "carries_affiliate_content": True,
        }

        m = metadata_from_file(meta, "B0TEST001", Platform.YOUTUBE)
        assert isinstance(m, PublishMetadata)
        assert m.title, "the repair must supply the title YouTube requires"

        unified = m.clamped_for(
            [Platform.YOUTUBE, Platform.TIKTOK, Platform.INSTAGRAM]
        ).format_content()
        assert len(unified) <= 2200

        youtube_only = m.clamped_for([Platform.YOUTUBE]).format_content()
        assert 2200 < len(youtube_only) <= 5000

    def test_short_caption_is_untouched(self):
        from src.publisher.schedule import metadata_from_file

        meta = self._meta(200)
        m = metadata_from_file(meta, "B0TEST001", Platform.TIKTOK)
        caption = m.clamped_for([Platform.TIKTOK]).format_content()

        assert meta["description"].strip() in caption


class TestImmediateBatchPathClamps:
    """The immediate batch publishes one post per platform, in a loop.

    Driving that loop needs a publisher, connected accounts and an uploaded
    media id, so the call site is read instead: it must name the platform it
    is publishing to rather than relying on whichever platform's metadata was
    loaded.
    """

    def test_publish_loop_clamps_for_the_destination(self):
        source = Path("src/publisher/batch.py").read_text()
        tree = ast.parse(source)

        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "clamp_for_platforms"
        ]
        assert (
            len(calls) == 1
        ), f"batch.py has {len(calls)} clamp_for_platforms call(s); expected 1"

        (arg,) = calls[0].args
        assert isinstance(arg, ast.List) and len(arg.elts) == 1
        assert isinstance(arg.elts[0], ast.Name) and arg.elts[0].id == "platform", (
            "the clamp must target the platform being published to, not the "
            "platform whose metadata happened to load"
        )

    def test_no_single_platform_clamp_remains(self):
        """`clamp_to_limits` reads the metadata's own platform, which is not
        necessarily the destination on a path with a metadata fallback.
        """
        assert "clamp_to_limits" not in Path("src/publisher/schedule.py").read_text()
        assert "clamp_to_limits" not in Path("src/publisher/batch.py").read_text()
        assert (
            "clamp_to_limits" not in Path("src/publisher/publish_modes.py").read_text()
        )


class TestEveryCaptionIsComposedFromAClampedCopy:
    """The one invariant the collapse leaves (#408).

    The builder returns unclamped metadata and each branch clamps at its
    point of use, so the wiring rule is simply: no `format_content()` in
    the scheduler except on a `clamped_for(...)` result. That replaces the
    `targets` default, the `metas_used` side record and the two
    source-reading tests that pinned them.
    """

    def test_no_unclamped_format_content_in_the_scheduler(self):
        tree = ast.parse(Path("src/publisher/schedule.py").read_text())
        offenders = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "format_content"
            and not (
                isinstance(node.func.value, ast.Call)
                and isinstance(node.func.value.func, ast.Attribute)
                and node.func.value.func.attr == "clamped_for"
            )
        ]
        assert not offenders, (
            f"format_content called on an unclamped object at lines "
            f"{offenders}; compose captions only from clamped_for(...)"
        )
