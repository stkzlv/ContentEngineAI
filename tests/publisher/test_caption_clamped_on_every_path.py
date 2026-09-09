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
    """`schedule auto` builds its caption through `caption_from_metadata`."""

    def _meta(self, chars: int) -> dict:
        return {
            "title": "A topic video",
            "description": "word " * (chars // 5),
            "hashtags": ["tech", "howto"],
            "keywords": [],
            "carries_affiliate_content": True,
        }

    def test_instagram_caption_fits_its_cap(self):
        from src.publisher.schedule import caption_from_metadata

        caption = caption_from_metadata(
            self._meta(3000), "B0TEST001", Platform.INSTAGRAM
        )

        assert len(caption) <= 2200

    def test_youtube_only_post_keeps_its_larger_budget(self):
        """A post that reaches only YouTube is clamped to YouTube's cap.

        This is the platform-specific branch, one post per platform.
        """
        from src.publisher.schedule import caption_from_metadata

        caption = caption_from_metadata(self._meta(3000), "B0TEST001", Platform.YOUTUBE)

        assert 2200 < len(caption) <= 5000

    def test_unified_caption_is_clamped_for_every_target(self):
        """The unified branch, which is the default, posts one caption to all.

        It used to reuse whichever platform's caption was built first, and
        `schedule` defaults to `youtube, tiktok, instagram`, so a caption
        clamped to YouTube's 5000 went to Instagram's 2200 and the post was
        refused on all three.
        """
        from src.publisher.schedule import caption_from_metadata

        caption = caption_from_metadata(
            self._meta(3000),
            "B0TEST001",
            Platform.YOUTUBE,
            targets=[Platform.YOUTUBE, Platform.TIKTOK, Platform.INSTAGRAM],
        )

        assert len(caption) <= 2200

    def test_the_fallback_caption_is_clamped_too(self):
        """The builder falls back when `PublishMetadata` refuses the record.

        That path hand-assembles a caption and returns before the clamp, so
        it ignored `targets`. Only one of the two refusal conditions matters:
        an empty description gives a short caption either way, but a YouTube
        entry with no title keeps its whole description.
        """
        from src.publisher.schedule import caption_from_metadata

        meta = {
            "title": None,  # refused for YouTube
            "description": "word " * 700,
            "hashtags": ["tech"],
            "carries_affiliate_content": True,
        }

        unified = caption_from_metadata(
            meta,
            "B0TEST001",
            Platform.YOUTUBE,
            targets=[Platform.YOUTUBE, Platform.TIKTOK, Platform.INSTAGRAM],
        )
        assert len(unified) <= 2200

        # ...and a YouTube-only post keeps YouTube's larger budget.
        youtube_only = caption_from_metadata(meta, "B0TEST001", Platform.YOUTUBE)
        assert 2200 < len(youtube_only) <= 5000

    def test_every_per_platform_caption_records_its_metadata(self):
        """The unified rebuild needs the metadata each caption came from.

        `auto_schedule` builds a per-platform caption in two places: from a
        metadata file, and from `data.json` when none exists. Only the first
        recorded what it built from, so on a product with no metadata file
        the unified branch found nothing to rebuild and fell back to reusing
        a caption clamped for one platform.
        """
        tree = ast.parse(Path("src/publisher/schedule.py").read_text())

        per_platform_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "caption_from_metadata"
            and not any(kw.arg == "targets" for kw in node.keywords)
        ]
        recorded = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Subscript)
                and isinstance(t.value, ast.Name)
                and t.value.id == "metas_used"
                for t in node.targets
            )
        ]
        assert len(recorded) == len(per_platform_calls), (
            f"{len(per_platform_calls)} per-platform caption(s) built but "
            f"{len(recorded)} recorded in metas_used; every branch that "
            "builds one must record what it built from"
        )

    def test_unified_branch_passes_its_full_target_list(self):
        """The parameter is only worth having if the branch supplies it.

        `auto_schedule` is not drivable without a scheduler, connected
        accounts and rendered files, so the call site is read instead.
        """
        source = Path("src/publisher/schedule.py").read_text()
        tree = ast.parse(source)

        with_targets = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "caption_from_metadata"
            and any(kw.arg == "targets" for kw in node.keywords)
        ]
        assert len(with_targets) == 1, (
            f"{len(with_targets)} caption_from_metadata call(s) pass targets; "
            "the unified branch must, and the per-platform branch must not"
        )

    def test_short_caption_is_untouched(self):
        from src.publisher.schedule import caption_from_metadata

        meta = self._meta(200)
        caption = caption_from_metadata(meta, "B0TEST001", Platform.TIKTOK)

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
