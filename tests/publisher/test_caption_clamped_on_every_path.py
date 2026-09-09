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

    def test_youtube_caption_keeps_its_larger_budget(self):
        """The clamp is per destination, not one global minimum."""
        from src.publisher.schedule import caption_from_metadata

        caption = caption_from_metadata(self._meta(3000), "B0TEST001", Platform.YOUTUBE)

        assert 2200 < len(caption) <= 5000

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
        assert "clamp_to_limits" not in Path("src/publisher/batch.py").read_text()
        assert (
            "clamp_to_limits" not in Path("src/publisher/publish_modes.py").read_text()
        )
