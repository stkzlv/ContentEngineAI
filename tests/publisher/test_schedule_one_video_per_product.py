"""`schedule` enumerates products, not video files.

A product rendered under a second profile keeps both files. Taking each of
them scheduled the same product twice — two posts on different days, each
carrying a different render, burning two slots — while `single` had always
resolved exactly one video per product through the selector.
"""

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from src.publisher.late.cli import _scan_and_filter_videos


def _args(outputs_dir):
    return argparse.Namespace(outputs_dir=outputs_dir, force=True, platforms=None)


def _product(outputs_dir, asin, *profiles):
    d = outputs_dir / asin
    d.mkdir(parents=True, exist_ok=True)
    for profile in profiles:
        (d / f"video_{asin}_{profile}.mp4").write_bytes(b"x")
    return d


@pytest.mark.unit
class TestOneVideoPerProduct:
    def test_two_renders_yield_one_scheduled_video(self, tmp_path):
        _product(tmp_path, "B0TWORENDS", "product_video_single", "slideshow_images1")
        found = _scan_and_filter_videos(_args(tmp_path))
        assert len(found) == 1
        assert found[0].parent.name == "B0TWORENDS"

    def test_each_product_still_contributes_one(self, tmp_path):
        _product(tmp_path, "B0AAAAAAAA", "slideshow_images1")
        _product(tmp_path, "B0BBBBBBBB", "product_video_single", "slideshow_images3")
        found = _scan_and_filter_videos(_args(tmp_path))
        assert {p.parent.name for p in found} == {"B0AAAAAAAA", "B0BBBBBBBB"}
        assert len(found) == 2

    def test_the_chosen_render_matches_what_single_would_pick(self, tmp_path):
        from src.publisher.video_selector import select_video_for_platform

        d = _product(
            tmp_path, "B0AGREEAAA", "product_video_single", "slideshow_images1"
        )
        scheduled = _scan_and_filter_videos(_args(tmp_path))[0]
        singled = select_video_for_platform(d, "B0AGREEAAA", "youtube", None)
        assert scheduled == singled

    def test_a_product_with_no_render_is_skipped(self, tmp_path):
        (tmp_path / "B0NORENDER").mkdir(parents=True)
        _product(tmp_path, "B0HASONE01", "slideshow_images1")
        found = _scan_and_filter_videos(_args(tmp_path))
        assert [p.parent.name for p in found] == ["B0HASONE01"]

    def test_the_ignored_render_is_named_in_the_log(self, tmp_path, caplog):
        import logging

        _product(tmp_path, "B0LOGGEDAA", "product_video_single", "slideshow_images1")
        with caplog.at_level(logging.INFO):
            _scan_and_filter_videos(_args(tmp_path))
        assert "slideshow_images1" in caplog.text
        assert "2 renders" in caplog.text


@pytest.mark.unit
class TestEveryDiscovererAgrees:
    """Three paths find videos; they must choose the same render.

    `single` resolves one per product through the selector. `schedule` and
    the immediate batch each had their own glob, so a product rendered under
    two profiles was published once per render — and with a per-platform
    profile configured, the paths chose different cuts of the same product.
    """

    def test_the_immediate_batch_takes_one_render_per_product(self, tmp_path):
        from unittest.mock import MagicMock

        from src.publisher.batch import BatchPublisher

        _product(tmp_path, "B0TWORENDS", "product_video_single", "slideshow_images1")
        publisher = MagicMock()
        publisher.config = None
        batch = BatchPublisher(publisher=publisher, outputs_dir=tmp_path)

        found = batch._discover_videos()
        assert len(found) == 1
        assert found[0]["product_id"] == "B0TWORENDS"

    def test_a_configured_profile_decides_which_render(self, tmp_path):
        from src.publisher.video_selector import sole_render_for_product

        d = _product(
            tmp_path, "B0PROFILES", "product_video_single", "slideshow_short_20s"
        )
        profiles = {"youtube": "slideshow_short_20s"}

        chosen = sole_render_for_product(d, profiles, "youtube")
        assert chosen is not None
        assert chosen.name == "video_B0PROFILES_slideshow_short_20s.mp4"

        # And it matches what `single` would upload for that platform.
        from src.publisher.video_selector import select_video_for_platform

        assert chosen == select_video_for_platform(d, "B0PROFILES", "youtube", profiles)

    def test_no_profile_falls_back_to_a_stable_choice(self, tmp_path):
        from src.publisher.video_selector import sole_render_for_product

        d = _product(tmp_path, "B0NOPROFIL", "b_second", "a_first")
        chosen = sole_render_for_product(d, None, "")
        assert chosen is not None
        assert chosen.name.endswith("a_first.mp4")


@pytest.mark.unit
class TestTheImmediateBatchPayload:
    """The immediate path builds its own payload, and it must be complete."""

    @staticmethod
    def _batch(tmp_path, profiles=None, platforms=None):
        from unittest.mock import MagicMock

        from src.publisher.batch import BatchPublisher

        return BatchPublisher(
            publisher=MagicMock(),
            outputs_dir=tmp_path,
            platforms=platforms,
            profiles=profiles,
        )

    def test_the_configured_profile_reaches_the_discoverer(self, tmp_path):
        from src.publisher.models import Platform

        _product(tmp_path, "B0BATCHPRO", "slideshow_images1", "video_sequential")
        batch = self._batch(
            tmp_path,
            profiles={"youtube": "video_sequential"},
            platforms=[Platform.YOUTUBE],
        )
        found = batch._discover_videos()
        assert len(found) == 1
        assert found[0]["path"].name.endswith("video_sequential.mp4")

    def test_the_retry_queue_uses_the_same_render(self, tmp_path):
        """Driven through `_get_retry_queue_videos`, which is a fourth
        discoverer with its own glob. It used an unsorted one, so it could
        republish a different cut than the run that queued it.
        """
        import json

        from src.publisher.models import Platform

        _product(tmp_path, "B0RETRYQUE", "aaa_first", "zzz_last")
        (tmp_path / "publish_history.json").write_text(
            json.dumps(
                {
                    "posts": {},
                    "retry_queue": {
                        "B0RETRYQUE": {
                            "product_id": "B0RETRYQUE",
                            "retry_count": 1,
                            "error": "transient",
                        }
                    },
                }
            )
        )
        batch = self._batch(
            tmp_path,
            profiles={"youtube": "zzz_last"},
            platforms=[Platform.YOUTUBE],
        )

        # Filesystem order is not sorted order, and the queue used to read
        # whichever the directory happened to yield first. Forced here, so
        # the assertion does not pass by luck on one filesystem.
        real_glob = Path.glob

        def adverse(self, pattern):
            # Alphabetical, which puts the *unconfigured* render first, so a
            # discoverer that takes whatever the directory yields picks the
            # wrong one and the assertion below catches it.
            return iter(sorted(real_glob(self, pattern)))

        with patch.object(Path, "glob", adverse):
            queued = batch._get_retry_queue_videos()

        assert len(queued) == 1
        assert queued[0]["path"].name.endswith("zzz_last.mp4")

    @pytest.mark.asyncio
    async def test_the_rate_limit_retry_sends_the_same_payload(self, tmp_path):
        """The retry rebuilt its own call. Omitting the payload sent a
        YouTube leg with no title, which the builder now refuses — turning a
        recoverable 429 into a failed publish.
        """
        import json
        from unittest.mock import AsyncMock, MagicMock

        from src.publisher.base import PublishError
        from src.publisher.models import Platform

        d = _product(tmp_path, "B0RATELIMT", "slideshow_images1")
        (d / "metadata.json").write_text(
            json.dumps(
                {
                    "mode": "unified",
                    "title": "A perfectly good title",
                    "description": "Body copy.",
                    "hashtags": ["tech"],
                }
            )
        )

        publisher = MagicMock()
        publisher.publish = AsyncMock(
            side_effect=[
                PublishError("HTTP 429 rate limit exceeded"),
                {"post_id": "p1", "status": "published"},
            ]
        )
        publisher.upload_media = AsyncMock(return_value="media1")
        publisher.get_status = AsyncMock(return_value={"status": "published"})

        batch = self._batch(tmp_path, platforms=[Platform.YOUTUBE])
        batch.publisher = publisher

        with patch("asyncio.sleep", new=AsyncMock()):
            await batch._publish_single_video(
                d / "video_B0RATELIMT_slideshow_images1.mp4",
                "B0RATELIMT",
                1,
                1,
                [{"platform": "youtube", "account_id": "acc"}],
            )

        assert publisher.publish.await_count == 2
        first, retry = publisher.publish.await_args_list
        assert first.kwargs["platform_contents"] == retry.kwargs["platform_contents"]
        assert retry.kwargs["platform_contents"]["youtube"]["title"]

    @pytest.mark.asyncio
    async def test_the_title_it_sends_is_clamped(self, tmp_path):
        """Driven through the publish call, because the clamp has to happen
        on this path and not merely exist on the model. A scraped Amazon
        title routinely runs past YouTube's 100-character cap.
        """
        import json
        from unittest.mock import AsyncMock, MagicMock

        from src.publisher.models import Platform

        d = _product(tmp_path, "B0LONGTITL", "slideshow_images1")
        (d / "metadata.json").write_text(
            json.dumps(
                {
                    "mode": "unified",
                    "title": "Wireless Earbuds " * 12,
                    "description": "Body copy.",
                    "hashtags": ["tech"],
                }
            )
        )

        publisher = MagicMock()
        publisher.publish = AsyncMock(
            return_value={"post_id": "p1", "status": "published"}
        )
        publisher.get_accounts = AsyncMock(
            return_value=[{"platform": "youtube", "account_id": "acc"}]
        )
        publisher.upload_media = AsyncMock(return_value="media1")

        batch = self._batch(tmp_path, platforms=[Platform.YOUTUBE])
        batch.publisher = publisher
        await batch._publish_single_video(
            d / "video_B0LONGTITL_slideshow_images1.mp4",
            "B0LONGTITL",
            1,
            1,
            [{"platform": "youtube", "account_id": "acc"}],
        )

        assert publisher.publish.await_count >= 1
        pcs = publisher.publish.await_args.kwargs["platform_contents"]
        assert len(pcs["youtube"]["title"]) <= 100


@pytest.mark.unit
class TestTheScannerReadsTheConfig:
    """The scanner has to be given the config to honour the profile."""

    def test_the_configured_profile_reaches_the_scanner(self, tmp_path):
        from types import SimpleNamespace

        from src.publisher.models import Platform

        _product(tmp_path, "B0SCANPROF", "slideshow_images1", "video_sequential")
        args = argparse.Namespace(
            outputs_dir=tmp_path, force=True, platforms=[Platform.YOUTUBE]
        )
        config = SimpleNamespace(profiles={"youtube": "video_sequential"})

        found = _scan_and_filter_videos(args, config)
        assert len(found) == 1
        assert found[0].name.endswith("video_sequential.mp4")

    def test_no_config_still_yields_one_render(self, tmp_path):
        _product(tmp_path, "B0NOCONFIG", "aaa_one", "zzz_two")
        found = _scan_and_filter_videos(_args(tmp_path))
        assert len(found) == 1


@pytest.mark.unit
class TestTheScheduleTitleIsClamped:
    """The schedule path builds its payload from raw JSON, never a
    `PublishMetadata`, so it gets no clamp from `clamp_to_limits`. A scraped
    Amazon title routinely runs past YouTube's 100-character cap, and the
    platform rejects an over-cap title.
    """

    def test_a_long_data_json_title_is_trimmed(self):
        from src.publisher.models import _trim_on_word_boundary

        raw = "Wireless Earbuds Bluetooth 5.4 Headphones " * 6
        assert len(raw) > 100
        assert len(_trim_on_word_boundary(raw, 100)) <= 100

    def test_the_schedule_path_applies_it(self):
        import inspect

        from src.publisher import schedule

        source = inspect.getsource(schedule)
        # Every title the payload carries goes through the trim. A raw
        # `fb.get("title")` or `meta.get("title")` reaching platform_contents
        # is the defect.
        assert source.count("_trim_on_word_boundary(") >= 3
