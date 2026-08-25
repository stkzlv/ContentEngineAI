"""`schedule` enumerates products, not video files.

A product rendered under a second profile keeps both files. Taking each of
them scheduled the same product twice — two posts on different days, each
carrying a different render, burning two slots — while `single` had always
resolved exactly one video per product through the selector.
"""

import argparse

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
