"""Unit tests for per-platform video file selection (Phase 1.3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.publisher.video_selector import select_video_for_platform


@pytest.fixture
def product_dir(tmp_path: Path) -> Path:
    """Build an outputs/<asin>/ directory with two profile renders."""
    asin = "B0TESTASIN"
    d = tmp_path / asin
    d.mkdir()
    (d / f"video_{asin}_slideshow_images1.mp4").write_bytes(b"long")
    (d / f"video_{asin}_slideshow_short_20s.mp4").write_bytes(b"short")
    return d


def test_routed_profile_returns_matching_file(product_dir: Path) -> None:
    """Per-platform routing picks the configured profile's render."""
    asin = product_dir.name
    profiles = {"youtube": "slideshow_short_20s", "tiktok": "slideshow_images1"}

    yt = select_video_for_platform(product_dir, asin, "youtube", profiles)
    tk = select_video_for_platform(product_dir, asin, "tiktok", profiles)

    assert yt is not None and yt.name == f"video_{asin}_slideshow_short_20s.mp4"
    assert tk is not None and tk.name == f"video_{asin}_slideshow_images1.mp4"


def test_platform_name_case_insensitive(product_dir: Path) -> None:
    """Platform lookup normalises to lowercase."""
    asin = product_dir.name
    profiles = {"youtube": "slideshow_short_20s"}

    result = select_video_for_platform(product_dir, asin, "YouTube", profiles)
    assert result is not None
    assert result.name == f"video_{asin}_slideshow_short_20s.mp4"


def test_routed_profile_missing_falls_back(product_dir: Path) -> None:
    """When the routed profile has no render, fall back to first match."""
    asin = product_dir.name
    profiles = {"youtube": "profile_that_does_not_exist"}

    result = select_video_for_platform(product_dir, asin, "youtube", profiles)
    assert result is not None
    # sorted glob returns slideshow_images1 alphabetically first
    assert result.name == f"video_{asin}_slideshow_images1.mp4"


def test_empty_profiles_uses_first_match(product_dir: Path) -> None:
    """Without routing config, legacy behaviour: first matching render."""
    asin = product_dir.name

    result = select_video_for_platform(product_dir, asin, "youtube", {})
    assert result is not None
    assert result.name == f"video_{asin}_slideshow_images1.mp4"


def test_none_profiles_uses_first_match(product_dir: Path) -> None:
    """`None` profiles is treated the same as empty."""
    asin = product_dir.name

    result = select_video_for_platform(product_dir, asin, "youtube", None)
    assert result is not None
    assert result.name == f"video_{asin}_slideshow_images1.mp4"


def test_no_renders_returns_none(tmp_path: Path) -> None:
    """No rendered videos in the product dir returns None."""
    asin = "B0EMPTYDIR"
    d = tmp_path / asin
    d.mkdir()

    result = select_video_for_platform(d, asin, "youtube", {"youtube": "x"})
    assert result is None


def test_platform_without_routing_falls_back(product_dir: Path) -> None:
    """A platform missing from `profiles` falls back to first match."""
    asin = product_dir.name
    profiles = {"youtube": "slideshow_short_20s"}

    result = select_video_for_platform(product_dir, asin, "instagram", profiles)
    assert result is not None
    assert result.name == f"video_{asin}_slideshow_images1.mp4"
