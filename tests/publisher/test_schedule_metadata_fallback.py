"""A scheduled post with no metadata file of its own borrows another's.

`schedule` auto-scheduling fell back to the scraped `data.json` while
`single` and the batches fell back to another platform's metadata, so a
platform whose optimized metadata was switched off posted the raw listing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.publisher.models import Platform, RecurringSlot, ScheduleConfig
from src.publisher.schedule import ScheduleManager

PRODUCT = "B0FALLBACK"


def product_with(tmp_path: Path, files: dict[str, dict]) -> Path:
    product = tmp_path / PRODUCT
    product.mkdir()
    video = product / f"video_{PRODUCT}_slideshow.mp4"
    video.write_bytes(b"video")
    for name, record in files.items():
        (product / name).write_text(json.dumps(record))
    (product / "data.json").write_text(
        json.dumps({"title": "Raw scraped listing", "description": "raw"})
    )
    return video


def captions(tmp_path: Path, video: Path, platforms: list[Platform]):
    manager = ScheduleManager(
        tmp_path / "schedule.json",
        ScheduleConfig(
            enabled=True,
            slots=[RecurringSlot("monday", "10:00:00", "UTC")],
            timezone="UTC",
        ),
    )
    return manager._captions_for(video, PRODUCT, platforms)


def record(title: str | None, caption: str) -> dict:
    return {"title": title, "description": caption, "hashtags": ["lamp"]}


@pytest.mark.req("REQ-CNT-137")
def test_a_platform_without_its_own_file_uses_another_platforms(
    tmp_path: Path,
) -> None:
    video = product_with(
        tmp_path, {"metadata_youtube.json": record("Optimized title", "yt")}
    )

    metas, titles, _ = captions(tmp_path, video, [Platform.TIKTOK, Platform.YOUTUBE])

    assert titles == {"tiktok": "Optimized title", "youtube": "Optimized title"}
    assert metas["tiktok"].description == "yt"


def test_a_platforms_own_file_wins(tmp_path: Path) -> None:
    video = product_with(
        tmp_path,
        {
            "metadata_youtube.json": record("YT title", "yt"),
            "metadata_tiktok.json": record(None, "tt"),
        },
    )

    metas, _, _ = captions(tmp_path, video, [Platform.TIKTOK])

    assert metas["tiktok"].description == "tt"


def test_borrowing_follows_the_order_and_keeps_a_title(tmp_path: Path) -> None:
    video = product_with(
        tmp_path,
        {
            "metadata_tiktok.json": record(None, "tt"),
            "metadata_instagram.json": record(None, "ig"),
        },
    )

    metas, titles, _ = captions(tmp_path, video, [Platform.YOUTUBE])

    assert metas["youtube"].description == "tt"
    assert titles == {"youtube": "Raw scraped listing"}
