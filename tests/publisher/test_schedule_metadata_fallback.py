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


@pytest.mark.req("REQ-CNT-137")
def test_a_platform_without_its_own_file_uses_another_platforms(
    tmp_path: Path,
) -> None:
    product = tmp_path / PRODUCT
    product.mkdir()
    video = product / f"video_{PRODUCT}_slideshow.mp4"
    video.write_bytes(b"video")
    (product / "metadata_youtube.json").write_text(
        json.dumps(
            {
                "title": "Optimized title",
                "description": "Optimized caption",
                "hashtags": ["lamp"],
            }
        )
    )
    (product / "data.json").write_text(
        json.dumps({"title": "Raw scraped listing", "description": "raw"})
    )
    manager = ScheduleManager(
        tmp_path / "schedule.json",
        ScheduleConfig(
            enabled=True,
            slots=[RecurringSlot("monday", "10:00:00", "UTC")],
            timezone="UTC",
        ),
    )

    _, titles, _ = manager._captions_for(
        video, PRODUCT, [Platform.TIKTOK, Platform.YOUTUBE]
    )

    assert titles == {"tiktok": "Optimized title", "youtube": "Optimized title"}
