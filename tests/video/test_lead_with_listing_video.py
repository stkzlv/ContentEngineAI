"""A product render opens on the listing's own video (REQ-VID-016)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.video.assembler.video_strategies import VideoStrategyFactory
from src.video.assembler.visual_builder import VisualFilterBuilder, lead_with
from src.video.config import config
from src.video.config.visual_models import VideoSettings
from src.video.producer.steps import _listing_video

LEAD = Path("B0TEST_video_0.mp4")
OTHER = Path("B0TEST_video_1.mp4")
STILLS = [Path(f"B0TEST_image_{i}.jpg") for i in range(4)]


@pytest.mark.req("REQ-VID-016")
def test_the_lead_segment_moves_to_the_start_and_the_length_holds() -> None:
    timeline = [(STILLS[0], 3.0, False), (OTHER, 4.0, True), (LEAD, 5.0, True)]
    timeline.append((LEAD, 2.0, True))

    moved, opened = lead_with(timeline, LEAD)

    assert opened
    assert moved[0] == (LEAD, 5.0, True)
    assert moved[1:] == [timeline[0], timeline[1], timeline[3]]
    assert sum(d for _, d, _ in moved) == sum(d for _, d, _ in timeline)


@pytest.mark.req("REQ-VID-016")
def test_a_timeline_without_the_clip_is_unchanged() -> None:
    timeline = [(STILLS[0], 3.0, False), (OTHER, 4.0, True)]

    assert lead_with(timeline, LEAD) == (timeline, False)


@pytest.mark.req("REQ-VID-016")
def test_the_switch_ships_on_and_defaults_off() -> None:
    assert config.video_settings.lead_with_listing_video is True
    assert VideoSettings.model_fields["lead_with_listing_video"].default is False


async def _timeline(on: bool, normalize: bool = False) -> list[Path]:
    cfg = config.model_copy(deep=True)
    cfg.video_settings.lead_with_listing_video = on
    cfg.video_settings.beat_snap.enabled = False
    cfg.video_settings.video_assembly_mode = "mixed_media"
    inspector = MagicMock()
    inspector.is_video.side_effect = lambda p: Path(p).suffix == ".mp4"
    inspector.get_media_duration = AsyncMock(return_value=4.0)
    inspector.get_media_dimensions = AsyncMock(return_value=(1080, 1920))

    async def renamed(path: Path) -> Path:
        return path.with_name(f"norm_{path.name}")

    builder = VisualFilterBuilder(
        media_inspector=inspector,
        config=cfg,
        strategy_factory=VideoStrategyFactory(inspector, cfg, "B0TEST"),
        profile_settings=None,
        normalize_video_callback=renamed if normalize else None,
    )
    builder.lead_video = LEAD
    _, _, timed, *_ = await builder.build_visual_chain(
        visual_inputs=[*STILLS, OTHER, LEAD],
        total_video_duration=20.0,
        is_relative_mode=True,
        video_settings_dict=cfg.video_settings.model_dump(),
    )
    return [path for path, _, _ in timed]


@pytest.mark.req("REQ-VID-016")
@pytest.mark.asyncio
async def test_the_render_opens_on_the_listing_clip_only_when_on() -> None:
    off = await _timeline(on=False)
    on = await _timeline(on=True)

    assert off[0] != LEAD
    assert on[0] == LEAD
    assert sorted(map(str, on)) == sorted(map(str, off))


@pytest.mark.req("REQ-VID-016")
@pytest.mark.asyncio
async def test_the_clip_is_found_after_format_normalization() -> None:
    on = await _timeline(on=True, normalize=True)

    assert on[0] == LEAD.with_name(f"norm_{LEAD.name}")


@pytest.mark.req("REQ-VID-016")
def test_the_listing_clip_is_the_first_scraped_video(tmp_path: Path) -> None:
    gathered = tmp_path / "gathered_visuals.json"
    gathered.write_text(
        json.dumps(
            {
                "scraped_images": [str(STILLS[0])],
                "scraped_videos": [str(LEAD), str(OTHER)],
                "stock_media": [],
            }
        )
    )
    ctx = MagicMock()
    ctx.run_paths = {"gathered_visuals_file": gathered}

    assert _listing_video(ctx) == LEAD

    gathered.write_text(json.dumps({"scraped_videos": []}))
    assert _listing_video(ctx) is None
