"""Clean product images first, text-heavy ones only to fill (design 0012)."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from src.video.config import config
from src.video.image_curation import (
    ImageScore,
    curate,
    parse_judgement,
    read_cached,
    score_images,
    write_cache,
)


@pytest.mark.parametrize(
    ("answer", "share", "composite"),
    [
        ('{"text_share": 0.3, "composite": true}', 0.3, True),
        ('```json\n{"text_share": 0.0, "composite": false}\n```', 0.0, False),
        ('noise "text_share": 0.25 more', 0.25, None),
        ('{"text_share": 1.7}', None, None),
        ("I cannot tell", None, None),
        (None, None, None),
    ],
)
def test_parse_judgement(answer, share, composite) -> None:
    score = parse_judgement(answer)

    assert score.text_share == share
    assert score.composite == composite


def _paths(n: int) -> list[Path]:
    return [Path(f"img_{i}.jpg") for i in range(n)]


@pytest.mark.req("REQ-VID-015")
def test_clean_images_come_first_and_heavy_ones_go_above_the_minimum() -> None:
    images = _paths(6)
    scores = [ImageScore(s) for s in (0.3, 0.0, 0.25, 0.05, 0.4, 0.1)]

    kept, dropped = curate(images, scores, 0.15, 3)

    assert kept == [images[1], images[3], images[5]]
    assert dropped == [images[2], images[0], images[4]]


@pytest.mark.req("REQ-VID-015")
def test_too_few_clean_ones_keep_the_least_text_heavy() -> None:
    images = _paths(5)
    scores = [ImageScore(s) for s in (0.3, 0.0, 0.25, 0.5, 0.2)]

    kept, dropped = curate(images, scores, 0.15, 3)

    assert kept == [images[1], images[4], images[2]]
    assert dropped == [images[0], images[3]]


@pytest.mark.req("REQ-VID-015")
def test_a_failed_judgement_never_removes_an_image() -> None:
    images = _paths(4)
    scores = [ImageScore(None), ImageScore(0.6), ImageScore(None), ImageScore(0.0)]

    kept, dropped = curate(images, scores, 0.15, 1)

    assert kept == [images[3], images[0], images[2]]
    assert dropped == [images[1]]
    kept_all, dropped_none = curate(images, [ImageScore(None)] * 4, 0.15, 1)
    assert kept_all == images and dropped_none == []


def _image(tmp_path: Path, name: str = "a.jpg") -> Path:
    path = tmp_path / name
    Image.new("RGB", (64, 64), "white").save(path)
    return path


def test_the_cache_holds_for_the_same_file_only(tmp_path: Path) -> None:
    image = _image(tmp_path)
    write_cache(image, ImageScore(0.2, True))

    assert read_cached(image) == ImageScore(0.2, True)
    Image.new("RGB", (64, 64), "black").save(image)
    os.utime(image, ns=(1, 1))
    assert read_cached(image) is None


def test_an_unknown_score_is_not_cached(tmp_path: Path) -> None:
    image = _image(tmp_path)
    write_cache(image, ImageScore(None))

    assert read_cached(image) is None


@pytest.mark.asyncio
async def test_cached_images_cost_no_judgement(tmp_path: Path) -> None:
    image = _image(tmp_path)
    write_cache(image, ImageScore(0.1, False))

    with patch("google.genai.Client") as client:
        scores = await score_images(
            [image], api_key="k", model="m", concurrency=1, timeout_seconds=1
        )

    client.assert_not_called()
    assert scores == [ImageScore(0.1, False)]


@pytest.mark.req("REQ-VID-015")
def test_the_shipped_config_keeps_curation_off() -> None:
    assert config.video_settings.image_curation.enabled is False


def _ctx(enabled: bool, secrets: dict) -> SimpleNamespace:
    cfg = config.model_copy(deep=True)
    cfg.video_settings.image_curation.enabled = enabled
    return SimpleNamespace(config=cfg, secrets=secrets, state={})


@pytest.mark.req("REQ-VID-015")
@pytest.mark.asyncio
async def test_the_gather_step_curates_and_records_the_choice() -> None:
    from src.video.producer import steps

    images = _paths(7)
    shares = (0.0, 0.3, 0.0, 0.25, 0.0, 0.2, 0.0)
    ctx = _ctx(True, {config.llm_settings.api_key_env_var: "k"})
    fake = AsyncMock(return_value=[ImageScore(s) for s in shares])

    with patch("src.video.image_curation.score_images", fake):
        kept = await steps._curate_images(ctx, images, [])

    # Four clean, then the least text-heavy up to the image-only minimum.
    floor = config.video_settings.min_images_if_no_video
    assert kept[:4] == [images[0], images[2], images[4], images[6]]
    assert len(kept) == max(4, floor)
    assert fake.call_args.kwargs["model"] == "gemini-2.5-flash"
    recorded = ctx.state["image_curation"]
    assert recorded["scores"]["img_1.jpg"]["text_share"] == 0.3
    assert recorded["kept"] == [p.name for p in kept]


@pytest.mark.asyncio
async def test_off_or_without_a_key_the_images_stay() -> None:
    from src.video.producer import steps

    images = _paths(4)
    fake = AsyncMock()
    with patch("src.video.image_curation.score_images", fake):
        assert await steps._curate_images(_ctx(False, {"X": "k"}), images, []) == (
            images
        )
        assert await steps._curate_images(_ctx(True, {}), images, []) == images

    fake.assert_not_called()


@pytest.mark.req("REQ-VID-015")
def test_the_gather_step_curates_after_media_validation() -> None:
    """Curation trims what validation counted, never the other way round."""
    import inspect

    from src.video.producer import steps

    source = inspect.getsource(steps.step_gather_visuals)
    validate = source.index("validate_media_requirements(")
    curated = source.index("await _curate_images(ctx, scraped_images, scraped_videos)")
    saved = source.index("save_visuals_info(")
    assert validate < curated < saved
