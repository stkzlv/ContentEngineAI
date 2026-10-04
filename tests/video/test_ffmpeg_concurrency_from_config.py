"""A render takes its FFmpeg concurrency limit from the config.

`optimization_settings.async_ffmpeg_max_concurrent` was declared and
documented, but the semaphore was built at import with a fixed limit, so the
key changed nothing. This runs the producer's entry point far enough to see
the limit applied, for the producer CLI and the batch alike.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.scraper.amazon.models import ProductData
from src.utils.async_io import ffmpeg_semaphore
from src.video.config import load_video_config_modular
from src.video.config.core_models import OptimizationSettings
from src.video.producer import orchestration


@pytest.mark.req("REQ-OPS-039")
@pytest.mark.asyncio
async def test_a_render_applies_the_configured_limit(tmp_path) -> None:
    config = load_video_config_modular()
    config.global_output_root_path = tmp_path
    config.optimization_settings = OptimizationSettings.model_validate(
        {"async_ffmpeg_max_concurrent": 1}
    )
    before = ffmpeg_semaphore.limit
    product = ProductData(title="Desk lamp", price="", url="", platform="test")

    try:
        with patch.object(
            orchestration, "PipelineContext", side_effect=RuntimeError("stop")
        ):
            await orchestration.create_video_for_product(
                config, product, "slideshow_images1", {}, None, False, False, None
            )
        assert ffmpeg_semaphore.limit == 1
    finally:
        ffmpeg_semaphore.set_limit(before)


def test_the_default_keeps_four() -> None:
    assert OptimizationSettings.model_validate({}).async_ffmpeg_max_concurrent == 4
