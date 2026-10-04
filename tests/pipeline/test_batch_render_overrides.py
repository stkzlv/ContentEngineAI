"""The global batch accepts the producer's render overrides and applies them.

Both parsers register the overrides from `shared_cli`, and both build the
dotted override keys from it, so a flag cannot parse on one entry point and
be missing or inert on the other.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from src.pipeline.cli import create_argument_parser as batch_parser
from src.pipeline.config import load_global_batch_config
from src.pipeline.global_batch import GlobalPipelineOrchestrator
from src.scraper.amazon.models import ProductData
from src.video.producer.cli import create_argument_parser as producer_parser
from src.video.producer.shared_cli import add_shared_render_args

# Producer flags that steer the run rather than the render: what to render,
# how to batch it, and where to write. The batch has its own spelling of
# these where it needs one.
PRODUCER_RUN_FLAGS = {
    "-h",
    "--help",
    "--batch",
    "--topic",
    "--topic-description",
    "--topic-keywords",
    "--topics-file",
    "--batch-profile",
    "--random-profile",
    "--profile-pool",
    "--product-ids",
    "--outputs-dir",
    "--fail-fast",
    "--strict",
    "--product-index",
    "--debug",
    "--step",
    "--clean",
    "--output-format",
}


def _options(parser: argparse.ArgumentParser) -> set[str]:
    return {opt for action in parser._actions for opt in action.option_strings}


def _shared_options() -> set[str]:
    parser = argparse.ArgumentParser(add_help=False)
    add_shared_render_args(parser)
    return _options(parser)


@pytest.mark.req("REQ-BAT-017")
def test_every_producer_render_flag_is_shared() -> None:
    """A render flag added to the producer alone fails here."""
    unshared = _options(producer_parser()) - _shared_options() - PRODUCER_RUN_FLAGS

    assert unshared == set()


@pytest.mark.req("REQ-BAT-017")
def test_the_batch_accepts_every_shared_render_flag() -> None:
    assert _shared_options() <= _options(batch_parser())


ARGV = [
    "--subtitle-anchor",
    "top",
    "--no-content-aware",
    "--max-words-per-line",
    "0",
    "--font-size-scale",
    "1.3",
    "--randomize-fonts",
    "--image-width-percent",
    "0.7",
    "--metadata-mode",
    "optimized",
    "--preset",
    "bold",
]

EXPECTED = {
    "subtitle_settings.anchor": "top",
    "subtitle_settings.content_aware": False,
    "subtitle_settings.max_words_per_line": 0,
    "subtitle_settings.font_size_scale": 1.3,
    "subtitle_settings.randomize_fonts": True,
    "video_settings.image_width_percent": 0.7,
    "description_settings.metadata_mode": "optimized",
    "subtitle_settings.style_preset": "bold",
}


@pytest.mark.req("REQ-BAT-017")
@pytest.mark.asyncio
async def test_the_batch_passes_the_overrides_to_the_render(tmp_path: Path) -> None:
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        yaml.dump(
            {
                "global_batch": {
                    "product_ids": ["B0TEST1"],
                    "outputs_dir": str(tmp_path),
                    "profile": "slideshow_images1",
                    "skip_publish": True,
                }
            }
        )
    )
    config = load_global_batch_config(batch_parser().parse_args(ARGV), config_path=path)
    product = ProductData(
        title="A product",
        price="$10",
        url="https://www.amazon.com/dp/B0TEST1",
        platform=None,
        asin="B0TEST1",
    )
    seen: dict = {}

    async def fake_create(*_args, **kwargs):
        seen["cli_overrides"] = kwargs.get("cli_overrides")

    video_config = SimpleNamespace(
        pipeline_timeout_sec=900, llm_settings=SimpleNamespace(api_key_env_var=None)
    )
    with (
        patch("src.video.producer.orchestration.create_video_for_product", fake_create),
        patch("src.video.config.load_video_config", return_value=video_config),
        patch("asyncio.sleep", return_value=None),
    ):
        await GlobalPipelineOrchestrator(config)._execute_production_phase(
            [(tmp_path / "B0TEST1", product)]
        )

    assert EXPECTED.items() <= (seen["cli_overrides"] or {}).items()


@pytest.mark.req("REQ-BAT-017")
def test_the_producer_builds_the_same_overrides() -> None:
    from src.video.producer.cli import _build_cli_overrides

    args = producer_parser().parse_args(["data.json", "profile", *ARGV])

    assert EXPECTED.items() <= _build_cli_overrides(args).items()
