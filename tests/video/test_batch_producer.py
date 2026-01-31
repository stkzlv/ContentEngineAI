import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.video.producer.cli import discover_products_for_batch, main
from src.video.producer.utils import (
    ProfileUsageTracker,
    load_profile_pool,
    select_profile_for_product,
)


@pytest.fixture
def mock_outputs_dir(temp_dir):
    outputs = temp_dir / "outputs"
    outputs.mkdir()

    # Product 1: Valid
    p1 = outputs / "B0VALID1"
    p1.mkdir()
    (p1 / "data.json").write_text(
        json.dumps(
            {
                "asin": "B0VALID1",
                "title": "Product 1",
                "price": "$10",
                "url": "http",
                "platform": "amazon",
            }
        )
    )

    # Product 2: Valid
    p2 = outputs / "B0VALID2"
    p2.mkdir()
    (p2 / "data.json").write_text(
        json.dumps(
            {
                "asin": "B0VALID2",
                "title": "Product 2",
                "price": "$20",
                "url": "http",
                "platform": "amazon",
            }
        )
    )

    # Global dir to skip
    (outputs / "cache").mkdir()

    # Invalid product (missing data.json)
    (outputs / "B0INVALID").mkdir()

    return outputs


def test_discover_products_for_batch(mock_outputs_dir):
    products = discover_products_for_batch(mock_outputs_dir)
    assert len(products) == 2
    ids = {p[1].asin for p in products}
    assert ids == {"B0VALID1", "B0VALID2"}


def test_profile_usage_tracker():
    tracker = ProfileUsageTracker()
    tracker.record_usage("profile1")
    tracker.record_usage("profile1")
    tracker.record_usage("profile2")

    counts = tracker.get_counts()
    assert counts == {"profile1": 2, "profile2": 1}

    summary = tracker.format_summary()
    assert "profile1: 2 (66.7%)" in summary
    assert "profile2: 1 (33.3%)" in summary


def test_select_profile_for_product_determinism(mock_config):
    mock_config.video_profiles = {"p1": {}, "p2": {}, "p3": {}}
    pool = ["p1", "p2", "p3"]

    choice1 = select_profile_for_product("B0TEST", pool, mock_config)
    choice2 = select_profile_for_product("B0TEST", pool, mock_config)
    assert choice1 == choice2

    choices = [
        select_profile_for_product(f"PROD_{i}", pool, mock_config) for i in range(10)
    ]
    assert len(choices) == 10


def test_load_profile_pool(mock_config):
    mock_config.video_profiles = {"p1": {}, "p2": {}}

    assert load_profile_pool(["p1"], ["p2"], mock_config) == ["p1"]
    assert load_profile_pool(None, ["p2"], mock_config) == ["p2"]
    assert set(load_profile_pool(None, None, mock_config)) == {"p1", "p2"}

    with pytest.raises(ValueError, match="Invalid profile"):
        load_profile_pool(["p3"], None, mock_config)


@pytest.mark.asyncio
async def test_batch_loop_scenarios(mock_outputs_dir, mock_config):
    # Setup base mock args
    mock_args = MagicMock()
    mock_args.batch = True
    mock_args.batch_profile = "test_profile"
    mock_args.random_profile = False
    mock_args.fail_fast = False
    mock_args.outputs_dir = mock_outputs_dir
    mock_args.debug = False
    mock_args.clean = False
    mock_args.step = None
    mock_args.output_format = "text"
    mock_args.product_index = None
    mock_args.products_file = None
    mock_args.profile = None
    mock_args.profile_pool = None

    # Subtitle and other args (all default to None or False)
    for attr in [
        "subtitle_format",
        "ass_karaoke",
        "ass_fade",
        "preset",
        "subtitle_anchor",
        "subtitle_margin",
        "subtitle_content_aware",
        "font_size_scale",
        "max_subtitle_width_fraction",
        "subtitle_alignment",
        "max_line_length",
        "max_words_per_line",
        "max_duration",
        "min_duration",
        "subtitle_randomize_fonts",
        "subtitle_randomize_colors",
        "subtitle_randomize_effects",
        "subtitle_font",
        "subtitle_font_color",
        "subtitle_outline_color",
        "subtitle_background_color",
        "image_width_percent",
        "image_top_position_percent",
        "image_vertical_align",
        "target_platform",
        "metadata_mode",
    ]:
        setattr(
            mock_args,
            attr,
            None
            if "percent" in attr
            or "scale" in attr
            or "margin" in attr
            or "fraction" in attr
            or "duration" in attr
            or "length" in attr
            or "words" in attr
            else False,
        )

    # Fix specific defaults
    mock_args.ass_karaoke = False
    mock_args.ass_fade = False
    mock_args.subtitle_content_aware = None
    mock_args.subtitle_randomize_fonts = None
    mock_args.subtitle_randomize_colors = None
    mock_args.subtitle_randomize_effects = None

    mock_config.video_profiles = {"test_profile": {}}
    mock_config.pipeline_timeout_sec = 10
    mock_config.ffmpeg_settings = MagicMock()
    mock_config.ffmpeg_settings.executable_path = "ffmpeg"

    # Scenario 1: Success
    with (
        patch("argparse.ArgumentParser.parse_args", return_value=mock_args),
        patch(
            "src.video.producer.cli.load_video_config_modular", return_value=mock_config
        ),
        patch("src.video.producer.cli.setup_logging", return_value=Path("test.log")),
        patch("src.video.producer.cli.validate_config_and_exit_on_error"),
        patch("src.video.producer.cli.load_dotenv"),
        patch("os.getenv", return_value="dummy_key"),
        patch("shutil.which", return_value="/usr/bin/ffmpeg"),
        patch(
            "src.video.producer.cli.create_video_for_product",
            new_callable=AsyncMock,
        ) as mock_create,
        patch("asyncio.sleep", return_value=None),
    ):
        mock_create.return_value = Path("video.mp4")
        await main()
        assert mock_create.call_count == 2

    # Scenario 2: Fail-fast
    mock_args.fail_fast = True
    with (
        patch("argparse.ArgumentParser.parse_args", return_value=mock_args),
        patch(
            "src.video.producer.cli.load_video_config_modular", return_value=mock_config
        ),
        patch("src.video.producer.cli.setup_logging", return_value=Path("test.log")),
        patch("src.video.producer.cli.validate_config_and_exit_on_error"),
        patch("src.video.producer.cli.load_dotenv"),
        patch("os.getenv", return_value="dummy_key"),
        patch("shutil.which", return_value="/usr/bin/ffmpeg"),
        patch(
            "src.video.producer.cli.create_video_for_product",
            new_callable=AsyncMock,
        ) as mock_create,
        patch("asyncio.sleep", return_value=None),
    ):
        mock_create.side_effect = Exception("Failed!")
        await main()
        assert mock_create.call_count == 1
