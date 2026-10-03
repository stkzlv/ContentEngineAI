"""`--product-index` renders that one product, and an out-of-range one is an error.

It fell back to every product, so the "out of range" error could not fire
for a non-empty file and a typo rendered the whole file.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.video.producer.cli import main, selected_indices


def test_no_index_is_every_product() -> None:
    assert selected_indices(None, 3) == [0, 1, 2]


def test_an_index_in_range_is_that_product() -> None:
    assert selected_indices(1, 3) == [1]


def test_an_index_out_of_range_is_refused() -> None:
    assert selected_indices(3, 3) is None
    assert selected_indices(-1, 3) is None


@pytest.mark.asyncio
async def test_main_exits_one_for_an_index_out_of_range(tmp_path: Path, mock_config):
    """Driven through `main`, so a fallback to every product can't come back."""
    products = tmp_path / "products.json"
    products.write_text(
        json.dumps(
            [
                {
                    "asin": f"B0INDEX00{i}",
                    "title": "P",
                    "price": "$1",
                    "url": "http",
                    "platform": "amazon",
                }
                for i in (1, 2)
            ]
        )
    )
    args = MagicMock()
    for name in (
        "batch",
        "random_profile",
        "debug",
        "clean",
        "fail_fast",
        "ass_karaoke",
        "ass_fade",
    ):
        setattr(args, name, False)
    for name in (
        "step",
        "product_ids",
        "profile_pool",
        "topic",
        "topic_description",
        "topic_keywords",
        "topics_file",
        "batch_profile",
        "subtitle_content_aware",
    ):
        setattr(args, name, None)
    args.products_file = products
    args.profile = "test_profile"
    args.product_index = 5
    args.output_format = "text"
    mock_config.video_profiles = {"test_profile": {}}

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=args),
        patch(
            "src.video.producer.cli.load_video_config_modular", return_value=mock_config
        ),
        patch("src.video.producer.cli.setup_logging", return_value=Path("test.log")),
        patch("src.video.producer.cli.validate_config_and_exit_on_error"),
        patch("src.video.producer.cli.load_dotenv"),
        patch(
            "src.video.producer.cli.create_video_for_product", new_callable=AsyncMock
        ) as create,
        pytest.raises(SystemExit) as exit_info,
    ):
        await main()

    assert exit_info.value.code == 1
    create.assert_not_called()
