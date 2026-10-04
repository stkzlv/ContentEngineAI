"""`api_settings` values in the YAML reach the model, and a nested block fails.

The shipped block nested its keys under `llm:`, `tts:` and `stock_media:`
while the model is flat, so every value was dropped and the code defaults
ran. These drive the loader the producer and the batch use.
"""

from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError

from src.config_manager import get_unified_config_manager
from src.video.config.core_models import ApiSettings
from src.video.config_adapter import load_video_config_modular

REPO = Path(__file__).resolve().parents[2]


def load_with(update) -> object:
    manager = get_unified_config_manager()
    merged = copy.deepcopy(manager.get_video_config(None))
    update(merged["api_settings"])
    with patch.object(manager, "get_video_config", return_value=merged):
        return load_video_config_modular()


def test_every_shipped_key_is_a_model_field() -> None:
    raw = yaml.safe_load((REPO / "config" / "performance.yaml").read_text())

    assert set(raw["api_settings"]) <= set(ApiSettings.model_fields)


def test_a_yaml_value_reaches_the_model() -> None:
    config = load_with(lambda a: a.update(llm_model_fetch_timeout_sec=7))

    assert config.api_settings.llm_model_fetch_timeout_sec == 7


def test_a_nested_block_fails_the_load_naming_it() -> None:
    with pytest.raises(ValidationError, match="llm"):
        load_with(lambda a: a.update(llm={"model_fetch_timeout_sec": 15}))


@pytest.mark.asyncio
async def test_stock_downloads_use_the_configured_retry(tmp_path) -> None:
    """The fetcher read these through `getattr` with its own fallbacks, so a
    removed or renamed field changed the wait without any error.
    """
    from unittest.mock import AsyncMock, MagicMock

    from src.video.stock_media import StockMediaFetcher

    config = load_with(lambda a: a.update(download_retry_min_wait_sec=5))
    fetcher = StockMediaFetcher(
        settings=config.stock_media_settings,
        secrets={"PEXELS_API_KEY": "test_key"},
        media_settings=config.media_settings,
        api_settings=config.api_settings,
    )
    found = [{"url": "https://images.example/a.jpg", "id": 1}]
    with (
        patch.object(fetcher, "_search_and_select_pexels", return_value=found),
        patch(
            "src.video.stock_media.download_file", new_callable=AsyncMock
        ) as download,
    ):
        download.return_value = False
        await fetcher.fetch_and_download_stock(
            keywords=["lamp"],
            image_count=1,
            video_count=0,
            download_dir=tmp_path,
            session=MagicMock(),
        )

    assert download.call_args.kwargs["retry_min_wait_sec"] == 5
