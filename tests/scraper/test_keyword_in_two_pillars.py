"""A keyword listed under two pillars is refused (REQ-CNT-107)."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from src.scraper.base.keyword_pillars import KeywordPillarError, read_keyword_pillars

TWO = {"value": ["USB C hub"], "utility": ["portable ssd", "usb c  HUB"]}


@pytest.mark.req("REQ-CNT-107")
def test_a_keyword_under_two_pillars_is_refused() -> None:
    assert read_keyword_pillars({"value": ["USB C hub"], "utility": ["ssd"]})

    with pytest.raises(KeywordPillarError, match="'value' and 'utility'"):
        read_keyword_pillars(TWO)


def test_a_repeat_under_one_pillar_is_not_a_conflict() -> None:
    keywords, pillars = read_keyword_pillars({"value": ["hub", "Hub"]})

    assert keywords == ["hub", "Hub"] and pillars == {"hub": "value"}


@pytest.mark.req("REQ-CNT-107")
def test_the_batch_loader_refuses_it(tmp_path: Path) -> None:
    from src.pipeline.config import load_global_batch_config

    config = tmp_path / "pipeline.yaml"
    config.write_text(
        "global_batch:\n  keywords:\n    value: ['USB C hub']\n"
        "    utility: ['usb c hub']\n"
    )

    with (
        patch("src.pipeline.config.format_for_run", return_value="product"),
        pytest.raises(KeywordPillarError),
    ):
        load_global_batch_config(argparse.Namespace(), config)


@pytest.mark.req("REQ-CNT-107")
def test_the_scraper_loader_refuses_it() -> None:
    from src.scraper.amazon import config as scraper_config

    batch = scraper_config.SETTINGS.batch.model_copy(update={"keywords": TWO})
    with (
        patch.object(scraper_config.SETTINGS, "batch", batch),
        pytest.raises(KeywordPillarError),
    ):
        scraper_config.load_batch_config()
