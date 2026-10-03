"""The scraper checks media against the producer's minimums (REQ-VID-125).

It read a copy in `scraper.yaml`, kept equal to `video_production.yaml` by a
"Must match" comment, so a change to the producer's threshold never reached
the scraper and it kept products the producer would then skip.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.scraper.config_models import producer_media_minimums

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.req("REQ-VID-125")
def test_it_reads_the_producer_config(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "video_production.yaml").write_text(
        yaml.safe_dump(
            {
                "video_settings": {
                    "min_total_media": 4,
                    "min_images_if_no_video": 7,
                    "min_images_with_video": 3,
                }
            }
        )
    )
    with patch("src.utils.outputs_paths.get_project_root", return_value=tmp_path):
        assert producer_media_minimums() == (4, 7, 3)


def test_the_shipped_values_are_the_producer_s() -> None:
    settings = yaml.safe_load(
        (REPO / "config" / "video_production.yaml").read_text(encoding="utf-8")
    )["video_settings"]
    assert producer_media_minimums() == (
        settings["min_total_media"],
        settings["min_images_if_no_video"],
        settings["min_images_with_video"],
    )


@pytest.mark.req("REQ-VID-125")
def test_the_scraper_s_verification_uses_it() -> None:
    tree = ast.parse((REPO / "src/scraper/amazon/scraper.py").read_text())
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "producer_media_minimums" in called
    source = (REPO / "src/scraper/amazon/scraper.py").read_text()
    assert '"min_images_if_no_video"' not in source
