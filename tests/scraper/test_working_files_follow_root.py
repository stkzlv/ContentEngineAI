"""The scraper's working files follow the run's outputs root (#742, REQ-OPS-031).

The media-download records and the debug image analysis were written under
the repository's `outputs/` whatever root the scraper was given.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.scraper.amazon import botasaurus_output


@pytest.fixture
def root(tmp_path, monkeypatch):
    root = tmp_path / "elsewhere"
    root.mkdir()
    # The override is module-level: monkeypatch restores it after the test.
    monkeypatch.setattr(botasaurus_output, "_output_dir_override", str(root))
    return root


@pytest.mark.unit
@pytest.mark.req("REQ-OPS-031")
def test_download_records_are_written_under_the_root(root):
    botasaurus_output.write_download_cache_output(
        {"asin": "B0TESTREC1"}, {"downloaded": 1}
    )

    written = list(root.rglob("B0TESTREC1_downloads*"))
    assert written, sorted(str(p) for p in root.rglob("*"))


@pytest.mark.unit
@pytest.mark.req("REQ-OPS-031")
def test_debug_analysis_is_written_under_the_root(root):
    from src.scraper.amazon.debug_analysis import _perform_advanced_debug_analysis

    driver = MagicMock()
    driver.current_url = "https://www.amazon.com/dp/B0TESTDBG1/"

    _perform_advanced_debug_analysis(driver, {}, MagicMock())

    assert (root / "temp" / "debug" / "image_analysis").is_dir()
