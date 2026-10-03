"""A scraper run that stops before scraping exits 1, like any empty run.

It returned normally when the input file was missing, no inputs were
configured or the search filters were invalid, so a wrapper read a config
error as a successful run.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.scraper.amazon import cli as scraper_cli


def run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *argv: str) -> int:
    monkeypatch.setattr("sys.argv", ["scraper", *argv])
    with (
        patch.object(scraper_cli, "_start_logging", return_value=tmp_path / "x.log"),
        patch.object(scraper_cli, "BotasaurusAmazonScraper") as scraper,
        pytest.raises(SystemExit) as exit_info,
    ):
        scraper_cli.main()
    scraper.assert_not_called()
    return int(exit_info.value.code or 0)


@pytest.mark.req("REQ-SCR-049")
def test_a_missing_input_file_exits_one(monkeypatch, tmp_path: Path) -> None:
    assert run(monkeypatch, tmp_path, "--input-file", str(tmp_path / "none.txt")) == 1


@pytest.mark.req("REQ-SCR-049")
def test_no_configured_inputs_exits_one(monkeypatch, tmp_path: Path) -> None:
    with patch.object(scraper_cli, "_load_scraper_config", return_value=None):
        assert run(monkeypatch, tmp_path) == 1


@pytest.mark.req("REQ-SCR-049")
def test_invalid_filters_exit_one(monkeypatch, tmp_path: Path) -> None:
    code = run(
        monkeypatch,
        tmp_path,
        "--keywords",
        "lamp",
        "--min-price",
        "50",
        "--max-price",
        "10",
    )
    assert code == 1
