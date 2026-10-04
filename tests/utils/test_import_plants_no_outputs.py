"""Importing the scraper's output module creates no directories.

It used to build `outputs/cache`, `logs` and `reports` at import, so a
`--help`, a dry run or test collection on a fresh checkout planted the tree.
The writers create the directory they write to.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.mark.req("REQ-OPS-013")
def test_importing_the_output_module_creates_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("src.utils.outputs_paths.get_project_root", lambda: tmp_path)
    import src.scraper.amazon.botasaurus_output as module

    importlib.reload(module)

    assert list(tmp_path.iterdir()) == []
