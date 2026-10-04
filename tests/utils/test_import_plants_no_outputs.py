"""Importing the scraper's modules creates no directories.

`botasaurus_output` built `outputs/cache`, `logs` and `reports` at import,
and the scraper config resolved its output path through `get_outputs_root`,
which creates `outputs/`, so a `--help`, a dry run or test collection on a
fresh checkout planted the tree. The writers create the directory they write
to. Each import runs in a fresh interpreter, with the project root pointed at
a temporary directory, so no module is reloaded under the rest of the suite.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PROBE = """
import sys
from pathlib import Path

import src.utils.outputs_paths as paths

paths.get_project_root = lambda: Path(sys.argv[1])
__import__(sys.argv[2])
"""


@pytest.mark.req("REQ-OPS-013")
@pytest.mark.parametrize(
    "name",
    [
        "src.scraper.amazon.botasaurus_output",
        # Builds the browser config at import, which resolves the output path.
        "src.scraper.amazon.config",
    ],
)
def test_importing_a_scraper_module_creates_nothing(tmp_path: Path, name: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(tmp_path), name],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr[-800:]
    assert list(tmp_path.iterdir()) == []
