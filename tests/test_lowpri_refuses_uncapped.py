"""The lowpri targets refuse to run without their memory cap.

When `systemd-run` was missing they printed a warning and ran uncapped,
which is the case the cap exists for. `ALLOW_UNCAPPED=1` is the opt-out.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    shutil.which("make") is None or shutil.which("ionice") is None,
    reason="needs make and ionice",
)


def path_without_systemd_run(tmp_path: Path) -> str:
    """A PATH holding every command on the current one but `systemd-run`."""
    shims = tmp_path / "bin"
    shims.mkdir()
    for directory in os.environ["PATH"].split(os.pathsep):
        folder = Path(directory)
        if not folder.is_dir():
            continue
        for entry in folder.iterdir():
            target = shims / entry.name
            if entry.name != "systemd-run" and not target.exists():
                target.symlink_to(entry)
    return str(shims)


def run_target(tmp_path: Path, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["make", "-s", "test-lowpri", "ARGS=--version"],
        cwd=REPO,
        env={**os.environ, "PATH": path_without_systemd_run(tmp_path), **env},
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.req("REQ-OPS-040")
def test_without_systemd_run_the_target_refuses(tmp_path: Path) -> None:
    result = run_target(tmp_path)
    assert result.returncode != 0
    assert "refusing to run without the memory cap" in result.stdout + result.stderr


@pytest.mark.req("REQ-OPS-040")
def test_allow_uncapped_runs_it_anyway(tmp_path: Path) -> None:
    result = run_target(tmp_path, ALLOW_UNCAPPED="1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "no memory cap" in result.stdout
