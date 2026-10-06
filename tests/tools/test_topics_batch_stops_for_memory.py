"""`make topics-batch` stops when the producer gives up waiting for memory.

The script starts one producer process per step and topic and moves on after
a failure, so without a stop every remaining topic would wait out the memory
guard in turn. The producer exits 75 for that case alone.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    shutil.which("ffprobe") is None, reason="the script requires ffprobe"
)


def _run(tmp_path: Path, producer_exit: int) -> tuple[int, str, list[str]]:
    topics = tmp_path / "topics.yaml"
    topics.write_text(
        yaml.safe_dump([{"title": f"Topic {n}", "description": "d"} for n in "ABC"])
    )
    calls = tmp_path / "calls"
    fake = tmp_path / "python"
    # The enumeration goes to the real interpreter; every producer call is
    # recorded and exits with the code under test.
    fake.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$2" = src.video.producer ]; then\n'
        f'  echo "$*" >> {calls}; exit {producer_exit}\n'
        "fi\n"
        f'exec {sys.executable} "$@"\n'
    )
    fake.chmod(0o755)
    result = subprocess.run(
        ["scripts/render-topics-batch.sh"],
        cwd=REPO,
        env={**os.environ, "TOPICS": str(topics), "LOWPRI_PYTHON": str(fake)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    lines = calls.read_text().splitlines() if calls.exists() else []
    return result.returncode, result.stdout, lines


@pytest.mark.req("REQ-OPS-105")
def test_a_memory_refusal_stops_the_topics_batch(tmp_path: Path) -> None:
    code, out, calls = _run(tmp_path, 75)

    assert code != 0
    assert len(calls) == 1  # Topic A's script step, then nothing
    assert "STOP  Topic A (not enough memory)" in out
    assert "stopped for memory" in out


def test_another_failure_still_moves_on(tmp_path: Path) -> None:
    code, out, calls = _run(tmp_path, 1)

    assert code != 0
    assert len(calls) == 3  # each topic's script step was tried
    assert "stopped for memory" not in out
