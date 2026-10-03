"""Logged durations are measured on a monotonic clock (REQ-OPS-023).

A wall-clock difference skews when the system clock steps, which an NTP sync
or a suspend does mid-run. The check reads the source: a subtraction with
`time.time()` on one side is a duration taken on the wall clock.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TIMED = [
    "src/pipeline/global_batch.py",
    "src/pipeline/phases/production.py",
    "src/pipeline/phases/scraping.py",
    "src/pipeline/phases/publishing.py",
    "src/publisher/batch.py",
    "src/scraper/amazon/batch_controller.py",
    "src/video/stt_functions.py",
]


def _is_time_time(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "time"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "time"
    )


@pytest.mark.req("REQ-OPS-023")
@pytest.mark.parametrize("path", TIMED)
def test_no_duration_is_taken_on_the_wall_clock(path: str) -> None:
    """No `time.time()` at all: a wall-clock start subtracted from
    `time.monotonic()` logs a duration of minus fifty years.
    """
    tree = ast.parse((REPO / path).read_text(encoding="utf-8"))
    wall = [node.lineno for node in ast.walk(tree) if _is_time_time(node)]
    assert not wall, f"{path} lines {wall} call time.time(); use time.monotonic()"


@pytest.mark.req("REQ-OPS-023")
def test_the_producer_s_durations_are_monotonic() -> None:
    """Step and total durations; the stored timestamps stay on the wall clock."""
    tree = ast.parse((REPO / "src/utils/performance.py").read_text(encoding="utf-8"))
    wall = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Sub)
        and (_is_time_time(node.left) or _is_time_time(node.right))
    ]
    assert not wall, f"performance.py lines {wall} subtract time.time()"
