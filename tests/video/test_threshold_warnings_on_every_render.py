"""Threshold warnings run for failed and skipped renders too.

They were logged only on the success branch, so the render a slow or
memory-heavy step broke was the one render that said nothing about it.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.video.producer.orchestration import log_threshold_warnings

SOURCE = Path("src/video/producer/orchestration.py")


@pytest.mark.req("REQ-OPS-057", "REQ-OPS-058")
def test_the_warnings_run_in_the_finally_block() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    (render,) = (
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "create_video_for_product"
    )
    finally_calls = {
        call.func.id
        for node in ast.walk(render)
        if isinstance(node, ast.Try)
        for stmt in node.finalbody
        for call in ast.walk(stmt)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    every_call = [
        call
        for call in ast.walk(render)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "log_threshold_warnings"
    ]
    assert "log_threshold_warnings" in finally_calls
    assert len(every_call) == 1, "call it once, from the finally block"


@pytest.mark.req("REQ-OPS-057", "REQ-OPS-058")
def test_each_exceeded_threshold_is_a_warning(caplog) -> None:
    monitor = MagicMock()
    monitor.check_thresholds.return_value = ["slow step", "big step"]
    config = MagicMock()
    config.debug_settings.operation_timing_threshold_sec = 180.0
    config.debug_settings.memory_usage_warning_mb = 5000

    with caplog.at_level(logging.WARNING):
        log_threshold_warnings(monitor, config)

    monitor.check_thresholds.assert_called_once_with(
        timing_threshold_sec=180.0, memory_warning_mb=5000
    )
    assert [r.getMessage() for r in caplog.records] == [
        "Performance threshold exceeded: slow step",
        "Performance threshold exceeded: big step",
    ]
