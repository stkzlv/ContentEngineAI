"""Third-party loggers stay quiet in every mode, and no log line is a bare rule.

`setup_debug_logging` quieted httpx, google and friends only when debug mode
was off, and every documented command passes `--debug`, so the suppression
never applied: Pillow's PNG plugin alone was 45% of a producer log, and the
Gemini SDK's per-call "AFC is enabled" line another 4,600 rows. Separator-only
lines (`"=" * 80`) broke one-event-per-line, which is what keeps grep and awk
useful on the files.
"""

from __future__ import annotations

import ast
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.utils.logging_setup import (
    DEBUG_INFO_LOGGERS,
    QUIET_LOGGERS,
    setup_debug_logging,
)
from src.utils.outputs_paths import get_project_root

REPO = get_project_root()
LOG_METHODS = frozenset({"debug", "info", "warning", "error", "critical"})


@pytest.fixture
def clean_root() -> Iterator[None]:
    """Release the file handler the setup attaches, so tmp_path can go."""
    yield
    root = logging.getLogger()
    for handler in root.handlers[:]:
        handler.close()
        root.removeHandler(handler)


@pytest.mark.parametrize("debug_mode", [False, True])
class TestNoisyLibrariesStayQuiet:
    def test_the_quiet_set_sits_at_warning(
        self, tmp_path: Path, debug_mode: bool, clean_root: None
    ):
        setup_debug_logging(tmp_path / "t.log", debug_mode=debug_mode, mark_run=False)
        for name in QUIET_LOGGERS:
            assert logging.getLogger(name).getEffectiveLevel() == logging.WARNING, name

    def test_the_measured_offenders_are_in_it(self, debug_mode: bool):
        assert {"PIL", "google_genai", "httpx", "httpcore"} <= set(QUIET_LOGGERS)

    def test_a_png_chunk_line_never_reaches_the_file(
        self, tmp_path: Path, debug_mode: bool, clean_root: None
    ):
        log_file = tmp_path / "t.log"
        setup_debug_logging(log_file, debug_mode=debug_mode, mark_run=False)
        logging.getLogger("PIL.PngImagePlugin").debug("STREAM b'IHDR' 16 13")
        logging.getLogger("google_genai.models").info("AFC is enabled")
        logging.getLogger("src.video.steps").info("kept")
        for handler in logging.getLogger().handlers:
            handler.flush()
        text = log_file.read_text()
        assert "STREAM" not in text
        assert "AFC is enabled" not in text
        assert "kept" in text


class TestDebugModeKeepsTheirInfoLines:
    def test_info_under_debug(self, tmp_path: Path, clean_root: None):
        setup_debug_logging(tmp_path / "t.log", debug_mode=True, mark_run=False)
        for name in DEBUG_INFO_LOGGERS:
            assert logging.getLogger(name).getEffectiveLevel() == logging.INFO, name

    def test_warning_otherwise(self, tmp_path: Path, clean_root: None):
        setup_debug_logging(tmp_path / "t.log", debug_mode=False, mark_run=False)
        for name in DEBUG_INFO_LOGGERS:
            assert logging.getLogger(name).getEffectiveLevel() == logging.WARNING, name


def _is_rule(node: ast.expr) -> bool:
    """`"=" * 80`, `"-" * n` and the like: a repeated single punctuation mark."""
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Mult)
        and isinstance(node.left, ast.Constant)
        and isinstance(node.left.value, str)
        and len(node.left.value) == 1
        and node.left.value in "=-*#_~"
    )


def _rule_log_calls(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in LOG_METHODS or not node.args:
            continue
        owner = node.func.value
        is_logger = (isinstance(owner, ast.Name) and owner.id.endswith("logger")) or (
            isinstance(owner, ast.Attribute) and owner.attr.endswith("logger")
        )
        if is_logger and _is_rule(node.args[0]):
            hits.append(node.lineno)
    return hits


class TestNoLogLineIsABareRule:
    def test_no_logger_call_logs_a_repeated_punctuation_mark(self):
        offenders = {
            str(p.relative_to(REPO)): lines
            for p in sorted((REPO / "src").rglob("*.py"))
            if (lines := _rule_log_calls(p))
        }
        assert offenders == {}, offenders

    def test_the_guard_sees_a_rule(self, tmp_path: Path):
        sample = tmp_path / "s.py"
        sample.write_text('logger.info("=" * 80)\nself.logger.info("-" * 40)\n')
        assert _rule_log_calls(sample) == [1, 2]

    def test_the_guard_ignores_real_messages(self, tmp_path: Path):
        sample = tmp_path / "s.py"
        sample.write_text('logger.info("Batch %s done", name)\nprint("=" * 80)\n')
        assert _rule_log_calls(sample) == []
