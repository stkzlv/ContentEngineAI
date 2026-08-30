"""Configuring logging must not destroy the previous run's log.

`setup_debug_logging` built a `FileHandler` with `mode="w"`, which truncates at
construction. So anything that merely imported a module configuring logging
wiped the log before writing a line -- and that is not hypothetical: a tool
reading the source truncated `outputs/logs/scraper.log` to zero bytes without
running the scraper at all, because an editable install resolved the import to
the working tree.

Appending alone would grow the file forever, which is what the overwrite was
buying. Rotation keeps the bound, so the fix does not trade one problem for
another.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.utils.logging_setup import (
    LOG_BACKUP_COUNT,
    LOG_MAX_BYTES,
    setup_debug_logging,
)


@pytest.fixture(autouse=True)
def _restore_root_handlers():
    """`setup_debug_logging` clears the root logger, which pytest also uses."""
    root = logging.getLogger()
    saved = root.handlers[:]
    level = root.level
    yield
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    for handler in saved:
        root.addHandler(handler)
    root.setLevel(level)


class TestAnEarlierRunSurvives:
    def test_existing_content_is_not_truncated(self, tmp_path: Path):
        """The defect: the file was empty before a single line was written."""
        log_file = tmp_path / "run.log"
        log_file.write_text("earlier run\n", encoding="utf-8")

        setup_debug_logging(log_file)

        assert "earlier run" in log_file.read_text(encoding="utf-8"), (
            "configuring logging truncated the file, so importing a module "
            "that does so destroys the previous run's log"
        )

    def test_the_new_run_is_appended_after_it(self, tmp_path: Path):
        log_file = tmp_path / "run.log"
        log_file.write_text("earlier run\n", encoding="utf-8")

        setup_debug_logging(log_file)
        logging.getLogger("probe").warning("this run")

        contents = log_file.read_text(encoding="utf-8")
        assert contents.index("earlier run") < contents.index("this run")

    def test_a_missing_file_is_created(self, tmp_path: Path):
        """Appending must not require the file to exist already."""
        log_file = tmp_path / "fresh.log"

        setup_debug_logging(log_file)
        logging.getLogger("probe").warning("first line")

        assert "first line" in log_file.read_text(encoding="utf-8")


class TestTheSizeStaysBounded:
    """Appending without a bound is what the overwrite was avoiding."""

    def test_the_handler_rotates(self, tmp_path: Path):
        log_file = tmp_path / "run.log"

        setup_debug_logging(log_file)
        handler = next(
            h
            for h in logging.getLogger().handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        )

        assert handler.maxBytes == LOG_MAX_BYTES
        assert handler.backupCount == LOG_BACKUP_COUNT

    def test_it_actually_rolls_over(self, tmp_path: Path, monkeypatch):
        """Asserting the attributes is not the same as the file being bounded."""
        import src.utils.logging_setup as module

        monkeypatch.setattr(module, "LOG_MAX_BYTES", 2048)
        log_file = tmp_path / "run.log"

        setup_debug_logging(log_file)
        logger = logging.getLogger("probe")
        for index in range(200):
            logger.warning("a line that takes up some room %d", index)

        assert log_file.stat().st_size <= 4096, "the live file grew past the cap"
        assert (tmp_path / "run.log.1").exists(), "nothing rotated"
