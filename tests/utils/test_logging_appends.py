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


class TestEachRunIsFindable:
    """Appending without a boundary makes a grep ambiguous.

    `CLAUDE.md` verifies a render by grepping the log for a completion line.
    Under the old overwrite the file held one run, so a match was unambiguous.
    Appending removes that guarantee, and the producer, scraper and publisher
    log no run banner of their own -- only the batch does. So the marker has to
    be visible at the default level, not just under `--debug`.
    """

    def test_a_run_marker_is_written_at_info(self, tmp_path: Path):
        log_file = tmp_path / "run.log"

        setup_debug_logging(log_file, component_name="Probe")

        contents = log_file.read_text(encoding="utf-8")
        assert "Probe run starting" in contents
        assert "INFO" in contents

    def test_it_appears_without_debug_mode(self, tmp_path: Path):
        """The case that matters: a default run, which is most of them."""
        log_file = tmp_path / "run.log"

        setup_debug_logging(log_file, debug_mode=False, component_name="Probe")

        assert "Probe run starting" in log_file.read_text(encoding="utf-8")

    def test_two_runs_leave_two_markers(self, tmp_path: Path):
        """So a reader can tell which run a later line belongs to."""
        log_file = tmp_path / "run.log"

        setup_debug_logging(log_file, component_name="Probe")
        logging.getLogger("probe").warning("first")
        setup_debug_logging(log_file, component_name="Probe")
        logging.getLogger("probe").warning("second")

        contents = log_file.read_text(encoding="utf-8")
        assert contents.count("Probe run starting") == 2
        assert contents.index("first") < contents.rindex("Probe run starting")


class TestTheMarkerMeansARunStarted:
    """Configuring logging is not the same event as starting a run.

    The scraper configures logging at module import, so importing it -- which
    every producer, publisher and batch invocation does transitively, and so
    does `--help` -- wrote a marker claiming a scrape had begun, with no
    completion line after it. That is worse than no marker: it is a boundary
    an operator would trust while reading the runbook.
    """

    def test_it_can_be_suppressed(self, tmp_path: Path):
        log_file = tmp_path / "run.log"

        setup_debug_logging(log_file, component_name="Probe", mark_run=False)

        assert "run starting" not in log_file.read_text(encoding="utf-8")

    def test_suppressing_it_still_configures_logging(self, tmp_path: Path):
        """The suppression must not turn the call into a no-op."""
        log_file = tmp_path / "run.log"

        setup_debug_logging(log_file, component_name="Probe", mark_run=False)
        logging.getLogger("probe").warning("a line")

        assert "a line" in log_file.read_text(encoding="utf-8")

    def test_the_scraper_suppresses_it_at_import(self):
        """Read from the source: the defect was a call site, not the helper.

        Driving it would need the module re-imported, which the suite cannot
        do without disturbing the root handlers every other test shares.
        """
        import ast

        source = Path("src/scraper/amazon/scraper.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        module_level = [
            node
            for node in tree.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", None) == "setup_debug_logging"
        ]
        assert len(module_level) == 1, "the import-time call moved or multiplied"

        passed = {
            keyword.arg: keyword.value
            for keyword in module_level[0].value.keywords
            if isinstance(keyword.value, ast.Constant)
        }
        assert passed.get("mark_run") is not None, "mark_run is not passed at all"
        assert passed["mark_run"].value is False, (
            "importing the scraper writes a run marker, so every producer, "
            "publisher and batch invocation logs a scrape that never happened"
        )

    def test_the_scraper_marks_its_own_run(self):
        """Suppressing at import is only correct if the run marks itself."""
        source = Path("src/scraper/amazon/scraper.py").read_text(encoding="utf-8")

        assert "=== AmazonScraper run starting ===" in source, (
            "the import-time marker was suppressed and nothing replaced it, "
            "so a real scrape now has no boundary at all"
        )
