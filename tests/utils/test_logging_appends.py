"""Configuring logging must not destroy the previous run's log.

`setup_debug_logging` built a `FileHandler` with `mode="w"`, which truncates at
construction. So anything that merely imported a module configuring logging
wiped the log before writing a line -- and that is not hypothetical: a tool
reading the source truncated `outputs/logs/scraper.log` to zero bytes without
running the scraper at all, because an editable install resolved the import to
the working tree.

Appending alone would grow the file forever, which is what the overwrite was
buying. A file per component per day, pruned by age, keeps the bound, so the
fix does not trade one problem for another.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest

from src.utils import logging_setup
from src.utils.logging_setup import (
    LOG_RETENTION_DAYS,
    dated_log_path,
    prune_old_logs,
    setup_debug_logging,
)
from src.utils.outputs_paths import get_project_root

REPO = get_project_root()
# The scraper's CLI, where `main()` lives; the scraper module itself
# re-exports it for `python -m src.scraper.amazon.scraper`.
_SCRAPER_SOURCE = (REPO / "src/scraper/amazon/cli.py").read_text(encoding="utf-8")


def _scraper_main() -> ast.FunctionDef:
    return next(
        node
        for node in ast.parse(_SCRAPER_SOURCE).body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )


def _is_logging_setup(call: ast.Call) -> bool:
    """Both spellings: the bare name, and `<module>.setup_debug_logging`.

    Matching only the name would let the attribute form reintroduce the
    import-time call with every guard in this file green.
    """
    func = call.func
    return (
        getattr(func, "id", None) == "setup_debug_logging"
        or getattr(func, "attr", None) == "setup_debug_logging"
    )


def _logging_setup_calls(node: ast.AST) -> list[ast.Call]:
    return [
        sub
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call) and _is_logging_setup(sub)
    ]


def _module_scope_imports(tree: ast.Module) -> list[tuple[ast.alias, bool]]:
    """Every `setup_debug_logging` alias bound while the module imports,
    each with whether it landed in a class namespace rather than the
    module's.

    Anything nested in a module-scope `try:`, `if:`, `match:` or class body
    binds the name just as a top-level import does. A function body does not:
    a function-local import deliberately resolves through the patched source
    module at call time.
    """
    found: list[tuple[ast.alias, bool]] = []
    stack: list[tuple[ast.AST, bool]] = [(node, False) for node in tree.body]
    while stack:
        node, in_class = stack.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if isinstance(node, ast.ImportFrom):
            found.extend(
                (alias, in_class)
                for alias in node.names
                if alias.name == "setup_debug_logging"
            )
        nested = in_class or isinstance(node, ast.ClassDef)
        for field in ("body", "orelse", "finalbody", "handlers", "cases"):
            stack.extend((child, nested) for child in getattr(node, field, None) or [])
    return found


def _dotted_name(path: Path) -> str:
    """The name `sys.modules` holds, which for a package is not `__init__`."""
    parts = list(path.relative_to(REPO).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _marks_run(call: ast.Call) -> bool:
    """The helper marks the run unless `mark_run=False` says otherwise."""
    for keyword in call.keywords:
        if keyword.arg == "mark_run":
            return not (
                isinstance(keyword.value, ast.Constant) and keyword.value.value is False
            )
    return True


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


class TestTheFilesAreDatedAndPruned:
    """Appending without a bound is what the overwrite was avoiding.

    Size rotation gave `scraper.log.2` with no idea what dates it covered,
    and two processes with a rotating handler each (the analytics timer and
    a manual publisher run) rename the same file from under each other. A
    file per day in append mode has neither problem.
    """

    def test_the_entry_points_name_the_file_by_the_day(self, monkeypatch):
        import datetime as dt

        monkeypatch.setattr(logging_setup, "_today", lambda: dt.date(2026, 9, 19))
        assert dated_log_path(Path("logs/producer.log")) == Path(
            "logs/producer-2026-09-19.log"
        )

    def test_a_dated_name_is_left_alone(self):
        """The scraper configures twice for one run; the second call must not
        stack a second date onto the name.
        """
        already = Path("logs/scraper-2026-09-19.log")
        assert dated_log_path(already) == already

    def test_old_dated_files_go_and_everything_else_stays(
        self, tmp_path: Path, monkeypatch
    ):
        import datetime as dt

        today = dt.date(2026, 9, 19)
        monkeypatch.setattr(logging_setup, "_today", lambda: today)
        old = (
            tmp_path
            / f"producer-{today - dt.timedelta(days=LOG_RETENTION_DAYS + 1):%Y-%m-%d}.log"
        )
        edge = (
            tmp_path
            / f"producer-{today - dt.timedelta(days=LOG_RETENTION_DAYS):%Y-%m-%d}.log"
        )
        recent = tmp_path / "producer-2026-09-18.log"
        legacy = tmp_path / "scraper.log.1"
        other = tmp_path / "analytics-failures.log"
        for path in (old, edge, recent, legacy, other):
            path.write_text("x", encoding="utf-8")

        removed = prune_old_logs(tmp_path)

        assert removed == [old]
        assert not old.exists()
        for path in (edge, recent, legacy, other):
            assert path.exists(), path.name

    def test_setup_prunes_the_directory_it_writes_to(self, tmp_path: Path, monkeypatch):
        import datetime as dt

        monkeypatch.setattr(logging_setup, "_today", lambda: dt.date(2026, 9, 19))
        stale = tmp_path / "run-2020-01-01.log"
        stale.write_text("x", encoding="utf-8")

        written = setup_debug_logging(tmp_path / "run-2026-09-19.log", mark_run=False)

        assert written == tmp_path / "run-2026-09-19.log"
        assert not stale.exists()

    @pytest.mark.parametrize(
        "rel, base",
        [
            ("src/pipeline/cli.py", "global_pipeline.log"),
            ("src/scraper/amazon/cli.py", "scraper.log"),
            ("src/video/producer/utils.py", "producer.log"),
            ("src/publisher/late/cli.py", "publisher.log"),
        ],
    )
    def test_each_entry_point_builds_a_dated_path(self, rel: str, base: str):
        source = (REPO / rel).read_text(encoding="utf-8")
        assert "dated_log_path(" in source and f'"{base}"' in source, rel


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

    The scraper used to configure logging at module import, so importing it --
    which every producer, publisher and batch invocation does transitively,
    and so does `--help` -- wrote a marker claiming a scrape had begun, with
    no completion line after it. That is worse than no marker: it is a
    boundary an operator would trust while reading the runbook. The import-time
    call is gone (see the class below); `mark_run` still gates the second,
    debug-level configuration of the same run.
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

    def test_the_scraper_marks_its_own_run_once_after_parsing(self):
        """Located structurally, not by searching the file for a string.

        A substring check passes when the literal survives only in a comment,
        and passes again when the call is moved back to module scope -- which
        is the defect this file exists to keep out. Both mutations were tried
        and both slipped through the substring form.
        """
        module = ast.parse(_SCRAPER_SOURCE)
        marking = [call for call in _logging_setup_calls(module) if _marks_run(call)]
        assert len(marking) == 1, (
            "the CLI must configure logging exactly once with the marker on: "
            "no marking call leaves a real scrape without a boundary in an "
            "appended log, two of them fake a second run"
        )

        main = _scraper_main()

        def index_of(predicate) -> int | None:
            for position, statement in enumerate(main.body):
                if predicate(ast.dump(statement)):
                    return position
            return None

        parsed_at = index_of(lambda dumped: "parse_args" in dumped)
        # The call that owns the marking `setup_debug_logging`, wherever the
        # CLI keeps it; what matters is that main reaches it after parsing.
        configured_at = index_of(lambda dumped: "_start_logging" in dumped)

        assert parsed_at is not None, "main() no longer parses arguments"
        assert configured_at is not None, "main() no longer configures logging"
        assert configured_at > parsed_at, (
            "logging is configured before argument parsing, so `--help` and "
            "an argparse error open the log and write a marker for a run "
            "that never started"
        )


class TestOnlyAnEntryPointConfiguresLogging:
    """An imported module must not point the root logger anywhere (#442).

    The scraper configured logging at module scope, and `ProductData` is
    imported from it by the producer, the publisher, the batch and the test
    suite, so every one of those processes had root aimed at the production
    scraper.log before its own `main()` ran. One pytest session appended
    430 KB of unrelated output to real scrape history and rotated the oldest
    copy out of existence; the producer's fallback `basicConfig` was a no-op
    for the same reason, because root already had handlers.
    """

    def test_the_scraper_configures_nothing_at_import(self):
        for name in ("cli.py", "scraper.py"):
            source = (REPO / "src/scraper/amazon" / name).read_text(encoding="utf-8")
            module_level = [
                node
                for node in ast.parse(source).body
                if isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and _is_logging_setup(node.value)
            ]
            assert not module_level, (
                f"{name} configures logging at import again, so importing "
                "ProductData points root at the production scraper.log"
            )

    def test_importing_the_module_leaves_root_alone(self):
        """Driven for real, in a subprocess.

        Re-importing in-process proves nothing (the module is already in
        `sys.modules`, and the suite cannot disturb the root handlers every
        other test shares). A fresh interpreter is the only honest check.
        """
        import subprocess
        import sys

        production_log = dated_log_path(REPO / "outputs/logs/scraper.log")
        before = (
            (production_log.stat().st_size, production_log.stat().st_mtime_ns)
            if production_log.exists()
            else None
        )
        probe = (
            "import logging, src.scraper.amazon.scraper;"
            "print(len(logging.getLogger().handlers))"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=180,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().splitlines()[-1] == "0", (
            "importing the scraper installed root handlers: every log record "
            "in the importing process now lands in the scraper's log file"
        )
        # The handler count is the cause; this is the damage. Opening the
        # file is enough to do it -- a rotating handler can roll the chain
        # over before a line is written.
        if before is None:
            assert not production_log.exists(), "a bare import created the log file"
        else:
            after = (production_log.stat().st_size, production_log.stat().st_mtime_ns)
            assert after == before, "a bare import touched the production log"

    def test_every_module_level_binding_is_neutralised_in_tests(self):
        """The suite's own guard against writing to outputs/logs/.

        `tests/conftest.py` patches the helper per module, because a module
        that binds the name at import keeps its own reference and patching
        the source module never reaches it. A new entry point binding it the
        same way would be missed, and the first test driving that entry
        point would append to production logs again. Function-local imports
        are not listed: they re-read the source module at call time.
        """
        from tests.conftest import _LOGGING_SETUP_SITES

        bound: set[str] = set()
        unreachable: set[str] = set()
        for path in (REPO / "src").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for alias, in_class in _module_scope_imports(tree):
                name = _dotted_name(path)
                bound.add(name)
                if alias.asname or in_class:
                    unreachable.add(f"{name}:{alias.asname or 'in a class body'}")

        # Both shapes bind the real helper somewhere the fixture's
        # `setattr(module, "setup_debug_logging", ...)` does not reach -- under
        # another name, or in a class namespace -- so listing the module would
        # turn this guard green while the entry point still wrote to the log.
        # They have to move, not be registered.
        assert not unreachable, (
            f"setup_debug_logging is bound where the test stand-in cannot "
            f"replace it: {sorted(unreachable)}; import it at module scope "
            f"under its own name, or inside the function that calls it"
        )
        missing = sorted(bound - set(_LOGGING_SETUP_SITES))
        assert not missing, (
            f"modules bind setup_debug_logging but tests do not neutralise "
            f"it there: {missing}; add them to _LOGGING_SETUP_SITES"
        )

    def test_a_module_imported_during_a_test_picks_up_the_stand_in(self):
        """Why the fixture may patch only what `sys.modules` already holds.

        Resolving all five eagerly imports the scraper, publisher and
        producer stacks into every session and costs a single-file run
        seconds. Anything imported later reads the name off the patched
        source module, which is the half this asserts.
        """
        latecomer: dict = {}
        exec(  # noqa: S102 - the point is to bind the name the way a module does
            "from src.utils.logging_setup import setup_debug_logging",
            latecomer,
        )

        assert latecomer["setup_debug_logging"] is not setup_debug_logging, (
            "a module imported mid-test bound the real helper, so the first "
            "entry point imported that late writes to outputs/logs/"
        )

    def test_the_producer_fallback_forces_its_configuration(self):
        """The one line that branch exists to emit must not be swallowed by
        a root logger something else already configured.
        """
        source = (REPO / "src/video/producer/cli.py").read_text(encoding="utf-8")
        call = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "basicConfig"
        )
        forced = {
            keyword.arg
            for keyword in call.keywords
            if isinstance(keyword.value, ast.Constant) and keyword.value.value is True
        }
        assert "force" in forced, "basicConfig is a no-op once root has handlers"
