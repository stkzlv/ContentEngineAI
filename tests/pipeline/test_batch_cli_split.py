"""The batch CLI, plan printer and phases live outside global_batch.py (#450).

`global_batch.py` was 2,792 lines: the orchestrator, a 279-line argument
parser, a 248-line `main`, a 226-line plan printer and three phases of 230
to 560 lines in one module, the heaviest single context load for any
question about the batch. The parser and `main` are in
`src/pipeline/cli.py`, the plan printer in `src/pipeline/plan.py`, the
scraping and production phases in `src/pipeline/phases/`, and
`python -m src.pipeline.global_batch` still runs `main` through the module's
own entry block.

Three things a move like this breaks quietly, each pinned below: a test that
patches a name on the module the code used to read it from (the scraper's
CLI split left four tests patching a class that was no longer read from that
module, and every test stayed green while the real class was constructed);
the module entry point, which nothing imports; and the size, which grows back
one small addition at a time.
"""

from __future__ import annotations

import ast
import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

from src.utils.outputs_paths import get_project_root

REPO = get_project_root()
BATCH = REPO / "src/pipeline/global_batch.py"
CLI = REPO / "src/pipeline/cli.py"
PLAN = REPO / "src/pipeline/plan.py"
SCRAPING = REPO / "src/pipeline/phases/scraping.py"
PRODUCTION = REPO / "src/pipeline/phases/production.py"

# The file came out of the CLI split at about 2,000 lines and the phase split
# at about 1,600. The issue's target is 1,000 after the publishing phase
# moves out (PR 3); this ratchet only says it must not grow back meanwhile.
MAX_BATCH_LINES = 1_650

MODULES = {
    "global_batch": "src.pipeline.global_batch",
    "cli": "src.pipeline.cli",
    "plan": "src.pipeline.plan",
    "scraping": "src.pipeline.phases.scraping",
    "production": "src.pipeline.phases.production",
}
PATCH_TARGET = re.compile(
    r"""["'](src\.pipeline\.(global_batch|cli|plan|phases\.scraping|phases\.production))"""
    r"""\.([A-Za-z_][A-Za-z0-9_]*)["']"""
)
# `patch.object(global_batch, "name")` or `monkeypatch.setattr(global_batch,
# "name", ...)` after `from src.pipeline import global_batch` names the same
# target without the dotted string.
PATCH_OBJECT = re.compile(
    r"""(?:patch\.object|monkeypatch\.setattr)\(\s*"""
    r"""(global_batch|cli|plan|scraping|production)\s*,"""
    r"""\s*["']([A-Za-z_][A-Za-z0-9_]*)["']"""
)
# What a test function does that makes it read a moved body: it drives `main`
# (as `cli.main()`, or a bare `main()` after importing it from `cli`, possibly
# under another name), or it runs a phase, directly or through `run_pipeline`.
DRIVERS = {
    CLI: re.compile(
        r"cli\.main\(|from src\.pipeline\.cli import[^\n]*\bmain\b|\bmain\(\)"
    ),
    SCRAPING: re.compile(r"_execute_scraping_phase\(|run_pipeline\("),
    PRODUCTION: re.compile(r"_execute_production_phase\(|run_pipeline\("),
}


def _defs(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        n.name
        for n in tree.body
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }


def _module_scope_names(path: Path) -> set[str]:
    """Every name bound at module scope: definitions and imports alike."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = _defs(path)
    for n in tree.body:
        if isinstance(n, ast.ImportFrom):
            names |= {a.asname or a.name for a in n.names}
        elif isinstance(n, ast.Import):
            names |= {(a.asname or a.name).split(".")[0] for a in n.names}
    return names


class TestTheMoveIsAMove:
    def test_the_cli_owns_the_parser_and_main(self):
        cli = _defs(CLI)
        assert {"create_argument_parser", "main"} <= cli
        assert not {"create_argument_parser", "main"} & _defs(BATCH)

    @pytest.mark.parametrize(
        ("method", "module", "function"),
        [
            ("display_execution_plan", PLAN, "display_execution_plan"),
            ("_execute_scraping_phase", SCRAPING, "run_scraping_phase"),
            ("_execute_production_phase", PRODUCTION, "run_production_phase"),
        ],
        ids=["plan", "scraping", "production"],
    )
    def test_the_body_is_a_function_the_orchestrator_delegates_to(
        self, method: str, module: Path, function: str
    ):
        assert function in _defs(module)
        assert function not in _defs(BATCH)
        tree = ast.parse(BATCH.read_text(encoding="utf-8"))
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "GlobalPipelineOrchestrator"
        )
        node = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
            and n.name == method
        )
        # A delegator: a docstring, at most an import, one call. Anything
        # longer means the body is growing back where it was moved from.
        assert len(node.body) <= 3, ast.unparse(node)

    def test_global_batch_is_not_growing_back(self):
        lines = len(BATCH.read_text(encoding="utf-8").splitlines())
        assert lines <= MAX_BATCH_LINES, (
            f"global_batch.py is {lines} lines, over {MAX_BATCH_LINES}. The next "
            "section to move is a phase (#450, PRs 2 and 3), not the bound"
        )


class TestTheEntryPointStillRuns:
    @pytest.mark.parametrize("module", ["src.pipeline", "src.pipeline.global_batch"])
    def test_python_m_help(self, module: str):
        """Nothing imports either entry block, so only running them proves it.

        The package form is the one the parser's own epilog prints; it
        imported `main` from `global_batch` and was the one the split broke.
        """
        result = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stderr[-800:]
        assert "--product-ids" in result.stdout


class TestPatchTargetsStillResolve:
    """A `patch("src.pipeline.global_batch.X")` is silently inert once X moved.

    `unittest.mock.patch` raises for a missing attribute, but a name the
    module still imports for its own use resolves fine while the caller now
    reads it from somewhere else, and the test keeps passing against the real
    object. Every patched name is checked against the module that is
    patched, and every name a moved function reads must be patched on the
    module it moved to.
    """

    @staticmethod
    def _patch_targets() -> list[tuple[Path, str, str]]:
        hits = []
        for path in sorted((REPO / "tests").rglob("*.py")):
            if path == Path(__file__).resolve():
                # This file's own docstring spells the pattern it hunts.
                continue
            text = path.read_text(encoding="utf-8")
            for module, _, name in PATCH_TARGET.findall(text):
                hits.append((path, module, name))
            for short, name in PATCH_OBJECT.findall(text):
                hits.append((path, MODULES[short], name))
        return hits

    def test_the_sweep_finds_something(self):
        assert len(self._patch_targets()) >= 10

    @pytest.mark.parametrize(
        ("module", "name"),
        sorted({(m, n) for _, m, n in _patch_targets.__func__()}),
    )
    def test_every_patched_name_exists_on_its_module(self, module: str, name: str):
        mod = importlib.import_module(module)
        assert hasattr(
            mod, name
        ), f"tests patch {module}.{name}, which it does not have"

    def test_no_test_patches_a_moved_name_on_global_batch(self):
        """The names the moved bodies define are bound in their new modules."""
        moved = _defs(CLI) | _defs(PLAN) | _defs(SCRAPING) | _defs(PRODUCTION)
        offenders = [
            f"{path.relative_to(REPO)}: {name}"
            for path, module, name in self._patch_targets()
            if module == "src.pipeline.global_batch" and name in moved
        ]
        assert not offenders, f"patched on the old module: {offenders}"

    def test_no_driver_patches_what_a_moved_body_reads_on_global_batch(self):
        """The inert form: a moved body reads a name from its own namespace.

        `cli` binds `GlobalPipelineOrchestrator` and `load_pipeline_state` at
        module scope; `phases.production` binds `select_profile_for_product`
        and `load_video_config_modular`. A test that patches one of those on
        `src.pipeline.global_batch` and then drives the body that moved
        patches a name it never looks up; when `global_batch` still has the
        attribute nothing raises, and the test runs the real object. Only
        the test function that drives the moved body is held to this, and
        only for names it does not also patch on the new module; a sibling
        test patching `global_batch` for what the orchestrator itself reads
        is patching the right place.

        One over-reach is accepted: a `run_pipeline` driver that stubs the
        production phase on the instance and patches
        `global_batch.load_video_config_modular` for the topics phase is
        named too. The advice it gets, to patch `phases.production` as well,
        is harmless, since that patch simply goes unread.
        """
        reads = {module: _module_scope_names(module) for module in DRIVERS}
        dotted = {
            CLI: MODULES["cli"],
            SCRAPING: MODULES["scraping"],
            PRODUCTION: MODULES["production"],
        }
        offenders = []
        for path in sorted((REPO / "tests").rglob("*.py")):
            if path == Path(__file__).resolve():
                continue
            text = path.read_text(encoding="utf-8")
            if "global_batch" not in text:
                continue
            for fn in ast.walk(ast.parse(text)):
                if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                segment = ast.get_source_segment(text, fn) or ""
                patched: dict[str, set[str]] = {}
                for module, _, name in PATCH_TARGET.findall(segment):
                    patched.setdefault(module, set()).add(name)
                for short, name in PATCH_OBJECT.findall(segment):
                    patched.setdefault(MODULES[short], set()).add(name)
                on_old = patched.get(MODULES["global_batch"], set())
                for module, driver in DRIVERS.items():
                    if not driver.search(segment):
                        continue
                    also_on_new = patched.get(dotted[module], set())
                    for name in sorted((on_old & reads[module]) - also_on_new):
                        offenders.append(
                            f"{path.relative_to(REPO)}::{fn.name}: {name} "
                            f"(read by {dotted[module]})"
                        )
        assert not offenders, f"patched on global_batch but read elsewhere: {offenders}"
