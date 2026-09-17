"""The batch CLI and plan printer live outside global_batch.py (#450, PR 1).

`global_batch.py` was 2,792 lines: the orchestrator, a 287-line argument
parser, a 250-line `main` and a 226-line plan printer in one module, the
heaviest single context load for any question about the batch. The parser
and `main` are in `src/pipeline/cli.py`, the plan printer in
`src/pipeline/plan.py`, and `python -m src.pipeline.global_batch` still runs
`main` through the module's own entry block.

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

# The file came out of the split at about 2,000 lines. The issue's target is
# 1,000 after the phases move out (PRs 2 and 3); this ratchet only says it
# must not grow back in the meantime.
MAX_BATCH_LINES = 2_100

PATCH_TARGET = re.compile(
    r"""["'](src\.pipeline\.(global_batch|cli|plan))\.([A-Za-z_][A-Za-z0-9_]*)["']"""
)
# `patch.object(global_batch, "name")` after `from src.pipeline import global_batch`
# names the same target without the dotted string.
PATCH_OBJECT = re.compile(
    r"""patch\.object\(\s*(global_batch|cli|plan)\s*,\s*["']([A-Za-z_][A-Za-z0-9_]*)["']"""
)


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

    def test_the_plan_printer_is_a_function_the_orchestrator_delegates_to(self):
        assert "display_execution_plan" in _defs(PLAN)
        tree = ast.parse(BATCH.read_text(encoding="utf-8"))
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "GlobalPipelineOrchestrator"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "display_execution_plan"
        )
        # A delegator: a docstring, an import, one call. Anything longer means
        # the body is growing back where it was moved from.
        assert len(method.body) <= 3, ast.unparse(method)

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
                hits.append((path, f"src.pipeline.{short}", name))
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
        """The names `main` and the parser read are now bound in `cli`."""
        moved = _defs(CLI) | _defs(PLAN)
        offenders = [
            f"{path.relative_to(REPO)}: {name}"
            for path, module, name in self._patch_targets()
            if module == "src.pipeline.global_batch" and name in moved
        ]
        assert not offenders, f"patched on the old module: {offenders}"

    def test_no_main_driver_patches_what_main_reads_on_global_batch(self):
        """The inert form: `main` reads a name from `cli`'s own namespace.

        `cli` binds `GlobalPipelineOrchestrator`, `load_pipeline_state`,
        `load_video_config_modular` and the rest at module scope. A test that
        patches one of those on `src.pipeline.global_batch` and then drives
        `cli.main()` patches a name `main` never looks up; `global_batch`
        still has the attribute, so nothing raises, and the test runs the
        real object. Only tests that drive `main` are held to this; an
        orchestrator test patching `global_batch` is patching what it reads.
        """
        reads = _module_scope_names(CLI)
        offenders = []
        for path, module, name in self._patch_targets():
            if module != "src.pipeline.global_batch" or name not in reads:
                continue
            text = path.read_text(encoding="utf-8")
            if "cli.main()" in text or "from src.pipeline.cli import main" in text:
                offenders.append(f"{path.relative_to(REPO)}: {name}")
        assert not offenders, f"patched on global_batch but read by cli: {offenders}"
