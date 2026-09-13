"""Output and config paths anchor on the repo, not the working directory (#443).

The damage this class did before it was closed: a stray `src/outputs/` tree
from a mis-counted `.parent` chain during a file move, stray `outputs/` trees
under `tests/`, and a batch log line pointing at whatever directory the
command ran from -- all invisible, because the unanchored `outputs/`
gitignore matched at any depth.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.utils.outputs_paths import get_project_root

REPO = get_project_root()


class TestNoHandCountedRootChains:
    def test_src_carries_no_parent_chain_reaching_for_the_root(self):
        """Every `.parent.parent.parent`+ chain is a hand-counted guess at
        the repo root, and one has already gone wrong during a file move.
        `get_project_root()` is the single implementation; only its own
        module may count parents.
        """
        # Both spellings: chained `.parent` and subscripted `.parents[N]`.
        # The regression's recorded trigger is a file move, and either form
        # goes stale the same way.
        pattern = re.compile(r"\.parent\s*\.\s*parent\s*\.\s*parent|\.parents\s*\[")
        offenders = []
        for path in (REPO / "src").rglob("*.py"):
            if path == REPO / "src" / "utils" / "outputs_paths.py":
                continue
            if pattern.search(path.read_text(encoding="utf-8")):
                offenders.append(str(path.relative_to(REPO)))
        assert not offenders, (
            f"hand-counted project-root chains: {offenders}; "
            "use src.utils.outputs_paths.get_project_root()"
        )

    def test_no_bare_outputs_path_defaults_in_the_publisher_cli(self):
        source = (REPO / "src/publisher/late/cli.py").read_text(encoding="utf-8")
        assert 'Path("outputs")' not in source, (
            "a cwd-relative outputs default came back; route it through "
            "DEFAULT_OUTPUTS_DIR"
        )


class TestAnchoringHoldsFromAForeignCwd:
    def test_defaults_resolve_into_the_repo(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        from src.publisher.constants import DEFAULT_OUTPUTS_DIR
        from src.utils.outputs_paths import get_logs_directory

        assert DEFAULT_OUTPUTS_DIR == REPO / "outputs"
        assert get_logs_directory().is_relative_to(REPO)

    def test_config_manager_default_root_is_the_repo_config(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)

        from src.config_manager import UnifiedConfigManager

        assert UnifiedConfigManager().config_root == REPO / "config"

    def test_the_batch_outputs_dir_anchors_relative_values(self, tmp_path, monkeypatch):
        """The central fix: a foreign-cwd batch run must not scatter its
        tree (state, renders, registry) beside wherever the command ran.
        An absolute operator path is taken as given.
        """
        import argparse

        monkeypatch.chdir(tmp_path)

        from src.pipeline.config import load_global_batch_config

        cfg = load_global_batch_config(argparse.Namespace())
        assert cfg.outputs_dir == REPO / "outputs"

        explicit = load_global_batch_config(
            argparse.Namespace(outputs_dir=str(tmp_path / "elsewhere"))
        )
        assert explicit.outputs_dir == tmp_path / "elsewhere"


class TestTheIgnoreIsAnchored:
    def test_a_stray_outputs_tree_is_visible_to_git(self):
        """The unanchored pattern hid every stray tree, which is why the
        class went unnoticed; only the root outputs/ may be ignored.
        """
        gitignore = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
        assert "/outputs/" in gitignore
        assert "outputs/" not in gitignore, "the unanchored form is back"
