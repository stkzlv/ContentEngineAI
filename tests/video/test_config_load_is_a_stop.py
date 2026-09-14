"""A config that does not load is a stop, not a substitution (#455).

`get_video_config` answered any exception with a hardcoded dict -- 1920x1080
at 30fps with Arial 48 subtitles -- so a typo in a YAML file did not fail the
run, it changed the output format. Those dicts were also a second declaration
of defaults the Pydantic models already own, which is where the
`remove_temp_on_success` and `sort_order` duplicates this issue names came
from.

The scraper half was hardened in #125 by re-raising two exception types; this
covers both halves by removing the substitution instead.
"""

from __future__ import annotations

import ast

import pytest
import yaml

from src.utils.outputs_paths import get_project_root

REPO = get_project_root()
MANAGER = REPO / "src/config_manager.py"


class TestNoSubstituteConfig:
    def test_a_broken_video_yaml_raises(self, tmp_path):
        """The failure has to reach the caller, whatever its kind."""
        from src.config_manager import UnifiedConfigManager

        (tmp_path / "core.yaml").write_text("this: [is, not, {valid", encoding="utf-8")
        manager = UnifiedConfigManager(str(tmp_path))

        with pytest.raises((yaml.YAMLError, OSError, ValueError, KeyError)):
            manager.get_video_config()

    def test_a_broken_scraper_yaml_raises(self, tmp_path):
        from src.config_manager import UnifiedConfigManager

        (tmp_path / "scraper.yaml").write_text("a: [b, {c", encoding="utf-8")
        manager = UnifiedConfigManager(str(tmp_path))

        with pytest.raises((yaml.YAMLError, OSError, ValueError, KeyError)):
            manager.get_scraper_config()

    def test_no_hardcoded_fallback_config_remains(self):
        """Read from the source, because a fallback is only reachable on a
        broken config and would otherwise sit there unexercised -- which is
        how this one survived: every test ran against a config that loads.
        """
        tree = ast.parse(MANAGER.read_text(encoding="utf-8"))
        offenders = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and "fallback" in node.name.lower()
        ]
        assert not offenders, (
            f"a substitute config is back: {offenders}. A config that does not "
            "load must raise; a hardcoded one changes the render silently"
        )

    def test_the_dotted_string_reader_is_gone(self):
        """`get_config_value` was the third way to read config, and unchecked:
        one call site asked the video config for a scraper-only key and got
        its own default on every run.
        """
        import src.config_manager as manager

        assert not hasattr(manager, "get_config_value")


class TestTheDefaultsHaveOneDeclaration:
    def test_the_amazon_sort_token_agrees_across_layers(self):
        """Two layers legitimately carry a default -- the YAML model and the
        runtime dataclass -- so they cannot collapse into one declaration.
        Drift between them is what mattered, so it is asserted instead.
        """
        from src.scraper.amazon.models import SearchParameters
        from src.scraper.config_models import (
            SearchParameters as TypedSearchParameters,
        )

        typed = TypedSearchParameters.model_fields["sort_order"].default
        assert SearchParameters().sort_order == typed

    def test_the_yaml_reader_declares_no_defaults_of_its_own(self):
        """`get_default_search_parameters` restated six of them, so a value
        absent from the file came from there rather than from the dataclass.
        """
        source = (REPO / "src/scraper/amazon/config.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        reader = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "get_default_search_parameters"
        )
        # Only reads off the loaded block; `CONFIG.get("scrapers", {})` and
        # its siblings are navigation, and their `{}` is not a value default.
        defaults = [
            ast.unparse(call)
            for call in ast.walk(reader)
            if isinstance(call, ast.Call)
            and getattr(call.func, "attr", None) == "get"
            and getattr(call.func.value, "id", None) == "defaults"
            and len(call.args) > 1
        ]
        assert not defaults, f"per-key defaults restated in the reader: {defaults}"
