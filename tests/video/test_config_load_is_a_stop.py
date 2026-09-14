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
        """`get_config_value` was the third way to read config, and unchecked."""
        import src.config_manager as manager

        assert not hasattr(manager, "get_config_value")

    def test_the_key_it_read_is_a_declared_field_now(self):
        """Removing an unchecked reader removes the settings only it could see.

        `system_timeouts` is in `config/core.yaml` and merged into the video
        config, but `VideoConfig` never declared it, so the dotted reader was
        the one thing that could reach it -- and the value it returned equalled
        the call site's own default, which is what made the knob look dead.
        """
        from src.video.config import VideoConfig, config
        from src.video.config_adapter import ModularConfigAdapter

        merged = ModularConfigAdapter().get_merged_config_dict()
        assert "system_timeouts" in merged, "core.yaml stopped carrying the block"
        assert config.system_timeouts.ffprobe_timeout > 0

        # Against a value the model does not default to: the shipped 10 equals
        # the field default, so comparing them would hold even if the block
        # stopped reaching the model at all -- which is the silent drop this
        # exists to catch.
        raised = VideoConfig(**{**merged, "system_timeouts": {"ffprobe_timeout": 45}})
        assert (
            raised.system_timeouts.ffprobe_timeout == 45
        ), "the YAML value no longer reaches the model"


class TestTheDefaultsHaveOneDeclaration:
    def test_the_two_layers_agree_field_by_field(self):
        """Two layers legitimately carry a default -- the YAML model and the
        runtime dataclass -- so they cannot collapse into one declaration.
        Drift between them is what mattered, so it is asserted instead, over
        every shared field rather than the one token that had drifted.
        """
        from src.scraper.amazon.models import SearchParameters
        from src.scraper.config_models import (
            SearchParameters as TypedSearchParameters,
        )

        runtime = SearchParameters()
        drifted = {
            name: (getattr(runtime, name), field.default)
            for name, field in TypedSearchParameters.model_fields.items()
            if hasattr(runtime, name)
            and field.default is not None
            and getattr(runtime, name) != field.default
        }
        assert not drifted, f"the two layers disagree: {drifted}"

    def test_the_yaml_reader_declares_no_defaults_of_its_own(self):
        """`get_default_search_parameters` restated four of them, so a value
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
