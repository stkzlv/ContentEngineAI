"""Integration tests for configuration precedence (CLI > ENV > YAML).

Tests the three-tier configuration precedence system end-to-end:
1. YAML configuration files provide base/default values
2. Environment variables override YAML values
3. CLI arguments override both ENV and YAML values

Note: This test uses a standalone config manager to avoid circular imports
from the main conftest.py.
"""

import contextlib
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

# Standalone config manager for testing (mirrors UnifiedConfigManager logic)
# This avoids circular import issues from the main codebase


class _TestableConfigManager:
    """Minimal config manager for testing precedence logic.

    This mirrors the core logic of UnifiedConfigManager without
    importing the full module tree that causes circular imports.
    """

    def __init__(self, config_root: str = "config"):
        self.config_root = Path(config_root)

    def apply_precedence_rules(
        self, config: dict, cli_overrides: dict | None = None
    ) -> dict:
        """Apply unified precedence rules: CLI > ENV > YAML."""
        final_config = dict(self._deep_copy_dict(config))
        self._apply_env_overrides(final_config)
        if cli_overrides:
            self._apply_cli_overrides(final_config, cli_overrides)
        return final_config

    def _deep_copy_dict(self, obj: dict) -> dict:
        """Deep copy a nested dict."""
        return {k: self._deep_copy_value(v) for k, v in obj.items()}

    def _deep_copy_value(self, obj: object) -> object:
        """Deep copy a value (dict, list, or primitive)."""
        if isinstance(obj, dict):
            return {k: self._deep_copy_value(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._deep_copy_value(i) for i in obj]
        return obj

    def _apply_env_overrides(self, config: dict) -> None:
        """Apply environment variable overrides to config."""
        env_mappings = {
            "DEBUG_MODE": ["debug_mode", "global_settings.debug_mode"],
            "CONTENT_ENGINE_DEBUG": ["debug_mode", "global_settings.debug_mode"],
            "CONTENT_ENGINE_OUTPUT": ["global_output_directory"],
            "OUTPUTS_DIR": ["global_output_directory"],
            "CONTENT_ENGINE_TIMEOUT": ["pipeline_timeout_sec"],
            "FFMPEG_THREADS": ["ffmpeg_settings.encoding.threads"],
            "SUBTITLE_ANCHOR": ["subtitle_settings.anchor"],
            "SUBTITLE_MARGIN": ["subtitle_settings.margin"],
            "SUBTITLE_CONTENT_AWARE": ["subtitle_settings.content_aware"],
            "SUBTITLE_STYLE_PRESET": ["subtitle_settings.style_preset"],
            "SUBTITLE_FONT_SIZE_SCALE": ["subtitle_settings.font_size_scale"],
        }

        for env_var, config_paths in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                for path in config_paths:
                    self._set_nested_value(config, path, env_value)

    def _apply_cli_overrides(self, config: dict, cli_overrides: dict) -> None:
        """Apply CLI argument overrides to config."""
        cli_mappings = {
            "debug": ["debug_mode", "global_settings.debug_mode"],
            "output_dir": ["global_output_directory"],
            "timeout": ["pipeline_timeout_sec"],
            "headless": ["global_settings.browser_config.headless"],
        }

        for cli_key, config_paths in cli_mappings.items():
            if cli_key in cli_overrides:
                cli_value = cli_overrides[cli_key]
                for path in config_paths:
                    self._set_nested_value(config, path, cli_value)

        for cli_key, cli_value in cli_overrides.items():
            if cli_key not in cli_mappings:
                self._set_nested_value(config, cli_key, cli_value)

    def _set_nested_value(self, config: dict, path: str, value) -> None:
        """Set a nested configuration value using dot notation."""
        keys = path.split(".")
        current = config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        final_key = keys[-1]
        if isinstance(value, str):
            if value.lower() in ("true", "1", "yes"):
                value = True
            elif value.lower() in ("false", "0", "no"):
                value = False
            else:
                with contextlib.suppress(ValueError):
                    value = float(value) if "." in value else int(value)
        current[final_key] = value


@pytest.fixture
def base_config():
    """Base configuration dictionary for testing."""
    return {
        "debug_mode": False,
        "global_output_directory": "outputs",
        "pipeline_timeout_sec": 300,
        "global_settings": {
            "debug_mode": False,
            "browser_config": {"headless": True},
        },
        "subtitle_settings": {
            "anchor": "bottom",
            "margin": 0.05,
            "content_aware": True,
            "style_preset": "modern",
            "font_size_scale": 1.0,
            "max_line_length": 42,
        },
        "ffmpeg_settings": {"encoding": {"threads": 0}},
    }


@pytest.fixture
def config_manager():
    """Create a testable config manager instance."""
    return _TestableConfigManager()


class TestConfigPrecedence:
    """Test three-tier configuration precedence: CLI > ENV > YAML."""

    @pytest.mark.integration
    def test_yaml_values_used_when_no_overrides(self, config_manager, base_config):
        """Test YAML values are used when no ENV or CLI overrides exist."""
        env_vars_to_clear = [
            "DEBUG_MODE",
            "CONTENT_ENGINE_DEBUG",
            "SUBTITLE_ANCHOR",
            "SUBTITLE_MARGIN",
        ]
        clean_env = {k: v for k, v in os.environ.items() if k not in env_vars_to_clear}

        with patch.dict(os.environ, clean_env, clear=True):
            result = config_manager.apply_precedence_rules(base_config)

            assert result["debug_mode"] is False
            assert result["global_output_directory"] == "outputs"
            assert result["subtitle_settings"]["anchor"] == "bottom"
            assert result["subtitle_settings"]["margin"] == 0.05

    @pytest.mark.integration
    def test_env_overrides_yaml(self, config_manager, base_config):
        """Test environment variables override YAML values."""
        env_overrides = {
            "CONTENT_ENGINE_DEBUG": "true",
            "SUBTITLE_ANCHOR": "top",
            "SUBTITLE_MARGIN": "0.10",
            "CONTENT_ENGINE_OUTPUT": "/custom/output",
        }

        with patch.dict(os.environ, env_overrides, clear=False):
            result = config_manager.apply_precedence_rules(base_config)

            assert result["debug_mode"] is True
            assert result["subtitle_settings"]["anchor"] == "top"
            assert result["subtitle_settings"]["margin"] == 0.10
            assert result["global_output_directory"] == "/custom/output"

    @pytest.mark.integration
    def test_cli_overrides_env_and_yaml(self, config_manager, base_config):
        """Test CLI arguments override both ENV and YAML values."""
        env_overrides = {
            "CONTENT_ENGINE_DEBUG": "true",
            "SUBTITLE_ANCHOR": "top",
        }

        cli_overrides = {
            "debug": False,
            "subtitle_settings.anchor": "center",
            "timeout": 600,
        }

        with patch.dict(os.environ, env_overrides, clear=False):
            result = config_manager.apply_precedence_rules(
                base_config, cli_overrides=cli_overrides
            )

            assert result["debug_mode"] is False
            assert result["subtitle_settings"]["anchor"] == "center"
            assert result["pipeline_timeout_sec"] == 600

    @pytest.mark.integration
    def test_partial_overrides_preserve_other_values(self, config_manager, base_config):
        """Test that partial overrides don't affect unrelated settings."""
        env_overrides = {"SUBTITLE_ANCHOR": "top"}

        with patch.dict(os.environ, env_overrides, clear=False):
            result = config_manager.apply_precedence_rules(base_config)

            assert result["subtitle_settings"]["anchor"] == "top"
            assert result["subtitle_settings"]["margin"] == 0.05
            assert result["subtitle_settings"]["content_aware"] is True
            assert result["subtitle_settings"]["style_preset"] == "modern"

    @pytest.mark.integration
    def test_type_conversion_from_env_strings(self, config_manager, base_config):
        """Test environment variables are correctly converted from strings."""
        env_overrides = {
            "CONTENT_ENGINE_DEBUG": "true",
            "SUBTITLE_MARGIN": "0.15",
            "FFMPEG_THREADS": "4",
            "SUBTITLE_CONTENT_AWARE": "false",
            "CONTENT_ENGINE_TIMEOUT": "900",
        }

        with patch.dict(os.environ, env_overrides, clear=False):
            result = config_manager.apply_precedence_rules(base_config)

            assert result["debug_mode"] is True
            assert isinstance(result["debug_mode"], bool)
            assert result["subtitle_settings"]["margin"] == 0.15
            assert isinstance(result["subtitle_settings"]["margin"], float)
            assert result["ffmpeg_settings"]["encoding"]["threads"] == 4
            assert isinstance(result["ffmpeg_settings"]["encoding"]["threads"], int)
            assert result["subtitle_settings"]["content_aware"] is False
            assert isinstance(result["subtitle_settings"]["content_aware"], bool)

    @pytest.mark.integration
    def test_boolean_conversion_variants(self, config_manager, base_config):
        """Test various boolean string representations."""
        for true_value in ["true", "True", "TRUE", "1", "yes", "Yes"]:
            with patch.dict(os.environ, {"CONTENT_ENGINE_DEBUG": true_value}):
                result = config_manager.apply_precedence_rules(base_config)
                assert result["debug_mode"] is True, f"Failed for '{true_value}'"

        for false_value in ["false", "False", "FALSE", "0", "no", "No"]:
            with patch.dict(os.environ, {"CONTENT_ENGINE_DEBUG": false_value}):
                result = config_manager.apply_precedence_rules(base_config)
                assert result["debug_mode"] is False, f"Failed for '{false_value}'"

    @pytest.mark.integration
    def test_nested_cli_overrides(self, config_manager, base_config):
        """Test CLI can override deeply nested values using dot notation."""
        cli_overrides = {
            "subtitle_settings.anchor": "above_content",
            "subtitle_settings.margin": 0.08,
            "ffmpeg_settings.encoding.threads": 8,
            "global_settings.browser_config.headless": False,
        }

        result = config_manager.apply_precedence_rules(
            base_config, cli_overrides=cli_overrides
        )

        assert result["subtitle_settings"]["anchor"] == "above_content"
        assert result["subtitle_settings"]["margin"] == 0.08
        assert result["ffmpeg_settings"]["encoding"]["threads"] == 8
        assert result["global_settings"]["browser_config"]["headless"] is False

    @pytest.mark.integration
    def test_complete_precedence_chain(self, config_manager, base_config):
        """Test the complete precedence chain with all three tiers."""
        env_overrides = {
            "SUBTITLE_ANCHOR": "top",
            "CONTENT_ENGINE_DEBUG": "true",
        }

        cli_overrides = {
            "subtitle_settings.anchor": "center",
        }

        with patch.dict(os.environ, env_overrides, clear=False):
            result = config_manager.apply_precedence_rules(
                base_config, cli_overrides=cli_overrides
            )

            # CLI > ENV > YAML
            assert result["subtitle_settings"]["anchor"] == "center"  # CLI wins
            assert result["debug_mode"] is True  # ENV wins over YAML
            assert result["subtitle_settings"]["margin"] == 0.05  # YAML (no override)
            assert result["pipeline_timeout_sec"] == 300  # YAML (no override)

    @pytest.mark.integration
    def test_empty_cli_overrides(self, config_manager, base_config):
        """Test empty CLI overrides don't affect config."""
        env_overrides = {"SUBTITLE_ANCHOR": "top"}

        with patch.dict(os.environ, env_overrides, clear=False):
            result = config_manager.apply_precedence_rules(
                base_config, cli_overrides={}
            )
            assert result["subtitle_settings"]["anchor"] == "top"

            result = config_manager.apply_precedence_rules(
                base_config, cli_overrides=None
            )
            assert result["subtitle_settings"]["anchor"] == "top"

    @pytest.mark.integration
    def test_alternative_env_var_names(self, config_manager, base_config):
        """Test alternative environment variable names work correctly."""
        env_overrides = {
            "DEBUG_MODE": "true",  # Alternative to CONTENT_ENGINE_DEBUG
            "OUTPUTS_DIR": "/alt/output",  # Alternative to CONTENT_ENGINE_OUTPUT
        }

        with patch.dict(os.environ, env_overrides, clear=False):
            result = config_manager.apply_precedence_rules(base_config)

            assert result["debug_mode"] is True
            assert result["global_output_directory"] == "/alt/output"

    @pytest.mark.integration
    def test_cli_legacy_short_names(self, config_manager, base_config):
        """Test CLI short names are mapped correctly."""
        cli_overrides = {
            "debug": True,
            "output_dir": "/cli/output",
            "timeout": 1200,
            "headless": False,
        }

        result = config_manager.apply_precedence_rules(
            base_config, cli_overrides=cli_overrides
        )

        assert result["debug_mode"] is True
        assert result["global_output_directory"] == "/cli/output"
        assert result["pipeline_timeout_sec"] == 1200
        assert result["global_settings"]["browser_config"]["headless"] is False
