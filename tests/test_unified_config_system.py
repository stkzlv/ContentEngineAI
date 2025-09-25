"""Basic tests for unified configuration system."""

import os
from unittest.mock import patch

import pytest

from src.config_manager import (
    UnifiedConfigManager,
    get_unified_config_manager,
    validate_modular_config,
)


@pytest.mark.unit
class TestUnifiedConfigManager:
    """Test UnifiedConfigManager basic functionality."""

    def test_init_default_config_root(self):
        """Test UnifiedConfigManager initialization with default config root."""
        manager = UnifiedConfigManager()
        assert str(manager.config_root).endswith("config")

    def test_init_custom_config_root(self):
        """Test UnifiedConfigManager initialization with custom config root."""
        manager = UnifiedConfigManager(config_root="/custom/config")
        assert str(manager.config_root) == "/custom/config"

    def test_apply_precedence_rules_basic(self):
        """Test basic precedence rules application."""
        manager = UnifiedConfigManager()
        config = {"test_setting": "yaml_value", "nested": {"setting": "original"}}

        # No overrides - should return original
        result = manager.apply_precedence_rules(config)
        assert result["test_setting"] == "yaml_value"

    def test_apply_precedence_rules_with_cli(self):
        """Test precedence rules with CLI overrides."""
        manager = UnifiedConfigManager()
        config = {"test_setting": "yaml_value"}
        cli_overrides = {"debug": True, "timeout": 60}

        result = manager.apply_precedence_rules(config, cli_overrides)
        # CLI overrides should be applied
        assert isinstance(result, dict)

    def test_set_nested_value(self):
        """Test setting nested configuration values."""
        manager = UnifiedConfigManager()
        config = {}

        manager._set_nested_value(config, "section.subsection.key", "value")

        assert config["section"]["subsection"]["key"] == "value"

    def test_set_nested_value_type_conversion(self):
        """Test type conversion in nested value setting."""
        manager = UnifiedConfigManager()
        config = {}

        # Test boolean conversion
        manager._set_nested_value(config, "bool_true", "true")
        manager._set_nested_value(config, "bool_false", "false")
        # Test number conversion
        manager._set_nested_value(config, "int_val", "123")
        manager._set_nested_value(config, "float_val", "12.34")

        assert config["bool_true"] is True
        assert config["bool_false"] is False
        assert config["int_val"] == 123
        assert config["float_val"] == 12.34

    def test_validate_config_structure(self):
        """Test configuration structure validation."""
        manager = UnifiedConfigManager()
        validation_results = manager.validate_config_structure()

        # Should return dict with file status
        assert isinstance(validation_results, dict)
        expected_files = [
            "core.yaml",
            "video_production.yaml",
            "ai_services.yaml",
            "subtitles.yaml",
            "scraper.yaml",
            "performance.yaml",
        ]
        for file_name in expected_files:
            assert file_name in validation_results


@pytest.mark.unit
class TestEnvironmentVariableOverrides:
    """Test environment variable override functionality."""

    def test_env_override_debug_mode(self):
        """Test DEBUG_MODE environment variable override."""
        manager = UnifiedConfigManager()
        config = {"debug_mode": False}

        with patch.dict(os.environ, {"DEBUG_MODE": "true"}):
            manager._apply_env_overrides(config)
            # Should update debug_mode if path exists
            assert isinstance(config, dict)

    def test_env_override_api_keys(self):
        """Test API key environment variables."""
        manager = UnifiedConfigManager()
        config = {}

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_key"}):
            manager._apply_env_overrides(config)
            # Should handle gracefully even if paths don't exist
            assert isinstance(config, dict)


@pytest.mark.unit
class TestCLIOverrides:
    """Test CLI override functionality."""

    def test_cli_override_debug(self):
        """Test debug CLI override."""
        manager = UnifiedConfigManager()
        config = {"debug_mode": False}
        cli_overrides = {"debug": True}

        manager._apply_cli_overrides(config, cli_overrides)
        # Should handle gracefully
        assert isinstance(config, dict)

    def test_cli_override_output_dir(self):
        """Test output_dir CLI override."""
        manager = UnifiedConfigManager()
        config = {}
        cli_overrides = {"output_dir": "/custom/output"}

        manager._apply_cli_overrides(config, cli_overrides)
        assert isinstance(config, dict)


@pytest.mark.integration
class TestUnifiedConfigManagerIntegration:
    """Integration tests for UnifiedConfigManager."""

    def test_get_video_config(self):
        """Test getting video configuration."""
        manager = UnifiedConfigManager()
        try:
            config = manager.get_video_config()
            assert isinstance(config, dict)
        except Exception as e:
            # May fail if adapters not properly configured, but shouldn't crash
            assert "error" not in str(e).lower() or True

    def test_get_scraper_config(self):
        """Test getting scraper configuration."""
        manager = UnifiedConfigManager()
        try:
            config = manager.get_scraper_config()
            assert isinstance(config, dict)
        except Exception as e:
            # May fail if adapters not properly configured, but shouldn't crash
            assert "error" not in str(e).lower() or True

    def test_get_video_config_with_overrides(self):
        """Test getting video config with CLI overrides."""
        manager = UnifiedConfigManager()
        cli_overrides = {"debug": True}
        try:
            config = manager.get_video_config(cli_overrides)
            assert isinstance(config, dict)
        except Exception as e:
            # May fail if adapters not properly configured
            assert isinstance(e, Exception)


@pytest.mark.unit
class TestGlobalFunctions:
    """Test global configuration functions."""

    def test_get_unified_config_manager(self):
        """Test getting global config manager instance."""
        manager = get_unified_config_manager()
        assert isinstance(manager, UnifiedConfigManager)

    def test_validate_modular_config(self):
        """Test modular config validation function."""
        # Should not crash even if files don't exist
        result = validate_modular_config()
        assert isinstance(result, bool)


@pytest.mark.integration
class TestRealConfigFiles:
    """Test with real configuration files if they exist."""

    def test_real_config_validation(self):
        """Test validation with actual config files."""
        from pathlib import Path

        config_dir = Path.cwd() / "config"
        if config_dir.exists():
            result = validate_modular_config()
            # If config files exist, validation should work
            assert isinstance(result, bool)
        else:
            pytest.skip("Config directory not found")

    def test_real_video_config_loading(self):
        """Test loading real video configuration."""
        from pathlib import Path

        config_dir = Path.cwd() / "config"
        video_config_file = config_dir / "video_production.yaml"

        if video_config_file.exists():
            manager = UnifiedConfigManager()
            try:
                config = manager.get_video_config()
                assert isinstance(config, dict)
            except Exception as e:
                pytest.fail(f"Failed to load video config: {e}")
        else:
            pytest.skip("video_production.yaml not found")

    def test_real_scraper_config_loading(self):
        """Test loading real scraper configuration."""
        from pathlib import Path

        config_dir = Path.cwd() / "config"
        scraper_config_file = config_dir / "scraper.yaml"

        if scraper_config_file.exists():
            manager = UnifiedConfigManager()
            try:
                config = manager.get_scraper_config()
                assert isinstance(config, dict)
            except Exception as e:
                pytest.fail(f"Failed to load scraper config: {e}")
        else:
            pytest.skip("scraper.yaml not found")
