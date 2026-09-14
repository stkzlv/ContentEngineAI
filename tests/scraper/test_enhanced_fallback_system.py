"""Tests for enhanced fallback logic in configuration system."""

import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
import yaml

from src.config_manager import UnifiedConfigManager
from src.scraper.amazon.config import (
    get_default_search_parameters,
    get_filename_pattern,
    get_output_path,
    load_browser_config_from_yaml,
)
from src.scraper.config_adapter import ScraperConfigAdapter


@pytest.mark.unit
class TestEnhancedFallbackSystem:
    """Test enhanced fallback logic across the configuration system."""

    def test_a_failed_video_config_load_propagates(self):
        """It used to answer any failure with a hardcoded config -- 1920x1080
        at 30fps with Arial 48 subtitles -- so a broken YAML changed the
        render instead of stopping it.
        """
        manager = UnifiedConfigManager(config_root="non/existent/path")

        with (
            patch.object(
                manager.video_adapter,
                "get_merged_config_dict",
                side_effect=Exception("Config error"),
            ),
            pytest.raises(Exception, match="Config error"),
        ):
            manager.get_video_config({"debug": True})

    def test_a_failed_scraper_config_load_propagates(self):
        """The scraper half already re-raised two exception types (#125); now
        it substitutes nothing for any of them.
        """
        manager = UnifiedConfigManager(config_root="non/existent/path")

        with (
            patch.object(
                manager.scraper_adapter,
                "get_merged_config_dict",
                side_effect=Exception("Config error"),
            ),
            pytest.raises(Exception, match="Config error"),
        ):
            manager.get_scraper_config({"output_dir": "/custom/path"})

    def test_scraper_config_adapter_refuses_malformed_yaml(self):
        """A file that is there but does not parse is a defect, not a state
        to run in: the parse error propagates like a misspelled key (#125).
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "scraper.yaml"
            config_path.write_text("invalid: yaml: content: [")

            adapter = ScraperConfigAdapter(config_root=temp_dir)
            with pytest.raises(yaml.YAMLError):
                adapter.get_merged_config_dict()

    def test_scraper_config_adapter_missing_file_fallback(self):
        """Test ScraperConfigAdapter handles missing config file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Don't create the config file
            adapter = ScraperConfigAdapter(config_root=temp_dir)
            config = adapter.get_merged_config_dict()

            # Should use minimal fallback
            assert "global_settings" in config
            assert "scrapers" in config

    def test_scraper_config_adapter_refuses_an_unknown_structure(self):
        """A file whose keys the models do not declare fails at load.

        It used to be swallowed into a hand-written fallback dict, which read
        every value as a default of the adapter's own (#125).
        """
        from pydantic import ValidationError

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "scraper.yaml"
            config_path.write_text("valid_yaml: true\nbut_wrong_structure: yes")

            adapter = ScraperConfigAdapter(config_root=temp_dir)
            with pytest.raises(ValidationError, match="valid_yaml"):
                adapter.get_merged_config_dict()

    def test_get_output_path_config_fallback(self):
        """Test get_output_path function fallback when config fails."""
        # Mock CONFIG to cause exception in primary path
        with patch("src.scraper.amazon.config.CONFIG", {}):
            path = get_output_path("platform", platform="amazon")

            # Should fall back to temp directory structure
            assert "temp" in path or "outputs" in path

    def test_get_output_path_ultimate_fallback(self):
        """Test get_output_path ultimate fallback when all imports fail."""
        # Mock CONFIG to be empty to trigger exception
        with (
            patch("src.scraper.amazon.config.CONFIG", {}),
            patch(
                "src.scraper.amazon.config.get_outputs_root",
                side_effect=ImportError("No module"),
                create=True,
            ),
        ):
            path = get_output_path("base")

            # Should use current directory fallback
            assert "outputs" in path

    def test_get_filename_pattern_fallback(self):
        """Test get_filename_pattern function fallback logic."""
        # Mock CONFIG to cause exception in primary path
        with patch("src.scraper.amazon.config.CONFIG", {}):
            filename = get_filename_pattern("product", keyword="test")

            # Should use fallback pattern
            assert filename == "test_products.json"

    def test_get_filename_pattern_with_missing_kwargs(self):
        """Test get_filename_pattern handles missing keyword arguments gracefully."""
        # Test that function works even when some keywords are missing
        with patch("src.scraper.amazon.config.CONFIG", {}):
            # This will use fallback pattern but may miss some kwargs
            filename = get_filename_pattern("unknown_type", keyword="test")

            # Should still work and produce some filename
            assert isinstance(filename, str)
            assert len(filename) > 0

    def test_get_default_search_parameters_fallback(self):
        """Test get_default_search_parameters function fallback."""
        # Mock CONFIG to be empty
        with patch("src.scraper.amazon.config.CONFIG", {}):
            params = get_default_search_parameters()

            # Should return basic SearchParameters instance
            assert params is not None
            assert hasattr(params, "prime_only")

    def test_load_browser_config_enhanced_fallback(self):
        """Test load_browser_config_from_yaml enhanced fallback logic."""
        # Test the fallback by providing invalid config path
        config = load_browser_config_from_yaml("non/existent/config.yaml")

        # Should return enhanced fallback configuration
        assert isinstance(config, dict)

    def test_path_generation_formatting_fix(self):
        """Test platform path generation doesn't cause duplicate errors."""
        # This tests the fix for the platform path formatting issue
        path = get_output_path("platform", platform="amazon", keyword="test")

        # Should not raise any formatting errors
        assert isinstance(path, str)
        assert len(path) > 0

    def test_enhanced_error_messages(self):
        """Test that fallback logic provides informative error messages."""
        import io
        import sys
        from contextlib import redirect_stdout

        # Capture stdout to check warning messages
        captured_output = io.StringIO()

        # Mock CONFIG to trigger fallback with error message
        with (
            patch("src.scraper.amazon.config.CONFIG", {}),
            redirect_stdout(captured_output),
        ):
            get_output_path("unknown_type", platform="test")

        # Should contain warning message (though it goes to print, not stdout)
        # This test ensures the fallback logic runs without crashing

    def test_type_conversion_in_nested_values(self):
        """Test type conversion in UnifiedConfigManager._set_nested_value."""
        manager = UnifiedConfigManager()
        config: dict[str, Any] = {}

        # Test string to boolean conversion
        manager._set_nested_value(config, "debug", "true")
        assert config["debug"] is True

        manager._set_nested_value(config, "enabled", "false")
        assert config["enabled"] is False

        # Test string to number conversion
        manager._set_nested_value(config, "count", "42")
        assert config["count"] == 42

        manager._set_nested_value(config, "rate", "3.14")
        assert config["rate"] == 3.14

        # Test nested path creation
        manager._set_nested_value(config, "section.subsection.setting", "value")
        assert config["section"]["subsection"]["setting"] == "value"

    @pytest.mark.integration
    def test_an_empty_config_directory_is_not_a_working_config(self):
        """The one case a substitute config made look healthy.

        With nothing to read, the merged dict is empty, and the models refuse
        it rather than a caller rendering on whatever the defaults happen to
        be. CLI overrides cannot rescue it either: an override sets one key
        on a config that still has none of the others.
        """
        from pydantic import ValidationError

        from src.video.config_adapter import load_video_config_modular

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = UnifiedConfigManager(config_root=temp_dir)

            assert manager.get_video_config() == {}
            # An override sets its own key and nothing else; `debug` writes
            # both spellings, which is the precedence layer's business.
            assert set(manager.get_video_config({"debug": True})) == {
                "debug_mode",
                "global_settings",
            }

            with (
                patch(
                    "src.video.config_adapter.ModularConfigAdapter.get_merged_config_dict",
                    return_value={},
                ),
                pytest.raises(ValidationError),
            ):
                load_video_config_modular()
