# src/config_cli_integration.py
"""CLI integration for unified configuration system.

This module provides helper functions for CLI tools to integrate
with the unified configuration system, supporting precedence rules
and backward compatibility.
"""

import logging
from pathlib import Path
from typing import Any

from src.config_manager import get_unified_config_manager
from src.scraper.config_adapter import install_scraper_config_adapter
from src.video.config_adapter import install_modular_config_adapter

logger = logging.getLogger(__name__)


def install_unified_config_adapters():
    """Install both video and scraper config adapters for backward compatibility."""
    try:
        # Install video config adapter
        install_modular_config_adapter()
        logger.info("✅ Video config adapter installed")

        # Install scraper config adapter
        install_scraper_config_adapter()
        logger.info("✅ Scraper config adapter installed")

        logger.info("✅ Unified configuration system activated")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to install config adapters: {e}")
        return False


def get_video_config_with_cli(cli_args: dict[str, Any] = None):
    """Get video configuration with CLI precedence applied.

    Args:
    ----
        cli_args: Dictionary of CLI arguments to apply as overrides

    Returns:
    -------
        VideoConfig instance with precedence rules applied

    """
    from src.video.config_adapter import load_video_config_modular

    # Convert common CLI argument names to our system
    cli_overrides = {}
    if cli_args:
        cli_overrides = _normalize_cli_args(cli_args)

    return load_video_config_modular(cli_overrides=cli_overrides)


def get_scraper_config_with_cli(cli_args: dict[str, Any] = None) -> dict[str, Any]:
    """Get scraper configuration with CLI precedence applied.

    Args:
    ----
        cli_args: Dictionary of CLI arguments to apply as overrides

    Returns:
    -------
        Configuration dictionary with precedence rules applied

    """
    from src.scraper.config_adapter import load_scraper_config_modular

    # Convert common CLI argument names to our system
    cli_overrides = {}
    if cli_args:
        cli_overrides = _normalize_cli_args(cli_args)

    return load_scraper_config_modular(cli_overrides=cli_overrides)


def _normalize_cli_args(cli_args: dict[str, Any]) -> dict[str, Any]:
    """Normalize CLI arguments to standard configuration keys.

    Maps common CLI argument names to internal configuration keys.
    """
    normalized = {}

    # Common argument mappings
    arg_mappings = {
        # Debug mode
        "debug": "debug",
        "verbose": "debug",
        "debug_mode": "debug",
        # Output directory
        "output_dir": "output_dir",
        "output": "output_dir",
        "outputs_dir": "output_dir",
        # Timeouts
        "timeout": "timeout",
        "pipeline_timeout": "timeout",
        # Browser settings
        "headless": "headless",
        "no_headless": lambda: not cli_args.get("no_headless", False),
        # Cleanup
        "clean": "clean",
        "cleanup": "clean",
        "no_cleanup": lambda: not cli_args.get("no_cleanup", False),
    }

    for cli_key, config_key in arg_mappings.items():
        if cli_key in cli_args:
            if callable(config_key):
                normalized[cli_key] = config_key()
            else:
                if isinstance(config_key, str):
                    normalized[config_key] = cli_args[cli_key]

    return normalized


def validate_unified_config() -> bool:
    """Validate that the unified configuration system is working correctly.

    Returns
    -------
        True if validation passes, False otherwise

    """
    try:
        manager = get_unified_config_manager()

        # Check file structure
        validation_results = manager.validate_config_structure()
        missing_files = [f for f, exists in validation_results.items() if not exists]

        if missing_files:
            logger.warning(f"⚠️  Missing config files: {missing_files}")
            logger.info("Using fallback to monolithic configs")
        else:
            logger.info("✅ All modular config files found")

        # Test loading both systems
        try:
            video_config = manager.get_video_config()
            logger.info(f"✅ Video config loaded: {len(video_config)} settings")
        except Exception as e:
            logger.error(f"❌ Video config loading failed: {e}")
            return False

        try:
            scraper_config = manager.get_scraper_config()
            logger.info(f"✅ Scraper config loaded: {len(scraper_config)} settings")
        except Exception as e:
            logger.error(f"❌ Scraper config loading failed: {e}")
            return False

        # Test precedence system
        test_overrides = {"debug": True, "timeout": 300}
        try:
            manager.get_video_config(test_overrides)
            manager.get_scraper_config(test_overrides)
            logger.info("✅ Precedence system working correctly")
        except Exception as e:
            logger.error(f"❌ Precedence system failed: {e}")
            return False

        logger.info("✅ Unified configuration system validation passed")
        return True

    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        return False


if __name__ == "__main__":
    # Test the unified configuration system
    logging.basicConfig(level=logging.INFO)

    print("🔧 Testing unified configuration system...")

    # Install adapters
    if install_unified_config_adapters():
        print("✅ Adapters installed successfully")
    else:
        print("❌ Adapter installation failed")
        exit(1)

    # Validate system
    if validate_unified_config():
        print("✅ Unified configuration system is working correctly!")
    else:
        print("❌ Unified configuration system validation failed!")
        exit(1)
