"""Configuration management for Amazon scraper.

This module handles YAML configuration loading, path management, and global
settings for the scraper.
"""

from pathlib import Path
from typing import Any

import yaml

# Global configuration storage
CONFIG: dict[str, Any] = {}
_BROWSER_CONFIG: dict[str, Any] = {}


def get_output_path(path_type: str, **kwargs) -> str:
    """Get configurable output path from YAML config

    Args:
    ----
        path_type: Type of path to get ('base', 'platform', 'products',
            'media', 'debug', 'botasaurus')
        **kwargs: Variables for path substitution (platform, keyword, asin, etc.)

    Returns:
    -------
        Configured path string with variables substituted

    """
    try:
        from ...utils.outputs_paths import get_outputs_root

        output_config = CONFIG.get("global_settings", {}).get("output_config", {})
        # Use centralized outputs path as default
        default_base = str(get_outputs_root())
        base_dir = output_config.get("base_directory", default_base)
        subdirs = output_config.get("subdirectories", {})

        if path_type == "base":
            return base_dir
        elif path_type == "platform":
            pattern = subdirs.get("platform_pattern", "{data}/{platform}/scraped_data")
            data_dir = subdirs.get("data", "data")
            platform = kwargs.get("platform", "amazon")
            full_pattern = f"{base_dir}/{pattern}"
            return full_pattern.format(data=data_dir, platform=platform, **kwargs)
        elif path_type == "products":
            platform_path = get_output_path("platform", **kwargs)
            products_dir = subdirs.get("products", "products")
            return f"{platform_path}/{products_dir}"
        elif path_type == "media":
            platform_path = get_output_path("platform", **kwargs)
            media_dir = subdirs.get("media", "media")
            return f"{platform_path}/{media_dir}"
        elif path_type == "debug":
            platform_path = get_output_path("platform", **kwargs)
            debug_dir = subdirs.get("debug", "debug")
            return f"{platform_path}/{debug_dir}"
        elif path_type == "botasaurus":
            botasaurus_dir = subdirs.get("botasaurus", "botasaurus")
            return f"{base_dir}/{botasaurus_dir}"
        else:
            # Fallback for unknown path types - use outputs base
            return base_dir
    except Exception:
        # Fallback paths if config fails - use centralized structure
        from ...utils.outputs_paths import (
            get_botasaurus_cache_directory,
            get_outputs_root,
            get_temp_directory,
        )

        outputs_base = str(get_outputs_root())
        fallback_paths = {
            "base": outputs_base,
            "platform": str(get_temp_directory()),
            "products": str(get_temp_directory() / "products"),
            "media": str(get_temp_directory() / "media"),
            "debug": str(get_temp_directory() / "debug"),
            "botasaurus": str(get_botasaurus_cache_directory()),
        }
        return fallback_paths.get(path_type, outputs_base)


def get_filename_pattern(file_type: str, **kwargs) -> str:
    """Get configurable filename pattern from YAML config

    Args:
    ----
        file_type: Type of file ('product', 'image', 'video')
        **kwargs: Variables for filename substitution

    Returns:
    -------
        Formatted filename string

    """
    try:
        output_config = CONFIG.get("global_settings", {}).get("output_config", {})
        file_patterns = output_config.get("file_patterns", {})

        if file_type == "product":
            pattern = file_patterns.get("product_file", "{keyword}_products.json")
        elif file_type == "image":
            pattern = file_patterns.get("image_file", "{asin}_image_{index}.{ext}")
        elif file_type == "video":
            pattern = file_patterns.get("video_file", "{asin}_video_{index}.{ext}")
        else:
            pattern = "{keyword}_{file_type}.{ext}"

        return pattern.format(**kwargs)
    except Exception:
        # Fallback patterns
        fallback_patterns = {
            "product": "{keyword}_products.json",
            "image": "{asin}_image_{index}.{ext}",
            "video": "{asin}_video_{index}.{ext}",
        }
        pattern = fallback_patterns.get(file_type, "{keyword}_{file_type}.{ext}")
        return pattern.format(**kwargs)


def get_default_search_parameters():
    """Get default search parameters from YAML config

    Returns
    -------
        SearchParameters instance with defaults from config

    """
    from .models import SearchParameters

    try:
        defaults = (
            CONFIG.get("scrapers", {})
            .get("amazon", {})
            .get("default_search_parameters", {})
        )
        return SearchParameters(
            min_price=defaults.get("min_price"),
            max_price=defaults.get("max_price"),
            min_rating=defaults.get("min_rating"),
            prime_only=defaults.get("prime_only", False),
            free_shipping=defaults.get("free_shipping", False),
            brands=defaults.get("brands", []),
            sort_order=defaults.get("sort_order", "relevanceblender"),
            category=defaults.get("category"),
        )
    except Exception:
        # Return basic defaults if config fails
        return SearchParameters()


def load_browser_config_from_yaml(config_path: str = "config/scrapers.yaml"):
    """Load and apply YAML configuration to global browser settings"""
    global CONFIG, _BROWSER_CONFIG

    try:
        # Handle both relative and absolute paths
        if not config_path.startswith("/"):
            project_root = Path(__file__).parent.parent.parent.parent
            config_file = project_root / config_path
        else:
            config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        with open(config_file, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
            CONFIG.update(config_data)

        # Import here to avoid circular imports
        try:
            from botasaurus import bt
            from botasaurus.user_agent import UserAgent
            from botasaurus.window_size import WindowSize

            # Extract browser-specific settings
            global_settings = CONFIG.get("global_settings", {})
            debug_mode = global_settings.get("debug_mode", False)

            # Build browser configuration from YAML with performance optimizations
            _BROWSER_CONFIG = {
                "parallel": bt.calc_max_parallel_browsers(),  # Dynamic calculation
                # for optimal resource usage
                "cache": False,  # Disabled for testing new image extraction
                "max_retry": global_settings.get("retries", 3),
                "block_images": True,  # Always block images for performance
                # (50-70% faster)
                "reuse_driver": True,
                "close_on_crash": not debug_mode,  # Debug mode keeps browser
                # open on crash
                "proxy": global_settings.get("proxy"),
                "user_agent": UserAgent.RANDOM,  # Randomize user agent for better
                # anti-detection
                "window_size": WindowSize.RANDOM,  # Randomize window size for
                # better anti-detection
                "headless": not debug_mode,  # Show browser in debug mode
                "output": get_output_path(
                    "botasaurus"
                ),  # Configurable output directory
            }
        except ImportError:
            # Fallback if botasaurus not available
            debug_mode = global_settings.get("debug_mode", False)
            _BROWSER_CONFIG = {
                "headless": not debug_mode,
                "close_on_crash": not debug_mode,
            }

        # Remove None values to prevent issues
        _BROWSER_CONFIG = {k: v for k, v in _BROWSER_CONFIG.items() if v is not None}

        return config_data

    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        print("Using fallback configuration...")
        CONFIG = {}
        _BROWSER_CONFIG = {
            "headless": True,
            "close_on_crash": True,
        }
        return {}


# Initialize on import
try:
    load_browser_config_from_yaml()
except Exception:
    # Fallback configuration if loading fails
    CONFIG = {}
    _BROWSER_CONFIG = {
        "headless": True,
        "close_on_crash": True,
    }
