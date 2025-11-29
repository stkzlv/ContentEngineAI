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
            return str(base_dir)
        elif path_type == "platform":
            pattern = subdirs.get("platform_pattern", "{data}/{platform}/scraped_data")
            data_dir = subdirs.get("data", "data")
            platform = kwargs.get("platform", "amazon")
            full_pattern = f"{base_dir}/{pattern}"
            # Remove platform from kwargs to avoid duplicate argument error
            safe_kwargs = {k: v for k, v in kwargs.items() if k != "platform"}
            return full_pattern.format(data=data_dir, platform=platform, **safe_kwargs)
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
            return str(base_dir)
    except Exception as e:
        # Enhanced fallback paths with better error handling
        try:
            from ...utils.outputs_paths import (
                get_botasaurus_cache_directory,
                get_outputs_root,
                get_temp_directory,
            )

            outputs_base = str(get_outputs_root())
            temp_base = get_temp_directory()

            fallback_paths = {
                "base": outputs_base,
                "platform": str(temp_base),
                "products": str(temp_base / "products"),
                "media": str(temp_base / "media"),
                "debug": str(temp_base / "debug"),
                "botasaurus": str(get_botasaurus_cache_directory()),
            }

            fallback_path = fallback_paths.get(path_type, outputs_base)
            print(
                f"⚠️  Config fallback: Using path '{fallback_path}' "
                f"for type '{path_type}' (error: {e})"
            )
            return fallback_path

        except Exception as fallback_error:
            # Ultimate fallback - use current directory with subdirectories
            import os

            current_dir = os.getcwd()
            ultimate_fallback = {
                "base": f"{current_dir}/outputs",
                "platform": f"{current_dir}/outputs/temp",
                "products": f"{current_dir}/outputs/temp/products",
                "media": f"{current_dir}/outputs/temp/media",
                "debug": f"{current_dir}/outputs/temp/debug",
                "botasaurus": f"{current_dir}/outputs/botasaurus",
            }

            ultimate_path = ultimate_fallback.get(path_type, f"{current_dir}/outputs")
            print(
                f"⚠️  Ultimate fallback: Using path '{ultimate_path}' "
                f"for type '{path_type}' (errors: {e}, {fallback_error})"
            )
            return ultimate_path


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

        return str(pattern).format(**kwargs)
    except Exception as e:
        # Enhanced fallback patterns with error handling
        try:
            fallback_patterns = {
                "product": "{keyword}_products.json",
                "image": "{asin}_image_{index}.{ext}",
                "video": "{asin}_video_{index}.{ext}",
            }
            pattern = fallback_patterns.get(file_type, "{keyword}_{file_type}.{ext}")
            formatted_pattern = str(pattern).format(**kwargs)
            print(
                f"⚠️  Config fallback: Using filename pattern "
                f"'{formatted_pattern}' for type '{file_type}' (error: {e})"
            )
            return formatted_pattern

        except Exception as fallback_error:
            # Ultimate fallback - create safe filename
            import time

            timestamp = int(time.time())
            safe_filename = f"{file_type}_{timestamp}"

            # Add appropriate extension
            if file_type == "product":
                safe_filename += ".json"
            elif file_type == "image":
                safe_filename += ".jpg"
            elif file_type == "video":
                safe_filename += ".mp4"
            else:
                safe_filename += ".txt"

            print(
                f"⚠️  Ultimate fallback: Using safe filename '{safe_filename}' "
                f"for type '{file_type}' (errors: {e}, {fallback_error})"
            )
            return safe_filename


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


def load_batch_config(
    cli_product_ids: list[str] | None = None,
    cli_keywords: list[str] | None = None,
    cli_fail_fast: bool | None = None,
    cli_max_products: int | None = None,
) -> "BatchConfig":  # type: ignore[name-defined] # noqa: F821
    """Load batch configuration with CLI > YAML > Defaults precedence.

    Implements 3-tier configuration precedence:
    1. CLI arguments (highest priority)
    2. YAML configuration
    3. Default values (lowest priority)

    Args:
    ----
        cli_product_ids: Product IDs from CLI --product-ids argument
        cli_keywords: Keywords from CLI --keywords argument
        cli_fail_fast: Fail-fast flag from CLI --fail-fast argument
        cli_max_products: Max products from CLI --max-products argument

    Returns:
    -------
        BatchConfig instance with merged configuration from all sources

    Raises:
    ------
        ValueError: If both product_ids and keywords are empty after merge

    """
    from .models import BatchConfig

    # Load YAML batch configuration with defaults
    yaml_batch = CONFIG.get("batch", {})
    yaml_product_ids = yaml_batch.get("product_ids", [])
    yaml_keywords = yaml_batch.get("keywords", [])
    yaml_fail_fast = yaml_batch.get("fail_fast", False)

    # Load max_products from scrapers.amazon config
    yaml_max_products = (
        CONFIG.get("scrapers", {}).get("amazon", {}).get("max_products", 10)
    )

    # Apply CLI > YAML > Defaults precedence
    product_ids = cli_product_ids if cli_product_ids is not None else yaml_product_ids
    keywords = cli_keywords if cli_keywords is not None else yaml_keywords
    fail_fast = cli_fail_fast if cli_fail_fast is not None else yaml_fail_fast
    max_products = (
        cli_max_products if cli_max_products is not None else yaml_max_products
    )

    # Validate lists
    if not isinstance(product_ids, list):
        raise ValueError(f"product_ids must be a list, got {type(product_ids)}")
    if not isinstance(keywords, list):
        raise ValueError(f"keywords must be a list, got {type(keywords)}")

    # Get search parameters for keyword searches
    search_params = get_default_search_parameters()

    # Create BatchConfig instance
    batch_config = BatchConfig(
        product_ids=product_ids,
        keywords=keywords,
        fail_fast=fail_fast,
        search_params=search_params,
        max_products=max_products,
    )

    return batch_config


def get_batch_logging_config() -> dict[str, str | int]:
    """Get batch logging configuration from YAML.

    Returns
    -------
        Dictionary with batch logging settings:
        - separator_char: str
        - separator_width: int
        - duration_decimal_places: int
        - media_stats_decimal_places: int

    """
    yaml_batch = CONFIG.get("batch", {})
    logging_config = yaml_batch.get("logging", {})

    separator_char: str = logging_config.get("separator_char", "=")
    separator_width: int = logging_config.get("separator_width", 60)
    duration_decimal: int = logging_config.get("duration_decimal_places", 2)
    media_stats_decimal: int = logging_config.get("media_stats_decimal_places", 2)

    return {
        "separator_char": separator_char,
        "separator_width": separator_width,
        "duration_decimal_places": duration_decimal,
        "media_stats_decimal_places": media_stats_decimal,
    }


def load_browser_config_from_yaml(config_path: str = "config/scraper.yaml"):
    """Load and apply YAML configuration to global browser settings using config
    adapter
    """
    global CONFIG, _BROWSER_CONFIG

    try:
        # Use the new config adapter for backward compatibility
        from ..config_adapter import ScraperConfigAdapter

        adapter = ScraperConfigAdapter()
        config_data = adapter.get_merged_config_dict()
        CONFIG.update(config_data)

        # Import here to avoid circular imports
        try:
            from botasaurus import bt
            from botasaurus.user_agent import UserAgent  # type: ignore[import-untyped]
            from botasaurus.window_size import (  # type: ignore[import-untyped]
                WindowSize,
            )

            # Extract browser-specific settings
            global_settings = CONFIG.get("global_settings", {})
            debug_mode = global_settings.get("debug_mode", False)

            # Build browser configuration from YAML with performance optimizations
            _BROWSER_CONFIG = {
                "parallel": bt.calc_max_parallel_browsers(),  # Dynamic calculation
                # for optimal resource usage
                "cache": False,  # Disabled for testing new image extraction
                "max_retry": global_settings.get("retries", 3),
                "block_images": False,  # Show images in browser
                # Disabled - causes StopIteration in headless mode
                "reuse_driver": False,
                "close_on_crash": not debug_mode,  # Debug mode keeps browser
                # open on crash
                "proxy": global_settings.get("proxy"),
                "user_agent": UserAgent.RANDOM,  # Randomize user agent for better
                # anti-detection
                "window_size": WindowSize.RANDOM,  # Randomize window size for
                # better anti-detection
                "headless": False,  # Disabled - Botasaurus bug in headless mode
                "output": get_output_path(
                    "botasaurus"
                ),  # Configurable output directory
            }
        except ImportError:
            # Fallback if botasaurus not available
            debug_mode = global_settings.get("debug_mode", False)
            _BROWSER_CONFIG = {
                "headless": False,  # Disabled - Botasaurus bug in headless mode
                "close_on_crash": not debug_mode,
            }

        # Remove None values to prevent issues
        _BROWSER_CONFIG = {k: v for k, v in _BROWSER_CONFIG.items() if v is not None}

        return config_data

    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        print("Using enhanced fallback configuration...")

        # Enhanced fallback configuration
        CONFIG = {
            "global_settings": {
                "debug_mode": True,
                "output_config": {
                    "base_directory": "outputs",
                    "file_patterns": {
                        "product_file": "{keyword}_products.json",
                        "image_file": "{asin}_image_{index}.{ext}",
                        "video_file": "{asin}_video_{index}.{ext}",
                    },
                },
                "retries": 3,
            },
            "scrapers": {
                "amazon": {
                    "enabled": True,
                    "base_url": "https://www.amazon.com",
                    "max_products": 3,
                    "default_search_parameters": {
                        "prime_only": False,
                        "sort_order": "relevanceblender",
                    },
                }
            },
        }

        _BROWSER_CONFIG = {
            "headless": False,  # Disabled - Botasaurus bug in headless mode
            "close_on_crash": True,
            "max_retry": 3,
            "cache": False,
            "block_images": False,
            "reuse_driver": True,
        }

        return CONFIG


# Initialize on import with enhanced fallback
try:
    load_browser_config_from_yaml()
except Exception as init_error:
    print(f"⚠️  Warning: Config initialization failed: {init_error}")
    print("Using enhanced initialization fallback...")

    # Enhanced initialization fallback configuration
    CONFIG = {
        "global_settings": {
            "debug_mode": True,
            "output_config": {
                "base_directory": "outputs",
                "file_patterns": {
                    "product_file": "{keyword}_products.json",
                    "image_file": "{asin}_image_{index}.{ext}",
                    "video_file": "{asin}_video_{index}.{ext}",
                },
            },
            "retries": 3,
        },
        "scrapers": {
            "amazon": {
                "enabled": True,
                "base_url": "https://www.amazon.com",
                "max_products": 3,
                "default_search_parameters": {
                    "prime_only": False,
                    "sort_order": "relevanceblender",
                },
            }
        },
    }

    _BROWSER_CONFIG = {
        "headless": False,  # Disabled - Botasaurus bug in headless mode
        "close_on_crash": True,
        "max_retry": 3,
        "cache": False,
        "block_images": False,
        "reuse_driver": True,
    }
