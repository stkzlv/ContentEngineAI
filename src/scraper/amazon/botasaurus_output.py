"""Botasaurus output configuration to direct all outputs to the project's outputs/ dir.

This module provides custom output functions and configuration for Botasaurus
to ensure all scraped data, screenshots, debug files, and cache are saved
to our standardized outputs/ directory structure.
"""

import logging
import os
from pathlib import Path
from typing import Any

from botasaurus import bt

from ...utils.outputs_paths import (
    ensure_outputs_structure,
    get_botasaurus_cache_directory,
    get_outputs_root,
    get_product_directory,
)

logger = logging.getLogger(__name__)

# Module-level output directory override. When set, all output functions
# use this instead of the config default ("outputs").
# Set via set_output_dir() before running the scraper.
_output_dir_override: str | None = None


def set_output_dir(output_dir: str | None) -> None:
    """Set the module-level output directory override."""
    global _output_dir_override
    _output_dir_override = output_dir


def _effective_dir(explicit: str | None = None) -> str | None:
    """Return effective output dir: explicit param > module override > None."""
    return explicit or _output_dir_override


def get_product_output_dir(product_id: str, custom_dir: str | None = None) -> Path:
    """Get the output directory for a specific product."""
    return get_product_directory(product_id, custom_dir=_effective_dir(custom_dir))


def get_global_botasaurus_dir() -> Path:
    """Get the global Botasaurus output directory.

    DEPRECATED: Use get_botasaurus_cache_directory from utils.outputs_paths instead.
    """
    return get_botasaurus_cache_directory()


def write_scraped_data_output(
    data: Any,
    result: list[dict[str, Any]],
    output_dir: str | None = None,
) -> None:
    """Custom output function for browser scraping tasks.
    Saves scraped product data to the appropriate product directory.

    Args:
    ----
        data: Input data containing scraping parameters
        result: Scraped product data
        output_dir: Custom output directory (overrides config base_directory)

    """
    logger.debug(
        "write_scraped_data_output called with %d products",
        len(result) if result else 0,
    )
    if not result:
        logger.debug("No result data to save")
        return

    # Process each product individually to create separate data.json files
    if isinstance(result, list) and len(result) > 0:
        import json
        import os

        os.getcwd()

        for product in result:
            # Add platform field for video producer compatibility
            if "platform" not in product:
                product["platform"] = "amazon"

            # Get product-specific directory
            product_id = product.get("asin") or product.get("id")
            if not product_id:
                logger.warning("Skipping product with no ASIN or ID")
                continue
            product_dir = get_product_output_dir(product_id, custom_dir=output_dir)

            try:
                # Save individual product as JSON in its own directory
                # Video producer expects a single product wrapped in a list
                output_file = product_dir / "data.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump([product], f, indent=4, ensure_ascii=False)
                logger.info("Saved scraped data: %s", output_file)
            except Exception:
                # The product's primary artifact failed to write; a print
                # never reached any log file, so the loss was invisible.
                logger.exception("Failed to save product data for %s", product_id)
    else:
        logger.warning("Scraper returned non-list result, skipping save")


def write_download_cache_output(data: Any, result: dict[str, Any]) -> None:
    """Custom output function for download tasks.
    Saves download metadata to the cache directory.

    Args:
    ----
        data: Input data containing download parameters (can be dict or list)
        result: Download result metadata

    """
    if not result:
        return

    # Save to global cache directory
    cache_dir = get_global_botasaurus_dir()

    # Handle both single dict and list inputs from Botasaurus
    # When @task receives a list, it processes each item but calls
    # output with original list
    if isinstance(data, list) and len(data) > 0:
        # Use the first item's data for naming
        first_item = data[0]
        product_id = first_item.get(
            "asin", first_item.get("product_id", "download_cache")
        )
    elif isinstance(data, dict):
        # Single dict input
        product_id = data.get("asin", data.get("product_id", "download_cache"))
    else:
        # Fallback
        product_id = "download_cache"

    # Change working directory temporarily to save in the right location

    original_cwd = os.getcwd()
    try:
        os.chdir(cache_dir)
        json_filename = bt.write_json(result, f"{product_id}_downloads")
        logger.debug("Saved download cache: %s", cache_dir / json_filename)
    finally:
        os.chdir(original_cwd)


def configure_botasaurus_outputs() -> None:
    """Configure Botasaurus to use our outputs directory.
    This should be called at module initialization.
    """
    try:
        # Use centralized outputs structure setup
        ensure_outputs_structure()
        outputs_root = get_outputs_root()
        logger.debug("Configured outputs directory: %s", outputs_root)

    except Exception:
        logger.warning("Could not create output directories", exc_info=True)


def get_browser_config_for_outputs() -> dict[str, Any]:
    """Browser configuration that disables Botasaurus's own output.

    Every scraper arm saves through ``BotasaurusAmazonScraper._save_products``
    after the media downloads, so a write from the decorator's output
    callback was at best redundant: on the standalone keyword and ASIN
    arms ``write_scraped_data_output`` wrote a raw extractor dict that
    ``_save_products`` then overwrote, and on the batch arm it received one
    ``{"input", "products"}`` envelope per input, found no ASIN and warned.
    ``None`` tells Botasaurus to write nothing.
    """
    return {"output": None}


def get_task_config_for_outputs() -> dict[str, Any]:
    """Get task configuration dict that includes our custom output function.

    Returns
    -------
        Dictionary with task configuration including output function

    """
    return {
        "output": write_download_cache_output,
        # Add any other task-specific output configurations here
    }


# Configure outputs when module is imported
configure_botasaurus_outputs()
