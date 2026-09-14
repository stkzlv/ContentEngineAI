"""Centralized outputs path management for ContentEngineAI.

This module provides a single source of truth for all output directory paths,
ensuring consistency between scraper and producer modules.
"""

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Durable tracking state lives in its own directory under the outputs root,
# so the deletable/durable boundary is structural rather than maintained by
# a pattern list: everything else under outputs/ is a run artifact, a cache
# or a diagnostic, and can be rebuilt or re-scraped. These files cannot --
# the publish history backs the duplicate guard, the registry's rows for
# cleaned product dirs exist nowhere else, and day-N metrics age out of the
# provider's retention and are then unrecoverable at any price.
STATE_DIR_NAME = "state"


def durable_state_path(outputs_dir: Path, filename: str) -> Path:
    """The path of a durable tracking file, under ``outputs/state/``.

    Migrates eagerly: a legacy copy at the outputs root is moved into
    ``state/`` the first time the path is resolved, so every reader and
    writer converges on one location after the first touch -- a lazy
    migration would leave a writer on the new path while an unmigrated
    reader still read the stale root copy. ``os.replace`` keeps the move
    atomic; both paths share a filesystem by construction.

    The state directory is created only when something exists to migrate or
    the outputs root already exists, so resolving a path on a fresh clone
    (a dry run, a test collecting defaults) does not plant directories.
    """
    state_dir = outputs_dir / STATE_DIR_NAME
    target = state_dir / filename
    legacy = outputs_dir / filename
    try:
        if legacy.is_file() and not target.exists():
            state_dir.mkdir(parents=True, exist_ok=True)
            os.replace(legacy, target)
            logger.info("Migrated %s to %s", legacy, target)
        elif outputs_dir.exists():
            state_dir.mkdir(exist_ok=True)
    except OSError as exc:
        # Two bounded causes, neither of which may break a READ path:
        # a concurrent first touch (the analytics timer firing during a
        # batch publish) lost the os.replace race -- the winner moved the
        # file, so the target is the answer; or the tree is read-only (a
        # mounted backup snapshot inspected in place) -- migration is an
        # optimization for writers, not a precondition for reading, so
        # the legacy copy keeps serving reads unmigrated.
        if legacy.is_file() and not target.exists():
            logger.warning(
                "Could not migrate %s to %s (%s); reading the legacy copy",
                legacy,
                target,
                exc,
            )
            return legacy
        logger.debug("State-dir setup raced or was refused: %s", exc)
    return target


def get_project_root() -> Path:
    """Get the project root directory."""
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent
    return project_root.resolve()


def get_outputs_root(custom_dir: str | None = None) -> Path:
    """Get the root outputs directory for the project.

    Args:
    ----
        custom_dir: Custom outputs directory name (defaults to "outputs")

    Returns:
    -------
        Path to the outputs directory

    """
    project_root = get_project_root()
    outputs_dir_name = custom_dir or "outputs"
    outputs_dir = project_root / outputs_dir_name
    outputs_dir.mkdir(exist_ok=True)
    return outputs_dir


def get_product_directory(product_id: str, custom_dir: str | None = None) -> Path:
    """Get the directory for a specific product.

    Args:
    ----
        product_id: Product identifier (e.g., ASIN)
        custom_dir: Custom outputs directory name

    Returns:
    -------
        Path to the product directory

    """
    outputs_root = get_outputs_root(custom_dir)
    product_dir = outputs_root / product_id
    product_dir.mkdir(exist_ok=True)
    return product_dir


def get_global_directory(dir_name: str, custom_outputs_dir: str | None = None) -> Path:
    """Get a global directory (cache, logs, reports, temp).

    Args:
    ----
        dir_name: Name of the global directory
        custom_outputs_dir: Custom outputs directory name

    Returns:
    -------
        Path to the global directory

    """
    outputs_root = get_outputs_root(custom_outputs_dir)
    global_dir = outputs_root / dir_name
    global_dir.mkdir(exist_ok=True)
    return global_dir


def get_cache_directory(custom_outputs_dir: str | None = None) -> Path:
    """Get the cache directory."""
    return get_global_directory("cache", custom_outputs_dir)


def get_logs_directory(custom_outputs_dir: str | None = None) -> Path:
    """Get the logs directory."""
    return get_global_directory("logs", custom_outputs_dir)


def get_reports_directory(custom_outputs_dir: str | None = None) -> Path:
    """Get the reports directory."""
    return get_global_directory("reports", custom_outputs_dir)


def get_temp_directory(custom_outputs_dir: str | None = None) -> Path:
    """Get the temp directory."""
    return get_global_directory("temp", custom_outputs_dir)


def get_performance_history_directory(custom_outputs_dir: str | None = None) -> Path:
    """Get the performance_history directory."""
    return get_global_directory("performance_history", custom_outputs_dir)


def get_botasaurus_cache_directory(custom_outputs_dir: str | None = None) -> Path:
    """Get the Botasaurus cache directory."""
    cache_dir = get_cache_directory(custom_outputs_dir)
    botasaurus_dir = cache_dir / "botasaurus"
    botasaurus_dir.mkdir(exist_ok=True)
    return botasaurus_dir


def get_product_images_directory(
    product_id: str, custom_outputs_dir: str | None = None
) -> Path:
    """Get the images directory for a product.

    Args:
    ----
        product_id: Product identifier
        custom_outputs_dir: Custom outputs directory name

    Returns:
    -------
        Path to the product's images directory

    """
    product_dir = get_product_directory(product_id, custom_outputs_dir)
    images_dir = product_dir / "images"
    images_dir.mkdir(exist_ok=True)
    return images_dir


def get_product_videos_directory(
    product_id: str, custom_outputs_dir: str | None = None
) -> Path:
    """Get the videos directory for a product.

    Args:
    ----
        product_id: Product identifier
        custom_outputs_dir: Custom outputs directory name

    Returns:
    -------
        Path to the product's videos directory

    """
    product_dir = get_product_directory(product_id, custom_outputs_dir)
    videos_dir = product_dir / "videos"
    videos_dir.mkdir(exist_ok=True)
    return videos_dir


def get_product_music_directory(
    product_id: str, custom_outputs_dir: str | None = None
) -> Path:
    """Get the music directory for a product.

    Args:
    ----
        product_id: Product identifier
        custom_outputs_dir: Custom outputs directory name

    Returns:
    -------
        Path to the product's music directory

    """
    product_dir = get_product_directory(product_id, custom_outputs_dir)
    music_dir = product_dir / "music"
    music_dir.mkdir(exist_ok=True)
    return music_dir


def get_product_temp_directory(
    product_id: str, custom_outputs_dir: str | None = None
) -> Path:
    """Get the temp directory for a product.

    Args:
    ----
        product_id: Product identifier
        custom_outputs_dir: Custom outputs directory name

    Returns:
    -------
        Path to the product's temp directory

    """
    product_dir = get_product_directory(product_id, custom_outputs_dir)
    temp_dir = product_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    return temp_dir


def ensure_outputs_structure(custom_outputs_dir: str | None = None) -> None:
    """Ensure the basic outputs directory structure exists.

    Args:
    ----
        custom_outputs_dir: Custom outputs directory name

    """
    try:
        # Create root outputs directory
        outputs_root = get_outputs_root(custom_outputs_dir)

        # Create global directories
        get_cache_directory(custom_outputs_dir)
        get_logs_directory(custom_outputs_dir)
        get_reports_directory(custom_outputs_dir)

        # Create Botasaurus cache
        get_botasaurus_cache_directory(custom_outputs_dir)

        logger.debug("Outputs structure ensured at: %s", outputs_root)

    except Exception as e:
        logger.error("Failed to ensure outputs structure: %s", e)
        raise


def get_relative_path_from_outputs(
    path: Path, custom_outputs_dir: str | None = None
) -> str:
    """Get relative path from outputs root.

    Args:
    ----
        path: Absolute path to convert
        custom_outputs_dir: Custom outputs directory name

    Returns:
    -------
        Relative path from outputs root

    """
    outputs_root = get_outputs_root(custom_outputs_dir)
    try:
        return str(path.relative_to(outputs_root))
    except ValueError:
        # Path is not relative to outputs root
        return str(path)


def validate_outputs_structure(
    custom_outputs_dir: str | None = None, strict: bool = False
) -> dict[str, list[str]]:
    """Validate the outputs directory structure against expected layout.

    Args:
    ----
        custom_outputs_dir: Custom outputs directory name
        strict: If True, report any unexpected files/dirs as errors

    Returns:
    -------
        Dictionary with validation results:
        - valid_products: List of valid product directories
        - invalid_products: List of invalid product directories
        - unexpected_items: List of unexpected files/directories
        - missing_global_dirs: List of missing global directories
        - errors: List of validation error messages

    """
    outputs_root = get_outputs_root(custom_outputs_dir)
    results: dict[str, list[str]] = {
        "valid_products": [],
        "invalid_products": [],
        "unexpected_items": [],
        "missing_global_dirs": [],
        "errors": [],
    }

    expected_global_dirs = {"cache", "logs", "reports"}
    expected_product_subdirs = {"images", "videos"}

    try:
        if not outputs_root.exists():
            results["errors"].append(
                f"Outputs directory does not exist: {outputs_root}"
            )
            return results

        for item in outputs_root.iterdir():
            if item.is_file():
                # Files in outputs root are unexpected
                results["unexpected_items"].append(str(item.name))
            elif item.is_dir():
                if item.name in expected_global_dirs:
                    # This is a valid global directory
                    continue
                elif item.name in {"temp", "performance_history"}:
                    # These are optional global directories
                    continue
                elif _is_valid_product_id(item.name):
                    # This looks like a product directory
                    if _validate_product_directory(item, expected_product_subdirs):
                        results["valid_products"].append(item.name)
                    else:
                        results["invalid_products"].append(item.name)
                else:
                    # Unknown directory
                    if strict:
                        results["unexpected_items"].append(str(item.name))

        # Check for missing global directories
        for global_dir in expected_global_dirs:
            if not (outputs_root / global_dir).exists():
                results["missing_global_dirs"].append(global_dir)

    except Exception as e:
        results["errors"].append(f"Validation error: {e}")

    return results


def _is_valid_product_id(name: str) -> bool:
    """Check if a directory name looks like a valid product ID (ASIN)."""
    # Simple heuristic: ASINs are typically 10 characters, alphanumeric
    return len(name) >= 8 and len(name) <= 15 and name.replace("_", "").isalnum()


def _validate_product_directory(product_dir: Path, expected_subdirs: set[str]) -> bool:
    """Validate a product directory structure."""
    try:
        # Check for required data.json file
        if not (product_dir / "data.json").exists():
            return False

        # Check for expected subdirectories (at least some should exist)
        has_media_dirs = any(
            (product_dir / subdir).exists() for subdir in expected_subdirs
        )

        return has_media_dirs
    except Exception:
        return False


def cleanup_invalid_outputs(
    custom_outputs_dir: str | None = None, dry_run: bool = True
) -> dict[str, list[str]]:
    """Clean up invalid files and directories from outputs.

    Args:
    ----
        custom_outputs_dir: Custom outputs directory name
        dry_run: If True, only report what would be removed

    Returns:
    -------
        Dictionary with cleanup results:
        - removed_items: List of items that were (or would be) removed
        - preserved_items: List of valid items that were preserved
        - errors: List of cleanup error messages

    """
    validation_results = validate_outputs_structure(custom_outputs_dir, strict=True)
    cleanup_results: dict[str, list[str]] = {
        "removed_items": [],
        "preserved_items": [],
        "errors": [],
    }

    outputs_root = get_outputs_root(custom_outputs_dir)

    try:
        # Items to remove
        items_to_remove = []
        items_to_remove.extend(validation_results["unexpected_items"])
        items_to_remove.extend(validation_results["invalid_products"])

        for item_name in items_to_remove:
            item_path = outputs_root / item_name
            if dry_run:
                cleanup_results["removed_items"].append(f"[DRY RUN] {item_name}")
            else:
                try:
                    if item_path.is_file():
                        item_path.unlink()
                    elif item_path.is_dir():
                        import shutil

                        shutil.rmtree(item_path)
                    cleanup_results["removed_items"].append(item_name)
                except Exception as e:
                    cleanup_results["errors"].append(
                        f"Failed to remove {item_name}: {e}"
                    )

        # Items to preserve
        cleanup_results["preserved_items"].extend(validation_results["valid_products"])

    except Exception as e:
        cleanup_results["errors"].append(f"Cleanup error: {e}")

    return cleanup_results
