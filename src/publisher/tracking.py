"""Track published posts to prevent duplicates."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from src.publisher.constants import DEFAULT_OUTPUTS_DIR

logger = logging.getLogger(__name__)

TRACKING_FILE = "publish_history.json"


def get_tracking_path(outputs_dir: Path = DEFAULT_OUTPUTS_DIR) -> Path:
    """Get path to tracking file."""
    return outputs_dir / TRACKING_FILE


def load_tracking(
    outputs_dir: Path = DEFAULT_OUTPUTS_DIR,
) -> dict[str, dict[str, dict]]:
    """Load publish tracking data."""
    path = get_tracking_path(outputs_dir)
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load tracking: %s", e)
    return {"posts": {}}


def save_tracking(data: dict, outputs_dir: Path = DEFAULT_OUTPUTS_DIR) -> None:
    """Save publish tracking data atomically via temp-file + rename."""
    path = get_tracking_path(outputs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    try:
        temp_path.write_text(json.dumps(data, indent=2, default=str))
        temp_path.replace(path)
    except OSError:
        if temp_path.exists():
            temp_path.unlink()
        raise


def is_already_published(
    product_id: str,
    platform: str,
    outputs_dir: Path = DEFAULT_OUTPUTS_DIR,
) -> bool:
    """Check if product was already published to platform."""
    tracking = load_tracking(outputs_dir)
    key = f"{product_id}:{platform}"
    return key in tracking.get("posts", {})


def record_publish(
    product_id: str,
    platform: str,
    post_id: str,
    outputs_dir: Path = DEFAULT_OUTPUTS_DIR,
) -> None:
    """Record a successful publish to prevent duplicates."""
    tracking = load_tracking(outputs_dir)
    if "posts" not in tracking:
        tracking["posts"] = {}

    key = f"{product_id}:{platform}"
    tracking["posts"][key] = {
        "product_id": product_id,
        "platform": platform,
        "post_id": post_id,
        "published_at": datetime.now(UTC).isoformat(),
    }
    save_tracking(tracking, outputs_dir)
    logger.info("Recorded publish: %s -> %s", key, post_id)


def get_publish_record(
    product_id: str,
    platform: str,
    outputs_dir: Path = DEFAULT_OUTPUTS_DIR,
) -> dict[str, str] | None:
    """Get publish record for product/platform."""
    tracking = load_tracking(outputs_dir)
    key = f"{product_id}:{platform}"
    posts = tracking.get("posts", {})
    record = posts.get(key)
    if isinstance(record, dict):
        return record
    return None


# =============================================================================
# RETRY QUEUE FUNCTIONS
# =============================================================================


def add_to_retry_queue(
    product_id: str,
    platforms: list[str],
    error: str,
    scheduled_time: str | None = None,
    outputs_dir: Path = DEFAULT_OUTPUTS_DIR,
) -> None:
    """Add a failed product to the retry queue.

    Args:
    ----
        product_id: Product identifier
        platforms: List of platform names that failed
        error: Error message describing the failure
        scheduled_time: Original scheduled time (ISO format) to preserve
        outputs_dir: Outputs directory path

    """
    tracking = load_tracking(outputs_dir)
    if "retry_queue" not in tracking:
        tracking["retry_queue"] = {}

    # Use product_id as key (one entry per product, may have multiple platforms)
    tracking["retry_queue"][product_id] = {
        "product_id": product_id,
        "platforms": platforms,
        "error": error,
        "scheduled_time": scheduled_time,
        "failed_at": datetime.now(UTC).isoformat(),
        "retry_count": tracking["retry_queue"].get(product_id, {}).get("retry_count", 0)
        + 1,
    }
    save_tracking(tracking, outputs_dir)
    logger.info("Added to retry queue: %s (platforms: %s)", product_id, platforms)


def get_retry_queue(outputs_dir: Path = DEFAULT_OUTPUTS_DIR) -> list[dict]:
    """Get all items in the retry queue.

    Returns
    -------
        List of retry queue entries with product_id, platforms, error, etc.

    """
    tracking = load_tracking(outputs_dir)
    queue = tracking.get("retry_queue", {})
    return list(queue.values())


def get_retry_queue_item(
    product_id: str,
    outputs_dir: Path = DEFAULT_OUTPUTS_DIR,
) -> dict | None:
    """Get a specific item from the retry queue.

    Args:
    ----
        product_id: Product identifier
        outputs_dir: Outputs directory path

    Returns:
    -------
        Retry queue entry or None if not found

    """
    tracking = load_tracking(outputs_dir)
    queue = tracking.get("retry_queue", {})
    return queue.get(product_id)


def remove_from_retry_queue(
    product_id: str,
    outputs_dir: Path = DEFAULT_OUTPUTS_DIR,
) -> bool:
    """Remove a product from the retry queue after successful publish.

    Args:
    ----
        product_id: Product identifier
        outputs_dir: Outputs directory path

    Returns:
    -------
        True if item was removed, False if not found

    """
    tracking = load_tracking(outputs_dir)
    queue = tracking.get("retry_queue", {})

    if product_id in queue:
        del queue[product_id]
        tracking["retry_queue"] = queue
        save_tracking(tracking, outputs_dir)
        logger.info("Removed from retry queue: %s", product_id)
        return True

    return False


def clear_retry_queue(outputs_dir: Path = DEFAULT_OUTPUTS_DIR) -> int:
    """Clear all items from the retry queue.

    Returns
    -------
        Number of items cleared

    """
    tracking = load_tracking(outputs_dir)
    queue = tracking.get("retry_queue", {})
    count = len(queue)

    if count > 0:
        tracking["retry_queue"] = {}
        save_tracking(tracking, outputs_dir)
        logger.info("Cleared retry queue: %d item(s)", count)

    return count


def get_retry_queue_count(outputs_dir: Path = DEFAULT_OUTPUTS_DIR) -> int:
    """Get number of items in retry queue.

    Returns
    -------
        Number of items in the retry queue

    """
    tracking = load_tracking(outputs_dir)
    return len(tracking.get("retry_queue", {}))
