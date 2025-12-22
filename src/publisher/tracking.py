"""Track published posts to prevent duplicates."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

TRACKING_FILE = "publish_history.json"


def get_tracking_path(outputs_dir: Path = Path("outputs")) -> Path:
    """Get path to tracking file."""
    return outputs_dir / TRACKING_FILE


def load_tracking(outputs_dir: Path = Path("outputs")) -> dict[str, dict[str, dict]]:
    """Load publish tracking data."""
    path = get_tracking_path(outputs_dir)
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                return data
        except Exception as e:
            logger.warning(f"Failed to load tracking: {e}")
    return {"posts": {}}


def save_tracking(data: dict, outputs_dir: Path = Path("outputs")) -> None:
    """Save publish tracking data."""
    path = get_tracking_path(outputs_dir)
    path.write_text(json.dumps(data, indent=2, default=str))


def is_already_published(
    product_id: str,
    platform: str,
    outputs_dir: Path = Path("outputs"),
) -> bool:
    """Check if product was already published to platform."""
    tracking = load_tracking(outputs_dir)
    key = f"{product_id}:{platform}"
    return key in tracking.get("posts", {})


def record_publish(
    product_id: str,
    platform: str,
    post_id: str,
    outputs_dir: Path = Path("outputs"),
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
    logger.info(f"Recorded publish: {key} -> {post_id}")


def get_publish_record(
    product_id: str,
    platform: str,
    outputs_dir: Path = Path("outputs"),
) -> dict[str, str] | None:
    """Get publish record for product/platform."""
    tracking = load_tracking(outputs_dir)
    key = f"{product_id}:{platform}"
    posts = tracking.get("posts", {})
    record = posts.get(key)
    if isinstance(record, dict):
        return record
    return None
