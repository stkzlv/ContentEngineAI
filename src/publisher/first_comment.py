"""Build first-comment text for supported platforms.

Moves affiliate links from post captions into the first comment,
keeping descriptions clean and avoiding algorithm penalties on
platforms that deprioritize posts with outbound links in captions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.publisher.metadata import PublishMetadata
    from src.publisher.models import FirstCommentConfig

logger = logging.getLogger(__name__)


def build_first_comment(
    config: FirstCommentConfig,
    platform: str,
    product_id: str,
    outputs_dir: Path,
    metadata: PublishMetadata | None = None,
) -> str | None:
    """Render the first-comment template for a given platform.

    Returns None (silently) when disabled, unsupported, or data is missing.

    Args:
    ----
        config: First-comment configuration from publisher.yaml.
        platform: Platform name (youtube, instagram, tiktok).
        product_id: Product identifier for data.json lookup.
        outputs_dir: Base outputs directory.
        metadata: Optional publish metadata (used for hashtags).

    """
    if not config.enabled:
        return None

    if platform == "tiktok":
        return None

    template = config.platforms.get(platform)
    if not template:
        return None

    # Load product data for affiliate link
    data_path = outputs_dir / product_id / "data.json"
    if not data_path.exists():
        logger.warning("No data.json for %s, skipping first comment", product_id)
        return None

    try:
        with open(data_path, encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read data.json for %s: %s", product_id, e)
        return None

    product = raw[0] if isinstance(raw, list) else raw
    affiliate_link = product.get("shortened_affiliate_link") or product.get(
        "affiliate_link"
    )
    if not affiliate_link:
        logger.warning("No affiliate link for %s, skipping first comment", product_id)
        return None

    product_title = product.get("title", "")

    # Collect hashtags if moving them to the comment
    hashtags = ""
    if config.move_hashtags_to_comment and platform == "instagram" and metadata:
        hashtags = " ".join(
            f"#{t}" if not t.startswith("#") else t for t in (metadata.hashtags or [])
        )

    return template.format(
        affiliate_link=affiliate_link,
        product_title=product_title,
        hashtags=hashtags,
    )
