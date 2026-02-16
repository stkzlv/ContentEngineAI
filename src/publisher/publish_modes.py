"""Publishing mode helpers for unified and platform-specific posts."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.publisher.metadata import load_platform_metadata
from src.publisher.models import Platform

if TYPE_CHECKING:
    from src.publisher.base import BasePublisher

logger = logging.getLogger(__name__)


async def publish_product(
    publisher: BasePublisher,
    media_id: str,
    product_id: str,
    platforms: list[dict[str, str]],
    outputs_dir: Path | str,
    platform_specific: bool = False,
    schedule_time: datetime | None = None,
) -> list[dict]:
    """Publish a product video in unified or platform-specific mode.

    Args:
    ----
        publisher: Authenticated publisher instance with publish() method.
        media_id: Uploaded media ID.
        product_id: Product identifier.
        platforms: List of {"platform": "...", "account_id": "..."} dicts.
        outputs_dir: Base outputs directory for metadata loading.
        platform_specific: If True, create separate posts per platform
            with per-platform optimized metadata.
        schedule_time: Optional scheduled time.

    Returns:
    -------
        List of publish results. Each item has:
            - "result": dict from publisher.publish()
            - "platform": platform name or "all" for unified mode

    """
    outputs_dir = Path(outputs_dir)
    results: list[dict] = []

    if platform_specific:
        results = await _publish_platform_specific(
            publisher, media_id, product_id, platforms, outputs_dir, schedule_time
        )
    else:
        results = await _publish_unified(
            publisher, media_id, product_id, platforms, outputs_dir, schedule_time
        )

    return results


async def _publish_unified(
    publisher: BasePublisher,
    media_id: str,
    product_id: str,
    platforms: list[dict[str, str]],
    outputs_dir: Path,
    schedule_time: datetime | None,
) -> list[dict]:
    """Single post for all platforms with unified metadata."""
    # Load metadata from first available platform
    metadata = None
    for try_platform in platforms:
        metadata = load_platform_metadata(
            product_id, try_platform["platform"], outputs_dir
        )
        if metadata:
            break

    if not metadata:
        raise ValueError(f"No metadata found for {product_id}")

    content = metadata.format_content()
    logger.info("Publishing to %d platform(s) in single post...", len(platforms))

    result = await publisher.publish(
        media_id=media_id,
        platforms=platforms,
        content=content,
        scheduled_time=schedule_time,
    )

    return [{"result": result, "platform": "all"}]


async def _publish_platform_specific(
    publisher: BasePublisher,
    media_id: str,
    product_id: str,
    platforms: list[dict[str, str]],
    outputs_dir: Path,
    schedule_time: datetime | None,
) -> list[dict]:
    """Separate post per platform with platform-specific metadata."""
    results: list[dict] = []

    for p_info in platforms:
        platform_name = p_info["platform"]

        # Load platform-specific metadata
        metadata = load_platform_metadata(product_id, platform_name, outputs_dir)

        if not metadata:
            # Fallback: try any available platform metadata
            for fallback in [Platform.YOUTUBE, Platform.TIKTOK, Platform.INSTAGRAM]:
                metadata = load_platform_metadata(product_id, fallback, outputs_dir)
                if metadata:
                    logger.info(
                        "Using %s metadata as fallback for %s",
                        fallback.value,
                        platform_name,
                    )
                    break

        if not metadata:
            raise ValueError(f"No metadata found for {product_id}/{platform_name}")

        content = metadata.format_content()

        logger.info("Publishing to %s (platform-specific post)...", platform_name)

        result = await publisher.publish(
            media_id=media_id,
            platforms=[p_info],
            content=content,
            scheduled_time=schedule_time,
        )

        results.append({"result": result, "platform": platform_name})

    return results
