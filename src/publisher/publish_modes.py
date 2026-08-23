"""Publishing mode helpers for unified and platform-specific posts."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.publisher.first_comment import build_first_comment
from src.publisher.metadata import load_platform_metadata
from src.publisher.models import Platform

if TYPE_CHECKING:
    from src.publisher.base import BasePublisher

logger = logging.getLogger(__name__)


def _build_platform_contents_with_comments(
    publisher: BasePublisher,
    platforms: list[dict[str, str]],
    product_id: str,
    outputs_dir: Path,
    metadata_by_platform: dict[str, Any] | None = None,
) -> dict[str, dict[str, str]] | None:
    """Build platform_contents dict with first comments injected.

    Each entry carries the full per-platform payload the publisher reads, not
    only the comment: ``content`` and ``title`` come from that platform's
    metadata. A partial entry is not safe, because the consumer treats the
    dict as authoritative and reads ``content`` and ``title`` from it, so a
    missing key silently blanks the field rather than falling back.

    Returns None if no first comments were generated (caller can skip
    platform_contents entirely).
    """
    fc_config = getattr(publisher, "first_comment_config", None)
    if not fc_config or not fc_config.enabled:
        return None

    platform_contents: dict[str, dict[str, str]] = {}

    for p_info in platforms:
        platform_name = p_info["platform"]
        meta = metadata_by_platform.get(platform_name) if metadata_by_platform else None
        comment = build_first_comment(
            fc_config, platform_name, product_id, outputs_dir, metadata=meta
        )
        if not comment:
            continue
        entry: dict[str, str] = {"first_comment": comment}
        if meta is not None:
            entry["content"] = meta.format_content()
            if getattr(meta, "title", None):
                entry["title"] = meta.title
        platform_contents[platform_name] = entry

    return platform_contents if platform_contents else None


async def publish_product(
    publisher: BasePublisher,
    media_id: str,
    product_id: str,
    platforms: list[dict[str, str]],
    outputs_dir: Path | str,
    platform_specific: bool = False,
    schedule_time: datetime | None = None,
    disclosure_phrase: str | None = None,
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
        disclosure_phrase: Optional affiliate program literal phrase to
            include in the caption between the disclosure and description.

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
            publisher,
            media_id,
            product_id,
            platforms,
            outputs_dir,
            schedule_time,
            disclosure_phrase=disclosure_phrase,
        )
    else:
        results = await _publish_unified(
            publisher,
            media_id,
            product_id,
            platforms,
            outputs_dir,
            schedule_time,
            disclosure_phrase=disclosure_phrase,
        )

    return results


async def _publish_unified(
    publisher: BasePublisher,
    media_id: str,
    product_id: str,
    platforms: list[dict[str, str]],
    outputs_dir: Path,
    schedule_time: datetime | None,
    disclosure_phrase: str | None = None,
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

    # Gated on the same recorded decision as `#ad`. The phrase asserts
    # membership of an affiliate program, so on a render with no material
    # connection it is the same false statement the `#ad` gate removes, and it
    # would land in the slot the cleared `#ad` line just vacated.
    if disclosure_phrase and metadata.carries_affiliate_content:
        metadata.affiliate_disclosure = disclosure_phrase

    trimmed = metadata.clamp_to_limits()
    if trimmed:
        logger.info(
            "Clamped %s for %s to platform limits",
            ", ".join(trimmed),
            metadata.platform.value,
        )

    content = metadata.format_content()
    logger.info("Publishing to %d platform(s) in single post...", len(platforms))

    # Build first comments for each platform
    platform_contents = _build_platform_contents_with_comments(
        publisher,
        platforms,
        product_id,
        outputs_dir,
        metadata_by_platform={p["platform"]: metadata for p in platforms},
    )

    result = await publisher.publish(
        media_id=media_id,
        platforms=platforms,
        content=content,
        scheduled_time=schedule_time,
        platform_contents=platform_contents,
        carries_affiliate_content=metadata.carries_affiliate_content,
    )

    return [{"result": result, "platform": "all"}]


async def _publish_platform_specific(
    publisher: BasePublisher,
    media_id: str,
    product_id: str,
    platforms: list[dict[str, str]],
    outputs_dir: Path,
    schedule_time: datetime | None,
    disclosure_phrase: str | None = None,
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

        # Gated on the same recorded decision as `#ad`; see the unified path.
        if disclosure_phrase and metadata.carries_affiliate_content:
            metadata.affiliate_disclosure = disclosure_phrase

        trimmed = metadata.clamp_to_limits()
        if trimmed:
            logger.info(
                "Clamped %s for %s to platform limits",
                ", ".join(trimmed),
                metadata.platform.value,
            )

        content = metadata.format_content()

        # Build first comment for this platform
        platform_contents = _build_platform_contents_with_comments(
            publisher,
            [p_info],
            product_id,
            outputs_dir,
            metadata_by_platform={platform_name: metadata},
        )

        logger.info("Publishing to %s (platform-specific post)...", platform_name)

        result = await publisher.publish(
            media_id=media_id,
            platforms=[p_info],
            content=content,
            scheduled_time=schedule_time,
            platform_contents=platform_contents,
            carries_affiliate_content=metadata.carries_affiliate_content,
        )

        results.append({"result": result, "platform": platform_name})

    return results
