"""Metadata loading utilities for video publishing.

This module loads platform-specific metadata from the producer's JSON files,
converting it to PublishMetadata for the publisher implementations.
"""

import json
import logging
import re
from pathlib import Path

from src.publisher.constants import DEFAULT_OUTPUTS_DIR
from src.publisher.models import Platform, PublishMetadata, disclosure_from_record

logger = logging.getLogger(__name__)


def load_platform_metadata(
    product_id: str,
    platform: Platform | str,
    outputs_dir: Path | str = DEFAULT_OUTPUTS_DIR,
) -> PublishMetadata | None:
    """Load platform-specific metadata for publishing.

    Reads the unified `metadata.json`, then `metadata_<platform>.json`.
    Validates content against platform-specific character limits.

    Args:
    ----
        product_id: Product identifier (e.g., ASIN "B0ASIN123")
        platform: Target platform (Platform enum or string like "youtube")
        outputs_dir: Base outputs directory (default: "outputs")

    Returns:
    -------
        PublishMetadata object if successful, None if metadata cannot be loaded

    Example:
    -------
        >>> metadata = load_platform_metadata("B0ASIN123", Platform.YOUTUBE)
        >>> if metadata:
        ...     print(f"Title: {metadata.title}")
        ...     print(f"Description: {metadata.description[:50]}...")

    """
    # Convert platform to enum if string
    if isinstance(platform, str):
        try:
            platform = Platform(platform.lower())
        except ValueError:
            logger.error("Invalid platform: %s", platform)
            return None

    # Convert outputs_dir to Path
    if isinstance(outputs_dir, str):
        outputs_dir = Path(outputs_dir)

    product_dir = outputs_dir / product_id

    logger.info(
        "Loading metadata for product %s, platform %s", product_id, platform.value
    )

    # Try loading from unified metadata.json first (unified mode)
    unified_path = product_dir / "metadata.json"
    metadata = _load_from_json(unified_path, platform, product_id)

    if metadata:
        logger.info("Loaded metadata from unified JSON: %s", unified_path)
        return metadata

    # Fallback to platform-specific JSON (optimized mode)
    platform_path = product_dir / f"metadata_{platform.value}.json"
    metadata = _load_from_json(platform_path, platform, product_id)

    if metadata:
        logger.info("Loaded metadata from platform JSON: %s", platform_path)
        return metadata

    # No fallback to UPLOAD_INSTRUCTIONS.txt: the producer writes it to its
    # text directory, never the product root, and only after writing the JSON
    # to the product root, so a fallback here could not fire.
    logger.error(
        "Could not load metadata for %s/%s (tried %s and %s)",
        product_id,
        platform.value,
        unified_path,
        platform_path,
    )
    return None


def _load_from_json(
    json_path: Path,
    platform: Platform,
    product_id: str,
) -> PublishMetadata | None:
    """Load metadata from platform-specific JSON file.

    Args:
    ----
        json_path: Path to metadata JSON file
        platform: Target platform
        product_id: Product identifier for validation

    Returns:
    -------
        PublishMetadata object if successful, None otherwise

    """
    if not json_path.exists():
        logger.debug("JSON file not found: %s", json_path)
        return None

    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        # Extract fields
        title = data.get("title")
        description_raw = data.get("description", "")
        hashtags_raw = data.get("hashtags", [])

        # Strip trailing hashtags from description (legacy metadata compatibility)
        description = re.sub(r"(\s*#\w+)+\s*$", "", description_raw).strip()
        keywords = data.get("keywords", [])

        # Normalize hashtags (remove # prefix if present)
        hashtags = [
            tag.lstrip("#") if tag.startswith("#") else tag for tag in hashtags_raw
        ]

        # Validate required fields
        if not description:
            logger.error("Missing description in %s", json_path)
            return None

        # The producer records whether the render has a material connection to
        # disclose. Absent or unreadable means disclose: a missing disclosure
        # misstates a material connection, which is the compliance failure;
        # a needless one merely asserts a connection that does not exist.
        disclose = data.get("carries_affiliate_content", True)

        # Passed to the constructor rather than assigned after it.
        # `__post_init__` is where the disclosure tokens are removed, so a
        # flag set afterwards is a flag the guard never saw -- which is how
        # the removal came to depend on `disclosure` still holding its
        # default at construction time.
        #
        # `disclosure` keeps its configured value either way: the strip needs
        # it to know which token to remove, and `format_content` gates on the
        # flag rather than on the field being blank.
        metadata = PublishMetadata(
            platform=platform,
            title=title,
            description=description,
            hashtags=hashtags,
            keywords=keywords,
            product_id=product_id,
            disclosure=disclosure_from_record(data),
            carries_affiliate_content=bool(disclose),
        )
        if not disclose:
            logger.info(
                "Caption disclosure omitted for %s: no affiliate content",
                product_id or json_path,
            )

        # Validate character limits
        is_valid, error_msg = metadata.validate_limits()
        if not is_valid:
            logger.warning(
                "Metadata validation failed for %s: %s", json_path, error_msg
            )
            # Return metadata anyway - publisher may truncate or reject

        logger.debug(
            "Loaded JSON metadata: title=%d chars, desc=%d chars, hashtags=%d",
            len(title) if title else 0,
            len(description),
            len(hashtags),
        )
        return metadata

    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in %s: %s", json_path, e)
        return None
    except OSError as e:
        logger.error("Error loading JSON from %s: %s", json_path, e)
        return None
