"""Metadata loading utilities for video publishing.

This module provides functions to load platform-specific metadata from JSON files
or fallback to UPLOAD_INSTRUCTIONS.txt, converting to PublishMetadata format for
use with publisher implementations.
"""

import json
import logging
import re
from pathlib import Path

from src.publisher.constants import DEFAULT_OUTPUTS_DIR
from src.publisher.models import Platform, PublishMetadata

logger = logging.getLogger(__name__)


def load_platform_metadata(
    product_id: str,
    platform: Platform | str,
    outputs_dir: Path | str = DEFAULT_OUTPUTS_DIR,
) -> PublishMetadata | None:
    """Load platform-specific metadata for publishing.

    Attempts to load from metadata JSON file first, falls back to parsing
    UPLOAD_INSTRUCTIONS.txt if JSON not found. Validates content against
    platform-specific character limits.

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

    # Fallback to UPLOAD_INSTRUCTIONS.txt
    instructions_path = product_dir / "UPLOAD_INSTRUCTIONS.txt"
    metadata = _load_from_instructions(instructions_path, platform, product_id)

    if metadata:
        logger.info(
            "Loaded metadata from UPLOAD_INSTRUCTIONS.txt: %s", instructions_path
        )
        return metadata

    logger.error(
        "Could not load metadata for %s/%s (tried %s, %s, and %s)",
        product_id,
        platform.value,
        unified_path,
        platform_path,
        instructions_path,
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

        # Create PublishMetadata
        metadata = PublishMetadata(
            platform=platform,
            title=title,
            description=description,
            hashtags=hashtags,
            keywords=keywords,
            product_id=product_id,
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
    except Exception as e:
        logger.error("Error loading JSON from %s: %s", json_path, e)
        return None


def _load_from_instructions(
    instructions_path: Path,
    platform: Platform,
    product_id: str,
) -> PublishMetadata | None:
    """Load metadata by parsing UPLOAD_INSTRUCTIONS.txt.

    Extracts platform-specific sections and parses title/description/hashtags.

    Args:
    ----
        instructions_path: Path to UPLOAD_INSTRUCTIONS.txt
        platform: Target platform
        product_id: Product identifier for validation

    Returns:
    -------
        PublishMetadata object if successful, None otherwise

    """
    if not instructions_path.exists():
        logger.debug("UPLOAD_INSTRUCTIONS.txt not found: %s", instructions_path)
        return None

    try:
        with open(instructions_path, encoding="utf-8") as f:
            content = f.read()

        # Extract platform-specific section
        platform_section = _extract_platform_section(content, platform)
        if not platform_section:
            logger.error(
                "Could not find %s section in %s", platform.value, instructions_path
            )
            return None

        # Parse title (YouTube only)
        title = None
        if platform == Platform.YOUTUBE:
            title = _extract_field(platform_section, "TITLE")

        # Parse description/caption
        description_field = "DESCRIPTION" if platform == Platform.YOUTUBE else "CAPTION"
        description = _extract_field(platform_section, description_field)

        if not description:
            logger.error(
                "Could not extract %s from %s section",
                description_field, platform.value,
            )
            return None

        # Extract hashtags from description
        hashtags = _extract_hashtags(description)

        # Create PublishMetadata (no keywords available from instructions)
        metadata = PublishMetadata(
            platform=platform,
            title=title,
            description=description,
            hashtags=hashtags,
            keywords=[],  # Not available in UPLOAD_INSTRUCTIONS.txt
            product_id=product_id,
        )

        # Validate character limits
        is_valid, error_msg = metadata.validate_limits()
        if not is_valid:
            logger.warning(
                "Metadata validation for %s: %s", platform.value, error_msg
            )

        logger.debug(
            "Parsed instructions metadata: title=%d chars, desc=%d chars, hashtags=%d",
            len(title) if title else 0,
            len(description),
            len(hashtags),
        )
        return metadata

    except Exception as e:
        logger.error("Error parsing %s: %s", instructions_path, e)
        return None


def _extract_platform_section(content: str, platform: Platform) -> str | None:
    """Extract platform-specific section from UPLOAD_INSTRUCTIONS.txt.

    Args:
    ----
        content: Full content of UPLOAD_INSTRUCTIONS.txt
        platform: Target platform

    Returns:
    -------
        Platform section content or None if not found

    """
    # Platform headers in the file
    headers = {
        Platform.YOUTUBE: "YOUTUBE SHORTS",
        Platform.TIKTOK: "TIKTOK",
        Platform.INSTAGRAM: "INSTAGRAM REELS",
    }

    header = headers.get(platform)
    if not header:
        return None

    # Find section between header and next section divider
    pattern = rf"🎬\s*{re.escape(header)}.*?━+\n(.*?)(?=\n━+|$)"
    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()

    # Try alternate pattern without emoji
    pattern = rf"{re.escape(header)}.*?\n(.*?)(?=\n━+|$)"
    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)

    return match.group(1).strip() if match else None


def _extract_field(section: str, field_name: str) -> str | None:
    """Extract a specific field from platform section.

    Args:
    ----
        section: Platform section content
        field_name: Field to extract (e.g., "TITLE", "DESCRIPTION", "CAPTION")

    Returns:
    -------
        Extracted field content or None if not found

    """
    # Pattern: 📋 TITLE (copy below):\nContent
    pattern = rf"📋\s*{re.escape(field_name)}.*?:\s*\n(.*?)(?=\n📄|\n📝|\n🏷️|\n⚙️|$)"
    match = re.search(pattern, section, re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()

    # Try pattern without emoji: TITLE (copy below):\nContent
    pattern = rf"{re.escape(field_name)}.*?:\s*\n(.*?)(?=\n[A-Z]+\s*\(|$)"
    match = re.search(pattern, section, re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()

    # Try alternate patterns for description/caption
    if field_name in ["DESCRIPTION", "CAPTION"]:
        pattern = rf"📄\s*{re.escape(field_name)}.*?:\s*\n(.*?)(?=\n🏷️|\n⚙️|$)"
        match = re.search(pattern, section, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        pattern = rf"📝\s*{re.escape(field_name)}.*?:\s*\n(.*?)(?=\n⚙️|$)"
        match = re.search(pattern, section, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


def _extract_hashtags(text: str) -> list[str]:
    """Extract hashtags from text.

    Args:
    ----
        text: Text containing hashtags

    Returns:
    -------
        List of hashtags without # prefix

    """
    # Find all hashtags (# followed by alphanumeric/underscore)
    hashtag_pattern = r"#(\w+)"
    matches = re.findall(hashtag_pattern, text)

    # Return unique hashtags preserving order
    seen = set()
    unique_hashtags = []
    for tag in matches:
        if tag.lower() not in seen:
            seen.add(tag.lower())
            unique_hashtags.append(tag)

    return unique_hashtags
