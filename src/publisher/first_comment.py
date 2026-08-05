"""Build first-comment text for supported platforms.

Moves affiliate links from post captions into the first comment,
keeping descriptions clean and avoiding algorithm penalties on
platforms that deprioritize posts with outbound links in captions.

Not every platform can carry a link here. YouTube renders URLs in Shorts
comments as plain text, and any 9:16 clip under the duration ceiling is
classified as a Short, so a destination URL in a YouTube first comment is
inert. The template for that platform uses ``{closing_line}`` instead: the
script's engagement-bait closing beat, which earns replies and profile visits.
The profile is the only clickable route off a Short.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.publisher.metadata import PublishMetadata
    from src.publisher.models import FirstCommentConfig

logger = logging.getLogger(__name__)

# Sentence boundary: terminator followed by whitespace or end of string.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Openers of the channel-wide CTA lines the narrator profile allows. The
# engagement-bait beat sits immediately before the CTA, so the CTA has to be
# stripped from the tail before the beat can be read off the end.
_CTA_MARKERS = (
    "link in bio",
    "follow for more",
    "drop a comment",
    "share with someone",
    "check the link",
)


def extract_closing_line(script: str) -> str | None:
    """Return the script's engagement-bait closing beat, or None.

    Every template closes with the engagement-bait beat immediately before one
    CTA line, so the beat is found by position: strip trailing CTA sentences and
    take what is left at the end. It is a two-option question on personal and
    storytelling templates and a debatable claim on analytical ones; both invite
    a reply, and both are the last thing said.

    Selecting on punctuation instead would be wrong. Question-led templates open
    with rhetorical questions in the body, so "the last question in the script"
    reaches back past the closing beat and pulls one of those out mid-script.
    """
    if not script or not script.strip():
        return None

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(script.strip()) if s.strip()]
    while sentences and any(m in sentences[-1].lower() for m in _CTA_MARKERS):
        sentences.pop()
    return sentences[-1] if sentences else None


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

    # Only the placeholders a template actually uses are required. A template
    # built from the script alone must not be skipped for a missing link.
    needs_product = "{affiliate_link}" in template or "{product_title}" in template

    affiliate_link = ""
    product_title = ""
    if needs_product:
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
            "affiliate_link", ""
        )
        if "{affiliate_link}" in template and not affiliate_link:
            logger.warning(
                "No affiliate link for %s, skipping first comment", product_id
            )
            return None
        product_title = product.get("title", "")

    closing_line = ""
    if "{closing_line}" in template:
        script_path = outputs_dir / product_id / "temp" / "script.txt"
        try:
            closing_line = extract_closing_line(script_path.read_text("utf-8")) or ""
        except OSError as e:
            logger.warning("Could not read script for %s: %s", product_id, e)
        if not closing_line:
            logger.warning(
                "No closing line for %s, skipping %s first comment",
                product_id,
                platform,
            )
            return None

    # Collect hashtags if moving them to the comment
    hashtags = ""
    if config.move_hashtags_to_comment and platform == "instagram" and metadata:
        hashtags = " ".join(
            f"#{t}" if not t.startswith("#") else t for t in (metadata.hashtags or [])
        )

    rendered = template.format(
        affiliate_link=affiliate_link,
        product_title=product_title,
        hashtags=hashtags,
        closing_line=closing_line,
    ).strip()
    return rendered or None
