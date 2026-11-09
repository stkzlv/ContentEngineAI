"""Build first-comment text for supported platforms.

Moves affiliate links from post captions into the first comment,
keeping descriptions clean and avoiding algorithm penalties on
platforms that deprioritize posts with outbound links in captions.

Not every platform can carry a link here. YouTube renders URLs in Shorts
comments as plain text, and any 9:16 clip under the duration ceiling is
classified as a Short, so a destination URL in a YouTube first comment is
inert. The template for that platform uses ``{closing_line}`` instead: the
script's closing question or claim, which earns replies and profile visits.
The profile is the only clickable route off a Short.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.publisher.metadata import PublishMetadata
    from src.publisher.models import FirstCommentConfig

from src.utils.script_sanitizer import split_sentences
from src.utils.script_signoff import normalise, read_script

logger = logging.getLogger(__name__)


# Openers of the channel-wide CTA lines the narrator profile allows. The
# engagement-bait beat sits immediately before the CTA, so the CTA has to be
# stripped from the tail before the beat can be read off the end.
# One opener per configured CTA line, product and topic, matched at the start
# of the sentence. The publisher does not load the video config, so this list
# is hand-kept and a test asserts every `script_templates.cta_options*` entry
# starts with one -- a CTA this misses is not stripped, and the YouTube first
# comment becomes the CTA itself.
_CTA_MARKERS = (
    "link in bio",
    "follow for more",
    "drop a comment",
    "share with someone",
    "share it with",
    "save this",
    "check the link",
)


def extract_closing_line(script: str, signoff: str | None = None) -> str | None:
    """Return the script's closing beat (question or claim), or None.

    Every template closes with that beat immediately before one
    CTA line, so the beat is found by position: strip trailing CTA sentences and
    take what is left at the end. It is a two-option question on personal and
    storytelling templates and a debatable claim on analytical ones; both invite
    a reply, and both are the last thing said.

    Selecting on punctuation instead would be wrong. Question-led templates open
    with rhetorical questions in the body, so "the last question in the script"
    reaches back past the closing beat and pulls one of those out mid-script.

    A render with an author sign-off speaks it between the closing beat and
    the CTA, so the recorded `signoff` is stripped next; otherwise the first
    comment would be the sign-off.
    """
    if not script or not script.strip():
        return None

    sentences = split_sentences(script)
    # Anchored to the start of the sentence. A substring match anywhere
    # popped a closing beat that merely *contained* a marker word -- "Save
    # this or skip it, which would you do?" -- and made the sentence before
    # it the first comment. Every configured CTA opens with its marker.
    while sentences and any(
        sentences[-1].lower().lstrip("\"'\u201c\u2018 ").startswith(m)
        for m in _CTA_MARKERS
    ):
        sentences.pop()
    if signoff and sentences and normalise(sentences[-1]) == normalise(signoff):
        sentences.pop()
    return sentences[-1] if sentences else None


def first_comment_override(product_dir: Path, platform: str) -> tuple[bool, str | None]:
    """Read a post's own first comment for ``platform`` from its metadata.json.

    ``first_comment`` maps a platform to a string, which replaces the template,
    or to null or "", which skips the comment (REQ-PUB-161). Returns
    ``(False, None)`` when the post sets nothing for the platform, and for a
    malformed field, which is logged and ignored.
    """
    path = product_dir / "metadata.json"
    if not path.exists():
        return False, None
    try:
        field = json.loads(path.read_text(encoding="utf-8")).get("first_comment")
    except (json.JSONDecodeError, OSError, AttributeError):
        return False, None
    if field is None:
        return False, None
    if not isinstance(field, dict):
        logger.warning("Ignoring first_comment in %s: not a platform map", path)
        return False, None
    if platform not in field:
        return False, None
    value = field[platform]
    if value is None:
        return True, None
    if not isinstance(value, str):
        logger.warning("Ignoring first_comment.%s in %s: not text", platform, path)
        return False, None
    return True, value.strip() or None


def correction_target(outputs_dir: Path, product_id: str) -> str | None:
    """The record a correction corrects, read from its data.json, else None."""
    try:
        raw = json.loads((outputs_dir / product_id / "data.json").read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    record = raw[0] if isinstance(raw, list) and raw else raw
    if not isinstance(record, dict) or record.get("topic_kind") != "correction":
        return None
    target = record.get("corrects")
    return target if isinstance(target, str) and target else None


async def resolve_correction_links(
    publisher: object, product_id: str, outputs_dir: Path, platforms: list[str]
) -> dict[str, tuple[str, str | None]]:
    """For a correction, each platform's original title and post URL.

    Empty for any other record. The title comes from the product registry,
    which outlives the original's folder; the URL from the provider, through
    the post id the publish history holds. A URL is None while the original
    is still scheduled, or when the provider can't be asked.
    """
    from src.publisher.product_registry import load_registry
    from src.publisher.tracking import get_publish_record

    target = correction_target(outputs_dir, product_id)
    if target is None:
        return {}
    title = next(
        (
            e.title
            for e in load_registry(outputs_dir)
            if e.product_id == target and e.title
        ),
        target,
    )
    fetch = getattr(publisher, "get_post_platforms", None)
    legs_by_post: dict[str, list[dict]] = {}
    links: dict[str, tuple[str, str | None]] = {}
    for platform in platforms:
        row = get_publish_record(target, platform, outputs_dir) or {}
        post_id = row.get("post_id")
        url = None
        if post_id and fetch is not None:
            if post_id not in legs_by_post:
                try:
                    legs_by_post[post_id] = await fetch(post_id)
                except Exception as exc:  # noqa: BLE001 - the title still posts
                    logger.warning(
                        "Could not read post %s for its URL: %s", post_id, exc
                    )
                    legs_by_post[post_id] = []
            url = next(
                (
                    leg.get("platform_post_url")
                    for leg in legs_by_post[post_id]
                    if leg.get("platform") == platform
                ),
                None,
            )
        links[platform] = (title, url)
    return links


def build_first_comment(
    config: FirstCommentConfig,
    platform: str,
    product_id: str,
    outputs_dir: Path,
    metadata: PublishMetadata | None = None,
    correction_links: dict[str, tuple[str, str | None]] | None = None,
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
        correction_links: For a correction, each platform's original title
            and URL (`resolve_correction_links`); its first comment links
            the original instead of using the platform template.

    """
    if not config.enabled:
        return None

    if platform == "tiktok":
        return None

    overridden, own = first_comment_override(outputs_dir / product_id, platform)
    if overridden:
        if own is None:
            logger.info("First comment on %s skipped by the post's metadata", platform)
        return own

    if correction_links and platform in correction_links:
        original_title, original_url = correction_links[platform]
        if not original_url:
            logger.warning(
                "No %s URL yet for the post %s corrects; naming it by title",
                platform,
                product_id,
            )
        return (
            config.correction_template.format(
                original_title=original_title, original_url=original_url or ""
            ).strip()
            or None
        )

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
        found = read_script(outputs_dir / product_id)
        if found is None:
            logger.warning("Could not read script for %s", product_id)
        else:
            closing_line = extract_closing_line(*found) or ""
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
