"""Build producer input records from a topic instead of a scraped product.

The producer normally starts from `outputs/<ASIN>/data.json`, written by the
scraper. Content that is about a problem rather than a product has nothing to
scrape, so this module builds the same record from a title and a description.

Nothing downstream changes: the record is the one the pipeline already consumes,
and the run lands in an outputs directory the same way a scraped product does,
because `orchestration.create_video_for_product` derives the directory from the
record's identifier.

Two fields exist on the record with nothing to put in them. `price` and `url`
describe a listing, and a topic has no listing, so both are left empty rather
than filled with a plausible-looking value. Removing them for topic records is
tracked separately.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.scraper.amazon.models import ProductData
from src.scraper.base.models import Platform

# Cap the slug well under the filesystem limit that `sanitize_filename` enforces.
# Long enough to stay readable in `outputs/`, short enough to keep paths workable.
MAX_SLUG_LENGTH = 60

# Prefix on the generated identifier. Makes topic directories obvious in
# `outputs/` at a glance, and greppable when they need migrating.
TOPIC_ID_PREFIX = "topic-"


class TopicInputError(ValueError):
    """Raised when a topic file is malformed.

    Deliberately raised rather than skipping the offending entry: a topics file
    is written by hand, and silently dropping one line means a batch renders
    fewer videos than asked for without saying so.
    """


@dataclass
class TopicSpec:
    """A topic to render a video about.

    Attributes
    ----------
        title: The subject, used as the record title and to derive the slug.
        description: Source material for the script. The script generator reads
            only the title and the description, so this is what shapes the video.
        keywords: Optional stock-media search terms for this topic.

    """

    title: str
    description: str = ""
    keywords: list[str] = field(default_factory=list)


def topic_slug(title: str) -> str:
    """Derive a filesystem-safe, deterministic slug from a topic title.

    Deterministic so that re-running the same topic resumes the existing run
    directory and its pipeline state rather than starting a second one beside it.

    Accents are folded and punctuation collapsed, so titles differing only in
    case or punctuation produce the same slug.
    """
    normalized = unicodedata.normalize("NFKD", title)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_only.lower()).strip("-")
    if not slug:
        # Titles that are entirely punctuation or non-Latin script leave nothing
        # behind. Fall back rather than produce a bare prefix or an empty path.
        return "untitled"
    return slug[:MAX_SLUG_LENGTH].strip("-")


def topic_product_id(title: str) -> str:
    """Identifier for a topic's run, which also names its outputs directory."""
    return f"{TOPIC_ID_PREFIX}{topic_slug(title)}"


def build_topic_product(spec: TopicSpec) -> ProductData:
    """Build the record the producer consumes from a topic.

    The identifier goes in `asin` because that is the field the producer reads
    first when choosing the run directory. It is the wrong name for a topic and
    is tracked for replacement; the alternative, leaving it unset, names the
    directory from a truncated title with no marker that it is a topic at all.
    """
    return ProductData(
        title=spec.title,
        price="",
        url="",
        # `platform` is a required field with no neutral value, and
        # `ProductData.__post_init__` would stamp AMAZON regardless. Set it
        # explicitly so the record is honest about what it says rather than
        # relying on a fallback. `topic` is what actually distinguishes it.
        platform=Platform.AMAZON,
        description=spec.description,
        # Comma-joined: `resolve_topic_keywords` splits on this, so phrases
        # like "wifi router" survive as one search term.
        keyword=", ".join(spec.keywords),
        asin=topic_product_id(spec.title),
        topic=spec.title,
    )


def _spec_from_mapping(entry: Any, index: int, source: Path) -> TopicSpec:
    """Validate one entry of a topics file into a `TopicSpec`."""
    where = f"{source}, entry {index + 1}"
    if not isinstance(entry, dict):
        raise TopicInputError(
            f"{where}: expected a mapping, got {type(entry).__name__}"
        )

    title = entry.get("title")
    if not isinstance(title, str) or not title.strip():
        raise TopicInputError(
            f"{where}: 'title' is required and must be a non-empty string"
        )

    description = entry.get("description", "")
    if not isinstance(description, str):
        raise TopicInputError(f"{where}: 'description' must be a string")

    keywords = entry.get("keywords", []) or []
    if not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords):
        raise TopicInputError(f"{where}: 'keywords' must be a list of strings")

    unknown = set(entry) - {"title", "description", "keywords"}
    if unknown:
        # Strict, because a typo in a hand-written file would otherwise cost a
        # render before anyone noticed the field never applied.
        raise TopicInputError(f"{where}: unknown key(s): {', '.join(sorted(unknown))}")

    return TopicSpec(
        title=title.strip(), description=description, keywords=list(keywords)
    )


def load_topics_file(path: Path) -> list[TopicSpec]:
    """Read a YAML list of topics.

    Expected shape::

        - title: "Why your wifi keeps dropping"
          description: "Router placement, channel congestion, 2.4 vs 5GHz."
          keywords: ["wifi router", "home network"]
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise TopicInputError(f"Could not read topics file {path}: {e}") from e
    except yaml.YAMLError as e:
        raise TopicInputError(f"Could not parse topics file {path}: {e}") from e

    if raw is None:
        raise TopicInputError(f"Topics file {path} is empty")
    if not isinstance(raw, list):
        raise TopicInputError(f"Topics file {path} must contain a list of topics")

    specs = [_spec_from_mapping(entry, i, path) for i, entry in enumerate(raw)]

    seen: dict[str, str] = {}
    for spec in specs:
        pid = topic_product_id(spec.title)
        if pid in seen:
            # Two titles collapsing to one slug would render into one directory,
            # the second overwriting the first.
            raise TopicInputError(
                f"{path}: '{spec.title}' and '{seen[pid]}' both produce the "
                f"identifier '{pid}'; give them more distinct titles"
            )
        seen[pid] = spec.title
    return specs
