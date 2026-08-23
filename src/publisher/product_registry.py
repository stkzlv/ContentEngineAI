"""Published products registry — tracks published products in JSON and CSV."""

import csv
import json
import logging
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

logger = logging.getLogger(__name__)

REGISTRY_JSON = "published_products.json"
REGISTRY_CSV = "published_products.csv"
ASIN_RE = re.compile(r"/dp/([A-Z0-9]{10})")


@dataclass
class RegistryEntry:
    """A single published product record."""

    product_id: str
    title: str
    url: str
    affiliate_url: str
    # Which content format the product was produced under, so two formats
    # published side by side can be told apart afterwards. Comparing formats
    # requires interleaving them day by day, which is exactly the case where
    # publish date cannot reconstruct the arm.
    content_format: str = ""


def get_registry_path(outputs_dir: Path, fmt: str = "json") -> Path:
    """Return path to the registry file."""
    filename = REGISTRY_JSON if fmt == "json" else REGISTRY_CSV
    return outputs_dir / filename


def load_registry(outputs_dir: Path) -> list[RegistryEntry]:
    """Load registry entries from JSON file."""
    path = get_registry_path(outputs_dir, "json")
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        logger.warning("Failed to load registry: %s", exc)
        return []

    # Per row, not per file. A row the schema cannot build must cost that row
    # and nothing else: the caller treats an unreadable registry as an empty
    # one and rewrites the file, so failing the whole load replaces the entire
    # publish history with whatever row was being added.
    #
    # Unknown keys are dropped rather than passed on, so removing a column
    # does not break every row written before the removal.
    known = {f.name for f in fields(RegistryEntry)}
    entries: list[RegistryEntry] = []
    for row in data:
        if not isinstance(row, dict):
            logger.warning("Skipping registry row that is not an object: %r", row)
            continue
        try:
            entries.append(
                RegistryEntry(**{k: v for k, v in row.items() if k in known})
            )
        except TypeError as exc:
            logger.warning("Skipping unreadable registry row: %s", exc)
    return entries


def save_registry(entries: list[RegistryEntry], outputs_dir: Path) -> None:
    """Write registry to both JSON and CSV.

    Each existing file is renamed to ``<name>.bak`` before the new file
    is written, so a write that drops or corrupts entries can be recovered
    from the backup.
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)

    json_path = get_registry_path(outputs_dir, "json")
    csv_path = get_registry_path(outputs_dir, "csv")

    # Back up existing files before overwriting.
    for p in (json_path, csv_path):
        if p.exists():
            p.replace(p.with_suffix(p.suffix + ".bak"))

    json_path.write_text(
        json.dumps([asdict(e) for e in entries], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        # Derived from the dataclass rather than restated: a hand-written list
        # goes stale the moment a field is added, and `DictWriter` raises on the
        # extra key instead of writing a column, so the whole registry write
        # fails rather than one column going missing.
        writer = csv.DictWriter(
            f, fieldnames=[fld.name for fld in fields(RegistryEntry)]
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(asdict(entry))

    logger.info(
        "Registry saved: %d entries (%s, %s)", len(entries), json_path, csv_path
    )


CONTENT_FORMAT_TOPIC = "topic"
CONTENT_FORMAT_PRODUCT = "product"


def _content_format(product: dict) -> str:
    """Which arm a record belongs to.

    Read from the record rather than from the profile or the publish date. The
    profile is a visual treatment and two arms can share one; the date cannot
    reconstruct an arm that was interleaved, which is the only way to run the
    comparison fairly.
    """
    topic = product.get("topic")
    if isinstance(topic, str) and topic.strip():
        return CONTENT_FORMAT_TOPIC
    return CONTENT_FORMAT_PRODUCT


def _read_product_data(product_id: str, outputs_dir: Path) -> RegistryEntry | None:
    """Read product data.json and extract registry fields."""
    data_path = outputs_dir / product_id / "data.json"
    if not data_path.exists():
        return None

    try:
        raw = json.loads(data_path.read_text(encoding="utf-8"))
        product = raw[0] if isinstance(raw, list) else raw

        title = product.get("title", "")
        url = product.get("url", "")
        affiliate_url = product.get("affiliate_link", "")

        # Normalize URL to canonical form
        if url:
            m = ASIN_RE.search(url)
            if m:
                url = f"https://www.amazon.com/dp/{m.group(1)}"

        return RegistryEntry(
            product_id=product_id,
            title=title,
            url=url,
            affiliate_url=affiliate_url,
            content_format=_content_format(product),
        )
    except (json.JSONDecodeError, OSError, KeyError, ValueError) as exc:
        logger.warning("Failed to read data.json for %s: %s", product_id, exc)
        return None


def add_to_registry(product_id: str, outputs_dir: Path) -> bool:
    """Add or refresh a product in the registry.

    When the product is new, append it. When the product already exists (e.g.
    after a `--force` republish), replace the existing entry with the latest
    data so fields like ``title`` and ``affiliate_url`` reflect what was
    actually published this time, not the original publish.

    Returns
    -------
    True if a new entry was added; False if an existing entry was refreshed
    or the read failed.

    """
    entries = load_registry(outputs_dir)

    entry = _read_product_data(product_id, outputs_dir)
    if not entry:
        logger.warning("Cannot add %s to registry: no data.json", product_id)
        return False

    for i, existing in enumerate(entries):
        if existing.product_id == product_id:
            if existing == entry:
                logger.debug(
                    "Product %s already in registry with current data, skipping save",
                    product_id,
                )
                return False
            logger.info(
                "Refreshing registry entry for %s (republish updates fields)",
                product_id,
            )
            entries[i] = entry
            save_registry(entries, outputs_dir)
            return False

    entries.append(entry)
    save_registry(entries, outputs_dir)
    return True


def rebuild_registry(outputs_dir: Path, *, scan_dir: Path | None = None) -> int:
    """Rebuild registry by merging scanned entries into the existing one.

    Existing entries stay in the registry even when their product directory
    has been cleaned up after publishing. Scanned entries update matching
    existing rows (keyed by product_id) and add new ones. Counts logged
    cover existing, scanned, and final entry totals.

    Args:
    ----
        outputs_dir: Directory to save registry files.
        scan_dir: Directory to scan for product data. Defaults to outputs_dir.

    """
    source = scan_dir or outputs_dir
    existing = {e.product_id: e for e in load_registry(outputs_dir)}
    existing_count = len(existing)

    scanned_count = 0
    for data_json in sorted(source.glob("*/data.json")):
        product_id = data_json.parent.name
        entry = _read_product_data(product_id, source)
        if entry and entry.title:
            existing[product_id] = entry
            scanned_count += 1

    entries = list(existing.values())
    save_registry(entries, outputs_dir)
    logger.info(
        "Registry rebuilt: %d entries (existing=%d, scanned=%d)",
        len(entries),
        existing_count,
        scanned_count,
    )
    return len(entries)


def summarize_by_content_format(entries: list[RegistryEntry]) -> dict[str, int]:
    """Count published products per content-format arm.

    Products, not videos: the registry holds one row per product id, and a
    republish replaces that row rather than appending. A product published
    twice is one row and two live videos, so an arm that republishes more is
    under-counted here. Counting videos needs the scheduler's own record, not
    local state, since the publish history is keyed the same way and is
    overwritten on republish too.

    The point of recording the arm is being able to segment by it without
    keeping a list outside the system, so the grouping lives here rather than
    being rebuilt by every caller.

    Entries written before the arm was recorded carry an empty string. They are
    reported under "unlabelled" rather than folded into either arm, because a
    comparison that silently counts unknown videos as one side is worse than one
    that shows how many it cannot place.
    """
    counts: dict[str, int] = {}
    for entry in entries:
        key = entry.content_format or "unlabelled"
        counts[key] = counts.get(key, 0) + 1
    return counts
