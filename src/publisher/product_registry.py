"""Published products registry — tracks published products in JSON and CSV."""

import csv
import json
import logging
import re
from dataclasses import asdict, dataclass
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
        return [RegistryEntry(**entry) for entry in data]
    except Exception as exc:
        logger.warning("Failed to load registry: %s", exc)
        return []


def save_registry(entries: list[RegistryEntry], outputs_dir: Path) -> None:
    """Write registry to both JSON and CSV."""
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = get_registry_path(outputs_dir, "json")
    json_path.write_text(
        json.dumps([asdict(e) for e in entries], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # CSV
    csv_path = get_registry_path(outputs_dir, "csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["product_id", "title", "url", "affiliate_url"]
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(asdict(entry))

    logger.info(
        "Registry saved: %d entries (%s, %s)", len(entries), json_path, csv_path
    )


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
        )
    except Exception as exc:
        logger.warning("Failed to read data.json for %s: %s", product_id, exc)
        return None


def add_to_registry(product_id: str, outputs_dir: Path) -> bool:
    """Append a product to the registry (skip if duplicate)."""
    entries = load_registry(outputs_dir)

    if any(e.product_id == product_id for e in entries):
        logger.debug("Product %s already in registry, skipping", product_id)
        return False

    entry = _read_product_data(product_id, outputs_dir)
    if not entry:
        logger.warning("Cannot add %s to registry: no data.json", product_id)
        return False

    entries.append(entry)
    save_registry(entries, outputs_dir)
    return True


def rebuild_registry(
    outputs_dir: Path, *, scan_dir: Path | None = None
) -> int:
    """Rebuild registry from all product data.json files.

    Args:
        outputs_dir: Directory to save registry files.
        scan_dir: Directory to scan for product data. Defaults to outputs_dir.
    """
    source = scan_dir or outputs_dir
    entries: list[RegistryEntry] = []
    seen: set[str] = set()

    for data_json in sorted(source.glob("*/data.json")):
        product_id = data_json.parent.name
        if product_id in seen:
            continue

        entry = _read_product_data(product_id, source)
        if entry and entry.title:
            entries.append(entry)
            seen.add(product_id)

    save_registry(entries, outputs_dir)
    logger.info("Registry rebuilt: %d products", len(entries))
    return len(entries)
