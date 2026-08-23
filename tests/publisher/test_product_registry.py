"""Unit tests for published products registry."""

import csv
import json
from pathlib import Path

import pytest

from src.publisher.product_registry import (
    RegistryEntry,
    add_to_registry,
    load_registry,
    rebuild_registry,
    save_registry,
)


@pytest.fixture
def outputs_dir(tmp_path: Path) -> Path:
    """Create a temporary outputs directory."""
    return tmp_path / "outputs"


@pytest.fixture
def sample_entry() -> RegistryEntry:
    return RegistryEntry(
        product_id="B0ABC12345",
        title="Test Product",
        url="https://www.amazon.com/dp/B0ABC12345",
        affiliate_url="https://www.amazon.com/dp/B0ABC12345?tag=test-20",
    )


def _write_data_json(outputs_dir: Path, product_id: str, data: dict) -> None:
    """Helper to create a product data.json file."""
    product_dir = outputs_dir / product_id
    product_dir.mkdir(parents=True, exist_ok=True)
    (product_dir / "data.json").write_text(
        json.dumps([data], ensure_ascii=False), encoding="utf-8"
    )


class TestLoadSaveRegistry:
    """Test load/save round-trip."""

    def test_load_empty_dir(self, outputs_dir: Path):
        assert load_registry(outputs_dir) == []

    def test_save_and_load(self, outputs_dir: Path, sample_entry: RegistryEntry):
        save_registry([sample_entry], outputs_dir)

        entries = load_registry(outputs_dir)
        assert len(entries) == 1
        assert entries[0].product_id == "B0ABC12345"
        assert entries[0].title == "Test Product"

    def test_save_creates_json_and_csv(
        self, outputs_dir: Path, sample_entry: RegistryEntry
    ):
        save_registry([sample_entry], outputs_dir)

        json_path = outputs_dir / "published_products.json"
        csv_path = outputs_dir / "published_products.csv"
        assert json_path.exists()
        assert csv_path.exists()

        # Verify CSV content
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["product_id"] == "B0ABC12345"

    def test_save_creates_parent_dirs(
        self, tmp_path: Path, sample_entry: RegistryEntry
    ):
        deep_dir = tmp_path / "a" / "b" / "c"
        save_registry([sample_entry], deep_dir)
        assert (deep_dir / "published_products.json").exists()

    def test_load_corrupt_json(self, outputs_dir: Path):
        outputs_dir.mkdir(parents=True)
        (outputs_dir / "published_products.json").write_text("not json")
        assert load_registry(outputs_dir) == []


class TestAddToRegistry:
    """Test add_to_registry function."""

    def test_add_new_product(self, outputs_dir: Path):
        _write_data_json(
            outputs_dir,
            "B0ABC12345",
            {
                "title": "Test Product",
                "url": "https://www.amazon.com/dp/B0ABC12345?th=1",
                "affiliate_link": "https://www.amazon.com/dp/B0ABC12345?tag=test-20",
            },
        )

        result = add_to_registry("B0ABC12345", outputs_dir)
        assert result is True

        entries = load_registry(outputs_dir)
        assert len(entries) == 1
        assert entries[0].url == "https://www.amazon.com/dp/B0ABC12345"

    def test_skip_duplicate(self, outputs_dir: Path):
        _write_data_json(
            outputs_dir,
            "B0ABC12345",
            {
                "title": "Test Product",
                "url": "https://www.amazon.com/dp/B0ABC12345",
                "affiliate_link": "",
            },
        )

        assert add_to_registry("B0ABC12345", outputs_dir) is True
        assert add_to_registry("B0ABC12345", outputs_dir) is False
        assert len(load_registry(outputs_dir)) == 1

    def test_missing_data_json(self, outputs_dir: Path):
        outputs_dir.mkdir(parents=True)
        assert add_to_registry("B0MISSING00", outputs_dir) is False

    def test_url_normalization(self, outputs_dir: Path):
        _write_data_json(
            outputs_dir,
            "B0ABC12345",
            {
                "title": "Product",
                "url": "https://www.amazon.com/Some-Long-Name/dp/B0ABC12345?th=1&ref=abc",
                "affiliate_link": "",
            },
        )

        add_to_registry("B0ABC12345", outputs_dir)
        entries = load_registry(outputs_dir)
        assert entries[0].url == "https://www.amazon.com/dp/B0ABC12345"


class TestRebuildRegistry:
    """Test rebuild_registry function."""

    def test_rebuild_from_data_files(self, outputs_dir: Path):
        for i, asin in enumerate(["B0PRODUCT1", "B0PRODUCT2", "B0PRODUCT3"]):
            _write_data_json(
                outputs_dir,
                asin,
                {
                    "title": f"Product {i + 1}",
                    "url": f"https://www.amazon.com/dp/{asin}",
                    "affiliate_link": f"https://www.amazon.com/dp/{asin}?tag=t-20",
                },
            )

        count = rebuild_registry(outputs_dir)
        assert count == 3
        entries = load_registry(outputs_dir)
        assert len(entries) == 3

    def test_rebuild_skips_empty_title(self, outputs_dir: Path):
        _write_data_json(
            outputs_dir,
            "B0NOTITLE0",
            {
                "title": "",
                "url": "https://www.amazon.com/dp/B0NOTITLE0",
                "affiliate_link": "",
            },
        )
        _write_data_json(
            outputs_dir,
            "B0HASTITLE",
            {
                "title": "Good",
                "url": "https://www.amazon.com/dp/B0HASTITLE",
                "affiliate_link": "",
            },
        )

        count = rebuild_registry(outputs_dir)
        assert count == 1

    def test_rebuild_idempotent(self, outputs_dir: Path):
        _write_data_json(
            outputs_dir,
            "B0ABC12345",
            {
                "title": "Product",
                "url": "https://www.amazon.com/dp/B0ABC12345",
                "affiliate_link": "",
            },
        )

        assert rebuild_registry(outputs_dir) == 1
        assert rebuild_registry(outputs_dir) == 1
        assert len(load_registry(outputs_dir)) == 1

    def test_rebuild_with_separate_scan_dir(self, tmp_path: Path):
        scan_dir = tmp_path / "scan"
        save_dir = tmp_path / "save"

        _write_data_json(
            scan_dir,
            "B0ABC12345",
            {
                "title": "Product",
                "url": "https://www.amazon.com/dp/B0ABC12345",
                "affiliate_link": "",
            },
        )

        count = rebuild_registry(save_dir, scan_dir=scan_dir)
        assert count == 1
        assert (save_dir / "published_products.json").exists()
        assert not (scan_dir / "published_products.json").exists()

    def test_rebuild_empty_dir(self, outputs_dir: Path):
        outputs_dir.mkdir(parents=True)
        assert rebuild_registry(outputs_dir) == 0


class TestRepublishRefresh:
    """add_to_registry refreshes existing rows so --force republish updates fields."""

    def test_republish_refreshes_affiliate_url(self, outputs_dir: Path):
        _write_data_json(
            outputs_dir,
            "B0REPUB0002",
            {
                "title": "Test",
                "url": "https://www.amazon.com/dp/B0REPUB0002",
                "affiliate_link": "https://amzn.to/old-link",
            },
        )
        add_to_registry("B0REPUB0002", outputs_dir)
        assert load_registry(outputs_dir)[0].affiliate_url == "https://amzn.to/old-link"

        # Republish with a new affiliate URL (e.g., Lnk.bio rotation)
        _write_data_json(
            outputs_dir,
            "B0REPUB0002",
            {
                "title": "Test",
                "url": "https://www.amazon.com/dp/B0REPUB0002",
                "affiliate_link": "https://amzn.to/new-link",
            },
        )
        add_to_registry("B0REPUB0002", outputs_dir)

        entries = load_registry(outputs_dir)
        assert len(entries) == 1
        assert entries[0].affiliate_url == "https://amzn.to/new-link"

    def test_republish_returns_false_on_refresh(self, outputs_dir: Path):
        # Return contract: True only when a new row is added; False otherwise
        # (preserves the pseudo-semantic the original implementation had).
        _write_data_json(
            outputs_dir,
            "B0REPUB0003",
            {
                "title": "Test",
                "url": "https://www.amazon.com/dp/B0REPUB0003",
                "affiliate_link": "",
            },
        )
        assert add_to_registry("B0REPUB0003", outputs_dir) is True
        # Same data → no save needed, returns False
        assert add_to_registry("B0REPUB0003", outputs_dir) is False

        # Different data → refresh, still returns False (not a new row)
        _write_data_json(
            outputs_dir,
            "B0REPUB0003",
            {
                "title": "A retitled product",
                "url": "https://www.amazon.com/dp/B0REPUB0003",
                "affiliate_link": "",
            },
        )
        assert add_to_registry("B0REPUB0003", outputs_dir) is False
        assert load_registry(outputs_dir)[0].title == "A retitled product"

    def test_republish_with_no_data_change_does_not_resave(
        self, outputs_dir: Path, tmp_path: Path
    ):
        # When the entry is identical to what's already in the registry,
        # short-circuit before save_registry runs (avoids needless disk writes).
        _write_data_json(
            outputs_dir,
            "B0REPUB0004",
            {
                "title": "Test",
                "url": "https://www.amazon.com/dp/B0REPUB0004",
                "affiliate_link": "",
            },
        )
        add_to_registry("B0REPUB0004", outputs_dir)
        json_path = outputs_dir / "published_products.json"
        mtime_before = json_path.stat().st_mtime_ns

        # Repeated identical call should NOT touch the file.
        add_to_registry("B0REPUB0004", outputs_dir)
        assert json_path.stat().st_mtime_ns == mtime_before


class TestSaveRegistryBackup:
    """save_registry must back up existing files before overwriting (#137)."""

    def test_creates_bak_for_existing_json(
        self, outputs_dir: Path, sample_entry: RegistryEntry
    ) -> None:
        save_registry([sample_entry], outputs_dir)
        json_path = outputs_dir / "published_products.json"
        original = json_path.read_text(encoding="utf-8")

        # Save a different set of entries; original must be preserved as .bak.
        other = RegistryEntry(
            product_id="B0OTHER0000",
            title="Other",
            url="https://www.amazon.com/dp/B0OTHER0000",
            affiliate_url="",
        )
        save_registry([other], outputs_dir)

        bak_path = json_path.with_suffix(".json.bak")
        assert bak_path.exists()
        assert bak_path.read_text(encoding="utf-8") == original

    def test_creates_bak_for_existing_csv(
        self, outputs_dir: Path, sample_entry: RegistryEntry
    ) -> None:
        save_registry([sample_entry], outputs_dir)
        csv_path = outputs_dir / "published_products.csv"
        original = csv_path.read_text(encoding="utf-8")

        save_registry([], outputs_dir)

        bak_path = csv_path.with_suffix(".csv.bak")
        assert bak_path.exists()
        assert bak_path.read_text(encoding="utf-8") == original

    def test_no_bak_on_first_write(self, outputs_dir: Path) -> None:
        save_registry([], outputs_dir)
        assert not (outputs_dir / "published_products.json.bak").exists()
        assert not (outputs_dir / "published_products.csv.bak").exists()


class TestRebuildRegistryMerge:
    """rebuild_registry must merge with existing entries, not replace (#137)."""

    def test_merges_existing_when_scan_empty(
        self, outputs_dir: Path, sample_entry: RegistryEntry
    ) -> None:
        # Pre-populate registry with two entries, no data.json on disk.
        other = RegistryEntry(
            product_id="B0OLD0000XX",
            title="Old Product",
            url="https://www.amazon.com/dp/B0OLD0000XX",
            affiliate_url="",
        )
        save_registry([sample_entry, other], outputs_dir)

        count = rebuild_registry(outputs_dir)

        assert count == 2
        ids = {e.product_id for e in load_registry(outputs_dir)}
        assert ids == {sample_entry.product_id, other.product_id}

    def test_merges_existing_with_scan(
        self, outputs_dir: Path, sample_entry: RegistryEntry
    ) -> None:
        # Existing: A, B. Scan finds: C. Result: A, B, C.
        other = RegistryEntry(
            product_id="B0EXIST0000",
            title="Existing",
            url="https://www.amazon.com/dp/B0EXIST0000",
            affiliate_url="",
        )
        save_registry([sample_entry, other], outputs_dir)

        _write_data_json(
            outputs_dir,
            "B0NEW000000",
            {
                "title": "New Product",
                "url": "https://www.amazon.com/dp/B0NEW000000",
                "affiliate_link": "",
            },
        )

        count = rebuild_registry(outputs_dir)

        assert count == 3
        ids = {e.product_id for e in load_registry(outputs_dir)}
        assert ids == {sample_entry.product_id, "B0EXIST0000", "B0NEW000000"}

    def test_scan_overwrites_existing_on_match(self, outputs_dir: Path) -> None:
        # Existing A with old title. Scan finds A with new title.
        old = RegistryEntry(
            product_id="B0SAME00000",
            title="Old Title",
            url="https://www.amazon.com/dp/B0SAME00000",
            affiliate_url="",
        )
        save_registry([old], outputs_dir)

        _write_data_json(
            outputs_dir,
            "B0SAME00000",
            {
                "title": "New Title",
                "url": "https://www.amazon.com/dp/B0SAME00000",
                "affiliate_link": "https://www.amazon.com/dp/B0SAME00000?tag=t-20",
            },
        )

        rebuild_registry(outputs_dir)

        entries = load_registry(outputs_dir)
        assert len(entries) == 1
        assert entries[0].title == "New Title"
        assert entries[0].affiliate_url.endswith("tag=t-20")


@pytest.mark.unit
class TestARemovedColumnDoesNotDestroyHistory:
    """Every row written before a column is dropped still carries its key.

    `load_registry` builds each entry by splatting the row, so an undeclared
    key raises. The caller treats an unreadable registry as an empty one and
    writes the file afresh, so a strict load would replace the whole history
    with whatever row was being added. The `.bak` covers one generation of
    that, and only if someone notices.

    Named for the general case rather than for `pillar`: the next column to
    go should not have to rediscover this.
    """

    def test_rows_carrying_a_dropped_column_still_load(self, tmp_path):
        from src.publisher.product_registry import load_registry

        (tmp_path / "published_products.json").write_text(
            json.dumps(
                [
                    {
                        "product_id": "B0OLD00001",
                        "title": "An older row",
                        "url": "https://example.com/1",
                        "affiliate_url": "https://example.com/1?tag=x",
                        "pillar": "value",
                        "content_format": "product",
                    }
                ]
            ),
            encoding="utf-8",
        )

        entries = load_registry(tmp_path)

        assert len(entries) == 1, "a dropped column emptied the registry"
        assert entries[0].product_id == "B0OLD00001"
        assert entries[0].content_format == "product"
        assert not hasattr(entries[0], "pillar")

    def test_a_dropped_column_does_not_survive_a_round_trip(self, tmp_path):
        """Loading tolerates the key; saving must not write it back."""
        from src.publisher.product_registry import load_registry, save_registry

        (tmp_path / "published_products.json").write_text(
            json.dumps(
                [
                    {
                        "product_id": "B0OLD00001",
                        "title": "An older row",
                        "url": "https://example.com/1",
                        "affiliate_url": "https://example.com/1?tag=x",
                        "pillar": "value",
                    }
                ]
            ),
            encoding="utf-8",
        )

        save_registry(load_registry(tmp_path), tmp_path)

        written = json.loads(
            (tmp_path / "published_products.json").read_text(encoding="utf-8")
        )
        assert "pillar" not in written[0]
        assert written[0]["product_id"] == "B0OLD00001"
