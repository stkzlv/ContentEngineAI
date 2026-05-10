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


def _write_pipeline_state(outputs_dir: Path, product_id: str, state: dict) -> None:
    """Helper to create a product pipeline_state.json file."""
    temp_dir = outputs_dir / product_id / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / "pipeline_state.json").write_text(
        json.dumps(state, ensure_ascii=False), encoding="utf-8"
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


class TestPillarColumn:
    """Test pillar field on registry rows (#83)."""

    def test_default_pillar_is_empty(self, sample_entry: RegistryEntry):
        assert sample_entry.pillar == ""

    def test_add_picks_pillar_from_pipeline_state(self, outputs_dir: Path):
        _write_data_json(
            outputs_dir,
            "B0PILLAR001",
            {
                "title": "Tagged Product",
                "url": "https://www.amazon.com/dp/B0PILLAR001",
                "affiliate_link": "",
            },
        )
        _write_pipeline_state(
            outputs_dir,
            "B0PILLAR001",
            {"pillar": "value", "script_template": "before_after"},
        )

        add_to_registry("B0PILLAR001", outputs_dir)
        entries = load_registry(outputs_dir)
        assert entries[0].pillar == "value"

    def test_add_with_no_state_file_leaves_pillar_empty(self, outputs_dir: Path):
        _write_data_json(
            outputs_dir,
            "B0NOSTATE00",
            {
                "title": "Untagged",
                "url": "https://www.amazon.com/dp/B0NOSTATE00",
                "affiliate_link": "",
            },
        )

        add_to_registry("B0NOSTATE00", outputs_dir)
        entries = load_registry(outputs_dir)
        assert entries[0].pillar == ""

    def test_rebuild_picks_pillar_from_state(self, outputs_dir: Path):
        for asin, pillar in [
            ("B0VALUEPRO", "value"),
            ("B0NOVELTY01", "novelty"),
            ("B0NOTAG0001", None),
        ]:
            _write_data_json(
                outputs_dir,
                asin,
                {
                    "title": f"Product {asin}",
                    "url": f"https://www.amazon.com/dp/{asin}",
                    "affiliate_link": "",
                },
            )
            if pillar is not None:
                _write_pipeline_state(outputs_dir, asin, {"pillar": pillar})

        rebuild_registry(outputs_dir)
        entries = {e.product_id: e for e in load_registry(outputs_dir)}
        assert entries["B0VALUEPRO"].pillar == "value"
        assert entries["B0NOVELTY01"].pillar == "novelty"
        assert entries["B0NOTAG0001"].pillar == ""

    def test_csv_includes_pillar_column(
        self, outputs_dir: Path, sample_entry: RegistryEntry
    ):
        sample_entry.pillar = "utility"
        save_registry([sample_entry], outputs_dir)

        with open(outputs_dir / "published_products.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
        assert "pillar" in fieldnames
        assert rows[0]["pillar"] == "utility"

    def test_load_legacy_json_without_pillar(self, outputs_dir: Path):
        # Existing registry rows from before #83 won't have a pillar field.
        # Loader must accept them and default pillar to "".
        outputs_dir.mkdir(parents=True)
        legacy = [
            {
                "product_id": "B0LEGACY001",
                "title": "Legacy",
                "url": "https://www.amazon.com/dp/B0LEGACY001",
                "affiliate_url": "",
            }
        ]
        (outputs_dir / "published_products.json").write_text(
            json.dumps(legacy), encoding="utf-8"
        )

        entries = load_registry(outputs_dir)
        assert len(entries) == 1
        assert entries[0].pillar == ""

    def test_corrupt_state_file_leaves_pillar_empty(self, outputs_dir: Path):
        _write_data_json(
            outputs_dir,
            "B0CORRUPT01",
            {
                "title": "Corrupt State",
                "url": "https://www.amazon.com/dp/B0CORRUPT01",
                "affiliate_link": "",
            },
        )
        temp_dir = outputs_dir / "B0CORRUPT01" / "temp"
        temp_dir.mkdir(parents=True)
        (temp_dir / "pipeline_state.json").write_text("{not json")

        add_to_registry("B0CORRUPT01", outputs_dir)
        entries = load_registry(outputs_dir)
        assert entries[0].pillar == ""


class TestRepublishRefresh:
    """add_to_registry refreshes existing rows so --force republish updates fields."""

    def test_republish_refreshes_pillar(self, outputs_dir: Path):
        _write_data_json(
            outputs_dir,
            "B0REPUB0001",
            {
                "title": "Test",
                "url": "https://www.amazon.com/dp/B0REPUB0001",
                "affiliate_link": "",
            },
        )
        # First publish: no pillar in state
        add_to_registry("B0REPUB0001", outputs_dir)
        assert load_registry(outputs_dir)[0].pillar == ""

        # Second publish (--force): producer ran with --pillar value
        _write_pipeline_state(outputs_dir, "B0REPUB0001", {"pillar": "value"})
        add_to_registry("B0REPUB0001", outputs_dir)

        entries = load_registry(outputs_dir)
        assert len(entries) == 1
        assert entries[0].pillar == "value"

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
        _write_pipeline_state(outputs_dir, "B0REPUB0003", {"pillar": "novelty"})
        assert add_to_registry("B0REPUB0003", outputs_dir) is False

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
