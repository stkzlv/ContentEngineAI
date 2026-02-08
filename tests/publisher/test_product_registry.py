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
