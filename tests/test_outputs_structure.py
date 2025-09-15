"""Tests for outputs directory structure consistency."""

import tempfile
from pathlib import Path

import pytest

from src.utils.outputs_paths import (
    cleanup_invalid_outputs,
    ensure_outputs_structure,
    get_outputs_root,
    get_product_directory,
    validate_outputs_structure,
)


@pytest.fixture
def temp_outputs_dir():
    """Create a temporary outputs directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_validate_empty_outputs_structure(temp_outputs_dir):
    """Test validation of an empty outputs directory."""
    results = validate_outputs_structure(temp_outputs_dir)

    # Should have no products but missing global directories
    assert results["valid_products"] == []
    assert results["invalid_products"] == []
    assert results["unexpected_items"] == []
    assert "cache" in results["missing_global_dirs"]
    assert "logs" in results["missing_global_dirs"]
    assert "reports" in results["missing_global_dirs"]
    assert results["errors"] == []


def test_validate_valid_outputs_structure(temp_outputs_dir):
    """Test validation of a valid outputs directory structure."""
    temp_path = Path(temp_outputs_dir)

    # Create global directories
    (temp_path / "cache").mkdir()
    (temp_path / "logs").mkdir()
    (temp_path / "reports").mkdir()

    # Create a valid product directory
    product_dir = temp_path / "B0TEST123"
    product_dir.mkdir()
    (product_dir / "data.json").write_text('{"asin": "B0TEST123"}')
    (product_dir / "images").mkdir()
    (product_dir / "videos").mkdir()

    results = validate_outputs_structure(temp_outputs_dir)

    assert "B0TEST123" in results["valid_products"]
    assert results["invalid_products"] == []
    assert results["unexpected_items"] == []
    assert results["missing_global_dirs"] == []
    assert results["errors"] == []


def test_validate_invalid_product_directory(temp_outputs_dir):
    """Test validation of invalid product directories."""
    temp_path = Path(temp_outputs_dir)

    # Create global directories
    (temp_path / "cache").mkdir()
    (temp_path / "logs").mkdir()
    (temp_path / "reports").mkdir()

    # Create an invalid product directory (missing data.json)
    invalid_product = temp_path / "B0INVALID"
    invalid_product.mkdir()
    (invalid_product / "images").mkdir()

    results = validate_outputs_structure(temp_outputs_dir)

    assert results["valid_products"] == []
    assert "B0INVALID" in results["invalid_products"]
    assert results["missing_global_dirs"] == []


def test_validate_unexpected_items(temp_outputs_dir):
    """Test validation with unexpected files and directories."""
    temp_path = Path(temp_outputs_dir)

    # Create global directories
    (temp_path / "cache").mkdir()
    (temp_path / "logs").mkdir()
    (temp_path / "reports").mkdir()

    # Create unexpected items
    (temp_path / "random_file.txt").write_text("unexpected")
    (temp_path / "random_directory").mkdir()

    results = validate_outputs_structure(temp_outputs_dir, strict=True)

    assert "random_file.txt" in results["unexpected_items"]
    assert "random_directory" in results["unexpected_items"]


def test_cleanup_invalid_outputs_dry_run(temp_outputs_dir):
    """Test dry run cleanup of invalid outputs."""
    temp_path = Path(temp_outputs_dir)

    # Create some invalid items
    (temp_path / "invalid_file.txt").write_text("invalid")
    invalid_dir = temp_path / "invalid_dir"
    invalid_dir.mkdir()

    # Create valid structure
    (temp_path / "cache").mkdir()
    (temp_path / "logs").mkdir()
    (temp_path / "reports").mkdir()

    product_dir = temp_path / "B0VALID123"
    product_dir.mkdir()
    (product_dir / "data.json").write_text('{"asin": "B0VALID123"}')
    (product_dir / "images").mkdir()

    cleanup_results = cleanup_invalid_outputs(temp_outputs_dir, dry_run=True)

    # Should identify items to remove but not actually remove them
    assert any("invalid_file.txt" in item for item in cleanup_results["removed_items"])
    assert any("invalid_dir" in item for item in cleanup_results["removed_items"])
    assert "B0VALID123" in cleanup_results["preserved_items"]

    # Items should still exist (dry run)
    assert (temp_path / "invalid_file.txt").exists()
    assert invalid_dir.exists()


def test_centralized_path_functions():
    """Test that centralized path functions work correctly."""
    # Test basic path creation
    outputs_root = get_outputs_root()
    assert outputs_root.name == "outputs"

    # Test product directory creation
    product_dir = get_product_directory("TEST123")
    assert product_dir.name == "TEST123"
    assert product_dir.parent == outputs_root

    # Test structure creation
    ensure_outputs_structure()

    # Verify expected directories exist
    assert (outputs_root / "cache").exists()
    assert (outputs_root / "logs").exists()


def test_structure_consistency_between_modules():
    """Test that both scraper and producer use the same structure."""
    from src.scraper.amazon.botasaurus_output import get_outputs_root as scraper_root
    from src.utils.outputs_paths import get_outputs_root as utils_root

    # Both should return the same path
    assert scraper_root() == utils_root()

    # Both should create the same global directories
    scraper_root()
    utils_root()

    cache_dir = utils_root() / "cache"
    assert cache_dir.exists()


@pytest.mark.parametrize("product_id", ["B0TEST123", "B08EXAMPLE", "ASINTEST01"])
def test_product_id_validation(product_id):
    """Test product ID validation with various formats."""
    from src.utils.outputs_paths import _is_valid_product_id

    assert _is_valid_product_id(product_id) is True


@pytest.mark.parametrize(
    "invalid_id",
    [
        "short",
        "way_too_long_to_be_valid",
        "invalid-chars!",
        "",
    ],
)
def test_invalid_product_id_validation(invalid_id):
    """Test product ID validation with invalid formats."""
    from src.utils.outputs_paths import _is_valid_product_id

    assert _is_valid_product_id(invalid_id) is False
