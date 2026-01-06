"""Unit tests for outputs_paths.py utilities.

Tests path getters, validation logic, and cleanup utilities using
isolated temp directories.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.utils.outputs_paths import (
    _is_valid_product_id,
    _validate_product_directory,
    cleanup_invalid_outputs,
    ensure_outputs_structure,
    get_botasaurus_cache_directory,
    get_cache_directory,
    get_global_directory,
    get_logs_directory,
    get_outputs_root,
    get_performance_history_directory,
    get_product_directory,
    get_product_images_directory,
    get_product_music_directory,
    get_product_temp_directory,
    get_product_videos_directory,
    get_project_root,
    get_relative_path_from_outputs,
    get_reports_directory,
    get_temp_directory,
    validate_outputs_structure,
)


class TestGetProjectRoot:
    """Tests for get_project_root()."""

    @pytest.mark.unit
    def test_returns_path_object(self):
        """Test that get_project_root returns a Path object."""
        result = get_project_root()
        assert isinstance(result, Path)

    @pytest.mark.unit
    def test_returns_resolved_path(self):
        """Test that returned path is resolved (absolute)."""
        result = get_project_root()
        assert result.is_absolute()

    @pytest.mark.unit
    def test_path_exists(self):
        """Test that returned project root exists."""
        result = get_project_root()
        assert result.exists()


class TestGetOutputsRoot:
    """Tests for get_outputs_root()."""

    @pytest.mark.unit
    def test_default_outputs_dir(self):
        """Test default outputs directory name is 'outputs'."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "src.utils.outputs_paths.get_project_root", return_value=Path(tmp_dir)
            ),
        ):
            result = get_outputs_root()
            assert result.name == "outputs"
            assert result.exists()

    @pytest.mark.unit
    def test_custom_outputs_dir(self):
        """Test custom outputs directory name."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "src.utils.outputs_paths.get_project_root", return_value=Path(tmp_dir)
            ),
        ):
            result = get_outputs_root("custom_outputs")
            assert result.name == "custom_outputs"
            assert result.exists()

    @pytest.mark.unit
    def test_creates_directory_if_not_exists(self):
        """Test that directory is created if it doesn't exist."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "src.utils.outputs_paths.get_project_root", return_value=Path(tmp_dir)
            ),
        ):
            outputs_path = Path(tmp_dir) / "new_outputs"
            assert not outputs_path.exists()
            result = get_outputs_root("new_outputs")
            assert result.exists()


class TestGetProductDirectory:
    """Tests for get_product_directory()."""

    @pytest.mark.unit
    def test_creates_product_directory(self):
        """Test product directory is created."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "src.utils.outputs_paths.get_project_root", return_value=Path(tmp_dir)
            ),
        ):
            result = get_product_directory("B0TEST12345")
            assert result.name == "B0TEST12345"
            assert result.exists()
            assert result.parent.name == "outputs"

    @pytest.mark.unit
    def test_with_custom_outputs_dir(self):
        """Test product directory with custom outputs dir."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "src.utils.outputs_paths.get_project_root", return_value=Path(tmp_dir)
            ),
        ):
            result = get_product_directory("B0PROD123", "my_outputs")
            assert result.name == "B0PROD123"
            assert result.parent.name == "my_outputs"
            assert result.exists()


class TestGetGlobalDirectory:
    """Tests for get_global_directory()."""

    @pytest.mark.unit
    def test_creates_global_directory(self):
        """Test global directory is created."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "src.utils.outputs_paths.get_project_root", return_value=Path(tmp_dir)
            ),
        ):
            result = get_global_directory("mydir")
            assert result.name == "mydir"
            assert result.exists()


class TestGlobalDirectoryGetters:
    """Tests for cache, logs, reports, temp, and performance_history getters."""

    @pytest.mark.unit
    def test_get_cache_directory(self):
        """Test get_cache_directory returns correct path."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "src.utils.outputs_paths.get_project_root", return_value=Path(tmp_dir)
            ),
        ):
            result = get_cache_directory()
            assert result.name == "cache"
            assert result.exists()

    @pytest.mark.unit
    def test_get_logs_directory(self):
        """Test get_logs_directory returns correct path."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "src.utils.outputs_paths.get_project_root", return_value=Path(tmp_dir)
            ),
        ):
            result = get_logs_directory()
            assert result.name == "logs"
            assert result.exists()

    @pytest.mark.unit
    def test_get_reports_directory(self):
        """Test get_reports_directory returns correct path."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "src.utils.outputs_paths.get_project_root", return_value=Path(tmp_dir)
            ),
        ):
            result = get_reports_directory()
            assert result.name == "reports"
            assert result.exists()

    @pytest.mark.unit
    def test_get_temp_directory(self):
        """Test get_temp_directory returns correct path."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "src.utils.outputs_paths.get_project_root", return_value=Path(tmp_dir)
            ),
        ):
            result = get_temp_directory()
            assert result.name == "temp"
            assert result.exists()

    @pytest.mark.unit
    def test_get_performance_history_directory(self):
        """Test get_performance_history_directory returns correct path."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "src.utils.outputs_paths.get_project_root", return_value=Path(tmp_dir)
            ),
        ):
            result = get_performance_history_directory()
            assert result.name == "performance_history"
            assert result.exists()

    @pytest.mark.unit
    def test_get_botasaurus_cache_directory(self):
        """Test get_botasaurus_cache_directory returns correct nested path."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "src.utils.outputs_paths.get_project_root", return_value=Path(tmp_dir)
            ),
        ):
            result = get_botasaurus_cache_directory()
            assert result.name == "botasaurus"
            assert result.parent.name == "cache"
            assert result.exists()


class TestProductSubdirectoryGetters:
    """Tests for product-specific subdirectory getters."""

    @pytest.mark.unit
    def test_get_product_images_directory(self):
        """Test get_product_images_directory creates images subdir."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "src.utils.outputs_paths.get_project_root", return_value=Path(tmp_dir)
            ),
        ):
            result = get_product_images_directory("B0IMG12345")
            assert result.name == "images"
            assert result.parent.name == "B0IMG12345"
            assert result.exists()

    @pytest.mark.unit
    def test_get_product_videos_directory(self):
        """Test get_product_videos_directory creates videos subdir."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "src.utils.outputs_paths.get_project_root", return_value=Path(tmp_dir)
            ),
        ):
            result = get_product_videos_directory("B0VID12345")
            assert result.name == "videos"
            assert result.parent.name == "B0VID12345"
            assert result.exists()

    @pytest.mark.unit
    def test_get_product_music_directory(self):
        """Test get_product_music_directory creates music subdir."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "src.utils.outputs_paths.get_project_root", return_value=Path(tmp_dir)
            ),
        ):
            result = get_product_music_directory("B0MUS12345")
            assert result.name == "music"
            assert result.parent.name == "B0MUS12345"
            assert result.exists()

    @pytest.mark.unit
    def test_get_product_temp_directory(self):
        """Test get_product_temp_directory creates temp subdir."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "src.utils.outputs_paths.get_project_root", return_value=Path(tmp_dir)
            ),
        ):
            result = get_product_temp_directory("B0TMP12345")
            assert result.name == "temp"
            assert result.parent.name == "B0TMP12345"
            assert result.exists()


class TestEnsureOutputsStructure:
    """Tests for ensure_outputs_structure()."""

    @pytest.mark.unit
    def test_creates_all_global_directories(self):
        """Test that all global directories are created."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "src.utils.outputs_paths.get_project_root", return_value=Path(tmp_dir)
            ),
        ):
            ensure_outputs_structure()

            outputs_root = Path(tmp_dir) / "outputs"
            assert (outputs_root / "cache").exists()
            assert (outputs_root / "logs").exists()
            assert (outputs_root / "reports").exists()
            assert (outputs_root / "cache" / "botasaurus").exists()


class TestGetRelativePathFromOutputs:
    """Tests for get_relative_path_from_outputs()."""

    @pytest.mark.unit
    def test_relative_path_within_outputs(self):
        """Test relative path for path within outputs."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "src.utils.outputs_paths.get_project_root", return_value=Path(tmp_dir)
            ),
        ):
            outputs_root = get_outputs_root()
            product_path = outputs_root / "B0TEST" / "images"
            result = get_relative_path_from_outputs(product_path)
            assert result == "B0TEST/images"

    @pytest.mark.unit
    def test_path_outside_outputs(self):
        """Test path outside outputs returns absolute path."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "src.utils.outputs_paths.get_project_root", return_value=Path(tmp_dir)
            ),
        ):
            external_path = Path("/tmp/some/other/path")  # noqa: S108 - test data
            result = get_relative_path_from_outputs(external_path)
            assert result == "/tmp/some/other/path"  # noqa: S108 - test data


class TestIsValidProductId:
    """Tests for _is_valid_product_id()."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "product_id,expected",
        [
            ("B0TEST1234", True),  # 10 char ASIN
            ("B0ABCD12345", True),  # 11 char ASIN
            ("ABCD1234", True),  # 8 char minimum
            ("B0ABCDEFGHIJK", True),  # 13 chars
            ("B0_TEST_123", True),  # Underscores allowed
            ("short", False),  # Too short (5 chars)
            ("AB", False),  # Way too short
            ("ABCDEFGHIJKLMNOP", False),  # Too long (16 chars)
            ("B0TEST!@#$", False),  # Special characters not allowed
            ("", False),  # Empty string
        ],
    )
    def test_product_id_validation(self, product_id, expected):
        """Test product ID validation with various inputs."""
        assert _is_valid_product_id(product_id) == expected


class TestValidateProductDirectory:
    """Tests for _validate_product_directory()."""

    @pytest.mark.unit
    def test_valid_product_directory(self):
        """Test validation of valid product directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            product_dir = Path(tmp_dir) / "B0TEST1234"
            product_dir.mkdir()
            (product_dir / "images").mkdir()
            (product_dir / "data.json").write_text("{}")

            result = _validate_product_directory(product_dir, {"images", "videos"})
            assert result is True

    @pytest.mark.unit
    def test_missing_data_json(self):
        """Test validation fails when data.json is missing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            product_dir = Path(tmp_dir) / "B0TEST1234"
            product_dir.mkdir()
            (product_dir / "images").mkdir()
            # No data.json

            result = _validate_product_directory(product_dir, {"images", "videos"})
            assert result is False

    @pytest.mark.unit
    def test_missing_media_directories(self):
        """Test validation fails when no media directories exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            product_dir = Path(tmp_dir) / "B0TEST1234"
            product_dir.mkdir()
            (product_dir / "data.json").write_text("{}")
            # No images or videos directory

            result = _validate_product_directory(product_dir, {"images", "videos"})
            assert result is False

    @pytest.mark.unit
    def test_empty_product_directory(self):
        """Test validation fails for empty directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            product_dir = Path(tmp_dir) / "B0TEST1234"
            product_dir.mkdir()

            result = _validate_product_directory(product_dir, {"images", "videos"})
            assert result is False


class TestValidateOutputsStructure:
    """Tests for validate_outputs_structure()."""

    @pytest.mark.unit
    def test_empty_outputs_directory(self):
        """Test validation of empty outputs directory."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "src.utils.outputs_paths.get_project_root", return_value=Path(tmp_dir)
            ),
        ):
            # Create empty outputs
            get_outputs_root()

            result = validate_outputs_structure()
            assert result["valid_products"] == []
            assert result["invalid_products"] == []
            assert "cache" in result["missing_global_dirs"]
            assert "logs" in result["missing_global_dirs"]

    @pytest.mark.unit
    def test_valid_structure(self):
        """Test validation of valid outputs structure."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "src.utils.outputs_paths.get_project_root", return_value=Path(tmp_dir)
            ),
        ):
            # Create valid structure
            ensure_outputs_structure()

            # Create valid product
            product_dir = get_product_directory("B0VALID123")
            (product_dir / "images").mkdir()
            (product_dir / "data.json").write_text("{}")

            result = validate_outputs_structure()
            assert "B0VALID123" in result["valid_products"]
            assert result["missing_global_dirs"] == []
            assert result["errors"] == []

    @pytest.mark.unit
    def test_invalid_product_detected(self):
        """Test that invalid products are detected."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "src.utils.outputs_paths.get_project_root", return_value=Path(tmp_dir)
            ),
        ):
            outputs_root = get_outputs_root()

            # Create invalid product (no data.json)
            invalid_dir = outputs_root / "B0INVALID1"
            invalid_dir.mkdir()
            (invalid_dir / "images").mkdir()

            result = validate_outputs_structure()
            assert "B0INVALID1" in result["invalid_products"]

    @pytest.mark.unit
    def test_unexpected_file_in_outputs(self):
        """Test that unexpected files are detected in strict mode."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "src.utils.outputs_paths.get_project_root", return_value=Path(tmp_dir)
            ),
        ):
            outputs_root = get_outputs_root()
            (outputs_root / "random_file.txt").write_text("unexpected")

            result = validate_outputs_structure(strict=True)
            assert "random_file.txt" in result["unexpected_items"]


class TestCleanupInvalidOutputs:
    """Tests for cleanup_invalid_outputs()."""

    @pytest.mark.unit
    def test_dry_run_mode(self):
        """Test cleanup in dry run mode doesn't delete files."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "src.utils.outputs_paths.get_project_root", return_value=Path(tmp_dir)
            ),
        ):
            outputs_root = get_outputs_root()

            # Create invalid product
            invalid_dir = outputs_root / "B0INVALID1"
            invalid_dir.mkdir()
            (invalid_dir / "images").mkdir()
            # No data.json

            result = cleanup_invalid_outputs(dry_run=True)

            # Should report but not delete
            assert any("DRY RUN" in item for item in result["removed_items"])
            assert invalid_dir.exists()

    @pytest.mark.unit
    def test_actual_cleanup(self):
        """Test actual cleanup removes invalid items."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "src.utils.outputs_paths.get_project_root", return_value=Path(tmp_dir)
            ),
        ):
            outputs_root = get_outputs_root()

            # Create invalid product
            invalid_dir = outputs_root / "B0INVALID1"
            invalid_dir.mkdir()
            (invalid_dir / "images").mkdir()
            # No data.json

            result = cleanup_invalid_outputs(dry_run=False)

            # Should be deleted
            assert "B0INVALID1" in result["removed_items"]
            assert not invalid_dir.exists()

    @pytest.mark.unit
    def test_preserves_valid_products(self):
        """Test cleanup preserves valid products."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "src.utils.outputs_paths.get_project_root", return_value=Path(tmp_dir)
            ),
        ):
            outputs_root = get_outputs_root()

            # Create valid product
            valid_dir = outputs_root / "B0VALID123"
            valid_dir.mkdir()
            (valid_dir / "images").mkdir()
            (valid_dir / "data.json").write_text("{}")

            result = cleanup_invalid_outputs(dry_run=False)

            assert "B0VALID123" in result["preserved_items"]
            assert valid_dir.exists()

    @pytest.mark.unit
    def test_cleanup_unexpected_files(self):
        """Test cleanup removes unexpected files."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "src.utils.outputs_paths.get_project_root", return_value=Path(tmp_dir)
            ),
        ):
            outputs_root = get_outputs_root()
            unexpected_file = outputs_root / "random.txt"
            unexpected_file.write_text("unexpected")

            result = cleanup_invalid_outputs(dry_run=False)

            assert "random.txt" in result["removed_items"]
            assert not unexpected_file.exists()


class TestCustomOutputsDirParameter:
    """Tests for custom_outputs_dir parameter across functions."""

    @pytest.mark.unit
    def test_all_functions_support_custom_dir(self):
        """Test that all path functions support custom_outputs_dir."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "src.utils.outputs_paths.get_project_root", return_value=Path(tmp_dir)
            ),
        ):
            custom_name = "my_custom_outputs"

            # Test all functions with custom dir
            assert get_outputs_root(custom_name).name == custom_name
            assert get_cache_directory(custom_name).parent.name == custom_name
            assert get_logs_directory(custom_name).parent.name == custom_name
            assert get_reports_directory(custom_name).parent.name == custom_name
            assert get_temp_directory(custom_name).parent.name == custom_name
            assert (
                get_performance_history_directory(custom_name).parent.name
                == custom_name
            )

            # Product directories
            assert (
                get_product_directory("B0TEST", custom_name).parent.name == custom_name
            )
            assert (
                get_product_images_directory("B0TEST", custom_name).parent.parent.name
                == custom_name
            )
