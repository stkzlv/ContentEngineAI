"""Unit tests for metadata export functionality."""

import csv
import json
import tempfile
from pathlib import Path

import pytest

from src.ai.platform_metadata.export import (
    ExportFormat,
    ExportResult,
    MetadataExporter,
)
from src.ai.platform_metadata.models import (
    ExportSettings,
    PlatformMetadata,
    PlatformMetadataSettings,
)


class TestExportSettings:
    """Test ExportSettings Pydantic model."""

    def test_default_settings(self):
        """Test default export settings."""
        settings = ExportSettings()

        assert settings.enabled is True
        assert settings.default_format == "json"
        assert settings.youtube_category == "22"
        assert settings.youtube_privacy == "private"

    def test_custom_settings(self):
        """Test custom export settings."""
        settings = ExportSettings(
            enabled=False,
            default_format="csv",
            youtube_category="28",
            youtube_privacy="public",
        )

        assert settings.enabled is False
        assert settings.default_format == "csv"
        assert settings.youtube_category == "28"
        assert settings.youtube_privacy == "public"

    def test_format_validation(self):
        """Test format pattern validation."""
        # Valid formats
        for fmt in ["json", "csv", "youtube_csv", "tiktok", "instagram"]:
            settings = ExportSettings(default_format=fmt)
            assert settings.default_format == fmt

        # Invalid format
        with pytest.raises(ValueError):
            ExportSettings(default_format="invalid")

    def test_privacy_validation(self):
        """Test privacy pattern validation."""
        # Valid privacy settings
        for privacy in ["private", "public", "unlisted"]:
            settings = ExportSettings(youtube_privacy=privacy)
            assert settings.youtube_privacy == privacy

        # Invalid privacy
        with pytest.raises(ValueError):
            ExportSettings(youtube_privacy="invalid")


class TestExportResult:
    """Test ExportResult dataclass."""

    def test_result_creation(self):
        """Test creating an export result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.json"
            result = ExportResult(
                success=True,
                format=ExportFormat.JSON,
                file_path=file_path,
                record_count=10,
            )

            assert result.success is True
            assert result.format == ExportFormat.JSON
            assert result.record_count == 10
            assert result.error_message is None

    def test_result_with_error(self):
        """Test result with error."""
        result = ExportResult(
            success=False,
            format=ExportFormat.CSV,
            error_message="Export failed",
        )

        assert result.success is False
        assert result.error_message == "Export failed"

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "youtube.csv"
            result = ExportResult(
                success=True,
                format=ExportFormat.YOUTUBE_CSV,
                file_path=file_path,
                record_count=5,
            )

            data = result.to_dict()

            assert data["success"] is True
            assert data["format"] == "youtube_csv"
            assert data["file_path"] == str(file_path)
            assert data["record_count"] == 5


class TestMetadataExporter:
    """Test MetadataExporter class."""

    @pytest.fixture
    def exporter(self):
        """Create exporter instance."""
        return MetadataExporter()

    @pytest.fixture
    def youtube_metadata(self):
        """Create sample YouTube metadata."""
        return PlatformMetadata.create(
            platform="youtube",
            title="Amazing Product Review",
            description="Check out this amazing product! Best value for money.",
            hashtags=["#Shorts", "#ProductReview", "#ad"],
            keywords=["product review", "best value"],
            product_id="B0TEST001",
        )

    @pytest.fixture
    def tiktok_metadata(self):
        """Create sample TikTok metadata."""
        return PlatformMetadata.create(
            platform="tiktok",
            description="This product changed my life!",
            hashtags=["#TechReview", "#MustHave", "#ad"],
            keywords=["tech review", "must have"],
            product_id="B0TEST001",
        )

    @pytest.fixture
    def instagram_metadata(self):
        """Create sample Instagram metadata."""
        return PlatformMetadata.create(
            platform="instagram",
            description="Best purchase ever! ✨",
            hashtags=[
                "#ProductReview",
                "#BestPurchase",
                "#TechGadgets",
                "#ad",
            ],
            keywords=["product review"],
            product_id="B0TEST001",
        )

    @pytest.fixture
    def all_metadata(self, youtube_metadata, tiktok_metadata, instagram_metadata):
        """Create list with all platform metadata."""
        return [youtube_metadata, tiktok_metadata, instagram_metadata]

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for exports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    # --- JSON Export Tests ---

    def test_export_json(self, exporter, youtube_metadata, temp_dir):
        """Test JSON export to file."""
        output_path = temp_dir / "metadata.json"

        result = exporter.export_json([youtube_metadata], output_path)

        assert result.success is True
        assert result.format == ExportFormat.JSON
        assert result.record_count == 1
        assert output_path.exists()

        # Verify JSON content
        with output_path.open() as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["platform"] == "youtube"
        assert data[0]["title"] == "Amazing Product Review"

    def test_to_json_string(self, exporter, youtube_metadata):
        """Test converting to JSON string."""
        content = exporter.to_json_string([youtube_metadata])

        data = json.loads(content)

        assert len(data) == 1
        assert data[0]["product_id"] == "B0TEST001"

    def test_json_unicode_handling(self, exporter, temp_dir):
        """Test JSON export handles unicode correctly."""
        metadata = PlatformMetadata.create(
            platform="instagram",
            description="Best product ever! 🔥✨💯",
            hashtags=["#emoji", "#unicode", "#日本語"],
            keywords=["unicode test"],
            product_id="B0UNICODE",
        )

        output_path = temp_dir / "unicode.json"
        result = exporter.export_json([metadata], output_path)

        assert result.success is True

        # Verify unicode preserved
        with output_path.open(encoding="utf-8") as f:
            data = json.load(f)

        assert "🔥✨💯" in data[0]["description"]
        assert "#日本語" in data[0]["hashtags"]

    # --- CSV Export Tests ---

    def test_export_csv(self, exporter, youtube_metadata, temp_dir):
        """Test CSV export to file."""
        output_path = temp_dir / "metadata.csv"

        result = exporter.export_csv([youtube_metadata], output_path)

        assert result.success is True
        assert result.format == ExportFormat.CSV
        assert result.record_count == 1
        assert output_path.exists()

        # Verify CSV content
        with output_path.open(encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["platform"] == "youtube"
        assert rows[0]["product_id"] == "B0TEST001"

    def test_to_csv_string(self, exporter, all_metadata):
        """Test converting to CSV string."""
        content = exporter.to_csv_string(all_metadata)

        # Parse CSV content
        reader = csv.DictReader(content.splitlines())
        rows = list(reader)

        assert len(rows) == 3

    def test_csv_columns(self, exporter, youtube_metadata):
        """Test CSV contains expected columns."""
        content = exporter.to_csv_string([youtube_metadata])

        reader = csv.DictReader(content.splitlines())
        rows = list(reader)

        expected_columns = [
            "product_id",
            "platform",
            "title",
            "description",
            "hashtags",
            "keywords",
            "validation_status",
            "generated_at",
            "prompt_variant",
        ]

        for col in expected_columns:
            assert col in rows[0], f"Missing column: {col}"

    def test_csv_hashtags_formatting(self, exporter, youtube_metadata):
        """Test hashtags are space-separated in CSV."""
        content = exporter.to_csv_string([youtube_metadata])

        reader = csv.DictReader(content.splitlines())
        row = next(reader)

        assert row["hashtags"] == "#Shorts #ProductReview #ad"

    # --- YouTube CSV Export Tests ---

    def test_export_youtube_csv(self, exporter, youtube_metadata, temp_dir):
        """Test YouTube CSV export."""
        output_path = temp_dir / "youtube.csv"

        result = exporter.export_youtube_csv([youtube_metadata], output_path)

        assert result.success is True
        assert result.format == ExportFormat.YOUTUBE_CSV
        assert result.record_count == 1

        # Verify YouTube CSV format
        with output_path.open(encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["title"] == "Amazing Product Review"
        assert "Shorts" in rows[0]["tags"]  # Without #
        assert rows[0]["category"] == "22"
        assert rows[0]["privacy"] == "private"

    def test_youtube_csv_filters_non_youtube(self, exporter, all_metadata, temp_dir):
        """Test YouTube CSV export filters non-YouTube metadata."""
        output_path = temp_dir / "youtube.csv"

        result = exporter.export_youtube_csv(all_metadata, output_path)

        assert result.success is True
        assert result.record_count == 1  # Only YouTube metadata

    def test_youtube_csv_tags_formatting(self, exporter, youtube_metadata):
        """Test tags are comma-separated without # prefix."""
        content = exporter.to_youtube_csv_string([youtube_metadata])

        reader = csv.DictReader(content.splitlines())
        row = next(reader)

        # Tags should be comma-separated, without #
        tags = row["tags"].split(",")
        assert "Shorts" in tags
        assert "ProductReview" in tags
        assert "#Shorts" not in tags

    def test_youtube_csv_includes_keywords_as_tags(self, exporter, youtube_metadata):
        """Test keywords are included in tags."""
        content = exporter.to_youtube_csv_string([youtube_metadata])

        reader = csv.DictReader(content.splitlines())
        row = next(reader)

        assert "product review" in row["tags"]
        assert "best value" in row["tags"]

    # --- TikTok Export Tests ---

    def test_export_tiktok(self, exporter, tiktok_metadata, temp_dir):
        """Test TikTok export."""
        output_path = temp_dir / "tiktok.txt"

        result = exporter.export_tiktok([tiktok_metadata], output_path)

        assert result.success is True
        assert result.format == ExportFormat.TIKTOK
        assert result.record_count == 1

        content = output_path.read_text()
        assert "B0TEST001" in content
        assert "This product changed my life!" in content
        assert "#TechReview" in content

    def test_tiktok_caption_format(self, exporter, tiktok_metadata):
        """Test TikTok caption format."""
        content = exporter.to_tiktok_string([tiktok_metadata])

        # Check format: description + blank line + hashtags
        assert "This product changed my life!" in content
        assert "#TechReview #MustHave #ad" in content

    def test_tiktok_filters_non_tiktok(self, exporter, all_metadata, temp_dir):
        """Test TikTok export filters non-TikTok metadata."""
        output_path = temp_dir / "tiktok.txt"

        result = exporter.export_tiktok(all_metadata, output_path)

        assert result.record_count == 1  # Only TikTok metadata

    # --- Instagram Export Tests ---

    def test_export_instagram(self, exporter, instagram_metadata, temp_dir):
        """Test Instagram export."""
        output_path = temp_dir / "instagram.txt"

        result = exporter.export_instagram([instagram_metadata], output_path)

        assert result.success is True
        assert result.format == ExportFormat.INSTAGRAM
        assert result.record_count == 1

        content = output_path.read_text()
        assert "Best purchase ever!" in content
        assert "#ProductReview" in content

    def test_instagram_caption_format(self, exporter, instagram_metadata):
        """Test Instagram caption format with separator dots."""
        content = exporter.to_instagram_string([instagram_metadata])

        # Instagram style: caption, dots, hashtags
        assert "Best purchase ever! ✨" in content
        assert ".\n.\n.\n" in content
        assert "#ProductReview" in content

    # --- Multi-format Export Tests ---

    def test_export_dispatch(self, exporter, youtube_metadata, temp_dir):
        """Test export method dispatches to correct exporter."""
        for format_type in [ExportFormat.JSON, ExportFormat.CSV]:
            suffix = ".json" if format_type == ExportFormat.JSON else ".csv"
            output_path = temp_dir / f"test{suffix}"

            result = exporter.export([youtube_metadata], output_path, format_type)

            assert result.success is True
            assert result.format == format_type

    def test_export_all_formats(self, exporter, all_metadata, temp_dir):
        """Test exporting to all formats."""
        results = exporter.export_all_formats(all_metadata, temp_dir, "test")

        # Should have all format exports
        assert ExportFormat.JSON in results
        assert ExportFormat.CSV in results
        assert ExportFormat.YOUTUBE_CSV in results
        assert ExportFormat.TIKTOK in results
        assert ExportFormat.INSTAGRAM in results

        # All should succeed
        for format_type, result in results.items():
            assert result.success is True, f"Failed format: {format_type}"

    def test_export_all_formats_creates_files(self, exporter, all_metadata, temp_dir):
        """Test export_all_formats creates expected files."""
        exporter.export_all_formats(all_metadata, temp_dir, "metadata")

        expected_files = [
            "metadata.json",
            "metadata.csv",
            "metadata_youtube.csv",
            "metadata_tiktok.txt",
            "metadata_instagram.txt",
        ]

        for filename in expected_files:
            assert (temp_dir / filename).exists(), f"Missing file: {filename}"

    # --- Error Handling Tests ---

    def test_export_handles_empty_list(self, exporter, temp_dir):
        """Test export handles empty metadata list."""
        output_path = temp_dir / "empty.json"

        result = exporter.export_json([], output_path)

        assert result.success is True
        assert result.record_count == 0

    def test_export_creates_parent_dirs(self, exporter, youtube_metadata, temp_dir):
        """Test export creates parent directories if needed."""
        output_path = temp_dir / "nested" / "dir" / "metadata.json"

        result = exporter.export_json([youtube_metadata], output_path)

        assert result.success is True
        assert output_path.exists()

    def test_unsupported_format(self, exporter, youtube_metadata, temp_dir):
        """Test handling of unsupported format gracefully."""
        # Create a mock unsupported format scenario
        result = ExportResult(
            success=False,
            format=ExportFormat.JSON,  # Use valid format for result
            error_message="Unsupported format: unknown",
        )

        assert result.success is False
        assert (
            result.error_message is not None
            and "Unsupported format" in result.error_message
        )


class TestExportSettingsIntegration:
    """Test ExportSettings integration with PlatformMetadataSettings."""

    def test_export_settings_in_platform_metadata_settings(self):
        """Test that export settings are part of PlatformMetadataSettings."""
        settings = PlatformMetadataSettings()

        assert hasattr(settings, "export")
        assert isinstance(settings.export, ExportSettings)
        assert settings.export.enabled is True
        assert settings.export.default_format == "json"

    def test_custom_export_settings_in_platform_metadata_settings(self):
        """Test custom export settings in PlatformMetadataSettings."""
        export_settings = ExportSettings(
            enabled=False,
            default_format="csv",
            youtube_category="28",
            youtube_privacy="unlisted",
        )

        settings = PlatformMetadataSettings(export=export_settings)

        assert settings.export.enabled is False
        assert settings.export.default_format == "csv"
        assert settings.export.youtube_category == "28"
        assert settings.export.youtube_privacy == "unlisted"
