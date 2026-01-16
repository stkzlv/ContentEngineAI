"""Metadata export functionality for multiple formats.

This module provides export capabilities for platform metadata including JSON,
CSV for spreadsheet analysis, and platform-specific formats optimized for
YouTube Studio and TikTok bulk uploads.
"""

import csv
import io
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from src.ai.platform_metadata.models import PlatformMetadata

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Supported export formats for metadata."""

    JSON = "json"
    CSV = "csv"
    YOUTUBE_CSV = "youtube_csv"
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"


@dataclass
class ExportResult:
    """Result of a metadata export operation.

    Attributes
    ----------
        success: Whether export completed successfully
        format: Export format used
        file_path: Path to exported file (if saved to disk)
        content: Exported content as string (if not saved to disk)
        record_count: Number of records exported
        error_message: Error message if export failed

    """

    success: bool
    format: ExportFormat
    file_path: Path | None = None
    content: str | None = None
    record_count: int = 0
    error_message: str | None = None

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "format": self.format.value,
            "file_path": str(self.file_path) if self.file_path else None,
            "record_count": self.record_count,
            "error_message": self.error_message,
        }


class MetadataExporter:
    """Exports platform metadata to various formats.

    Supports exporting to:
    - JSON: Full metadata with all fields (default)
    - CSV: Tabular format for spreadsheet analysis
    - YouTube CSV: YouTube Studio compatible format for bulk uploads
    - TikTok: Caption + hashtags format for TikTok posting
    - Instagram: Caption with hashtags optimized for Instagram

    Example usage:
        exporter = MetadataExporter()

        # Export to JSON
        result = exporter.export_json(metadata_list, Path("output.json"))

        # Export to CSV
        result = exporter.export_csv(metadata_list, Path("output.csv"))

        # Export to YouTube format
        result = exporter.export_youtube_csv(metadata_list, Path("youtube.csv"))

        # Get content as string instead of file
        content = exporter.to_json_string(metadata_list)

    """

    # YouTube CSV column headers
    YOUTUBE_COLUMNS = [
        "product_id",
        "title",
        "description",
        "tags",
        "category",
        "privacy",
    ]

    # Generic CSV column headers
    CSV_COLUMNS = [
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

    def __init__(
        self,
        default_category: str = "22",
        default_privacy: str = "private",
        csv_encoding: str = "utf-8-sig",
        json_indent: int = 2,
        youtube_title_fallback_length: int = 60,
    ):
        """Initialize exporter.

        Args:
        ----
            default_category: Default YouTube category ID (22 = People & Blogs)
            default_privacy: Default privacy setting for YouTube exports
            csv_encoding: CSV encoding (utf-8-sig for Excel, utf-8 for others)
            json_indent: JSON indentation spaces (0 for compact)
            youtube_title_fallback_length: Max length when using description as title

        """
        self.default_category = default_category
        self.default_privacy = default_privacy
        self.csv_encoding = csv_encoding
        self.json_indent = json_indent
        self.youtube_title_fallback_length = youtube_title_fallback_length

    # --- JSON Export ---

    def export_json(
        self,
        metadata_list: list[PlatformMetadata],
        output_path: Path,
        indent: int | None = None,
    ) -> ExportResult:
        """Export metadata to JSON file.

        Args:
        ----
            metadata_list: List of PlatformMetadata objects to export
            output_path: Path to output JSON file
            indent: JSON indentation level (None uses instance default)

        Returns:
        -------
            ExportResult with success status and file path

        """
        try:
            effective_indent = indent if indent is not None else self.json_indent
            content = self.to_json_string(metadata_list, effective_indent)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding="utf-8")

            logger.info(f"Exported {len(metadata_list)} records to {output_path}")
            return ExportResult(
                success=True,
                format=ExportFormat.JSON,
                file_path=output_path,
                record_count=len(metadata_list),
            )
        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            return ExportResult(
                success=False,
                format=ExportFormat.JSON,
                error_message=str(e),
            )

    def to_json_string(
        self,
        metadata_list: list[PlatformMetadata],
        indent: int = 2,
    ) -> str:
        """Convert metadata to JSON string.

        Args:
        ----
            metadata_list: List of PlatformMetadata objects
            indent: JSON indentation level

        Returns:
        -------
            JSON string representation

        """
        data = [m.to_dict() for m in metadata_list]
        return json.dumps(data, indent=indent, ensure_ascii=False)

    # --- CSV Export ---

    def export_csv(
        self,
        metadata_list: list[PlatformMetadata],
        output_path: Path,
    ) -> ExportResult:
        """Export metadata to CSV file for spreadsheet analysis.

        Args:
        ----
            metadata_list: List of PlatformMetadata objects to export
            output_path: Path to output CSV file

        Returns:
        -------
            ExportResult with success status and file path

        """
        try:
            content = self.to_csv_string(metadata_list)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding=self.csv_encoding)

            logger.info(f"Exported {len(metadata_list)} records to CSV: {output_path}")
            return ExportResult(
                success=True,
                format=ExportFormat.CSV,
                file_path=output_path,
                record_count=len(metadata_list),
            )
        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            return ExportResult(
                success=False,
                format=ExportFormat.CSV,
                error_message=str(e),
            )

    def to_csv_string(self, metadata_list: list[PlatformMetadata]) -> str:
        """Convert metadata to CSV string.

        Args:
        ----
            metadata_list: List of PlatformMetadata objects

        Returns:
        -------
            CSV string with headers and data rows

        """
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=self.CSV_COLUMNS,
            quoting=csv.QUOTE_ALL,
            extrasaction="ignore",
        )
        writer.writeheader()

        for metadata in metadata_list:
            row = self._metadata_to_csv_row(metadata)
            writer.writerow(row)

        return output.getvalue()

    def _metadata_to_csv_row(self, metadata: PlatformMetadata) -> dict[str, str]:
        """Convert single metadata to CSV row dictionary.

        Args:
        ----
            metadata: PlatformMetadata object

        Returns:
        -------
            Dictionary with CSV column values

        """
        return {
            "product_id": metadata.product_id,
            "platform": metadata.platform,
            "title": metadata.title or "",
            "description": metadata.description,
            "hashtags": " ".join(metadata.hashtags),
            "keywords": ", ".join(metadata.keywords),
            "validation_status": metadata.validation_status,
            "generated_at": metadata.generated_at,
            "prompt_variant": metadata.prompt_variant or "",
        }

    # --- YouTube CSV Export ---

    def export_youtube_csv(
        self,
        metadata_list: list[PlatformMetadata],
        output_path: Path,
    ) -> ExportResult:
        """Export metadata to YouTube Studio compatible CSV format.

        YouTube Studio supports CSV imports with columns:
        - Title, Description, Tags (comma-separated), Category, Privacy

        Args:
        ----
            metadata_list: List of PlatformMetadata objects (YouTube platform)
            output_path: Path to output CSV file

        Returns:
        -------
            ExportResult with success status and file path

        Note:
        ----
            Non-YouTube metadata will be filtered out with a warning.

        """
        try:
            youtube_metadata = [m for m in metadata_list if m.platform == "youtube"]
            if len(youtube_metadata) < len(metadata_list):
                skipped = len(metadata_list) - len(youtube_metadata)
                logger.warning(
                    f"Skipped {skipped} non-YouTube records in YouTube CSV export"
                )

            content = self.to_youtube_csv_string(youtube_metadata)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding=self.csv_encoding)

            logger.info(
                f"Exported {len(youtube_metadata)} records to YouTube CSV: "
                f"{output_path}"
            )
            return ExportResult(
                success=True,
                format=ExportFormat.YOUTUBE_CSV,
                file_path=output_path,
                record_count=len(youtube_metadata),
            )
        except Exception as e:
            logger.error(f"YouTube CSV export failed: {e}")
            return ExportResult(
                success=False,
                format=ExportFormat.YOUTUBE_CSV,
                error_message=str(e),
            )

    def to_youtube_csv_string(self, metadata_list: list[PlatformMetadata]) -> str:
        """Convert metadata to YouTube Studio compatible CSV string.

        Args:
        ----
            metadata_list: List of YouTube PlatformMetadata objects

        Returns:
        -------
            CSV string formatted for YouTube Studio import

        """
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=self.YOUTUBE_COLUMNS,
            quoting=csv.QUOTE_ALL,
            extrasaction="ignore",
        )
        writer.writeheader()

        for metadata in metadata_list:
            row = self._metadata_to_youtube_row(metadata)
            writer.writerow(row)

        return output.getvalue()

    def _metadata_to_youtube_row(self, metadata: PlatformMetadata) -> dict[str, str]:
        """Convert single metadata to YouTube CSV row.

        Args:
        ----
            metadata: YouTube PlatformMetadata object

        Returns:
        -------
            Dictionary with YouTube CSV column values

        """
        # Convert hashtags to comma-separated tags (remove # prefix)
        tags = [h.lstrip("#") for h in metadata.hashtags]
        # Add keywords as additional tags
        tags.extend(metadata.keywords)

        # Use title or truncate description as fallback
        fallback_len = self.youtube_title_fallback_length
        title = metadata.title or metadata.description[:fallback_len]

        return {
            "product_id": metadata.product_id,
            "title": title,
            "description": metadata.description,
            "tags": ",".join(tags),
            "category": self.default_category,
            "privacy": self.default_privacy,
        }

    # --- TikTok Export ---

    def export_tiktok(
        self,
        metadata_list: list[PlatformMetadata],
        output_path: Path,
    ) -> ExportResult:
        """Export metadata to TikTok-optimized format.

        TikTok format includes caption with hashtags appended, one per line.
        Each entry is separated by a blank line for easy copy-paste.

        Args:
        ----
            metadata_list: List of PlatformMetadata objects (TikTok platform)
            output_path: Path to output text file

        Returns:
        -------
            ExportResult with success status and file path

        """
        try:
            tiktok_metadata = [m for m in metadata_list if m.platform == "tiktok"]
            if len(tiktok_metadata) < len(metadata_list):
                skipped = len(metadata_list) - len(tiktok_metadata)
                logger.warning(f"Skipped {skipped} non-TikTok records in TikTok export")

            content = self.to_tiktok_string(tiktok_metadata)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding="utf-8")

            logger.info(
                f"Exported {len(tiktok_metadata)} records to TikTok format: "
                f"{output_path}"
            )
            return ExportResult(
                success=True,
                format=ExportFormat.TIKTOK,
                file_path=output_path,
                record_count=len(tiktok_metadata),
            )
        except Exception as e:
            logger.error(f"TikTok export failed: {e}")
            return ExportResult(
                success=False,
                format=ExportFormat.TIKTOK,
                error_message=str(e),
            )

    def to_tiktok_string(self, metadata_list: list[PlatformMetadata]) -> str:
        """Convert metadata to TikTok-ready caption format.

        Args:
        ----
            metadata_list: List of TikTok PlatformMetadata objects

        Returns:
        -------
            Text with captions and hashtags, entries separated by blank lines

        """
        entries = []
        for metadata in metadata_list:
            caption = self._format_tiktok_caption(metadata)
            entry = f"--- {metadata.product_id} ---\n{caption}"
            entries.append(entry)

        return "\n\n".join(entries)

    def _format_tiktok_caption(self, metadata: PlatformMetadata) -> str:
        """Format single TikTok caption with hashtags.

        Args:
        ----
            metadata: TikTok PlatformMetadata object

        Returns:
        -------
            Caption text with hashtags appended

        """
        hashtags_str = " ".join(metadata.hashtags)
        return f"{metadata.description}\n\n{hashtags_str}"

    # --- Instagram Export ---

    def export_instagram(
        self,
        metadata_list: list[PlatformMetadata],
        output_path: Path,
    ) -> ExportResult:
        """Export metadata to Instagram-optimized format.

        Instagram format includes caption with hashtags, optimized for posting.
        Hashtags can be in caption or as a separate block based on settings.

        Args:
        ----
            metadata_list: List of PlatformMetadata objects (Instagram platform)
            output_path: Path to output text file

        Returns:
        -------
            ExportResult with success status and file path

        """
        try:
            ig_metadata = [m for m in metadata_list if m.platform == "instagram"]
            if len(ig_metadata) < len(metadata_list):
                skipped = len(metadata_list) - len(ig_metadata)
                logger.warning(
                    f"Skipped {skipped} non-Instagram records in Instagram export"
                )

            content = self.to_instagram_string(ig_metadata)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding="utf-8")

            logger.info(
                f"Exported {len(ig_metadata)} records to Instagram format: "
                f"{output_path}"
            )
            return ExportResult(
                success=True,
                format=ExportFormat.INSTAGRAM,
                file_path=output_path,
                record_count=len(ig_metadata),
            )
        except Exception as e:
            logger.error(f"Instagram export failed: {e}")
            return ExportResult(
                success=False,
                format=ExportFormat.INSTAGRAM,
                error_message=str(e),
            )

    def to_instagram_string(self, metadata_list: list[PlatformMetadata]) -> str:
        """Convert metadata to Instagram-ready caption format.

        Args:
        ----
            metadata_list: List of Instagram PlatformMetadata objects

        Returns:
        -------
            Text with captions and hashtags, entries separated by blank lines

        """
        entries = []
        for metadata in metadata_list:
            caption = self._format_instagram_caption(metadata)
            entry = f"--- {metadata.product_id} ---\n{caption}"
            entries.append(entry)

        return "\n\n".join(entries)

    def _format_instagram_caption(self, metadata: PlatformMetadata) -> str:
        """Format single Instagram caption with hashtags.

        Instagram best practice: Hashtags in caption (not first comment).

        Args:
        ----
            metadata: Instagram PlatformMetadata object

        Returns:
        -------
            Caption text with hashtags in a separate block

        """
        hashtags_str = " ".join(metadata.hashtags)
        # Use line breaks to separate caption from hashtags (Instagram style)
        return f"{metadata.description}\n.\n.\n.\n{hashtags_str}"

    # --- Multi-format Export ---

    def export(
        self,
        metadata_list: list[PlatformMetadata],
        output_path: Path,
        format: ExportFormat = ExportFormat.JSON,
    ) -> ExportResult:
        """Export metadata in specified format.

        Convenience method that dispatches to format-specific exporters.

        Args:
        ----
            metadata_list: List of PlatformMetadata objects
            output_path: Path to output file
            format: Export format (default: JSON)

        Returns:
        -------
            ExportResult with success status

        """
        exporters: dict[
            ExportFormat,
            Callable[[list[PlatformMetadata], Path], ExportResult],
        ] = {
            ExportFormat.JSON: self.export_json,
            ExportFormat.CSV: self.export_csv,
            ExportFormat.YOUTUBE_CSV: self.export_youtube_csv,
            ExportFormat.TIKTOK: self.export_tiktok,
            ExportFormat.INSTAGRAM: self.export_instagram,
        }

        exporter = exporters.get(format)
        if not exporter:
            return ExportResult(
                success=False,
                format=format,
                error_message=f"Unsupported format: {format}",
            )

        return exporter(metadata_list, output_path)

    def export_all_formats(
        self,
        metadata_list: list[PlatformMetadata],
        output_dir: Path,
        base_name: str = "metadata",
    ) -> dict[ExportFormat, ExportResult]:
        """Export metadata to all supported formats.

        Creates one file per format in the specified directory.

        Args:
        ----
            metadata_list: List of PlatformMetadata objects
            output_dir: Directory to save exported files
            base_name: Base filename without extension

        Returns:
        -------
            Dictionary mapping format to export result

        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {}

        # JSON export (all platforms)
        results[ExportFormat.JSON] = self.export_json(
            metadata_list, output_dir / f"{base_name}.json"
        )

        # CSV export (all platforms)
        results[ExportFormat.CSV] = self.export_csv(
            metadata_list, output_dir / f"{base_name}.csv"
        )

        # Platform-specific exports
        youtube_data = [m for m in metadata_list if m.platform == "youtube"]
        if youtube_data:
            results[ExportFormat.YOUTUBE_CSV] = self.export_youtube_csv(
                youtube_data, output_dir / f"{base_name}_youtube.csv"
            )

        tiktok_data = [m for m in metadata_list if m.platform == "tiktok"]
        if tiktok_data:
            results[ExportFormat.TIKTOK] = self.export_tiktok(
                tiktok_data, output_dir / f"{base_name}_tiktok.txt"
            )

        instagram_data = [m for m in metadata_list if m.platform == "instagram"]
        if instagram_data:
            results[ExportFormat.INSTAGRAM] = self.export_instagram(
                instagram_data, output_dir / f"{base_name}_instagram.txt"
            )

        success_count = sum(1 for r in results.values() if r.success)
        logger.info(
            f"Exported to {success_count}/{len(results)} formats in {output_dir}"
        )

        return results
