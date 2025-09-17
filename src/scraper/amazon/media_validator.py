"""Media validation utilities for Amazon scraper.

This module provides comprehensive validation for downloaded images and videos,
ensuring they meet quality standards and are actual media files.
"""

import contextlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from .config import CONFIG

logger = logging.getLogger(__name__)


class MediaValidationResult:
    """Represents the result of media file validation."""

    def __init__(
        self,
        file_path: Path,
        is_valid: bool,
        validation_data: dict,
        issues: list = None,
    ):
        self.file_path = file_path
        self.is_valid = is_valid
        self.validation_data = validation_data
        self.issues = issues or []

    def to_dict(self) -> dict:
        """Convert validation result to dictionary."""
        return {
            "file_path": str(self.file_path),
            "is_valid": self.is_valid,
            "validation_data": self.validation_data,
            "issues": self.issues,
        }


def verify_image_file(
    file_path: Path, min_dimension: int = None, min_file_size: int = None
) -> MediaValidationResult:
    """Verify image format, dimensions, and quality.

    Args:
    ----
        file_path: Path to the image file
        min_dimension: Minimum width or height (defaults to config value)
        min_file_size: Minimum file size in bytes (defaults to config value)

    Returns:
    -------
        MediaValidationResult with detailed validation information

    """
    # Get config values if not provided
    if min_dimension is None:
        min_dimension = (
            CONFIG.get("global_settings", {})
            .get("image_config", {})
            .get("min_high_res_dimension", 1500)
        )

    if min_file_size is None:
        min_file_size = (
            CONFIG.get("global_settings", {})
            .get("image_config", {})
            .get("min_high_res_file_size", 10000)
        )

    validation_data = {
        "file_type": "image",
        "expected_min_dimension": min_dimension,
        "expected_min_file_size": min_file_size,
    }
    issues = []

    try:
        # Check if file exists
        if not file_path.exists():
            issues.append(f"File does not exist: {file_path}")
            return MediaValidationResult(file_path, False, validation_data, issues)

        # Check file size
        file_size = file_path.stat().st_size
        validation_data["actual_file_size"] = file_size

        if file_size < min_file_size:
            issues.append(
                f"File size {file_size} bytes is below minimum "
                f"{min_file_size} bytes"
            )
        elif file_size == 0:
            issues.append("File is empty (0 bytes)")
            return MediaValidationResult(file_path, False, validation_data, issues)

        # Try to open and validate image
        try:
            with Image.open(file_path) as img:
                # Get image properties
                width, height = img.size
                format_name = img.format
                mode = img.mode

                validation_data.update(
                    {
                        "width": width,
                        "height": height,
                        "format": format_name,
                        "mode": mode,
                        "max_dimension": max(width, height),
                    }
                )

                # Check dimensions
                if max(width, height) < min_dimension:
                    issues.append(
                        f"Image dimensions {width}x{height} do not meet minimum "
                        f"requirement "
                        f"of {min_dimension}px in largest dimension"
                    )

                # Check format
                if format_name not in ["JPEG", "PNG", "WEBP"]:
                    issues.append(f"Unsupported image format: {format_name}")

                # Check if image is corrupted by attempting basic operations
                try:
                    img.verify()  # Verify image integrity
                    # Re-open for further checks (verify() closes the file)
                    with Image.open(file_path) as img2:
                        # Try to get a pixel to ensure image data is readable
                        img2.getpixel((0, 0))
                        validation_data["integrity_check"] = "passed"
                except Exception as e:
                    issues.append(f"Image integrity check failed: {e}")
                    validation_data["integrity_check"] = "failed"

        except Exception as e:
            issues.append(f"Failed to open/process image: {e}")
            validation_data["image_processing_error"] = str(e)

        # Check file extension matches content
        expected_extensions = {
            "JPEG": [".jpg", ".jpeg"],
            "PNG": [".png"],
            "WEBP": [".webp"],
        }

        if "format" in validation_data:
            format_name = validation_data["format"]
            if format_name in expected_extensions and not any(
                str(file_path).lower().endswith(ext)
                for ext in expected_extensions[format_name]
            ):
                issues.append(f"File extension doesn't match format {format_name}")

    except Exception as e:
        issues.append(f"Unexpected error during validation: {e}")
        validation_data["validation_error"] = str(e)

    is_valid = len(issues) == 0
    return MediaValidationResult(file_path, is_valid, validation_data, issues)


def verify_video_file(
    file_path: Path, min_duration: float = None, min_dimension: int = None
) -> MediaValidationResult:
    """Verify video format, duration, and quality.

    Args:
    ----
        file_path: Path to the video file
        min_duration: Minimum duration in seconds (defaults to config value)
        min_dimension: Minimum width or height in pixels (defaults to config value)

    Returns:
    -------
        MediaValidationResult with detailed validation information

    """
    # Get config values if not provided
    if min_dimension is None:
        min_dimension = (
            CONFIG.get("global_settings", {})
            .get("video_config", {})
            .get("min_dimension", 640)
        )

    if min_duration is None:
        min_duration = (
            CONFIG.get("global_settings", {})
            .get("video_config", {})
            .get("min_duration", 1.0)
        )

    validation_data = {
        "file_type": "video",
        "expected_min_duration": min_duration,
        "expected_min_dimension": min_dimension,
    }
    issues = []

    try:
        # Check if file exists
        if not file_path.exists():
            issues.append(f"File does not exist: {file_path}")
            return MediaValidationResult(file_path, False, validation_data, issues)

        # Check file size
        file_size = file_path.stat().st_size
        validation_data["actual_file_size"] = file_size

        if file_size == 0:
            issues.append("File is empty (0 bytes)")
            return MediaValidationResult(file_path, False, validation_data, issues)

        # Use FFprobe to get video information
        try:
            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "stream=codec_name,codec_type,width,height,duration,bit_rate",
                "-show_entries",
                "format=duration,size,bit_rate",
                "-of",
                "json",
                str(file_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                issues.append(f"FFprobe failed with error: {result.stderr}")
                validation_data["ffprobe_error"] = result.stderr
                return MediaValidationResult(file_path, False, validation_data, issues)

            try:
                probe_data = json.loads(result.stdout)
            except json.JSONDecodeError as e:
                issues.append(f"Failed to parse FFprobe output: {e}")
                return MediaValidationResult(file_path, False, validation_data, issues)

            # Extract stream information
            streams = probe_data.get("streams", [])
            format_info = probe_data.get("format", {})

            validation_data["probe_data"] = probe_data

            # Find video and audio streams
            video_streams = [s for s in streams if s.get("codec_type") == "video"]
            audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

            validation_data["video_stream_count"] = len(video_streams)
            validation_data["audio_stream_count"] = len(audio_streams)

            if not video_streams:
                issues.append("No video streams found in file")
                return MediaValidationResult(file_path, False, validation_data, issues)

            # Check primary video stream
            video_stream = video_streams[0]

            # Get dimensions
            width = video_stream.get("width")
            height = video_stream.get("height")

            if width and height:
                validation_data["width"] = width
                validation_data["height"] = height
                validation_data["max_dimension"] = max(width, height)

                if max(width, height) < min_dimension:
                    issues.append(
                        f"Video dimensions {width}x{height} do not meet minimum "
                        f"requirement "
                        f"of {min_dimension}px in largest dimension"
                    )
            else:
                issues.append("Could not determine video dimensions")

            # Get duration
            duration = None

            # Try to get duration from stream first
            if "duration" in video_stream:
                with contextlib.suppress(ValueError, TypeError):
                    duration = float(video_stream["duration"])

            # Fall back to format duration
            if duration is None and "duration" in format_info:
                with contextlib.suppress(ValueError, TypeError):
                    duration = float(format_info["duration"])

            if duration is not None:
                validation_data["duration"] = duration

                if duration < min_duration:
                    issues.append(
                        f"Video duration {duration:.2f}s is below minimum "
                        f"{min_duration}s"
                    )
            else:
                issues.append("Could not determine video duration")

            # Check codec
            codec_name = video_stream.get("codec_name")
            validation_data["video_codec"] = codec_name

            if codec_name not in ["h264", "h265", "vp8", "vp9", "av1"]:
                issues.append(f"Unsupported video codec: {codec_name}")

            # Check for audio (optional)
            if audio_streams:
                audio_codec = audio_streams[0].get("codec_name")
                validation_data["audio_codec"] = audio_codec
            else:
                validation_data["audio_codec"] = None
                # Note: Audio is optional for product videos

        except subprocess.TimeoutExpired:
            issues.append("FFprobe timeout - file may be corrupted or very large")
        except subprocess.SubprocessError as e:
            issues.append(f"FFprobe subprocess error: {e}")
        except Exception as e:
            issues.append(f"Unexpected error during video analysis: {e}")

        # Check file extension
        valid_extensions = [".mp4", ".mov", ".webm", ".avi", ".mkv"]
        if not any(str(file_path).lower().endswith(ext) for ext in valid_extensions):
            issues.append(f"Invalid video file extension: {file_path.suffix}")

        # Additional content type check
        try:
            # Read first few bytes to check for valid video file signatures
            with open(file_path, "rb") as f:
                header = f.read(12)

            # Check for common video file signatures
            video_signatures = [
                b"\x00\x00\x00\x18ftypmp4",  # MP4
                b"\x00\x00\x00 ftypisom",  # MP4/ISO
                b"\x1aE\xdf\xa3",  # WebM/Matroska
                b"RIFF",  # AVI (first 4 bytes)
            ]

            has_valid_signature = any(
                header.startswith(sig) for sig in video_signatures
            )
            validation_data["has_valid_signature"] = has_valid_signature

            if not has_valid_signature:
                # Check if it looks like HTML or text (common issue)
                try:
                    header_text = header.decode("utf-8", errors="ignore")
                    if any(
                        tag in header_text.lower()
                        for tag in ["<html", "<!doc", "<head"]
                    ):
                        issues.append(
                            "File appears to be HTML content, not a video file"
                        )
                    elif header_text.isprintable() and len(header_text.strip()) > 0:
                        issues.append(
                            "File appears to be text content, not a video file"
                        )
                    else:
                        issues.append(
                            "File does not have a recognized video file signature"
                        )
                except Exception:
                    issues.append(
                        "File does not have a recognized video file signature"
                    )

        except Exception as e:
            issues.append(f"Could not verify file signature: {e}")

    except Exception as e:
        issues.append(f"Unexpected error during validation: {e}")
        validation_data["validation_error"] = str(e)

    is_valid = len(issues) == 0
    return MediaValidationResult(file_path, is_valid, validation_data, issues)


def validate_media_batch(
    file_paths: list[Path], media_type: str = "auto"
) -> list[MediaValidationResult]:
    """Validate a batch of media files.

    Args:
    ----
        file_paths: List of file paths to validate
        media_type: "auto", "image", or "video"

    Returns:
    -------
        List of MediaValidationResult objects

    """
    results = []

    for file_path in file_paths:
        try:
            # Auto-detect media type if not specified
            if media_type == "auto":
                if file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    detected_type = "image"
                elif file_path.suffix.lower() in [
                    ".mp4",
                    ".mov",
                    ".webm",
                    ".avi",
                    ".mkv",
                ]:
                    detected_type = "video"
                else:
                    # Try to detect by content
                    try:
                        with Image.open(file_path):
                            detected_type = "image"
                    except Exception:
                        detected_type = "video"  # Default to video for unknown types
            else:
                detected_type = media_type

            # Validate based on detected/specified type
            if detected_type == "image":
                result = verify_image_file(file_path)
            else:
                result = verify_video_file(file_path)

            results.append(result)

        except Exception as e:
            # Create error result for files that couldn't be processed
            error_result = MediaValidationResult(
                file_path,
                False,
                {"file_type": media_type, "validation_error": str(e)},
                [f"Failed to validate file: {e}"],
            )
            results.append(error_result)

    return results


def generate_validation_report(
    results: list[MediaValidationResult], output_path: Path = None
) -> dict[str, Any]:
    """Generate a comprehensive validation report.

    Args:
    ----
        results: List of validation results
        output_path: Optional path to save the report

    Returns:
    -------
        Dictionary containing the validation report

    """
    total_files = len(results)
    valid_files = sum(1 for r in results if r.is_valid)
    invalid_files = total_files - valid_files

    # Categorize by file type
    image_results = [
        r for r in results if r.validation_data.get("file_type") == "image"
    ]
    video_results = [
        r for r in results if r.validation_data.get("file_type") == "video"
    ]

    # Collect common issues
    all_issues = []
    for result in results:
        all_issues.extend(result.issues)

    issue_counts: dict[str, int] = {}
    for issue in all_issues:
        issue_counts[issue] = issue_counts.get(issue, 0) + 1

    report: dict[str, Any] = {
        "summary": {
            "total_files": total_files,
            "valid_files": valid_files,
            "invalid_files": invalid_files,
            "success_rate": (valid_files / total_files * 100) if total_files > 0 else 0,
            "image_files": len(image_results),
            "video_files": len(video_results),
        },
        "image_validation": {
            "total": len(image_results),
            "valid": sum(1 for r in image_results if r.is_valid),
            "average_dimensions": None,
            "average_file_size": None,
        },
        "video_validation": {
            "total": len(video_results),
            "valid": sum(1 for r in video_results if r.is_valid),
            "average_duration": None,
            "average_file_size": None,
        },
        "common_issues": dict(
            sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ),
        "detailed_results": [r.to_dict() for r in results],
    }

    # Calculate averages for images
    valid_images = [
        r for r in image_results if r.is_valid and "width" in r.validation_data
    ]
    if valid_images:
        avg_width = sum(r.validation_data["width"] for r in valid_images) / len(
            valid_images
        )
        avg_height = sum(r.validation_data["height"] for r in valid_images) / len(
            valid_images
        )
        report["image_validation"]["average_dimensions"] = (
            f"{avg_width:.0f}x{avg_height:.0f}"
        )

        avg_size = sum(
            r.validation_data.get("actual_file_size", 0) for r in valid_images
        ) / len(valid_images)
        report["image_validation"]["average_file_size"] = f"{avg_size / 1024:.1f}KB"

    # Calculate averages for videos
    valid_videos = [
        r for r in video_results if r.is_valid and "duration" in r.validation_data
    ]
    if valid_videos:
        avg_duration = sum(r.validation_data["duration"] for r in valid_videos) / len(
            valid_videos
        )
        report["video_validation"]["average_duration"] = f"{avg_duration:.1f}s"

        avg_size = sum(
            r.validation_data.get("actual_file_size", 0) for r in valid_videos
        ) / len(valid_videos)
        report["video_validation"]["average_file_size"] = (
            f"{avg_size / (1024*1024):.1f}MB"
        )

    # Save report if path provided
    if output_path:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Validation report saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save validation report: {e}")

    return report


def validate_url_accessibility(
    url: str, timeout: int = 10
) -> tuple[bool, dict[str, Any]]:
    """Validate that a URL is accessible and returns expected content.

    Args:
    ----
        url: URL to validate
        timeout: Request timeout in seconds

    Returns:
    -------
        Tuple of (is_accessible, metadata)

    """
    metadata = {"url": url}

    try:
        # Get headers from config
        headers = (
            CONFIG.get("scrapers", {})
            .get("amazon", {})
            .get("http_headers", {})
            .get(
                "media_download",
                {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
                    ),
                    "Accept": "*/*",
                },
            )
        )

        # Make HEAD request to check accessibility
        response = requests.head(
            url, timeout=timeout, allow_redirects=True, headers=headers
        )

        metadata.update(
            {
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type", "unknown"),
                "content_length": response.headers.get("content-length", "unknown"),
                "final_url": str(response.url),
                "redirected": response.url != url,
            }
        )

        is_accessible = response.status_code < 400

        # Additional checks for content type
        content_type = response.headers.get("content-type", "").lower()
        if "text/html" in content_type:
            metadata["warning"] = "URL returns HTML content (may not be direct media)"
        elif "application/json" in content_type:
            metadata["warning"] = "URL returns JSON content (may not be direct media)"

        return is_accessible, metadata

    except requests.exceptions.Timeout:
        metadata["error"] = "Request timeout"
        return False, metadata
    except requests.exceptions.RequestException as e:
        metadata["error"] = str(e)
        return False, metadata
    except Exception as e:
        metadata["error"] = f"Unexpected error: {e}"
        return False, metadata
