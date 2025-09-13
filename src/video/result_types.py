"""Unified result types for ContentEngineAI video module.

This module provides standardized result objects that replace
the inconsistent return patterns across subtitle and video generation functions.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class OperationResult:
    """Base result type for all video module operations."""

    success: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_error(self, error: str) -> None:
        """Add an error message to the result."""
        self.errors.append(error)
        self.success = False

    def add_warning(self, warning: str) -> None:
        """Add a warning message to the result."""
        self.warnings.append(warning)

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the result."""
        self.metadata[key] = value

    @property
    def has_errors(self) -> bool:
        """Check if the result has any errors."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if the result has any warnings."""
        return len(self.warnings) > 0


@dataclass
class FileOperationResult(OperationResult):
    """Result type for operations that produce files."""

    path: Path | None = None
    file_size_bytes: int = 0

    def set_file(self, path: Path) -> None:
        """Set the result file path and update size."""
        self.path = path
        if path and path.exists():
            self.file_size_bytes = path.stat().st_size
            self.success = True
        else:
            self.add_error(f"Result file not found: {path}")


@dataclass
class SubtitleResult(FileOperationResult):
    """Result type for subtitle generation operations."""

    format: str = "srt"  # 'srt' or 'ass'
    segments_created: int = 0
    generation_method: str = ""
    timing_source: str = ""  # 'whisper', 'google_stt', 'script', etc.
    duration_ms: int = 0

    def set_segments(self, count: int, method: str = "", source: str = "") -> None:
        """Set segment information."""
        self.segments_created = count
        self.generation_method = method
        self.timing_source = source
        self.add_metadata("segments", count)


@dataclass
class VideoResult(FileOperationResult):
    """Result type for video assembly operations."""

    format: str = "mp4"
    resolution: tuple[int, int] = (0, 0)
    duration_sec: float = 0.0
    frame_rate: int = 0
    has_subtitles: bool = False
    has_audio: bool = False
    visual_count: int = 0

    def set_video_info(
        self,
        resolution: tuple[int, int],
        duration: float,
        frame_rate: int = 30,
        subtitle_path: Path | None = None,
        audio_path: Path | None = None,
        visual_count: int = 0,
    ) -> None:
        """Set video information."""
        self.resolution = resolution
        self.duration_sec = duration
        self.frame_rate = frame_rate
        self.has_subtitles = subtitle_path is not None and subtitle_path.exists()
        self.has_audio = audio_path is not None and audio_path.exists()
        self.visual_count = visual_count

        # Add to metadata
        self.add_metadata("resolution", resolution)
        self.add_metadata("duration_sec", duration)
        self.add_metadata("frame_rate", frame_rate)


@dataclass
class AudioResult(FileOperationResult):
    """Result type for audio processing operations."""

    format: str = "wav"
    duration_sec: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    bitrate_kbps: int = 0

    def set_audio_info(
        self,
        duration: float,
        sample_rate: int = 44100,
        channels: int = 2,
        bitrate: int = 128,
    ) -> None:
        """Set audio information."""
        self.duration_sec = duration
        self.sample_rate = sample_rate
        self.channels = channels
        self.bitrate_kbps = bitrate

        # Add to metadata
        self.add_metadata("duration_sec", duration)
        self.add_metadata("sample_rate", sample_rate)
        self.add_metadata("channels", channels)


@dataclass
class MultiStepResult(OperationResult):
    """Result type for operations with multiple steps."""

    steps_completed: list[str] = field(default_factory=list)
    steps_failed: list[str] = field(default_factory=list)
    current_step: str = ""
    total_steps: int = 0

    def start_step(self, step_name: str) -> None:
        """Mark the start of a processing step."""
        self.current_step = step_name

    def complete_step(self, step_name: str | None = None) -> None:
        """Mark completion of a processing step."""
        step = step_name or self.current_step
        if step and step not in self.steps_completed:
            self.steps_completed.append(step)
        self.current_step = ""

    def fail_step(self, step_name: str | None = None, error: str = "") -> None:
        """Mark failure of a processing step."""
        step = step_name or self.current_step
        if step and step not in self.steps_failed:
            self.steps_failed.append(step)
        if error:
            self.add_error(f"Step '{step}' failed: {error}")
        self.success = False
        self.current_step = ""

    @property
    def progress_percent(self) -> float:
        """Get completion percentage."""
        if self.total_steps == 0:
            return 0.0
        return (len(self.steps_completed) / self.total_steps) * 100.0


@dataclass
class PipelineResult(MultiStepResult):
    """Result type for complete video production pipeline."""

    product_id: str = ""
    output_files: dict[str, Path] = field(default_factory=dict)
    processing_time_sec: float = 0.0

    def add_output_file(self, file_type: str, path: Path) -> None:
        """Add an output file to the result."""
        self.output_files[file_type] = path
        if path.exists():
            self.add_metadata(f"{file_type}_size_bytes", path.stat().st_size)

    def get_output_file(self, file_type: str) -> Path | None:
        """Get an output file by type."""
        return self.output_files.get(file_type)

    @property
    def video_path(self) -> Path | None:
        """Get the main video output file."""
        return self.get_output_file("video")

    @property
    def subtitle_path(self) -> Path | None:
        """Get the subtitle output file."""
        return self.get_output_file("subtitles")

    @property
    def script_path(self) -> Path | None:
        """Get the script output file."""
        return self.get_output_file("script")


# Utility functions for backward compatibility


def create_legacy_subtitle_result(
    success: bool,
    path: Path | None,
    method_used: str = "",
    error_chain: list[str] | None = None,
    timing_count: int = 0,
) -> SubtitleResult:
    """Create a SubtitleResult from legacy function parameters.

    This helps convert existing functions to use the new result types
    while maintaining backward compatibility.
    """
    result = SubtitleResult(
        success=success,
        path=path,
        generation_method=method_used,
        segments_created=timing_count,
    )

    if path:
        result.set_file(path)

    if error_chain:
        result.errors.extend(error_chain)

    return result


def extract_legacy_values(
    result: OperationResult,
) -> tuple[bool, Path | None, list[str]]:
    """Extract values needed for legacy return patterns.

    This helps transition existing code that expects (bool, Path, list[str]) tuples.
    """
    path = getattr(result, "path", None)
    return result.success, path, result.errors
