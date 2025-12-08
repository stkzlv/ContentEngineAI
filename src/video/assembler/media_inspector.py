"""Media inspection utilities for FFmpeg operations.

This module provides standalone utilities for inspecting media files using
FFprobe and FFmpeg. These utilities have zero dependencies on other assembler
components and can be used independently.
"""

import asyncio
import logging
import mimetypes
from pathlib import Path

from src.utils.async_io import async_get_media_duration
from src.utils.caching import cache_media_metadata, get_cached_media_metadata
from src.video.config import VideoConfig

logger = logging.getLogger(__name__)


class MediaInspector:
    """Standalone media file inspection using FFprobe/FFmpeg.

    This class provides utilities for inspecting media files to determine their
    properties such as type (video/image), dimensions, and duration.
    """

    def __init__(
        self, config: VideoConfig | None = None, ffprobe_path: str = "ffprobe"
    ):
        """Initialize MediaInspector.

        Args:
        ----
            config: Optional VideoConfig for configuration settings
            ffprobe_path: Path to ffprobe executable (default: "ffprobe")

        """
        self.config = config
        self.ffprobe_path = ffprobe_path

    @staticmethod
    def is_video(path: Path) -> bool:
        """Determine if a file is a video based on its MIME type.

        Args:
        ----
            path: Path to the media file

        Returns:
        -------
            True if the file is a video, False otherwise

        """
        content_type, _ = mimetypes.guess_type(path)
        return content_type is not None and content_type.startswith("video")

    async def get_media_dimensions(self, file_path: Path) -> tuple[int, int]:
        """Extract the width and height of a media file using FFprobe.

        This method uses FFprobe to analyze the media file and extract its
        dimensions. It works for both images and videos.

        Args:
        ----
            file_path: Path to the media file

        Returns:
        -------
            Tuple of (width, height) in pixels

        Raises:
        ------
            ValueError: If the dimensions cannot be extracted

        """
        # Use configurable FFprobe parameters
        streams = (
            self.config.video_processing.ffmpeg_probe_streams
            if self.config
            and hasattr(self.config, "video_processing")
            and self.config.video_processing
            else "v:0"
        )
        entries = (
            self.config.video_processing.ffmpeg_probe_entries
            if self.config
            and hasattr(self.config, "video_processing")
            and self.config.video_processing
            else "stream=width,height"
        )
        format_spec = (
            self.config.video_processing.ffmpeg_probe_format
            if self.config
            and hasattr(self.config, "video_processing")
            and self.config.video_processing
            else "csv=s=x:p=0"
        )

        cmd = [
            self.ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            streams,
            "-show_entries",
            entries,
            "-of",
            format_spec,
            str(file_path),
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.warning(
                    f"ffprobe failed to get dimensions for {file_path.name}: "
                    f"{stderr.decode()}"
                )
                return 0, 0
            w_str, h_str = stdout.decode().strip().split("x")
            return int(w_str), int(h_str)
        except Exception as e:
            logger.error(f"Error getting dimensions for {file_path.name}: {e}")
            return 0, 0

    async def get_media_duration(self, file_path: Path) -> float:
        """Get media duration in seconds using FFprobe.

        This method retrieves the duration of a media file, utilizing caching
        to improve performance for repeated queries.

        Args:
        ----
            file_path: Path to the media file

        Returns:
        -------
            Duration in seconds, or 0.0 if duration cannot be determined

        """
        # Check cache first
        cached_metadata = get_cached_media_metadata(file_path)
        if cached_metadata and "duration" in cached_metadata:
            duration_value: float = cached_metadata["duration"]
            return duration_value

        # Get duration and cache it
        timeout_sec = (
            self.config.video_settings.verification_probe_timeout_sec
            if self.config and hasattr(self.config, "video_settings")
            else 30
        )

        duration = await async_get_media_duration(
            file_path,
            self.ffprobe_path,
            timeout_sec=timeout_sec,
        )

        # Cache the metadata
        if duration > 0:
            cache_media_metadata(file_path, {"duration": duration})

        return duration
