"""Video assembly mode strategies using Strategy Pattern.

This module provides different strategies for assembling videos and images
into a final product video, each with different logic for handling timing
and sequencing.
"""

import asyncio
import logging
from pathlib import Path
from typing import Protocol

from src.video.assembler.media_inspector import MediaInspector
from src.video.config import VideoConfig

logger = logging.getLogger(__name__)


class VideoModeStrategy(Protocol):
    """Protocol for video assembly mode strategies."""

    async def assemble(
        self,
        video_files: list[Path],
        image_files: list[Path],
        target_duration: float,
    ) -> tuple[list[tuple[Path, float, bool]], str]:
        """Build visual timeline for this video mode.

        Args:
        ----
            video_files: List of product video file paths
            image_files: List of product image file paths
            target_duration: Target duration in seconds to match

        Returns:
        -------
            Tuple of (timed_visuals, mode_info) where:
            - timed_visuals: List of (Path, duration, is_video) tuples
            - mode_info: String describing the assembly mode used

        """
        ...


class SequentialStrategy:
    """Sequential mode: concatenate all videos/images in order."""

    def __init__(
        self,
        media_inspector: MediaInspector,
        config: VideoConfig,
        product_id: str,
    ):
        """Initialize SequentialStrategy.

        Args:
        ----
            media_inspector: MediaInspector instance for media operations
            config: VideoConfig containing settings
            product_id: Product identifier for logging

        """
        self.inspector = media_inspector
        self.config = config
        self.product_id = product_id

    async def assemble(
        self,
        video_files: list[Path],
        image_files: list[Path],
        target_duration: float,
    ) -> tuple[list[tuple[Path, float, bool]], str]:
        """Assemble videos sequentially with duration matching.

        Concatenates all product videos end-to-end with crossfade transitions.
        Handles insufficient/excessive duration by looping videos, adding images,
        or trimming with fade-out.

        Args:
        ----
            video_files: List of product video file paths
            image_files: List of product image file paths
            target_duration: Target duration in seconds to match

        Returns:
        -------
            Tuple of (timed_visuals, mode_info) where:
            - timed_visuals: List of (Path, duration, is_video) tuples
            - mode_info: String describing the assembly mode used

        """
        timed_visuals: list[tuple[Path, float, bool]] = []
        video_settings = self.config.video_settings
        tolerance = video_settings.video_duration_tolerance_sec

        # Edge case: No videos at all - use images only
        if not video_files:
            if image_files:
                image_duration = target_duration / len(image_files)
                for img in image_files:
                    timed_visuals.append((img, image_duration, False))
                mode_info = f"sequential (no videos, {len(image_files)} images)"
                return timed_visuals, mode_info
            else:
                mode_info = "sequential (no media available)"
                return timed_visuals, mode_info

        # Get durations of all videos
        video_durations = await asyncio.gather(
            *[self.inspector.get_media_duration(vf) for vf in video_files]
        )

        # Edge case: Single video
        if len(video_files) == 1:
            video_duration = video_durations[0]
            duration_diff = target_duration - video_duration

            if abs(duration_diff) <= tolerance:
                # Perfect match - use as-is
                timed_visuals.append((video_files[0], video_duration, True))
                mode_info = "sequential (1 video, perfect match)"
            elif duration_diff > tolerance:
                # Need more duration - loop video or add images
                loops_needed = int(
                    (target_duration + video_duration - 1) / video_duration
                )
                current_duration = 0.0

                for _ in range(loops_needed):
                    remaining = target_duration - current_duration
                    if remaining >= video_duration:
                        timed_visuals.append((video_files[0], video_duration, True))
                        current_duration += video_duration
                    else:
                        # Partial loop for last iteration
                        timed_visuals.append((video_files[0], remaining, True))
                        current_duration += remaining
                        break

                # Fill remaining with images if needed
                if current_duration < target_duration - tolerance and image_files:
                    remaining_duration = target_duration - current_duration
                    image_duration = remaining_duration / len(image_files)
                    for img in image_files:
                        timed_visuals.append((img, image_duration, False))

                mode_info = f"sequential (1 video looped {loops_needed}x)"
            else:
                # Too long - trim video
                timed_visuals.append((video_files[0], target_duration, True))
                mode_info = "sequential (1 video trimmed)"

            return timed_visuals, mode_info

        # Multiple videos: concatenate in order
        total_video_duration = sum(video_durations)

        # Add all videos in sequence
        for video_path, duration in zip(video_files, video_durations, strict=False):
            timed_visuals.append((video_path, duration, True))

        duration_diff = target_duration - total_video_duration

        # Check if duration is within tolerance
        if abs(duration_diff) <= tolerance:
            # Perfect match
            mode_info = f"sequential ({len(video_files)} videos, perfect match)"
        elif duration_diff > tolerance:
            # Insufficient duration - loop or add images
            remaining_duration = duration_diff

            # Try looping the last video
            if video_files:
                last_video = video_files[-1]
                last_duration = video_durations[-1]

                while remaining_duration > tolerance:
                    if remaining_duration >= last_duration:
                        timed_visuals.append((last_video, last_duration, True))
                        remaining_duration -= last_duration
                    else:
                        # Partial loop
                        timed_visuals.append((last_video, remaining_duration, True))
                        remaining_duration = 0
                        break

            # If still short and have images, add them
            if remaining_duration > tolerance and image_files:
                image_duration = remaining_duration / len(image_files)
                for img in image_files:
                    timed_visuals.append((img, image_duration, False))
                mode_info = f"sequential ({len(video_files)} videos + images)"
            else:
                mode_info = f"sequential ({len(video_files)} videos looped)"
        else:
            # Excessive duration - trim last video
            excess = abs(duration_diff)
            last_idx = len(timed_visuals) - 1

            if last_idx >= 0:
                last_path, last_duration, last_is_video = timed_visuals[last_idx]
                min_duration = self.config.video_settings.min_trimmed_video_duration
                new_duration = max(min_duration, last_duration - excess)
                timed_visuals[last_idx] = (last_path, new_duration, last_is_video)

            mode_info = f"sequential ({len(video_files)} videos, last trimmed)"

        return timed_visuals, mode_info


class SingleBestStrategy:
    """Single best mode: use longest/highest quality video."""

    def __init__(
        self,
        media_inspector: MediaInspector,
        config: VideoConfig,
        product_id: str,
    ):
        """Initialize SingleBestStrategy.

        Args:
        ----
            media_inspector: MediaInspector instance for media operations
            config: VideoConfig containing settings
            product_id: Product identifier for logging

        """
        self.inspector = media_inspector
        self.config = config
        self.product_id = product_id

    async def assemble(
        self,
        video_files: list[Path],
        image_files: list[Path],
        target_duration: float,
    ) -> tuple[list[tuple[Path, float, bool]], str]:
        """Select the longest video and loop/trim to match target duration.

        This strategy finds the longest available video and uses it exclusively,
        looping or trimming as needed. Falls back to images if no videos exist.

        Args:
        ----
            video_files: List of product video file paths
            image_files: List of product image file paths
            target_duration: Target duration in seconds to match

        Returns:
        -------
            Tuple of (timed_visuals, mode_info)

        """
        timed_visuals: list[tuple[Path, float, bool]] = []
        video_settings = self.config.video_settings
        tolerance = video_settings.video_duration_tolerance_sec

        # Edge case: No videos - fallback to sequential images
        if not video_files:
            if image_files:
                img_duration = target_duration / len(image_files)
                for img in image_files:
                    timed_visuals.append((img, img_duration, False))
                mode_info = f"single_best (no videos, {len(image_files)} images)"
                return timed_visuals, mode_info
            else:
                mode_info = "single_best (no media available)"
                return timed_visuals, mode_info

        # Find longest video
        video_durations = await asyncio.gather(
            *[self.inspector.get_media_duration(vf) for vf in video_files]
        )
        longest_idx = video_durations.index(max(video_durations))
        best_video = video_files[longest_idx]
        best_duration = video_durations[longest_idx]

        # Case 1: Video matches target within tolerance
        if abs(best_duration - target_duration) <= tolerance:
            timed_visuals.append((best_video, best_duration, True))
            mode_info = f"single_best (1 video, {best_duration:.1f}s)"
            return timed_visuals, mode_info

        # Case 2: Video exceeds target - trim it
        if best_duration > target_duration:
            timed_visuals.append((best_video, target_duration, True))
            mode_info = (
                f"single_best (1 video trimmed from {best_duration:.1f}s to "
                f"{target_duration:.1f}s)"
            )
            return timed_visuals, mode_info

        # Case 3: Video shorter than target - loop it
        total_duration = 0.0
        while total_duration < target_duration - tolerance:
            remaining = target_duration - total_duration
            if remaining >= best_duration:
                # Full video fits
                timed_visuals.append((best_video, best_duration, True))
                total_duration += best_duration
            else:
                # Partial video to finish
                timed_visuals.append((best_video, remaining, True))
                total_duration += remaining

        loop_count = len(timed_visuals)
        mode_info = f"single_best (1 video looped {loop_count}x, {total_duration:.1f}s)"
        return timed_visuals, mode_info


class MixedMediaStrategy:
    """Mixed mode: interleave videos and images."""

    def __init__(
        self,
        media_inspector: MediaInspector,
        config: VideoConfig,
        product_id: str,
    ):
        """Initialize MixedMediaStrategy.

        Args:
        ----
            media_inspector: MediaInspector instance for media operations
            config: VideoConfig containing settings
            product_id: Product identifier for logging

        """
        self.inspector = media_inspector
        self.config = config
        self.product_id = product_id

    async def assemble(
        self,
        video_files: list[Path],
        image_files: list[Path],
        target_duration: float,
    ) -> tuple[list[tuple[Path, float, bool]], str]:
        """Assemble videos and images interleaved across timeline.

        Distributes videos evenly across the timeline at calculated intervals,
        filling gaps with images. This creates visual variety by alternating
        between video content and static images.

        Args:
        ----
            video_files: List of product video file paths
            image_files: List of product image file paths
            target_duration: Target duration in seconds to match

        Returns:
        -------
            Tuple of (timed_visuals, mode_info)

        """
        timed_visuals: list[tuple[Path, float, bool]] = []
        video_settings = self.config.video_settings
        tolerance = video_settings.video_duration_tolerance_sec

        # Edge case: No videos - fallback to sequential images
        if not video_files:
            if image_files:
                img_duration = target_duration / len(image_files)
                for img in image_files:
                    timed_visuals.append((img, img_duration, False))
                mode_info = f"mixed_media (no videos, {len(image_files)} images)"
                return timed_visuals, mode_info
            else:
                mode_info = "mixed_media (no media available)"
                return timed_visuals, mode_info

        # Edge case: No images - fallback to sequential videos
        if not image_files:
            # Get video durations
            video_durations = await asyncio.gather(
                *[self.inspector.get_media_duration(vf) for vf in video_files]
            )
            total_video_duration = sum(video_durations)

            # Add all videos
            for video_path, duration in zip(video_files, video_durations, strict=False):
                timed_visuals.append((video_path, duration, True))

            # Handle duration mismatch
            if total_video_duration < target_duration - tolerance:
                # Loop last video
                last_video = video_files[-1]
                last_duration = video_durations[-1]
                remaining = target_duration - total_video_duration

                while remaining > tolerance:
                    if remaining >= last_duration:
                        timed_visuals.append((last_video, last_duration, True))
                        remaining -= last_duration
                    else:
                        timed_visuals.append((last_video, remaining, True))
                        break

            mode_info = f"mixed_media (no images, {len(video_files)} videos)"
            return timed_visuals, mode_info

        # Mixed mode: Interleave videos and images
        # Get video durations
        video_durations = await asyncio.gather(
            *[self.inspector.get_media_duration(vf) for vf in video_files]
        )
        total_video_duration = sum(video_durations)

        # Calculate how much time is available for images
        remaining_for_images = target_duration - total_video_duration

        if remaining_for_images <= 0:
            # Videos already exceed target - just use videos sequentially
            for video_path, duration in zip(video_files, video_durations, strict=False):
                timed_visuals.append((video_path, duration, True))
            mode_info = f"mixed_media ({len(video_files)} videos, no space for images)"
            return timed_visuals, mode_info

        # Distribute videos evenly across timeline
        # Create slots: image, video, image, video, ..., image
        num_videos = len(video_files)
        num_image_slots = num_videos + 1

        # Calculate image duration for each slot
        if num_image_slots > len(image_files):
            # Not enough images to fill all slots - use available images
            img_per_slot = len(image_files) // num_image_slots
            if img_per_slot == 0:
                # Very few images - distribute them across slots
                img_duration = remaining_for_images / len(image_files)
            else:
                img_duration = remaining_for_images / len(image_files)
        else:
            # More images than slots - use subset and distribute evenly
            img_duration = remaining_for_images / len(image_files)

        # Build interleaved timeline
        image_idx = 0
        images_per_slot = max(1, len(image_files) // num_image_slots)
        remaining_images = len(image_files)

        # Add images before first video
        if image_idx < len(image_files):
            for _ in range(min(images_per_slot, remaining_images)):
                if image_idx < len(image_files):
                    timed_visuals.append((image_files[image_idx], img_duration, False))
                    image_idx += 1
                    remaining_images -= 1

        # Interleave videos and images
        for video_idx, (video_path, video_duration) in enumerate(
            zip(video_files, video_durations, strict=False)
        ):
            # Add video
            timed_visuals.append((video_path, video_duration, True))

            # Add images after this video (except after last video)
            if video_idx < num_videos - 1 or remaining_images > 0:
                for _ in range(min(images_per_slot, remaining_images)):
                    if image_idx < len(image_files):
                        timed_visuals.append(
                            (image_files[image_idx], img_duration, False)
                        )
                        image_idx += 1
                        remaining_images -= 1

        # Add any remaining images at the end
        while image_idx < len(image_files):
            timed_visuals.append((image_files[image_idx], img_duration, False))
            image_idx += 1

        # Calculate actual total duration
        total_duration = sum(dur for _, dur, _ in timed_visuals)

        mode_info = (
            f"mixed_media ({len(video_files)} videos, "
            f"{len(image_files)} images, {total_duration:.1f}s)"
        )
        return timed_visuals, mode_info


class VideoFirstFallbackStrategy:
    """Video-first fallback: use videos first, fill with images."""

    def __init__(
        self,
        media_inspector: MediaInspector,
        config: VideoConfig,
        product_id: str,
    ):
        """Initialize VideoFirstFallbackStrategy.

        Args:
        ----
            media_inspector: MediaInspector instance for media operations
            config: VideoConfig containing settings
            product_id: Product identifier for logging

        """
        self.inspector = media_inspector
        self.config = config
        self.product_id = product_id

    async def assemble(
        self,
        video_files: list[Path],
        image_files: list[Path],
        target_duration: float,
    ) -> tuple[list[tuple[Path, float, bool]], str]:
        """Assemble videos first, fallback to images for remaining duration.

        This strategy prioritizes product videos by playing all of them first
        sequentially, then fills any remaining duration with images.

        Strategy:
        1. Play all videos sequentially at start
        2. Calculate remaining duration after videos
        3. Fill remaining time with images (if any)
        4. Apply transition at video-to-image boundary

        Args:
        ----
            video_files: List of video file paths
            image_files: List of image file paths
            target_duration: Target total duration in seconds

        Returns:
        -------
            Tuple of (timed_visuals, mode_info)

        """
        timed_visuals: list[tuple[Path, float, bool]] = []
        video_settings = self.config.video_settings
        tolerance = video_settings.video_duration_tolerance_sec

        # Edge case: No videos - fallback to images only
        if not video_files:
            if image_files:
                img_duration = target_duration / len(image_files)
                for img in image_files:
                    timed_visuals.append((img, img_duration, False))
                mode_info = (
                    f"video_first_fallback (no videos, {len(image_files)} images)"
                )
                return timed_visuals, mode_info
            else:
                mode_info = "video_first_fallback (no media available)"
                return timed_visuals, mode_info

        # Get all video durations
        video_durations = await asyncio.gather(
            *[self.inspector.get_media_duration(vf) for vf in video_files]
        )
        total_video_duration = sum(video_durations)

        # Add all videos first (priority content)
        for video_path, duration in zip(video_files, video_durations, strict=False):
            timed_visuals.append((video_path, duration, True))

        # Calculate remaining duration for images
        remaining_duration = target_duration - total_video_duration

        # Case 1: Videos exceed or match target duration
        if remaining_duration <= tolerance:
            # Videos fill entire duration (no space for images)
            if total_video_duration > target_duration + tolerance:
                # Videos exceed - trim last video
                if len(timed_visuals) > 0:
                    last_path, last_dur, _ = timed_visuals[-1]
                    overage = total_video_duration - target_duration
                    min_last_duration = (
                        self.config.video_settings.min_last_video_duration
                    )
                    trimmed_duration = max(min_last_duration, last_dur - overage)
                    timed_visuals[-1] = (last_path, trimmed_duration, True)
                    mode_info = (
                        f"video_first_fallback ({len(video_files)} videos "
                        f"trimmed, {total_video_duration:.1f}s)"
                    )
                else:
                    mode_info = "video_first_fallback (no videos to trim)"
            else:
                # Videos match target within tolerance
                mode_info = (
                    f"video_first_fallback ({len(video_files)} videos, "
                    f"{total_video_duration:.1f}s)"
                )
            return timed_visuals, mode_info

        # Case 2: Videos don't fill target - add images for remainder
        if image_files and remaining_duration > tolerance:
            # Calculate image duration to fill remaining time
            img_duration = remaining_duration / len(image_files)

            # Add all images after videos
            for img in image_files:
                timed_visuals.append((img, img_duration, False))

            total_duration = total_video_duration + (img_duration * len(image_files))
            mode_info = (
                f"video_first_fallback ({len(video_files)} videos + "
                f"{len(image_files)} images, {total_duration:.1f}s)"
            )
        elif not image_files:
            # No images available - videos only
            mode_info = (
                f"video_first_fallback ({len(video_files)} videos only, "
                f"{total_video_duration:.1f}s < target)"
            )
        else:
            # Remaining duration too small for images
            mode_info = (
                f"video_first_fallback ({len(video_files)} videos, "
                f"{total_video_duration:.1f}s)"
            )

        return timed_visuals, mode_info


class VideoStrategyFactory:
    """Factory for creating video mode strategies."""

    def __init__(
        self,
        media_inspector: MediaInspector,
        config: VideoConfig,
        product_id: str,
    ):
        """Initialize VideoStrategyFactory.

        Args:
        ----
            media_inspector: MediaInspector instance for media operations
            config: VideoConfig containing settings
            product_id: Product identifier for logging

        """
        self.strategies: dict[str, VideoModeStrategy] = {
            "sequential": SequentialStrategy(media_inspector, config, product_id),
            "single_best": SingleBestStrategy(media_inspector, config, product_id),
            "mixed": MixedMediaStrategy(media_inspector, config, product_id),
            "video_first_fallback": VideoFirstFallbackStrategy(
                media_inspector, config, product_id
            ),
        }

    def get_strategy(self, mode: str) -> VideoModeStrategy:
        """Get strategy for the given video mode.

        Args:
        ----
            mode: Video mode name (sequential, single_best, etc.)

        Returns:
        -------
            VideoModeStrategy instance for the mode

        Raises:
        ------
            KeyError: If mode is not recognized

        """
        return self.strategies[mode]
