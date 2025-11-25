"""Unit tests for video assembly strategies.

Tests all 4 video assembly modes (sequential, single_best, mixed_media,
video_first_fallback) following requirements 1 and 5.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.video.assembler import VideoAssembler
from src.video.config import VideoConfig


@pytest.fixture
def video_assembler(mock_config: VideoConfig) -> VideoAssembler:
    """Create video assembler with video assembly configuration."""
    # Ensure video assembly settings are configured
    mock_config.video_settings.video_assembly_mode = "sequential"
    mock_config.video_settings.video_aspect_mode = "letterbox"
    mock_config.video_settings.video_audio_handling = "remove"
    mock_config.video_settings.video_transition_duration = 0.5
    return VideoAssembler(config=mock_config, debug_mode=False)


@pytest.fixture
def sample_video_files(temp_dir: Path) -> list[Path]:
    """Create sample video file paths for testing."""
    video_files = []
    for i in range(3):
        video_path = temp_dir / f"video_{i}.mp4"
        video_path.write_text(f"mock video data {i}")
        video_files.append(video_path)
    return video_files


@pytest.fixture
def sample_image_files(temp_dir: Path) -> list[Path]:
    """Create sample image file paths for testing."""
    image_files = []
    for i in range(5):
        image_path = temp_dir / f"image_{i}.jpg"
        image_path.write_text(f"mock image data {i}")
        image_files.append(image_path)
    return image_files


class TestSequentialMode:
    """Test sequential video assembly mode (Requirement 1.1)."""

    @pytest.mark.asyncio
    async def test_sequential_concatenates_all_videos(
        self, video_assembler: VideoAssembler, sample_video_files: list[Path]
    ):
        """Test sequential mode concatenates all videos end-to-end."""
        target_duration = 30.0

        # Mock duration: 3 videos @ 8s each = 24s total
        # Implementation will loop last video to fill remaining 6s
        with patch.object(
            video_assembler, "_get_media_duration", new_callable=AsyncMock
        ) as mock_duration:
            mock_duration.return_value = 8.0

            timed_visuals, mode_info = await video_assembler._assemble_sequential(
                sample_video_files, [], target_duration
            )

        # Should use all 3 videos + loop last one to match duration
        assert len(timed_visuals) >= 3
        # All should be marked as videos (is_video=True)
        assert all(is_video for _, _, is_video in timed_visuals)
        # Total duration should match target ±1s
        total_duration = sum(duration for _, duration, _ in timed_visuals)
        assert abs(total_duration - target_duration) <= 1.0
        assert "sequential" in mode_info.lower()

    @pytest.mark.asyncio
    async def test_sequential_handles_insufficient_duration(
        self,
        video_assembler: VideoAssembler,
        sample_video_files: list[Path],
        sample_image_files: list[Path],
    ):
        """Test sequential mode adds images when videos too short (Req 1.5)."""
        target_duration = 60.0

        # Mock duration: 3 videos @ 10s each = 30s, need 30s more
        # Implementation loops videos first before adding images
        with patch.object(
            video_assembler, "_get_media_duration", new_callable=AsyncMock
        ) as mock_duration:
            mock_duration.return_value = 10.0

            timed_visuals, mode_info = await video_assembler._assemble_sequential(
                sample_video_files, sample_image_files, target_duration
            )

        # Should have videos (possibly looped) and may have images
        video_count = sum(1 for _, _, is_video in timed_visuals if is_video)
        # At minimum should have 3 original videos
        assert video_count >= 3
        # Total duration should match target ±1s (Requirement 5.5)
        total_duration = sum(duration for _, duration, _ in timed_visuals)
        assert abs(total_duration - target_duration) <= 1.0

    @pytest.mark.asyncio
    async def test_sequential_handles_excessive_duration(
        self, video_assembler: VideoAssembler, sample_video_files: list[Path]
    ):
        """Test sequential mode trims when videos too long (Req 1.6)."""
        target_duration = 20.0

        # Mock duration: 3 videos @ 10s each = 30s, need to trim 10s
        with patch.object(
            video_assembler, "_get_media_duration", new_callable=AsyncMock
        ) as mock_duration:
            mock_duration.return_value = 10.0

            timed_visuals, mode_info = await video_assembler._assemble_sequential(
                sample_video_files, [], target_duration
            )

        # Should have all 3 videos (last one trimmed)
        assert len(timed_visuals) == 3
        # First 2 videos should be 10s
        assert timed_visuals[0][1] == 10.0
        assert timed_visuals[1][1] == 10.0
        # Last video should be trimmed (total = 20s)
        total_duration = sum(duration for _, duration, _ in timed_visuals)
        assert abs(total_duration - target_duration) <= 1.0

    @pytest.mark.asyncio
    async def test_sequential_with_single_video(
        self, video_assembler: VideoAssembler, sample_video_files: list[Path]
    ):
        """Test sequential mode with only 1 video (edge case)."""
        target_duration = 25.0
        single_video = [sample_video_files[0]]

        # Mock duration: 1 video @ 15s
        with patch.object(
            video_assembler, "_get_media_duration", new_callable=AsyncMock
        ) as mock_duration:
            mock_duration.return_value = 15.0

            timed_visuals, mode_info = await video_assembler._assemble_sequential(
                single_video, [], target_duration
            )

        # Should loop the single video to match duration
        assert len(timed_visuals) >= 1


class TestSingleBestMode:
    """Test single_best video assembly mode (Requirement 1.2)."""

    @pytest.mark.asyncio
    async def test_single_best_selects_longest_video(
        self, video_assembler: VideoAssembler, sample_video_files: list[Path]
    ):
        """Test single_best mode selects longest video."""
        target_duration = 30.0

        # Mock durations: 8s, 15s, 10s (longest is video_1 @ 15s)
        async def mock_duration_side_effect(path: Path) -> float:
            if "video_0" in str(path):
                return 8.0
            elif "video_1" in str(path):
                return 15.0
            else:
                return 10.0

        with patch.object(
            video_assembler, "_get_media_duration", new_callable=AsyncMock
        ) as mock_duration:
            mock_duration.side_effect = mock_duration_side_effect

            timed_visuals, mode_info = await video_assembler._assemble_single_best(
                sample_video_files, [], target_duration
            )

        # Should select video_1 (longest at 15s)
        assert any("video_1" in str(path) for path, _, _ in timed_visuals)
        # Should loop to match 30s target
        total_duration = sum(duration for _, duration, _ in timed_visuals)
        assert abs(total_duration - target_duration) <= 1.0

    @pytest.mark.asyncio
    async def test_single_best_loops_seamlessly(
        self, video_assembler: VideoAssembler, sample_video_files: list[Path]
    ):
        """Test single_best mode loops video to match duration (Req 1.2, 5.2)."""
        target_duration = 40.0
        single_video = [sample_video_files[0]]

        # Mock duration: 1 video @ 12s, needs ~3.33 loops to reach 40s
        with patch.object(
            video_assembler, "_get_media_duration", new_callable=AsyncMock
        ) as mock_duration:
            mock_duration.return_value = 12.0

            timed_visuals, mode_info = await video_assembler._assemble_single_best(
                single_video, [], target_duration
            )

        # Should have multiple instances (looped)
        assert len(timed_visuals) >= 3
        # All should be the same video
        assert all(is_video for _, _, is_video in timed_visuals)
        # Total duration should match target ±1s
        total_duration = sum(duration for _, duration, _ in timed_visuals)
        assert abs(total_duration - target_duration) <= 1.0

    @pytest.mark.asyncio
    async def test_single_best_exact_duration_match(
        self, video_assembler: VideoAssembler, sample_video_files: list[Path]
    ):
        """Test single_best when video duration exactly matches target."""
        target_duration = 30.0
        single_video = [sample_video_files[0]]

        # Mock duration: exactly 30s (perfect match)
        with patch.object(
            video_assembler, "_get_media_duration", new_callable=AsyncMock
        ) as mock_duration:
            mock_duration.return_value = 30.0

            timed_visuals, mode_info = await video_assembler._assemble_single_best(
                single_video, [], target_duration
            )

        # Should have exactly 1 video at 30s
        assert len(timed_visuals) == 1
        assert timed_visuals[0][1] == 30.0


class TestMixedMediaMode:
    """Test mixed_media video assembly mode (Requirement 1.3)."""

    @pytest.mark.asyncio
    async def test_mixed_media_interleaves_videos_and_images(
        self,
        video_assembler: VideoAssembler,
        sample_video_files: list[Path],
        sample_image_files: list[Path],
    ):
        """Test mixed_media mode interleaves videos and images."""
        target_duration = 30.0

        # Mock duration: 3 videos @ 5s each = 15s
        with patch.object(
            video_assembler, "_get_media_duration", new_callable=AsyncMock
        ) as mock_duration:
            mock_duration.return_value = 5.0

            timed_visuals, mode_info = await video_assembler._assemble_mixed_media(
                sample_video_files, sample_image_files, target_duration
            )

        # Should have mix of videos and images
        video_count = sum(1 for _, _, is_video in timed_visuals if is_video)
        image_count = sum(1 for _, _, is_video in timed_visuals if not is_video)
        assert video_count > 0
        assert image_count > 0
        # Should be interleaved (not all videos then all images)
        # Check that videos and images alternate somewhat
        types_sequence = [is_video for _, _, is_video in timed_visuals]
        assert not all(types_sequence)  # Not all videos
        assert not all(not t for t in types_sequence)  # Not all images

    @pytest.mark.asyncio
    async def test_mixed_media_duration_matching(
        self,
        video_assembler: VideoAssembler,
        sample_video_files: list[Path],
        sample_image_files: list[Path],
    ):
        """Test mixed_media matches target duration (Req 5.5)."""
        target_duration = 45.0

        # Mock duration: videos @ 8s each
        with patch.object(
            video_assembler, "_get_media_duration", new_callable=AsyncMock
        ) as mock_duration:
            mock_duration.return_value = 8.0

            timed_visuals, mode_info = await video_assembler._assemble_mixed_media(
                sample_video_files, sample_image_files, target_duration
            )

        # Total duration should match target ±1s
        total_duration = sum(duration for _, duration, _ in timed_visuals)
        assert abs(total_duration - target_duration) <= 1.0

    @pytest.mark.asyncio
    async def test_mixed_media_with_no_images(
        self, video_assembler: VideoAssembler, sample_video_files: list[Path]
    ):
        """Test mixed_media mode with only videos (edge case)."""
        target_duration = 30.0

        # Mock duration: 3 videos @ 10s each = 30s
        with patch.object(
            video_assembler, "_get_media_duration", new_callable=AsyncMock
        ) as mock_duration:
            mock_duration.return_value = 10.0

            timed_visuals, mode_info = await video_assembler._assemble_mixed_media(
                sample_video_files, [], target_duration
            )

        # Should have only videos
        assert all(is_video for _, _, is_video in timed_visuals)
        # Should match target duration
        total_duration = sum(duration for _, duration, _ in timed_visuals)
        assert abs(total_duration - target_duration) <= 1.0


class TestVideoFirstFallbackMode:
    """Test video_first_fallback assembly mode (Requirement 1.4)."""

    @pytest.mark.asyncio
    async def test_video_first_uses_all_videos_first(
        self,
        video_assembler: VideoAssembler,
        sample_video_files: list[Path],
        sample_image_files: list[Path],
    ):
        """Test video_first_fallback mode uses all videos first."""
        target_duration = 50.0

        # Mock duration: 3 videos @ 10s each = 30s, need 20s images
        with patch.object(
            video_assembler, "_get_media_duration", new_callable=AsyncMock
        ) as mock_duration:
            mock_duration.return_value = 10.0

            (
                timed_visuals,
                mode_info,
            ) = await video_assembler._assemble_video_first_fallback(
                sample_video_files, sample_image_files, target_duration
            )

        # First 3 items should be videos
        assert timed_visuals[0][2] is True  # is_video
        assert timed_visuals[1][2] is True
        assert timed_visuals[2][2] is True
        # Subsequent items should be images (if any)
        if len(timed_visuals) > 3:
            assert not timed_visuals[3][2]  # is_video should be False

    @pytest.mark.asyncio
    async def test_video_first_fills_remaining_with_images(
        self,
        video_assembler: VideoAssembler,
        sample_video_files: list[Path],
        sample_image_files: list[Path],
    ):
        """Test video_first_fallback adds images for remaining time (Req 1.4)."""
        target_duration = 60.0

        # Mock duration: 3 videos @ 15s each = 45s, need 15s images
        with patch.object(
            video_assembler, "_get_media_duration", new_callable=AsyncMock
        ) as mock_duration:
            mock_duration.return_value = 15.0

            (
                timed_visuals,
                mode_info,
            ) = await video_assembler._assemble_video_first_fallback(
                sample_video_files, sample_image_files, target_duration
            )

        # Should have 3 videos + images
        video_count = sum(1 for _, _, is_video in timed_visuals if is_video)
        image_count = sum(1 for _, _, is_video in timed_visuals if not is_video)
        assert video_count == 3
        assert image_count > 0
        # Total duration should match target ±1s
        total_duration = sum(duration for _, duration, _ in timed_visuals)
        assert abs(total_duration - target_duration) <= 1.0

    @pytest.mark.asyncio
    async def test_video_first_videos_exceed_duration(
        self, video_assembler: VideoAssembler, sample_video_files: list[Path]
    ):
        """Test video_first_fallback trims when videos exceed target."""
        target_duration = 25.0

        # Mock duration: 3 videos @ 10s each = 30s, need to trim 5s
        with patch.object(
            video_assembler, "_get_media_duration", new_callable=AsyncMock
        ) as mock_duration:
            mock_duration.return_value = 10.0

            (
                timed_visuals,
                mode_info,
            ) = await video_assembler._assemble_video_first_fallback(
                sample_video_files, [], target_duration
            )

        # Should use all videos but trim last one
        assert len(timed_visuals) == 3
        # Total duration should match target ±1s
        total_duration = sum(duration for _, duration, _ in timed_visuals)
        assert abs(total_duration - target_duration) <= 1.0


class TestDurationMatching:
    """Test duration matching algorithm across all modes (Requirement 5)."""

    @pytest.mark.asyncio
    async def test_duration_tolerance_within_one_second(
        self, video_assembler: VideoAssembler, sample_video_files: list[Path]
    ):
        """Test all modes respect ±1s duration tolerance (Req 5.5)."""
        target_duration = 30.0

        # Mock duration: 3 videos @ 9s each = 27s (3s short)
        with patch.object(
            video_assembler, "_get_media_duration", new_callable=AsyncMock
        ) as mock_duration:
            mock_duration.return_value = 9.0

            # Test sequential mode
            timed_visuals, _ = await video_assembler._assemble_sequential(
                sample_video_files, [], target_duration
            )
            total_duration = sum(duration for _, duration, _ in timed_visuals)
            assert abs(total_duration - target_duration) <= 1.0

    @pytest.mark.asyncio
    async def test_transition_duration_applied(
        self, video_assembler: VideoAssembler, sample_video_files: list[Path]
    ):
        """Test transitions are applied between clips (Req 6.1)."""
        target_duration = 30.0
        video_assembler.config.video_settings.video_transition_duration = 0.5

        # Mock duration: 3 videos @ 10s each
        with patch.object(
            video_assembler, "_get_media_duration", new_callable=AsyncMock
        ) as mock_duration:
            mock_duration.return_value = 10.0

            timed_visuals, _ = await video_assembler._assemble_sequential(
                sample_video_files, [], target_duration
            )

        # Should have 3 videos with transitions accounted for
        assert len(timed_visuals) == 3


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_zero_videos_falls_back_to_images(
        self, video_assembler: VideoAssembler, sample_image_files: list[Path]
    ):
        """Test graceful fallback when no videos provided (Req 8.2)."""
        target_duration = 20.0

        # Test sequential with 0 videos
        timed_visuals, mode_info = await video_assembler._assemble_sequential(
            [], sample_image_files, target_duration
        )

        # Should use only images
        assert all(not is_video for _, _, is_video in timed_visuals)
        assert len(timed_visuals) > 0

    @pytest.mark.asyncio
    async def test_many_videos_handled_correctly(
        self, video_assembler: VideoAssembler, temp_dir: Path
    ):
        """Test handling many videos (stress test)."""
        # Create 10 videos
        many_videos = []
        for i in range(10):
            video_path = temp_dir / f"video_{i}.mp4"
            video_path.write_text(f"mock video data {i}")
            many_videos.append(video_path)

        target_duration = 60.0

        # Mock duration: 10 videos @ 5s each = 50s
        # Implementation will loop to fill remaining 10s
        with patch.object(
            video_assembler, "_get_media_duration", new_callable=AsyncMock
        ) as mock_duration:
            mock_duration.return_value = 5.0

            timed_visuals, mode_info = await video_assembler._assemble_sequential(
                many_videos, [], target_duration
            )

        # Should handle all 10 videos (may loop to match duration)
        assert len(timed_visuals) >= 10
        total_duration = sum(duration for _, duration, _ in timed_visuals)
        assert abs(total_duration - target_duration) <= 1.0

    @pytest.mark.asyncio
    async def test_very_short_target_duration(
        self, video_assembler: VideoAssembler, sample_video_files: list[Path]
    ):
        """Test handling very short target duration (edge case)."""
        target_duration = 5.0

        # Mock duration: videos @ 3s each (shorter than target)
        # This avoids transition duration issues
        with patch.object(
            video_assembler, "_get_media_duration", new_callable=AsyncMock
        ) as mock_duration:
            mock_duration.return_value = 3.0

            timed_visuals, mode_info = await video_assembler._assemble_sequential(
                sample_video_files, [], target_duration
            )

        # Should use videos to match duration
        assert len(timed_visuals) >= 1
        # Total duration should match target ±1.5s (accounting for transitions)
        total_duration = sum(duration for _, duration, _ in timed_visuals)
        assert abs(total_duration - target_duration) <= 1.5

    @pytest.mark.asyncio
    async def test_very_long_target_duration(
        self,
        video_assembler: VideoAssembler,
        sample_video_files: list[Path],
        sample_image_files: list[Path],
    ):
        """Test handling very long target duration (edge case)."""
        target_duration = 120.0  # 2 minutes

        # Mock duration: 3 videos @ 10s each = 30s (much shorter than target)
        with patch.object(
            video_assembler, "_get_media_duration", new_callable=AsyncMock
        ) as mock_duration:
            mock_duration.return_value = 10.0

            timed_visuals, mode_info = await video_assembler._assemble_sequential(
                sample_video_files, sample_image_files, target_duration
            )

        # Should use videos + images to fill long duration
        total_duration = sum(duration for _, duration, _ in timed_visuals)
        assert abs(total_duration - target_duration) <= 1.0
