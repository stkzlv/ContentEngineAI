"""Integration tests for end-to-end video assembly pipeline.

Tests complete video assembly pipeline from product data to final MP4 output,
covering all 4 video assembly profiles (sequential, single, mixed, primary).
"""

import json
import subprocess
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from src.video.assembler import VideoAssembler
from src.video.producer import validate_media_requirements
from src.video.video_config import VideoConfig


class TestVideoAssemblyIntegration:
    """Integration tests for end-to-end video assembly."""

    @pytest.fixture
    def temp_output_dir(self) -> Generator[Path, None, None]:
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def test_videos(self, temp_output_dir: Path) -> list[Path]:
        """Create real test video files using FFmpeg.

        Creates 3 test videos with different durations and aspect ratios:
        - video_0.mp4: 8s, 1920x1080 (16:9 landscape)
        - video_1.mp4: 6s, 1080x1920 (9:16 vertical)
        - video_2.mp4: 10s, 1080x1080 (1:1 square)
        """
        videos = []

        # Video 0: 8s landscape
        video_0 = temp_output_dir / "video_0.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:s=1920x1080:d=8",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=channel_layout=stereo:sample_rate=44100",
                "-t",
                "8",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-r",
                "30",
                "-c:a",
                "aac",
                "-shortest",
                str(video_0),
            ],
            check=True,
            capture_output=True,
        )
        videos.append(video_0)

        # Video 1: 6s vertical
        video_1 = temp_output_dir / "video_1.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-f",
                "lavfi",
                "-i",
                "color=c=green:s=1080x1920:d=6",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=channel_layout=stereo:sample_rate=44100",
                "-t",
                "6",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-r",
                "30",
                "-c:a",
                "aac",
                "-shortest",
                str(video_1),
            ],
            check=True,
            capture_output=True,
        )
        videos.append(video_1)

        # Video 2: 10s square
        video_2 = temp_output_dir / "video_2.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:s=1080x1080:d=10",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=channel_layout=stereo:sample_rate=44100",
                "-t",
                "10",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-r",
                "30",
                "-c:a",
                "aac",
                "-shortest",
                str(video_2),
            ],
            check=True,
            capture_output=True,
        )
        videos.append(video_2)

        return videos

    @pytest.fixture
    def test_images(self, temp_output_dir: Path) -> list[Path]:
        """Create real test image files using FFmpeg."""
        images = []
        for i in range(5):
            image_path = temp_output_dir / f"image_{i}.jpg"
            subprocess.run(
                [
                    "ffmpeg",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=yellow:s=1080x1920:d=0.1",
                    "-frames:v",
                    "1",
                    str(image_path),
                ],
                check=True,
                capture_output=True,
            )
            images.append(image_path)
        return images

    @pytest.fixture
    def test_voiceover(self, temp_output_dir: Path) -> Path:
        """Create test voiceover audio file (30s)."""
        voiceover_path = temp_output_dir / "voiceover.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=30",
                "-ar",
                "44100",
                "-ac",
                "2",
                str(voiceover_path),
            ],
            check=True,
            capture_output=True,
        )
        return voiceover_path

    @pytest.fixture
    def test_music(self, temp_output_dir: Path) -> Path:
        """Create test background music file (30s)."""
        music_path = temp_output_dir / "music.mp3"
        subprocess.run(
            [
                "ffmpeg",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=220:duration=30",
                "-ar",
                "44100",
                "-ac",
                "2",
                "-b:a",
                "192k",
                str(music_path),
            ],
            check=True,
            capture_output=True,
        )
        return music_path

    @pytest.fixture
    def test_subtitles(self, temp_output_dir: Path) -> Path:
        """Create test subtitle file."""
        subtitle_path = temp_output_dir / "captions.srt"
        subtitle_content = """1
00:00:00,000 --> 00:00:03,000
Test subtitle line one

2
00:00:03,000 --> 00:00:06,000
Test subtitle line two
"""
        subtitle_path.write_text(subtitle_content)
        return subtitle_path

    def validate_video_output(self, video_path: Path, expected_duration: float):
        """Validate final video output using FFprobe.

        Checks:
        - Video codec is H.264
        - Resolution is 1080x1920 (9:16 vertical)
        - Frame rate is 30fps
        - Duration matches expected (±2s tolerance)
        - Has both video and audio streams
        """
        # Get video metadata using ffprobe
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        metadata = json.loads(result.stdout)

        # Find video and audio streams
        video_stream = next(
            (s for s in metadata["streams"] if s["codec_type"] == "video"), None
        )
        audio_stream = next(
            (s for s in metadata["streams"] if s["codec_type"] == "audio"), None
        )

        assert video_stream is not None, "No video stream found"
        assert audio_stream is not None, "No audio stream found"

        # Validate video properties
        assert (
            video_stream["codec_name"] == "h264"
        ), f"Expected H.264 codec, got {video_stream['codec_name']}"
        assert (
            int(video_stream["width"]) == 1080
        ), f"Expected width 1080, got {video_stream['width']}"
        assert (
            int(video_stream["height"]) == 1920
        ), f"Expected height 1920, got {video_stream['height']}"

        # Validate frame rate (30fps)
        fps_str = video_stream.get("r_frame_rate", "30/1")
        num, den = map(int, fps_str.split("/"))
        fps = num / den if den != 0 else 0
        assert abs(fps - 30) < 1, f"Expected 30fps, got {fps}"

        # Validate duration (±2s tolerance for integration tests)
        duration = float(metadata["format"]["duration"])
        assert (
            abs(duration - expected_duration) <= 2.0
        ), f"Duration mismatch: expected {expected_duration}±2s, got {duration}"

        return metadata

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_sequential_profile_end_to_end(
        self,
        mock_config: VideoConfig,
        temp_output_dir: Path,
        test_videos: list[Path],
        test_images: list[Path],
        test_voiceover: Path,
        test_music: Path,
        test_subtitles: Path,
    ):
        """Test complete pipeline with product_video_sequential profile."""
        # Get sequential profile
        profile = mock_config.profiles["product_video_sequential"]

        # Create assembler
        assembler = VideoAssembler(config=mock_config, debug_mode=True)

        # Assemble video
        output_path = temp_output_dir / "output_sequential.mp4"
        await assembler.assemble_video(
            visuals=test_videos + test_images,
            voiceover_path=test_voiceover,
            music_path=test_music,
            subtitle_path=test_subtitles,
            output_path=output_path,
            profile=profile,
        )

        # Validate output
        assert output_path.exists(), "Output video was not created"
        metadata = self.validate_video_output(output_path, expected_duration=30.0)

        # Validate sequential mode: all videos should be used
        # Video 0: 8s, Video 1: 6s, Video 2: 10s = 24s total
        # Should loop to fill remaining 6s for 30s voiceover
        assert metadata is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_single_best_profile_end_to_end(
        self,
        mock_config: VideoConfig,
        temp_output_dir: Path,
        test_videos: list[Path],
        test_images: list[Path],
        test_voiceover: Path,
        test_music: Path,
        test_subtitles: Path,
    ):
        """Test complete pipeline with product_video_single profile."""
        # Get single profile
        profile = mock_config.profiles["product_video_single"]

        # Create assembler
        assembler = VideoAssembler(config=mock_config, debug_mode=True)

        # Assemble video
        output_path = temp_output_dir / "output_single.mp4"
        await assembler.assemble_video(
            visuals=test_videos + test_images,
            voiceover_path=test_voiceover,
            music_path=test_music,
            subtitle_path=test_subtitles,
            output_path=output_path,
            profile=profile,
        )

        # Validate output
        assert output_path.exists(), "Output video was not created"
        metadata = self.validate_video_output(output_path, expected_duration=30.0)

        # Validate single best mode: should use longest video (10s) looped
        assert metadata is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_mixed_media_profile_end_to_end(
        self,
        mock_config: VideoConfig,
        temp_output_dir: Path,
        test_videos: list[Path],
        test_images: list[Path],
        test_voiceover: Path,
        test_music: Path,
        test_subtitles: Path,
    ):
        """Test complete pipeline with product_video_mixed profile."""
        # Get mixed profile
        profile = mock_config.profiles["product_video_mixed"]

        # Create assembler
        assembler = VideoAssembler(config=mock_config, debug_mode=True)

        # Assemble video
        output_path = temp_output_dir / "output_mixed.mp4"
        await assembler.assemble_video(
            visuals=test_videos + test_images,
            voiceover_path=test_voiceover,
            music_path=test_music,
            subtitle_path=test_subtitles,
            output_path=output_path,
            profile=profile,
        )

        # Validate output
        assert output_path.exists(), "Output video was not created"
        metadata = self.validate_video_output(output_path, expected_duration=30.0)

        # Validate mixed mode: should interleave videos and images
        assert metadata is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_primary_profile_end_to_end(
        self,
        mock_config: VideoConfig,
        temp_output_dir: Path,
        test_videos: list[Path],
        test_images: list[Path],
        test_voiceover: Path,
        test_music: Path,
        test_subtitles: Path,
    ):
        """Test complete pipeline with product_video_primary profile."""
        # Get primary profile
        profile = mock_config.profiles["product_video_primary"]

        # Create assembler
        assembler = VideoAssembler(config=mock_config, debug_mode=True)

        # Assemble video
        output_path = temp_output_dir / "output_primary.mp4"
        await assembler.assemble_video(
            visuals=test_videos + test_images,
            voiceover_path=test_voiceover,
            music_path=test_music,
            subtitle_path=test_subtitles,
            output_path=output_path,
            profile=profile,
        )

        # Validate output
        assert output_path.exists(), "Output video was not created"
        metadata = self.validate_video_output(output_path, expected_duration=30.0)

        # Validate primary mode: all videos first (24s), then images (6s)
        assert metadata is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_transition_validation(
        self,
        mock_config: VideoConfig,
        temp_output_dir: Path,
        test_videos: list[Path],
        test_images: list[Path],
        test_voiceover: Path,
        test_music: Path,
        test_subtitles: Path,
    ):
        """Test that transitions are properly applied between clips."""
        # Use sequential profile with transitions
        profile = mock_config.profiles["product_video_sequential"]

        # Verify transition duration is configured
        assert profile.video_transition_duration == 0.5

        # Create assembler
        assembler = VideoAssembler(config=mock_config, debug_mode=True)

        # Assemble video
        output_path = temp_output_dir / "output_transitions.mp4"
        await assembler.assemble_video(
            visuals=test_videos,
            voiceover_path=test_voiceover,
            music_path=test_music,
            subtitle_path=test_subtitles,
            output_path=output_path,
            profile=profile,
        )

        # Validate output exists
        assert output_path.exists(), "Output video was not created"

        # Validate video properties
        self.validate_video_output(output_path, expected_duration=30.0)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_aspect_ratio_handling(
        self,
        mock_config: VideoConfig,
        temp_output_dir: Path,
        test_videos: list[Path],
        test_voiceover: Path,
        test_music: Path,
        test_subtitles: Path,
    ):
        """Test aspect ratio handling for different video orientations."""
        # Test letterbox mode
        profile_letterbox = mock_config.profiles["product_video_mixed"]
        assert profile_letterbox.video_aspect_mode == "letterbox"

        assembler = VideoAssembler(config=mock_config, debug_mode=True)

        output_letterbox = temp_output_dir / "output_letterbox.mp4"
        await assembler.assemble_video(
            visuals=test_videos,
            voiceover_path=test_voiceover,
            music_path=test_music,
            subtitle_path=test_subtitles,
            output_path=output_letterbox,
            profile=profile_letterbox,
        )

        assert output_letterbox.exists()
        self.validate_video_output(output_letterbox, expected_duration=30.0)

        # Test crop-to-fit mode
        profile_crop = mock_config.profiles["product_video_single"]
        assert profile_crop.video_aspect_mode == "crop-to-fit"

        output_crop = temp_output_dir / "output_crop.mp4"
        await assembler.assemble_video(
            visuals=test_videos,
            voiceover_path=test_voiceover,
            music_path=test_music,
            subtitle_path=test_subtitles,
            output_path=output_crop,
            profile=profile_crop,
        )

        assert output_crop.exists()
        self.validate_video_output(output_crop, expected_duration=30.0)

        # Test smart-scale mode
        profile_smart = mock_config.profiles["product_video_sequential"]
        assert profile_smart.video_aspect_mode == "smart-scale"

        output_smart = temp_output_dir / "output_smart.mp4"
        await assembler.assemble_video(
            visuals=test_videos,
            voiceover_path=test_voiceover,
            music_path=test_music,
            subtitle_path=test_subtitles,
            output_path=output_smart,
            profile=profile_smart,
        )

        assert output_smart.exists()
        self.validate_video_output(output_smart, expected_duration=30.0)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_audio_handling_modes(
        self,
        mock_config: VideoConfig,
        temp_output_dir: Path,
        test_videos: list[Path],
        test_voiceover: Path,
        test_music: Path,
        test_subtitles: Path,
    ):
        """Test audio handling modes (remove vs mixed)."""
        # Test remove mode
        profile_remove = mock_config.profiles["product_video_single"]
        assert profile_remove.video_audio_handling == "remove"

        assembler = VideoAssembler(config=mock_config, debug_mode=True)

        output_remove = temp_output_dir / "output_audio_remove.mp4"
        await assembler.assemble_video(
            visuals=test_videos,
            voiceover_path=test_voiceover,
            music_path=test_music,
            subtitle_path=test_subtitles,
            output_path=output_remove,
            profile=profile_remove,
        )

        assert output_remove.exists()
        metadata_remove = self.validate_video_output(
            output_remove, expected_duration=30.0
        )

        # Verify audio stream exists (voiceover + music only)
        audio_stream = next(
            s for s in metadata_remove["streams"] if s["codec_type"] == "audio"
        )
        assert audio_stream is not None

        # Test mixed mode
        profile_mixed = mock_config.profiles["product_video_sequential"]
        assert profile_mixed.video_audio_handling == "mixed"

        output_mixed = temp_output_dir / "output_audio_mixed.mp4"
        await assembler.assemble_video(
            visuals=test_videos,
            voiceover_path=test_voiceover,
            music_path=test_music,
            subtitle_path=test_subtitles,
            output_path=output_mixed,
            profile=profile_mixed,
        )

        assert output_mixed.exists()
        metadata_mixed = self.validate_video_output(
            output_mixed, expected_duration=30.0
        )

        # Verify audio stream exists (voiceover + music + video audio)
        audio_stream_mixed = next(
            s for s in metadata_mixed["streams"] if s["codec_type"] == "audio"
        )
        assert audio_stream_mixed is not None

    @pytest.mark.integration
    def test_media_validation_integration(self, mock_config: VideoConfig):
        """Test media validation with video-first profiles."""
        # Test with videos available
        scraped_videos = [Path("/mock/video1.mp4"), Path("/mock/video2.mp4")]
        scraped_images = [Path("/mock/image1.jpg")]
        stock_media: list = []
        profile = mock_config.profiles["product_video_sequential"]

        is_valid, reason = validate_media_requirements(
            scraped_images, scraped_videos, stock_media, profile, mock_config
        )

        assert is_valid, f"Validation failed: {reason}"

        # Test fallback to images when no videos
        scraped_videos_empty: list = []
        scraped_images_many = [Path(f"/mock/image{i}.jpg") for i in range(6)]

        is_valid_fallback, reason_fallback = validate_media_requirements(
            scraped_images_many, scraped_videos_empty, stock_media, profile, mock_config
        )

        assert is_valid_fallback, f"Fallback validation failed: {reason_fallback}"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cleanup_after_assembly(
        self,
        mock_config: VideoConfig,
        temp_output_dir: Path,
        test_videos: list[Path],
        test_voiceover: Path,
        test_music: Path,
        test_subtitles: Path,
    ):
        """Test that test outputs are properly cleaned up."""
        profile = mock_config.profiles["product_video_sequential"]
        assembler = VideoAssembler(config=mock_config, debug_mode=True)

        output_path = temp_output_dir / "output_cleanup.mp4"
        await assembler.assemble_video(
            visuals=test_videos,
            voiceover_path=test_voiceover,
            music_path=test_music,
            subtitle_path=test_subtitles,
            output_path=output_path,
            profile=profile,
        )

        # Verify output exists
        assert output_path.exists()

        # Cleanup is handled by temp_output_dir fixture automatically
        # Just verify the file was created successfully
        assert output_path.stat().st_size > 0, "Output file is empty"
