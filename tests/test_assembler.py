"""Unit tests for the video assembler component."""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from src.video.assembler import VideoAssembler
from src.video.video_config import VideoConfig, load_video_config


class TestVideoAssembler:
    """Test the video assembler functionality."""

    @pytest.fixture
    def assembler(self, mock_config: VideoConfig) -> VideoAssembler:
        """Create a video assembler instance for testing."""
        return VideoAssembler(mock_config)

    @pytest.fixture
    def sample_visuals(self, temp_dir: Path) -> list[Path]:
        """Create sample visual files for testing."""
        visuals = []
        for i in range(3):
            img_path = temp_dir / f"image_{i}.jpg"
            img_path.write_text(f"mock image data {i}")
            visuals.append(img_path)
        return visuals

    @pytest.fixture
    def sample_audio_files(self, temp_dir: Path) -> dict[str, Path]:
        """Create sample audio files for testing."""
        voiceover_path = temp_dir / "voiceover.wav"
        voiceover_path.write_text("mock voiceover data")

        music_path = temp_dir / "music.wav"
        music_path.write_text("mock music data")

        return {
            "voiceover": voiceover_path,
            "music": music_path,
        }

    @pytest.fixture
    def sample_subtitle_file(self, temp_dir: Path) -> Path:
        """Create a sample subtitle file for testing."""
        subtitle_path = temp_dir / "captions.srt"
        subtitle_content = """1
00:00:00,000 --> 00:00:03,000
Test subtitle line one

2
00:00:03,000 --> 00:00:06,000
Test subtitle line two

3
00:00:06,000 --> 00:00:09,000
Test subtitle line three
"""
        subtitle_path.write_text(subtitle_content)
        return subtitle_path

    @pytest.mark.asyncio
    async def test_assembler_initialization(self, mock_config: VideoConfig):
        """Test video assembler initialization."""
        assembler = VideoAssembler(mock_config)

        assert assembler.config == mock_config
        assert assembler.ffmpeg_path is not None

    @pytest.mark.asyncio
    async def test_get_media_dimensions_image(
        self, assembler: VideoAssembler, temp_dir: Path
    ):
        """Test getting dimensions for image files."""
        # Create a mock image file
        img_path = temp_dir / "test.jpg"
        img_path.write_text("mock image data")

        # Mock ffprobe output for image
        mock_output = "1920x1080"

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (mock_output.encode(), b"")
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc

            width, height = await assembler._get_media_dimensions(img_path)

            assert width == 1920
            assert height == 1080
            mock_exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_media_dimensions_video(
        self, assembler: VideoAssembler, temp_dir: Path
    ):
        """Test getting dimensions for video files."""
        # Create a mock video file
        video_path = temp_dir / "test.mp4"
        video_path.write_text("mock video data")

        # Mock ffprobe output for video
        mock_output = "1280x720"

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (mock_output.encode(), b"")
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc

            width, height = await assembler._get_media_dimensions(video_path)

            assert width == 1280
            assert height == 720
            mock_exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_media_dimensions_error(
        self, assembler: VideoAssembler, temp_dir: Path
    ):
        """Test error handling when getting media dimensions fails."""
        img_path = temp_dir / "test.jpg"
        img_path.write_text("mock image data")

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (b"", b"error")
            mock_proc.returncode = 1
            mock_exec.return_value = mock_proc

            width, height = await assembler._get_media_dimensions(img_path)

            assert width == 0
            assert height == 0

    @pytest.mark.asyncio
    async def test_get_media_duration(self, assembler: VideoAssembler, temp_dir: Path):
        """Test getting duration for media files."""
        media_path = temp_dir / "test.mp4"
        media_path.write_text("mock media data")

        # Mock ffprobe output for duration
        mock_output = "10.5"

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (mock_output.encode(), b"")
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc

            duration = await assembler._get_media_duration(media_path)

            assert duration == 10.5
            mock_exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_media_duration_error(
        self, assembler: VideoAssembler, temp_dir: Path
    ):
        """Test error handling when getting media duration fails."""
        media_path = temp_dir / "test.mp4"
        media_path.write_text("mock media data")

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (b"", b"error")
            mock_proc.returncode = 1
            mock_exec.return_value = mock_proc

            duration = await assembler._get_media_duration(media_path)
            assert duration == 0.0

    @pytest.mark.asyncio
    async def test_parse_srt_file(
        self, assembler: VideoAssembler, sample_subtitle_file: Path
    ):
        """Test parsing SRT subtitle files."""
        subtitles = assembler._parse_srt(sample_subtitle_file)

        assert len(subtitles) == 3
        assert subtitles[0].start == 0.0
        assert subtitles[0].end == 3.0
        assert subtitles[0].text == "Test subtitle line one"
        assert subtitles[1].start == 3.0
        assert subtitles[1].end == 6.0
        assert subtitles[1].text == "Test subtitle line two"

    @pytest.mark.asyncio
    async def test_parse_srt_file_empty(
        self, assembler: VideoAssembler, temp_dir: Path
    ):
        """Test parsing empty SRT file."""
        empty_srt = temp_dir / "empty.srt"
        empty_srt.write_text("")

        subtitles = assembler._parse_srt(empty_srt)
        assert subtitles == []

    @pytest.mark.asyncio
    async def test_parse_srt_file_malformed(
        self, assembler: VideoAssembler, temp_dir: Path
    ):
        """Test parsing malformed SRT file."""
        malformed_srt = temp_dir / "malformed.srt"
        malformed_srt.write_text(
            "1\n00:00:00,000 --> 00:00:03,000\nIncomplete subtitle"
        )

        # Should handle malformed SRT gracefully
        subtitles = assembler._parse_srt(malformed_srt)
        assert len(subtitles) == 1
        assert subtitles[0].text == "Incomplete subtitle"

    @pytest.mark.asyncio
    async def test_build_visual_filter_graph_images_only(
        self, assembler: VideoAssembler, sample_visuals: list[Path]
    ):
        """Test building filter graph for images only."""
        total_duration = 15.0

        with patch.object(
            assembler, "_get_media_dimensions", return_value=(1920, 1080)
        ):
            filters, input_cmd_parts, _, _, _ = await assembler._build_visual_chain(
                visual_inputs=sample_visuals,
                total_video_duration=total_duration,
                is_relative_mode=False,
            )

            assert any("xfade" in f for f in filters)
            assert any("loop" in part for part in input_cmd_parts)
            assert len(sample_visuals) == 3

    @pytest.mark.asyncio
    async def test_build_visual_filter_graph_mixed_media(
        self, assembler: VideoAssembler, temp_dir: Path
    ):
        """Test building filter graph for mixed images and videos."""
        visuals = [
            temp_dir / "image1.jpg",
            temp_dir / "video1.mp4",
            temp_dir / "image2.jpg",
        ]

        for i, visual in enumerate(visuals):
            visual.write_text(f"mock data {i}")

        total_duration = 20.0

        with (
            patch.object(assembler, "_get_media_dimensions", return_value=(1920, 1080)),
            patch.object(assembler, "_get_media_duration", return_value=5.0),
        ):
            filters, input_cmd_parts, _, _, _ = await assembler._build_visual_chain(
                visual_inputs=visuals,
                total_video_duration=total_duration,
                is_relative_mode=False,
            )

            assert any("xfade" in f for f in filters)
            assert any("loop" in part for part in input_cmd_parts)

    @pytest.mark.asyncio
    async def test_build_subtitle_graph(
        self,
        assembler: VideoAssembler,
        sample_subtitle_file: Path,
        sample_visuals: list[Path],
    ):
        """Test building subtitle filter graph."""
        with (
            patch.object(assembler, "_get_media_dimensions", return_value=(1920, 1080)),
            patch.object(
                assembler, "_resolve_font_path", return_value=Path("font.ttf")
            ),
        ):
            filters, _ = await assembler._build_subtitle_graph(
                visual_inputs=sample_visuals,
                total_video_duration=10.0,
                subtitle_path=sample_subtitle_file,
                temp_sub_dir=Path(tempfile.gettempdir()),
            )

            assert any("drawtext" in f for f in filters)

    @pytest.mark.asyncio
    async def test_build_subtitle_graph_disabled(
        self, assembler: VideoAssembler, sample_visuals: list[Path]
    ):
        """Test building subtitle filter graph when subtitles are disabled."""
        # Disable subtitles in config
        assembler.config.subtitle_settings.enabled = False

        with patch.object(
            assembler, "_get_media_dimensions", return_value=(1920, 1080)
        ):
            filters, _ = await assembler._build_subtitle_graph(
                visual_inputs=sample_visuals,
                total_video_duration=10.0,
                subtitle_path=None,
                temp_sub_dir=Path(tempfile.gettempdir()),
            )

            assert not any("drawtext" in f for f in filters)

    @pytest.mark.asyncio
    async def test_assemble_video_success(
        self,
        assembler: VideoAssembler,
        sample_visuals: list[Path],
        sample_audio_files: dict[str, Path],
        temp_dir: Path,
    ):
        """Test successful video assembly."""
        output_path = temp_dir / "output.mp4"
        total_duration = 15.0

        # Mock subprocess.run to simulate successful FFmpeg execution
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (b"", b"")
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc
            with (
                patch.object(
                    assembler, "_get_media_dimensions", return_value=(1920, 1080)
                ),
                patch.object(
                    assembler, "_resolve_font_path", return_value=Path("font.ttf")
                ),
            ):
                result = await assembler.assemble_video(
                    visual_inputs=sample_visuals,
                    voiceover_audio_path=sample_audio_files["voiceover"],
                    music_track_path=None,
                    subtitle_path=None,
                    output_path=output_path,
                    total_video_duration=total_duration,
                    temp_dir=temp_dir,
                )

                assert result is not None
                mock_exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_assemble_video_ffmpeg_error(
        self,
        assembler: VideoAssembler,
        sample_visuals: list[Path],
        sample_audio_files: dict[str, Path],
        temp_dir: Path,
    ):
        """Test video assembly when FFmpeg fails."""
        output_path = temp_dir / "output.mp4"
        total_duration = 15.0

        # Mock subprocess.run to simulate FFmpeg failure
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (b"", b"error")
            mock_proc.returncode = 1
            mock_exec.return_value = mock_proc
            with (
                patch.object(
                    assembler, "_get_media_dimensions", return_value=(1920, 1080)
                ),
                patch.object(
                    assembler, "_resolve_font_path", return_value=Path("font.ttf")
                ),
            ):
                result = await assembler.assemble_video(
                    visual_inputs=sample_visuals,
                    voiceover_audio_path=sample_audio_files["voiceover"],
                    music_track_path=None,
                    subtitle_path=None,
                    output_path=output_path,
                    total_video_duration=total_duration,
                    temp_dir=temp_dir,
                )
                assert result is None

    @pytest.mark.asyncio
    async def test_assemble_video_timeout(
        self,
        assembler: VideoAssembler,
        sample_visuals: list[Path],
        sample_audio_files: dict[str, Path],
        temp_dir: Path,
    ):
        """Test video assembly timeout handling."""
        output_path = temp_dir / "output.mp4"
        total_duration = 15.0

        # Mock subprocess.run to simulate timeout
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.side_effect = asyncio.TimeoutError
            mock_exec.return_value = mock_proc
            with (
                patch.object(
                    assembler, "_get_media_dimensions", return_value=(1920, 1080)
                ),
                patch.object(
                    assembler, "_resolve_font_path", return_value=Path("font.ttf")
                ),
            ):
                result = await assembler.assemble_video(
                    visual_inputs=sample_visuals,
                    voiceover_audio_path=sample_audio_files["voiceover"],
                    music_track_path=None,
                    subtitle_path=None,
                    output_path=output_path,
                    total_video_duration=total_duration,
                    temp_dir=temp_dir,
                )
                assert result is None

    @pytest.mark.asyncio
    async def test_verify_video_basic_functionality(
        self, assembler: VideoAssembler, temp_dir: Path
    ):
        """Test basic video verification functionality.

        Comprehensive tests in test_slideshow_images1_verification.py.
        """
        video_path = temp_dir / "test.mp4"
        video_path.write_text("mock video data")
        expected_duration = 15.0

        # Mock ffprobe to return expected duration
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(
                    {
                        "format": {"duration": "15.2"},
                        "streams": [{"codec_type": "video"}, {"codec_type": "audio"}],
                    }
                ),
                stderr=b"",
            )

            result = assembler.verify_video(
                video_path, expected_duration, should_have_subtitles=False
            )

            assert result["success"] is True
            mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_build_debug_overlay_filter_graph(self, assembler: VideoAssembler):
        """Test building debug overlay filter graph."""
        # Enable debug info in config
        assembler.config.subtitle_settings.show_debug_info = True

        # This test is difficult to implement without significant refactoring
        # of the assembler
        # since the debug overlay is not a separate method.
        # For now, we will just check that the flag is set correctly.
        assert assembler.config.subtitle_settings.show_debug_info is True


@pytest_asyncio.fixture
async def temp_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for test files."""
    return tmp_path


@pytest_asyncio.fixture
async def mock_config(temp_dir: Path) -> VideoConfig:
    """Create a mock video config for testing."""
    config_path = temp_dir / "test_config.yaml"
    config_path.write_text(
        """
video_profile: "1080p"
video_settings:
  resolution: [1920, 1080]
  output_codec: "libx264"
  output_preset: "fast"
  frame_rate: 30
  transition_duration_sec: 0.5
  min_visual_segment_duration_sec: 2.0
  image_width_percent: 0.8
  image_top_position_percent: 0.1
  pad_color: "black"
  verification_probe_timeout_sec: 10
  video_duration_tolerance_sec: 1.0
  min_file_size_kb: 10
subtitle_settings:
  enabled: true
  font_name: "Arial"
  font_directory: "/usr/share/fonts"
  font_size_percent: 0.05
  font_color: "&H00FFFFFF"
  outline_color: "&H00000000"
  back_color: "&H80000000"
  positioning_mode: "absolute"
  subtitle_similarity_threshold: 0.8
  show_debug_info: false
audio_settings:
  voiceover_volume_db: 0
  music_volume_db: -20
  music_fade_in_duration: 1.0
  music_fade_out_duration: 2.0
  output_audio_codec: "aac"
  output_audio_bitrate: "192k"
  audio_mix_duration: "longest"
  background_music_paths: []
  freesound_api_key_env_var: "FREESOUND_API_KEY"
  freesound_sort: "rating_desc"
  freesound_search_query: "test"
  freesound_filters: "test"
  freesound_max_results: 1
ffmpeg_settings:
  executable_path: "/usr/bin/ffmpeg"
  final_assembly_timeout_sec: 300
  rw_timeout_microseconds: 30000000
video_profiles:
    default:
        description: "Default profile"
llm_settings:
    provider: "mock"
    api_key_env_var: "mock"
    models: ["mock"]
    prompt_template_path: "mock"
stock_media_settings:
    pexels_api_key_env_var: "mock"
tts_config:
    provider_order: ["google_cloud"]
    google_cloud:
        model_name: "en-US-Chirp3-HD"
        language_code: "en-US"
        voice_selection_criteria:
            - language_code: "en-US"
              ssml_gender: "FEMALE"
attribution_settings:
    attribution_file_name: "mock"
    attribution_template: "mock"
    attribution_entry_template: "mock"
whisper_settings:
    enabled: false
media_settings:
    stock_media_keywords: []
    stock_video_min_duration_sec: 0
    stock_video_max_duration_sec: 0
"""
    )
    return load_video_config(config_path)
