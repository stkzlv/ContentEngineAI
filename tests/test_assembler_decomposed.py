"""Unit tests for decomposed video assembler functions.

This module tests the individual functions that were extracted from the
large assemble_video function to improve testability and maintainability.
"""

from pathlib import Path
from unittest.mock import MagicMock

from src.video.assembler import VideoAssembler


class TestVideoAssemblerDecomposed:
    """Test cases for decomposed VideoAssembler functions."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock config with default values
        self.mock_config = MagicMock()

        # Create mock audio settings
        mock_audio = MagicMock()
        mock_audio.voiceover_volume_db = -3.0
        mock_audio.music_volume_db = -12.0
        mock_audio.music_fade_in_duration = 2.0
        mock_audio.music_fade_out_duration = 3.0
        mock_audio.audio_mix_duration = "longest"
        mock_audio.output_audio_codec = "aac"
        mock_audio.output_audio_bitrate = "128k"
        self.mock_config.audio_settings = mock_audio

        # Create mock video settings
        mock_video = MagicMock()
        mock_video.output_codec = "libx264"
        mock_video.output_preset = "medium"
        mock_video.output_pixel_format = "yuv420p"
        mock_video.frame_rate = 30
        self.mock_config.video_settings = mock_video

        # Create mock ffmpeg settings
        mock_ffmpeg = MagicMock()
        mock_ffmpeg.rw_timeout_microseconds = 30000000
        self.mock_config.ffmpeg_settings = mock_ffmpeg

        self.mock_config.debug_settings = {"create_ffmpeg_command_logs": True}

        # Create assembler instance
        self.assembler = VideoAssembler(self.mock_config)

    def test_prepare_audio_inputs_with_both_audio_types(self):
        """Test preparing audio inputs with both voiceover and music."""
        input_cmd_parts = ["-i", "visual1.mp4", "-i", "visual2.mp4"]
        voiceover_path = Path("voiceover.wav")
        music_path = Path("music.mp3")
        visual_count = 2

        voiceover_idx, music_idx = self.assembler._prepare_audio_inputs(
            input_cmd_parts, voiceover_path, music_path, visual_count
        )

        assert voiceover_idx == 2  # First audio input after 2 visuals
        assert music_idx == 3  # Second audio input

        expected_cmd = [
            "-i",
            "visual1.mp4",
            "-i",
            "visual2.mp4",
            "-i",
            str(voiceover_path),
            "-i",
            str(music_path),
        ]
        assert input_cmd_parts == expected_cmd

    def test_prepare_audio_inputs_voiceover_only(self):
        """Test preparing audio inputs with only voiceover."""
        input_cmd_parts = ["-i", "visual1.mp4"]
        voiceover_path = Path("voiceover.wav")
        visual_count = 1

        voiceover_idx, music_idx = self.assembler._prepare_audio_inputs(
            input_cmd_parts, voiceover_path, None, visual_count
        )

        assert voiceover_idx == 1
        assert music_idx is None

        expected_cmd = ["-i", "visual1.mp4", "-i", str(voiceover_path)]
        assert input_cmd_parts == expected_cmd

    def test_prepare_audio_inputs_music_only(self):
        """Test preparing audio inputs with only music."""
        input_cmd_parts = ["-i", "visual1.mp4"]
        music_path = Path("music.mp3")
        visual_count = 1

        voiceover_idx, music_idx = self.assembler._prepare_audio_inputs(
            input_cmd_parts, None, music_path, visual_count
        )

        assert voiceover_idx is None
        assert music_idx == 1

        expected_cmd = ["-i", "visual1.mp4", "-i", str(music_path)]
        assert input_cmd_parts == expected_cmd

    def test_prepare_audio_inputs_no_audio(self):
        """Test preparing audio inputs with no audio files."""
        input_cmd_parts = ["-i", "visual1.mp4"]
        visual_count = 1

        voiceover_idx, music_idx = self.assembler._prepare_audio_inputs(
            input_cmd_parts, None, None, visual_count
        )

        assert voiceover_idx is None
        assert music_idx is None

        # Command should remain unchanged
        assert input_cmd_parts == ["-i", "visual1.mp4"]

    def test_build_audio_filters_voiceover_and_music(self):
        """Test building audio filters with both voiceover and music."""
        voiceover_idx = 2
        music_idx = 3
        total_duration = 30.0

        # Set specific audio settings for testing
        self.mock_config.audio_settings.voiceover_volume_db = -3.0
        self.mock_config.audio_settings.music_volume_db = -12.0
        self.mock_config.audio_settings.music_fade_in_duration = 2.0
        self.mock_config.audio_settings.music_fade_out_duration = 3.0
        self.mock_config.audio_settings.audio_mix_duration = "longest"

        audio_filters, final_label = self.assembler._build_audio_filters(
            voiceover_idx, music_idx, total_duration
        )

        assert len(audio_filters) == 3  # voiceover + music + mix
        assert final_label == "[a_mixed]"

        # Check voiceover filter
        assert "[2:a]volume=-3.0dB[a_voice_proc]" in audio_filters

        # Check music filter with fades
        music_filter = audio_filters[1]
        assert "[3:a]volume=-12.0dB" in music_filter
        assert "afade=t=in:st=0:d=2.0" in music_filter
        assert "afade=t=out:st=27.000:d=3.0" in music_filter  # 30-3=27
        assert "[a_music_proc]" in music_filter

        # Check mix filter
        mix_filter = audio_filters[2]
        assert (
            mix_filter
            == "[a_voice_proc][a_music_proc]amix=inputs=2:duration=longest[a_mixed]"
        )

    def test_build_audio_filters_voiceover_only(self):
        """Test building audio filters with only voiceover."""
        voiceover_idx = 1
        music_idx = None
        total_duration = 20.0

        audio_filters, final_label = self.assembler._build_audio_filters(
            voiceover_idx, music_idx, total_duration
        )

        assert len(audio_filters) == 1
        assert final_label == "[a_voice_proc]"
        assert "[1:a]volume=" in audio_filters[0]

    def test_build_audio_filters_no_audio(self):
        """Test building audio filters with no audio inputs."""
        audio_filters, final_label = self.assembler._build_audio_filters(
            None, None, 15.0
        )

        assert audio_filters == []
        assert final_label == ""

    def test_build_ffmpeg_command_with_audio(self):
        """Test building complete FFmpeg command with audio."""
        input_cmd_parts = ["-i", "video.mp4", "-i", "audio.wav"]
        video_filters = ["[0:v]scale=1920:1080[v_out]"]
        audio_filters = ["[1:a]volume=-3dB[a_proc]"]
        final_audio_label = "[a_proc]"
        total_duration = 25.5
        output_path = Path("output.mp4")

        # Set specific settings for testing - need ffmpeg_path
        self.mock_config.ffmpeg_settings.executable_path = "ffmpeg"
        self.mock_config.ffmpeg_settings.rw_timeout_microseconds = 30000000
        self.mock_config.audio_settings.output_audio_codec = "aac"
        self.mock_config.audio_settings.output_audio_bitrate = "128k"
        self.mock_config.video_settings.output_codec = "libx264"
        self.mock_config.video_settings.output_preset = "medium"
        self.mock_config.video_settings.output_pixel_format = "yuv420p"
        self.mock_config.video_settings.frame_rate = 30

        cmd = self.assembler._build_ffmpeg_command(
            input_cmd_parts,
            video_filters,
            audio_filters,
            final_audio_label,
            total_duration,
            output_path,
        )

        # Check that we get a valid command list
        assert isinstance(cmd, list)
        assert len(cmd) > 10

        # Convert command elements to strings for verification
        cmd_str = " ".join(str(x) for x in cmd)

        # Verify essential FFmpeg parameters
        assert "ffmpeg" in cmd_str
        assert "-y" in cmd_str
        assert "-rw_timeout" in cmd_str
        assert "30000000" in cmd_str
        assert "video.mp4" in cmd_str
        assert "audio.wav" in cmd_str
        assert "-filter_complex" in cmd_str
        assert "[0:v]scale=1920:1080[v_out];[1:a]volume=-3dB[a_proc]" in cmd_str
        assert "-map" in cmd_str
        assert "[v_out]" in cmd_str
        assert "[a_proc]" in cmd_str
        assert "-c:a" in cmd_str
        assert "aac" in cmd_str
        assert "-b:a" in cmd_str
        assert "128k" in cmd_str
        assert "-c:v" in cmd_str
        assert "libx264" in cmd_str
        assert "-preset" in cmd_str
        assert "medium" in cmd_str
        assert "-pix_fmt" in cmd_str
        assert "yuv420p" in cmd_str
        assert "-r" in cmd_str
        assert "30" in cmd_str
        assert "-t" in cmd_str
        assert "25.5" in cmd_str
        assert str(output_path) in cmd_str

    def test_build_ffmpeg_command_video_only(self):
        """Test building FFmpeg command with video only (no audio)."""
        input_cmd_parts = ["-i", "video.mp4"]
        video_filters = ["[0:v]scale=1920:1080[v_out]"]
        audio_filters = []
        final_audio_label = ""
        total_duration = 10.0
        output_path = Path("output.mp4")

        cmd = self.assembler._build_ffmpeg_command(
            input_cmd_parts,
            video_filters,
            audio_filters,
            final_audio_label,
            total_duration,
            output_path,
        )

        # Convert to string for verification
        cmd_str = " ".join(str(x) for x in cmd)

        # Should not contain audio mapping or codec options
        assert "-map [a_proc]" not in cmd_str
        assert "-c:a" not in cmd_str
        assert "-b:a" not in cmd_str

        # Should contain video options
        assert "-map" in cmd_str
        assert "[v_out]" in cmd_str
        assert "-c:v" in cmd_str

    def test_should_create_ffmpeg_logs_true(self):
        """Test FFmpeg log creation when enabled in config."""
        self.mock_config.debug_settings = {"create_ffmpeg_command_logs": True}

        result = self.assembler._should_create_ffmpeg_logs()
        assert result is True

    def test_should_create_ffmpeg_logs_false(self):
        """Test FFmpeg log creation when disabled in config."""
        self.mock_config.debug_settings = {"create_ffmpeg_command_logs": False}

        result = self.assembler._should_create_ffmpeg_logs()
        assert result is False

    def test_should_create_ffmpeg_logs_no_debug_settings(self):
        """Test FFmpeg log creation when no debug settings exist."""
        # Remove debug_settings attribute
        if hasattr(self.mock_config, "debug_settings"):
            delattr(self.mock_config, "debug_settings")

        result = self.assembler._should_create_ffmpeg_logs()
        assert result is True  # Should default to True

    def test_should_create_ffmpeg_logs_exception_handling(self):
        """Test FFmpeg log creation with exception in config access."""
        # Make debug_settings access raise an exception
        self.mock_config.debug_settings = MagicMock()
        self.mock_config.debug_settings.get.side_effect = Exception("Config error")

        result = self.assembler._should_create_ffmpeg_logs()
        assert result is True  # Should fallback to True

    def test_music_fade_calculation_edge_cases(self):
        """Test music fade calculation with edge cases."""
        music_idx = 1

        # Test when fade out duration is longer than total duration
        self.mock_config.audio_settings.music_fade_out_duration = 10.0
        total_duration = 5.0  # Shorter than fade duration

        audio_filters, _ = self.assembler._build_audio_filters(
            None, music_idx, total_duration
        )

        # Fade out should start at 0 (max of 0 and negative value)
        music_filter = audio_filters[0]
        assert "afade=t=out:st=0.000:d=10.0" in music_filter

    def test_audio_filter_integration(self):
        """Test integration of audio filters with realistic settings."""
        voiceover_idx = 2
        music_idx = 3
        total_duration = 45.0

        # Set realistic audio settings
        self.mock_config.audio_settings.voiceover_volume_db = -1.5
        self.mock_config.audio_settings.music_volume_db = -15.0
        self.mock_config.audio_settings.music_fade_in_duration = 1.5
        self.mock_config.audio_settings.music_fade_out_duration = 2.5
        self.mock_config.audio_settings.audio_mix_duration = "first"

        audio_filters, final_label = self.assembler._build_audio_filters(
            voiceover_idx, music_idx, total_duration
        )

        assert len(audio_filters) == 3
        assert final_label == "[a_mixed]"

        # Verify realistic fade timing: 45 - 2.5 = 42.5
        assert "afade=t=out:st=42.500:d=2.5" in audio_filters[1]
        assert "amix=inputs=2:duration=first[a_mixed]" in audio_filters[2]


# Integration tests
class TestAssemblerIntegration:
    """Integration tests for decomposed assembler functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = MagicMock()

        # Create mock settings
        mock_audio = MagicMock()
        mock_audio.voiceover_volume_db = -1.5
        mock_audio.music_volume_db = -15.0
        mock_audio.music_fade_in_duration = 1.5
        mock_audio.music_fade_out_duration = 2.5
        mock_audio.audio_mix_duration = "first"
        self.mock_config.audio_settings = mock_audio

        self.mock_config.video_settings = MagicMock()
        self.mock_config.ffmpeg_settings = MagicMock()
        self.mock_config.ffmpeg_settings.executable_path = "ffmpeg"
        self.assembler = VideoAssembler(self.mock_config)

    def test_complete_audio_pipeline_integration(self):
        """Test the complete audio processing pipeline."""
        # Setup inputs
        input_cmd_parts = ["-i", "visual.mp4"]
        voiceover_path = Path("voice.wav")
        music_path = Path("bgm.mp3")
        total_duration = 30.0
        output_path = Path("final.mp4")

        # Execute audio preparation
        voiceover_idx, music_idx = self.assembler._prepare_audio_inputs(
            input_cmd_parts, voiceover_path, music_path, 1
        )

        # Execute filter building
        audio_filters, final_audio_label = self.assembler._build_audio_filters(
            voiceover_idx, music_idx, total_duration
        )

        # Execute command building
        video_filters = ["[0:v]scale=1920:1080[v_out]"]
        final_cmd = self.assembler._build_ffmpeg_command(
            input_cmd_parts,
            video_filters,
            audio_filters,
            final_audio_label,
            total_duration,
            output_path,
        )

        # Verify integration results
        assert len(input_cmd_parts) == 6  # 1 visual + 2 audio = 3 inputs * 2 parts each
        assert voiceover_idx == 1
        assert music_idx == 2
        assert len(audio_filters) == 3  # voice + music + mix
        assert final_audio_label == "[a_mixed]"

        # Verify command contains all expected parts
        cmd_str = " ".join(str(x) for x in final_cmd)
        assert str(voiceover_path) in cmd_str
        assert str(music_path) in cmd_str
        assert "[a_mixed]" in cmd_str
        assert str(output_path) in cmd_str
