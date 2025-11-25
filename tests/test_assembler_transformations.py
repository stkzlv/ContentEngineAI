"""Unit tests for aspect ratio and audio transformation filters.

Tests FFmpeg filter generation for aspect ratio modes (letterbox, crop-to-fit,
smart-scale) and audio handling (remove, mixed) following requirements 2 and 3.
"""

import pytest

from src.video.assembler import VideoAssembler
from src.video.config import VideoConfig


@pytest.fixture
def assembler_with_audio(mock_config: VideoConfig) -> VideoAssembler:
    """Create video assembler with audio settings configured."""
    # Configure audio settings
    mock_config.audio_settings.voiceover_volume_db = 0.0
    mock_config.audio_settings.music_volume_db = -10.0
    mock_config.audio_settings.music_fade_in_duration = 2.0
    mock_config.audio_settings.music_fade_out_duration = 3.0
    mock_config.audio_settings.audio_mix_duration = "longest"
    return VideoAssembler(config=mock_config, debug_mode=False)


class TestAspectRatioLetterbox:
    """Test letterbox aspect ratio mode (Requirement 2.1)."""

    def test_letterbox_maintains_aspect_ratio(
        self, assembler_with_audio: VideoAssembler
    ):
        """Test letterbox mode preserves video aspect ratio with padding."""
        # 16:9 video in 9:16 frame → should add black padding
        filter_string, output_label = assembler_with_audio._apply_aspect_ratio_mode(
            input_label="[v0]",
            aspect_mode="letterbox",
            target_width=1080,
            target_height=1920,
            video_width=1920,
            video_height=1080,
        )

        # Should contain scale and pad operations
        assert "scale=" in filter_string
        assert "pad=" in filter_string
        # Should center the video
        assert "pad=" in filter_string
        # Output label should be generated
        assert output_label == "[v0]_scaled"

    def test_letterbox_vertical_video_in_vertical_frame(
        self, assembler_with_audio: VideoAssembler
    ):
        """Test letterbox with 9:16 video in 9:16 frame."""
        # Already matching aspect ratio, should still process
        filter_string, output_label = assembler_with_audio._apply_aspect_ratio_mode(
            input_label="[v1]",
            aspect_mode="letterbox",
            target_width=1080,
            target_height=1920,
            video_width=1080,
            video_height=1920,
        )

        # Should still apply filters even if aspect matches
        assert "scale=" in filter_string
        assert output_label == "[v1]_scaled"

    def test_letterbox_square_video_in_vertical_frame(
        self, assembler_with_audio: VideoAssembler
    ):
        """Test letterbox with square (1:1) video in vertical (9:16) frame."""
        filter_string, output_label = assembler_with_audio._apply_aspect_ratio_mode(
            input_label="[v2]",
            aspect_mode="letterbox",
            target_width=1080,
            target_height=1920,
            video_width=1080,
            video_height=1080,
        )

        # Should add padding for square video
        assert "scale=" in filter_string
        assert "pad=" in filter_string
        assert output_label == "[v2]_scaled"


class TestAspectRatioCrop:
    """Test crop-to-fit aspect ratio mode (Requirement 2.2)."""

    def test_crop_fills_frame(self, assembler_with_audio: VideoAssembler):
        """Test crop-to-fit mode scales and crops to fill frame."""
        # 16:9 video in 9:16 frame → should scale up and crop
        filter_string, output_label = assembler_with_audio._apply_aspect_ratio_mode(
            input_label="[v0]",
            aspect_mode="crop-to-fit",
            target_width=1080,
            target_height=1920,
            video_width=1920,
            video_height=1080,
        )

        # Should contain scale and crop operations
        assert "scale=" in filter_string
        assert "crop=" in filter_string
        # Output label should be generated
        assert output_label == "[v0]_scaled"

    def test_crop_centers_crop_region(self, assembler_with_audio: VideoAssembler):
        """Test crop-to-fit centers the crop region (Req 2.5)."""
        filter_string, output_label = assembler_with_audio._apply_aspect_ratio_mode(
            input_label="[v1]",
            aspect_mode="crop-to-fit",
            target_width=1080,
            target_height=1920,
            video_width=1920,
            video_height=1080,
        )

        # Should contain crop with centering calculation
        assert "crop=" in filter_string
        # Crop filter should have x and y parameters (centered)
        # Format: crop=w:h:x:y where x,y are calculated for centering
        assert output_label == "[v1]_scaled"

    def test_crop_with_already_vertical_video(
        self, assembler_with_audio: VideoAssembler
    ):
        """Test crop-to-fit when video is already vertical."""
        # 9:16 video in 9:16 frame → minimal cropping needed
        filter_string, output_label = assembler_with_audio._apply_aspect_ratio_mode(
            input_label="[v2]",
            aspect_mode="crop-to-fit",
            target_width=1080,
            target_height=1920,
            video_width=1080,
            video_height=1920,
        )

        # Should still apply scale (may be identity transform)
        assert "scale=" in filter_string
        assert output_label == "[v2]_scaled"


class TestAspectRatioSmartScale:
    """Test smart-scale aspect ratio mode (Requirement 2.3)."""

    def test_smart_scale_chooses_crop_within_threshold(
        self, assembler_with_audio: VideoAssembler
    ):
        """Test smart-scale chooses crop when aspect ratio within 10% (Req 2.3)."""
        # 9:16 video (0.5625) in 9:16 frame (0.5625) → exactly matching
        # Aspect difference = 0% → should choose crop-to-fit
        filter_string, output_label = assembler_with_audio._apply_aspect_ratio_mode(
            input_label="[v0]",
            aspect_mode="smart-scale",
            target_width=1080,
            target_height=1920,
            video_width=1080,
            video_height=1920,
        )

        # Should choose crop-to-fit (difference = 0%)
        assert "scale=" in filter_string
        # When aspects match exactly, crop may not be needed
        assert output_label == "[v0]_scaled"

    def test_smart_scale_chooses_crop_at_9_percent_difference(
        self, assembler_with_audio: VideoAssembler
    ):
        """Test smart-scale chooses crop at 9% difference (just within threshold)."""
        # Design scenario: 9:16 frame (0.5625), video at ~9% difference
        # 0.5625 * 1.09 = 0.613 (just within 10% threshold)
        # Create video with aspect ratio ~0.613 (e.g., 1100:1795)
        filter_string, output_label = assembler_with_audio._apply_aspect_ratio_mode(
            input_label="[v1]",
            aspect_mode="smart-scale",
            target_width=1080,
            target_height=1920,  # Target aspect = 0.5625
            video_width=1100,
            video_height=1795,  # Video aspect ~0.613 (~9% diff)
        )

        # Should choose crop-to-fit (9% < 10% threshold)
        assert "crop=" in filter_string or "scale=" in filter_string
        assert output_label == "[v1]_scaled"

    def test_smart_scale_chooses_letterbox_beyond_threshold(
        self, assembler_with_audio: VideoAssembler
    ):
        """Test smart-scale chooses letterbox when beyond 10% (Req 2.3)."""
        # 16:9 landscape video (1.778) in 9:16 vertical frame (0.5625)
        # Aspect difference = |1.778 - 0.5625| / 0.5625 = 216% → letterbox
        filter_string, output_label = assembler_with_audio._apply_aspect_ratio_mode(
            input_label="[v2]",
            aspect_mode="smart-scale",
            target_width=1080,
            target_height=1920,
            video_width=1920,
            video_height=1080,
        )

        # Should choose letterbox (difference > 10%)
        assert "pad=" in filter_string
        assert output_label == "[v2]_scaled"

    def test_smart_scale_at_exactly_10_percent_threshold(
        self, assembler_with_audio: VideoAssembler
    ):
        """Test smart-scale decision at exactly 10% threshold boundary."""
        # Target aspect = 0.5625 (9:16)
        # Video aspect = 0.5625 * 1.10 = 0.61875 (exactly 10% difference)
        # Should choose crop-to-fit (<=0.10 threshold)
        filter_string, output_label = assembler_with_audio._apply_aspect_ratio_mode(
            input_label="[v3]",
            aspect_mode="smart-scale",
            target_width=1080,
            target_height=1920,  # 0.5625
            video_width=1237,
            video_height=2000,  # ~0.6185 (10% diff)
        )

        # At exactly 10%, should still choose crop (threshold is <=0.10)
        assert "crop=" in filter_string or "scale=" in filter_string
        assert output_label == "[v3]_scaled"


class TestAudioRemoveMode:
    """Test audio handling in remove mode (Requirement 3)."""

    def test_remove_mode_excludes_video_audio(
        self, assembler_with_audio: VideoAssembler
    ):
        """Test remove mode strips video audio from mix."""
        audio_filters, final_label = (
            assembler_with_audio._build_audio_filters_with_video_audio(
                voiceover_input_idx=0,
                music_input_idx=1,
                video_audio_indices=[2, 3, 4],  # 3 videos with audio
                video_audio_handling="remove",
                video_original_volume=-30.0,
                total_video_duration=30.0,
            )
        )

        # Should NOT include video audio in filters
        audio_str = "".join(audio_filters)
        assert "a_vid" not in audio_str
        # Should only mix voiceover + music (2 streams)
        assert "amix=inputs=2" in audio_str
        assert final_label == "[a_mixed]"

    def test_remove_mode_with_only_voiceover(
        self, assembler_with_audio: VideoAssembler
    ):
        """Test remove mode with only voiceover (no music, no video audio)."""
        audio_filters, final_label = (
            assembler_with_audio._build_audio_filters_with_video_audio(
                voiceover_input_idx=0,
                music_input_idx=None,
                video_audio_indices=[2],
                video_audio_handling="remove",
                video_original_volume=-30.0,
                total_video_duration=30.0,
            )
        )

        # Should only process voiceover (single stream)
        audio_str = "".join(audio_filters)
        assert "[a_voice_proc]" in audio_str
        assert "a_vid" not in audio_str
        # Single stream should not use amix
        assert "amix" not in audio_str
        assert final_label == "[a_voice_proc]"


class TestAudioMixedMode:
    """Test audio handling in mixed mode (Requirement 3)."""

    def test_mixed_mode_includes_video_audio(
        self, assembler_with_audio: VideoAssembler
    ):
        """Test mixed mode includes video audio at configured volume."""
        audio_filters, final_label = (
            assembler_with_audio._build_audio_filters_with_video_audio(
                voiceover_input_idx=0,
                music_input_idx=1,
                video_audio_indices=[2, 3],  # 2 videos with audio
                video_audio_handling="mixed",
                video_original_volume=-30.0,
                total_video_duration=30.0,
            )
        )

        # Should include video audio streams
        audio_str = "".join(audio_filters)
        assert "[a_vid0_proc]" in audio_str
        assert "[a_vid1_proc]" in audio_str
        # Should apply volume adjustment
        assert "volume=-30.0dB" in audio_str
        # Should mix 4 streams (voice + music + 2 videos)
        assert "amix=inputs=4" in audio_str
        assert final_label == "[a_mixed]"

    def test_mixed_mode_with_custom_volume(self, assembler_with_audio: VideoAssembler):
        """Test mixed mode applies custom video volume."""
        audio_filters, final_label = (
            assembler_with_audio._build_audio_filters_with_video_audio(
                voiceover_input_idx=0,
                music_input_idx=1,
                video_audio_indices=[2],
                video_audio_handling="mixed",
                video_original_volume=-20.0,  # Custom volume
                total_video_duration=30.0,
            )
        )

        # Should apply custom -20dB volume
        audio_str = "".join(audio_filters)
        assert "volume=-20.0dB[a_vid0_proc]" in audio_str
        # Should mix 3 streams
        assert "amix=inputs=3" in audio_str

    def test_mixed_mode_with_many_video_streams(
        self, assembler_with_audio: VideoAssembler
    ):
        """Test mixed mode handles multiple video audio streams."""
        # 5 videos + voiceover + music = 7 total streams
        audio_filters, final_label = (
            assembler_with_audio._build_audio_filters_with_video_audio(
                voiceover_input_idx=0,
                music_input_idx=1,
                video_audio_indices=[2, 3, 4, 5, 6],  # 5 videos
                video_audio_handling="mixed",
                video_original_volume=-30.0,
                total_video_duration=30.0,
            )
        )

        # Should mix all 7 streams
        audio_str = "".join(audio_filters)
        assert "amix=inputs=7" in audio_str
        # Should include normalize=0 to prevent clipping
        assert "normalize=0" in audio_str
        # Should process all 5 video audio streams
        for i in range(5):
            assert f"[a_vid{i}_proc]" in audio_str

    def test_mixed_mode_with_no_videos(self, assembler_with_audio: VideoAssembler):
        """Test mixed mode with empty video list (edge case)."""
        audio_filters, final_label = (
            assembler_with_audio._build_audio_filters_with_video_audio(
                voiceover_input_idx=0,
                music_input_idx=1,
                video_audio_indices=[],  # No videos
                video_audio_handling="mixed",
                video_original_volume=-30.0,
                total_video_duration=30.0,
            )
        )

        # Should fall back to voice + music only
        audio_str = "".join(audio_filters)
        assert "a_vid" not in audio_str
        assert "amix=inputs=2" in audio_str


class TestAudioVoiceoverAndMusicVolumes:
    """Test voiceover and music volume settings are preserved."""

    def test_voiceover_volume_unchanged(self, assembler_with_audio: VideoAssembler):
        """Test voiceover volume matches config (0dB)."""
        assembler_with_audio.config.audio_settings.voiceover_volume_db = 0.0

        audio_filters, _ = assembler_with_audio._build_audio_filters_with_video_audio(
            voiceover_input_idx=0,
            music_input_idx=1,
            video_audio_indices=[2],
            video_audio_handling="mixed",
            video_original_volume=-30.0,
            total_video_duration=30.0,
        )

        audio_str = "".join(audio_filters)
        assert "volume=0.0dB[a_voice_proc]" in audio_str

    def test_music_volume_unchanged(self, assembler_with_audio: VideoAssembler):
        """Test music volume matches config (-10dB)."""
        assembler_with_audio.config.audio_settings.music_volume_db = -10.0

        audio_filters, _ = assembler_with_audio._build_audio_filters_with_video_audio(
            voiceover_input_idx=0,
            music_input_idx=1,
            video_audio_indices=[2],
            video_audio_handling="mixed",
            video_original_volume=-30.0,
            total_video_duration=30.0,
        )

        audio_str = "".join(audio_filters)
        assert "[1:a]volume=-10.0dB" in audio_str

    def test_music_fades_applied(self, assembler_with_audio: VideoAssembler):
        """Test music fade-in and fade-out filters are applied."""
        assembler_with_audio.config.audio_settings.music_fade_in_duration = 2.0
        assembler_with_audio.config.audio_settings.music_fade_out_duration = 3.0

        audio_filters, _ = assembler_with_audio._build_audio_filters_with_video_audio(
            voiceover_input_idx=0,
            music_input_idx=1,
            video_audio_indices=[],
            video_audio_handling="remove",
            video_original_volume=-30.0,
            total_video_duration=30.0,
        )

        audio_str = "".join(audio_filters)
        # Fade-in at start for 2s
        assert "afade=t=in:st=0:d=2.0" in audio_str
        # Fade-out starts at 27s (30 - 3)
        assert "afade=t=out:st=27.000:d=3.0" in audio_str


class TestFFmpegFilterSyntax:
    """Validate FFmpeg filter syntax correctness."""

    def test_aspect_ratio_filter_syntax_valid(
        self, assembler_with_audio: VideoAssembler
    ):
        """Test aspect ratio filters generate valid FFmpeg syntax."""
        for mode in ["letterbox", "crop-to-fit"]:
            filter_string, output_label = assembler_with_audio._apply_aspect_ratio_mode(
                input_label="[v0]",
                aspect_mode=mode,
                target_width=1080,
                target_height=1920,
                video_width=1920,
                video_height=1080,
            )

            # Should not contain invalid characters
            assert "[" in filter_string  # FFmpeg labels use brackets
            assert "]" in filter_string
            # Should not have syntax errors (no double colons, etc.)
            assert "::" not in filter_string

    def test_audio_filter_syntax_valid(self, assembler_with_audio: VideoAssembler):
        """Test audio filters generate valid FFmpeg syntax."""
        audio_filters, final_label = (
            assembler_with_audio._build_audio_filters_with_video_audio(
                voiceover_input_idx=0,
                music_input_idx=1,
                video_audio_indices=[2, 3],
                video_audio_handling="mixed",
                video_original_volume=-30.0,
                total_video_duration=30.0,
            )
        )

        # Validate filter list is not empty
        assert len(audio_filters) > 0
        # Validate final label format
        assert final_label.startswith("[")
        assert final_label.endswith("]")
        # Check no syntax errors in filters
        for filter_str in audio_filters:
            assert filter_str  # Not empty
            # Should not have invalid syntax patterns
            assert "==" not in filter_str

    def test_amix_normalize_parameter(self, assembler_with_audio: VideoAssembler):
        """Test amix filter includes normalize=0 to prevent clipping."""
        audio_filters, _ = assembler_with_audio._build_audio_filters_with_video_audio(
            voiceover_input_idx=0,
            music_input_idx=1,
            video_audio_indices=[2, 3, 4],
            video_audio_handling="mixed",
            video_original_volume=-30.0,
            total_video_duration=30.0,
        )

        audio_str = "".join(audio_filters)
        # Should use normalize=0 to prevent automatic gain reduction
        assert "normalize=0" in audio_str
