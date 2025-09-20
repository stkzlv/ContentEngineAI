"""Unit tests for video configuration module."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.video.video_config import (
    AudioSettings,
    LLMSettings,
    SubtitleSettings,
    TTSConfig,
    VideoConfig,
    VideoProfile,
    VideoSettings,
    load_video_config,
)


class TestVideoProfile:
    """Test VideoProfile model."""

    def test_video_profile_valid(self):
        """Test valid video profile creation."""
        profile = VideoProfile(
            description="Test profile",
            use_scraped_images=True,
            use_scraped_videos=False,
            use_stock_images=True,
            use_stock_videos=False,
            stock_image_count=5,
            stock_video_count=2,
            use_dynamic_image_count=False,
        )

        assert profile.description == "Test profile"
        assert profile.use_scraped_images is True
        assert profile.use_scraped_videos is False
        assert profile.stock_image_count == 5
        assert profile.stock_video_count == 2

    def test_video_profile_defaults(self):
        """Test video profile with default values."""
        profile = VideoProfile(
            description="Test profile",
            use_scraped_images=False,
            use_scraped_videos=False,
            use_stock_images=False,
            use_stock_videos=False,
            stock_image_count=0,
            stock_video_count=0,
            use_dynamic_image_count=False,
        )

        assert profile.use_scraped_images is False
        assert profile.use_scraped_videos is False
        assert profile.use_stock_images is False
        assert profile.use_stock_videos is False
        assert profile.stock_image_count == 0
        assert profile.stock_video_count == 0

    def test_video_profile_invalid_stock_count(self):
        """Test video profile with invalid stock counts."""
        with pytest.raises(ValidationError):
            VideoProfile(
                description="Test profile",
                use_scraped_images=False,
                use_scraped_videos=False,
                use_stock_images=False,
                use_stock_videos=False,
                stock_image_count=-1,
                stock_video_count=0,
                use_dynamic_image_count=False,
            )


class TestSubtitleSettings:
    """Test SubtitleSettings model."""

    def test_subtitle_settings_valid(self):
        """Test valid subtitle settings creation."""
        settings = SubtitleSettings(
            enabled=True,
            positioning_mode="absolute",
            font_name="Arial",
            font_directory="static/fonts",
            font_size_percent=0.05,
            font_width_to_height_ratio=0.5,
            font_color="&H00FFFFFF",
            outline_color="&HFF000000",
            back_color="&H99000000",
            alignment=2,
            margin_v_percent=0.05,
            relative_positioning=None,
            absolute_positioning=None,
            use_random_font=False,
            use_random_colors=False,
            available_fonts=["Arial"],
            available_color_combinations=[("&H00FFFFFF", "&HFF000000")],
            temp_subtitle_dir="temp",
            temp_subtitle_filename="subtitle.srt",
            save_srt_with_video=True,
            subtitle_format="srt",
            script_paths=["info/script.txt"],
            max_subtitle_duration=4.5,
            max_line_length=40,
            min_subtitle_duration=0.5,
            subtitle_split_on_punctuation=True,
            punctuation_marks=[".", "!", "?", ";", ":", ","],
            subtitle_similarity_threshold=0.70,
            subtitle_overlap_threshold=65.0,
            word_timestamp_pause_threshold=0.4,
            bold=True,
            outline_thickness=1,
            shadow=True,
        )

        assert settings.enabled is True
        assert settings.positioning_mode == "absolute"
        assert settings.font_name == "Arial"
        assert settings.font_size_percent == 0.05

    def test_subtitle_settings_invalid_positioning_mode_warning(self):
        """Test subtitle settings with invalid positioning mode - should log
        warning but not raise error.
        """
        # This should now succeed but log a deprecation warning
        settings = SubtitleSettings(
            enabled=True,
            positioning_mode="invalid_mode",  # Should log warning but not fail
            font_name="Arial",
            font_directory="static/fonts",
            font_size_percent=0.05,
            font_width_to_height_ratio=0.5,
            font_color="&H00FFFFFF",
            outline_color="&HFF000000",
            back_color="&H99000000",
            alignment=2,
            margin_v_percent=0.05,
            relative_positioning=None,
            absolute_positioning=None,
            use_random_font=False,
            use_random_colors=False,
            available_fonts=["Arial"],
            available_color_combinations=[("&H00FFFFFF", "&HFF000000")],
            temp_subtitle_dir="temp",
            temp_subtitle_filename="subtitle.srt",
            save_srt_with_video=True,
            subtitle_format="srt",
            script_paths=["info/script.txt"],
            max_subtitle_duration=4.5,
            max_line_length=40,
            min_subtitle_duration=0.5,
            subtitle_split_on_punctuation=True,
            punctuation_marks=[".", "!", "?", ";", ":", ","],
            subtitle_similarity_threshold=0.70,
            subtitle_overlap_threshold=65.0,
            word_timestamp_pause_threshold=0.4,
            bold=True,
            outline_thickness=1,
            shadow=True,
        )
        # Should succeed with deprecated mode
        assert settings.positioning_mode == "invalid_mode"

    def test_subtitle_settings_defaults(self):
        """Test subtitle settings with default values."""
        settings = SubtitleSettings()  # type: ignore[call-arg]

        assert settings.enabled is True
        assert settings.positioning_mode == "absolute"
        assert settings.font_name == "Arial"
        assert settings.font_size_percent == 0.05


class TestVideoSettings:
    """Test VideoSettings model."""

    def test_video_settings_valid(self):
        """Test valid video settings creation."""
        settings = VideoSettings(
            resolution=(1080, 1920),
            frame_rate=30,
            output_codec="libx264",
            output_pixel_format="yuv420p",
            image_width_percent=0.8,
            image_top_position_percent=0.05,
            default_image_duration_sec=3.0,
            transition_duration_sec=0.5,
            total_duration_limit_sec=90,
            video_duration_tolerance_sec=1.0,
            min_video_file_size_mb=0.1,
            inter_product_delay_min_sec=1.5,
            inter_product_delay_max_sec=4.0,
            min_visual_segment_duration_sec=0.1,
            dynamic_image_count_limit=25,
            verification_probe_timeout_sec=30,
            preserve_aspect_ratio=True,
            default_max_chars_per_line=20,
            subtitle_box_border_width=5,
            image_loop=1,
            pad_color="black",
        )

        assert settings.resolution == (1080, 1920)
        assert settings.frame_rate == 30
        assert settings.output_codec == "libx264"
        assert settings.image_width_percent == 0.8

    def test_video_settings_invalid_resolution(self):
        """Test video settings with invalid resolution."""
        with pytest.raises(ValidationError):
            VideoSettings(
                resolution=(0, 1920),  # Invalid width
                frame_rate=30,
            )


class TestAudioSettings:
    """Test AudioSettings model."""

    def test_audio_settings_valid(self):
        """Test valid audio settings creation."""
        settings = AudioSettings(
            music_volume_db=-20.0,
            voiceover_volume_db=0.0,
            audio_mix_duration="longest",
            background_music_paths=[],
            freesound_api_key_env_var="FREESOUND_API_KEY",
            freesound_client_id_env_var="FREESOUND_CLIENT_ID",
            freesound_client_secret_env_var="FREESOUND_CLIENT_SECRET",  # noqa: S106
            freesound_refresh_token_env_var="FREESOUND_REFRESH_TOKEN",  # noqa: S106
            freesound_sort="rating_desc",
            freesound_search_query="background music",
            freesound_filters="type:wav duration:[5 TO 30]",
            freesound_max_results=10,
            freesound_max_search_duration_sec=9999,
            output_audio_codec="aac",
            output_audio_bitrate="192k",
            music_fade_in_duration=2.0,
            music_fade_out_duration=3.0,
        )

        assert settings.music_volume_db == -20.0
        assert settings.voiceover_volume_db == 0.0
        assert settings.audio_mix_duration == "longest"
        assert settings.freesound_api_key_env_var == "FREESOUND_API_KEY"

    def test_audio_settings_defaults(self):
        """Test audio settings with default values."""
        settings = AudioSettings(
            music_volume_db=-20.0,
            voiceover_volume_db=0.0,
            background_music_paths=[],
            freesound_api_key_env_var="FREESOUND_API_KEY",
            freesound_sort="rating_desc",
            freesound_search_query="background music",
            freesound_filters="type:wav duration:[5 TO 30]",
            freesound_max_results=10,
        )

        assert settings.audio_mix_duration == "longest"
        assert settings.output_audio_codec == "aac"
        assert settings.output_audio_bitrate == "192k"


class TestTTSConfig:
    """Test TTSConfig model."""

    def test_tts_config_valid(self):
        """Test valid TTS config creation."""
        config = TTSConfig(
            provider_order=["google_cloud", "coqui"],
            google_cloud={
                "model_name": "en-US-Chirp3-HD",
                "language_code": "en-US",
                "voice_selection_criteria": [
                    {"language_code": "en-US", "ssml_gender": "FEMALE"},
                ],
                "speaking_rate": 1.0,
                "pitch": 0.0,
                "volume_gain_db": 0.0,
                "debug": False,
                "api_timeout_sec": 60,
                "api_max_retries": 2,
                "api_retry_delay_sec": 5,
                "last_word_buffer_sec": 0.3,
            },
            coqui={
                "model_name": "tts_models/en/ljspeech/tacotron2-DDC",
                "speaker_name": None,
            },
        )

        assert config.provider_order == ["google_cloud", "coqui"]
        assert config.google_cloud is not None
        assert config.coqui is not None

    def test_tts_config_empty_provider_order(self):
        """Test TTS config with empty provider order."""
        with pytest.raises(ValidationError):
            TTSConfig(provider_order=[])

    def test_tts_config_missing_provider_settings(self):
        """Test TTS config with missing provider settings.

        Should filter out unavailable providers.
        """
        # This should no longer raise an error, instead filter out unavailable providers
        config = TTSConfig(
            provider_order=["google_cloud"],
            # Missing google_cloud settings
        )
        # Should result in empty provider list since google_cloud is not configured
        assert config.provider_order == []


class TestLLMSettings:
    """Test LLMSettings model."""

    def test_llm_settings_valid(self):
        """Test valid LLM settings creation."""
        settings = LLMSettings(
            provider="openrouter",
            api_key_env_var="OPENROUTER_API_KEY",
            models=["gpt-3.5-turbo", "gpt-4"],
            prompt_template_path="src/ai/prompts/product_script.txt",
            target_audience="General audience",
            base_url="https://openrouter.ai/api/v1",
            auto_select_free_model=True,
            max_tokens=350,
            temperature=0.7,
            timeout_seconds=90,
        )

        assert settings.provider == "openrouter"
        assert settings.api_key_env_var == "OPENROUTER_API_KEY"
        assert settings.models == ["gpt-3.5-turbo", "gpt-4"]
        assert settings.max_tokens == 350

    def test_llm_settings_empty_models(self):
        """Test LLM settings with empty models list."""
        with pytest.raises(ValidationError):
            LLMSettings(
                provider="openrouter",
                api_key_env_var="OPENROUTER_API_KEY",
                models=[],  # Empty models list
                prompt_template_path="src/ai/prompts/product_script.txt",
            )

    def test_llm_settings_defaults(self):
        """Test LLM settings with default values."""
        settings = LLMSettings(
            provider="openrouter",
            api_key_env_var="OPENROUTER_API_KEY",
            models=["gpt-3.5-turbo"],
            prompt_template_path="src/ai/prompts/product_script.txt",
        )

        assert settings.target_audience == "General audience"
        assert settings.max_tokens == 350
        assert settings.temperature == 0.7
        assert settings.timeout_seconds == 90


class TestVideoConfig:
    """Test VideoConfig model."""

    def test_video_config_valid(self, temp_dir: Path):
        """Test valid video config creation."""
        config_data = {
            "global_output_directory": str(temp_dir / "outputs"),
            "video_runs_subdirectory": "video_files",
            "video_intermediate_files_subdirectory": "working_files",
            "video_producer_general_logs_subdirectory": "logs",
            "video_settings": {
                "resolution": [1080, 1920],
                "frame_rate": 30,
                "output_codec": "libx264",
                "output_pixel_format": "yuv420p",
                "output_preset": "ultrafast",
                "image_width_percent": 0.8,
                "image_top_position_percent": 0.05,
                "default_image_duration_sec": 3.0,
                "transition_duration_sec": 0.5,
                "total_duration_limit_sec": 90,
                "video_duration_tolerance_sec": 1.0,
                "min_video_file_size_mb": 0.1,
                "inter_product_delay_min_sec": 1.5,
                "inter_product_delay_max_sec": 4.0,
                "min_visual_segment_duration_sec": 0.1,
                "dynamic_image_count_limit": 25,
                "verification_probe_timeout_sec": 30,
                "preserve_aspect_ratio": True,
                "default_max_chars_per_line": 20,
                "subtitle_box_border_width": 5,
                "image_loop": 1,
                "pad_color": "black",
            },
            "media_settings": {
                "stock_media_keywords": ["product", "lifestyle"],
                "stock_video_min_duration_sec": 5,
                "stock_video_max_duration_sec": 30,
                "temp_media_dir": "downloaded_media_assets",
                "product_title_keyword_min_length": 3,
            },
            "audio_settings": {
                "music_volume_db": -20.0,
                "voiceover_volume_db": 0.0,
                "audio_mix_duration": "longest",
                "background_music_paths": [],
                "freesound_api_key_env_var": "FREESOUND_API_KEY",
                "freesound_client_id_env_var": "FREESOUND_CLIENT_ID",
                "freesound_client_secret_env_var": "FREESOUND_CLIENT_SECRET",  # noqa: S106
                "freesound_refresh_token_env_var": "FREESOUND_REFRESH_TOKEN",  # noqa: S106
                "freesound_sort": "rating_desc",
                "freesound_search_query": "background music",
                "freesound_filters": "type:wav duration:[5 TO 30]",
                "freesound_max_results": 10,
                "freesound_max_search_duration_sec": 9999,
                "output_audio_codec": "aac",
                "output_audio_bitrate": "192k",
                "music_fade_in_duration": 2.0,
                "music_fade_out_duration": 3.0,
            },
            "tts_config": {
                "provider_order": ["google_cloud", "coqui"],
                "google_cloud": {
                    "model_name": "en-US-Chirp3-HD",
                    "language_code": "en-US",
                    "voice_selection_criteria": [
                        {"language_code": "en-US", "ssml_gender": "FEMALE"},
                    ],
                    "speaking_rate": 1.0,
                    "pitch": 0.0,
                    "volume_gain_db": 0.0,
                    "debug": False,
                    "api_timeout_sec": 60,
                    "api_max_retries": 2,
                    "api_retry_delay_sec": 5,
                    "last_word_buffer_sec": 0.3,
                },
                "coqui": {
                    "model_name": "tts_models/en/ljspeech/tacotron2-DDC",
                    "speaker_name": None,
                },
            },
            "llm_settings": {
                "provider": "openrouter",
                "api_key_env_var": "OPENROUTER_API_KEY",
                "models": ["gpt-3.5-turbo", "gpt-4"],
                "prompt_template_path": "src/ai/prompts/product_script.txt",
                "target_audience": "General audience",
                "base_url": "https://openrouter.ai/api/v1",
                "auto_select_free_model": True,
                "max_tokens": 350,
                "temperature": 0.7,
                "timeout_seconds": 90,
            },
            "stock_media_settings": {
                "pexels_api_key_env_var": "PEXELS_API_KEY",
                "source": "Pexels",
            },
            "ffmpeg_settings": {
                "executable_path": None,
                "temp_ffmpeg_dir": "ffmpeg_work",
                "intermediate_segment_preset": "ultrafast",
                "final_assembly_timeout_sec": 300,
                "rw_timeout_microseconds": 30000000,
            },
            "attribution_settings": {
                "attribution_file_name": "ATTRIBUTIONS.txt",
                "attribution_template": (
                    "Video created with ContentEngineAI\n\nThird-party media used:"
                ),
                "attribution_entry_template": (
                    "- {name} by {author} ({license}) - {url}"
                ),
            },
            "subtitle_settings": {
                "enabled": True,
                "positioning_mode": "absolute",
                "font_name": "Arial",
                "font_directory": "static/fonts",
                "font_size_percent": 0.05,
                "font_width_to_height_ratio": 0.5,
                "font_color": "&H00FFFFFF",
                "outline_color": "&HFF000000",
                "alignment": 2,
                "margin_v_percent": 0.05,
                "use_random_font": False,
                "use_random_colors": False,
                "available_fonts": ["Montserrat", "Rubik", "Poppins", "Gabarito"],
                "available_color_combinations": [
                    ["&H00FFFFFF", "&HFF000000"],
                    ["&H0000FFFF", "&HFF000000"],
                ],
                "temp_subtitle_dir": "subtitle_processing",
                "temp_subtitle_filename": "captions.srt",
                "save_srt_with_video": True,
                "subtitle_format": "srt",
                "script_paths": ["info/script.txt"],
                "max_subtitle_duration": 4.5,
                "max_line_length": 38,
                "min_subtitle_duration": 0.4,
                "subtitle_split_on_punctuation": True,
                "punctuation_marks": [".", "!", "?", ";", ":", ","],
                "subtitle_similarity_threshold": 0.70,
                "subtitle_overlap_threshold": 65.0,
                "word_timestamp_pause_threshold": 0.4,
                "bold": True,
                "outline_thickness": 1,
                "shadow": True,
            },
            "whisper_settings": {
                "enabled": True,
                "model_size": "small",
                "model_device": "cpu",
                "model_in_memory": False,
                "model_download_root": "",
                "temperature": 0.0,
                "language": "en",
                "beam_size": 5,
                "fp16": False,
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.4,
                "condition_on_previous_text": True,
                "task": "transcribe",
                "patience": None,
            },
            "google_cloud_stt_settings": {
                "enabled": True,
                "language_code": "en-US",
                "encoding": "LINEAR16",
                "sample_rate_hertz": 24000,
                "use_enhanced": True,
                "api_timeout_sec": 120,
                "api_max_retries": 2,
                "api_retry_delay_sec": 10,
                "use_speech_adaptation_if_script_provided": True,
                "adaptation_boost_value": 15.0,
            },
            "description_settings": {
                "enabled": True,
                "prompt_template_path": "src/ai/prompts/video_description.md",
                "target_platforms": ["tiktok", "youtube", "instagram"],
                "max_tokens": 200,
                "min_description_chars": 50,
                "min_description_words": 10,
                "require_hashtags": True,
                "require_ad_hashtag": True,
            },
            "video_profiles": {
                "test_profile": {
                    "description": "Test profile for unit testing",
                    "use_scraped_images": True,
                    "use_scraped_videos": False,
                    "use_stock_images": True,
                    "use_stock_videos": False,
                    "stock_image_count": 2,
                    "stock_video_count": 0,
                    "use_dynamic_image_count": False,
                }
            },
        }

        config = VideoConfig(**config_data)

        assert config.global_output_directory == str(temp_dir / "outputs")
        assert config.video_settings.resolution == (1080, 1920)
        assert config.audio_settings.music_volume_db == -20.0
        assert "test_profile" in config.video_profiles

    def test_video_config_get_profile(self, mock_config: VideoConfig):
        """Test getting a video profile from config."""
        profile = mock_config.get_profile("test_profile")

        assert profile.description == "Test profile for unit testing"
        assert profile.use_scraped_images is True
        assert profile.stock_image_count == 2

    def test_video_config_get_nonexistent_profile(self, mock_config: VideoConfig):
        """Test getting a non-existent profile."""
        with pytest.raises(KeyError):
            mock_config.get_profile("nonexistent_profile")


class TestLoadVideoConfig:
    """Test video config loading functionality."""

    def test_load_video_config_valid_file(self, temp_dir: Path):
        """Test loading valid config from file."""
        config_data = {
            "global_output_directory": str(temp_dir / "outputs"),
            "video_runs_subdirectory": "video_files",
            "video_intermediate_files_subdirectory": "working_files",
            "video_producer_general_logs_subdirectory": "logs",
            "video_settings": {
                "resolution": [1080, 1920],
                "frame_rate": 30,
                "output_codec": "libx264",
                "output_pixel_format": "yuv420p",
                "output_preset": "ultrafast",
                "image_width_percent": 0.8,
                "image_top_position_percent": 0.05,
                "default_image_duration_sec": 3.0,
                "transition_duration_sec": 0.5,
                "total_duration_limit_sec": 90,
                "video_duration_tolerance_sec": 1.0,
                "min_video_file_size_mb": 0.1,
                "inter_product_delay_min_sec": 1.5,
                "inter_product_delay_max_sec": 4.0,
                "min_visual_segment_duration_sec": 0.1,
                "dynamic_image_count_limit": 25,
                "verification_probe_timeout_sec": 30,
                "preserve_aspect_ratio": True,
                "default_max_chars_per_line": 20,
                "subtitle_box_border_width": 5,
                "image_loop": 1,
                "pad_color": "black",
            },
            "media_settings": {
                "stock_media_keywords": ["product", "lifestyle"],
                "stock_video_min_duration_sec": 5,
                "stock_video_max_duration_sec": 30,
                "temp_media_dir": "downloaded_media_assets",
                "product_title_keyword_min_length": 3,
            },
            "audio_settings": {
                "music_volume_db": -20.0,
                "voiceover_volume_db": 0.0,
                "audio_mix_duration": "longest",
                "background_music_paths": [],
                "freesound_api_key_env_var": "FREESOUND_API_KEY",
                "freesound_client_id_env_var": "FREESOUND_CLIENT_ID",
                "freesound_client_secret_env_var": "FREESOUND_CLIENT_SECRET",  # noqa: S106
                "freesound_refresh_token_env_var": "FREESOUND_REFRESH_TOKEN",  # noqa: S106
                "freesound_sort": "rating_desc",
                "freesound_search_query": "background music",
                "freesound_filters": "type:wav duration:[5 TO 30]",
                "freesound_max_results": 10,
                "freesound_max_search_duration_sec": 9999,
                "output_audio_codec": "aac",
                "output_audio_bitrate": "192k",
                "music_fade_in_duration": 2.0,
                "music_fade_out_duration": 3.0,
            },
            "tts_config": {
                "provider_order": ["google_cloud", "coqui"],
                "google_cloud": {
                    "model_name": "en-US-Chirp3-HD",
                    "language_code": "en-US",
                    "voice_selection_criteria": [
                        {"language_code": "en-US", "ssml_gender": "FEMALE"},
                    ],
                    "speaking_rate": 1.0,
                    "pitch": 0.0,
                    "volume_gain_db": 0.0,
                    "debug": False,
                    "api_timeout_sec": 60,
                    "api_max_retries": 2,
                    "api_retry_delay_sec": 5,
                    "last_word_buffer_sec": 0.3,
                },
                "coqui": {
                    "model_name": "tts_models/en/ljspeech/tacotron2-DDC",
                    "speaker_name": None,
                },
            },
            "llm_settings": {
                "provider": "openrouter",
                "api_key_env_var": "OPENROUTER_API_KEY",
                "models": ["gpt-3.5-turbo", "gpt-4"],
                "prompt_template_path": "src/ai/prompts/product_script.txt",
                "target_audience": "General audience",
                "base_url": "https://openrouter.ai/api/v1",
                "auto_select_free_model": True,
                "max_tokens": 350,
                "temperature": 0.7,
                "timeout_seconds": 90,
            },
            "stock_media_settings": {
                "pexels_api_key_env_var": "PEXELS_API_KEY",
                "source": "Pexels",
            },
            "ffmpeg_settings": {
                "executable_path": None,
                "temp_ffmpeg_dir": "ffmpeg_work",
                "intermediate_segment_preset": "ultrafast",
                "final_assembly_timeout_sec": 300,
                "rw_timeout_microseconds": 30000000,
            },
            "attribution_settings": {
                "attribution_file_name": "ATTRIBUTIONS.txt",
                "attribution_template": (
                    "Video created with ContentEngineAI\n\nThird-party media used:"
                ),
                "attribution_entry_template": (
                    "- {name} by {author} ({license}) - {url}"
                ),
            },
            "subtitle_settings": {
                "enabled": True,
                "positioning_mode": "absolute",
                "font_name": "Arial",
                "font_directory": "static/fonts",
                "font_size_percent": 0.05,
                "font_width_to_height_ratio": 0.5,
                "font_color": "&H00FFFFFF",
                "outline_color": "&HFF000000",
                "alignment": 2,
                "margin_v_percent": 0.05,
                "use_random_font": False,
                "use_random_colors": False,
                "available_fonts": ["Montserrat", "Rubik", "Poppins", "Gabarito"],
                "available_color_combinations": [
                    ["&H00FFFFFF", "&HFF000000"],
                    ["&H0000FFFF", "&HFF000000"],
                ],
                "temp_subtitle_dir": "subtitle_processing",
                "temp_subtitle_filename": "captions.srt",
                "save_srt_with_video": True,
                "subtitle_format": "srt",
                "script_paths": ["info/script.txt"],
                "max_subtitle_duration": 4.5,
                "max_line_length": 38,
                "min_subtitle_duration": 0.4,
                "subtitle_split_on_punctuation": True,
                "punctuation_marks": [".", "!", "?", ";", ":", ","],
                "subtitle_similarity_threshold": 0.70,
                "subtitle_overlap_threshold": 65.0,
                "word_timestamp_pause_threshold": 0.4,
                "bold": True,
                "outline_thickness": 1,
                "shadow": True,
            },
            "whisper_settings": {
                "enabled": True,
                "model_size": "small",
                "model_device": "cpu",
                "model_in_memory": False,
                "model_download_root": "",
                "temperature": 0.0,
                "language": "en",
                "beam_size": 5,
                "fp16": False,
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.4,
                "condition_on_previous_text": True,
                "task": "transcribe",
                "patience": None,
            },
            "google_cloud_stt_settings": {
                "enabled": True,
                "language_code": "en-US",
                "encoding": "LINEAR16",
                "sample_rate_hertz": 24000,
                "use_enhanced": True,
                "api_timeout_sec": 120,
                "api_max_retries": 2,
                "api_retry_delay_sec": 10,
                "use_speech_adaptation_if_script_provided": True,
                "adaptation_boost_value": 15.0,
            },
            "description_settings": {
                "enabled": True,
                "prompt_template_path": "src/ai/prompts/video_description.md",
                "target_platforms": ["tiktok", "youtube", "instagram"],
                "max_tokens": 200,
                "min_description_chars": 50,
                "min_description_words": 10,
                "require_hashtags": True,
                "require_ad_hashtag": True,
            },
            "video_profiles": {
                "test_profile": {
                    "description": "Test profile for unit testing",
                    "use_scraped_images": True,
                    "use_scraped_videos": False,
                    "use_stock_images": True,
                    "use_stock_videos": False,
                    "stock_image_count": 2,
                    "stock_video_count": 0,
                    "use_dynamic_image_count": False,
                }
            },
        }

        config_file = temp_dir / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_video_config(config_file)

        assert config.global_output_directory == str(temp_dir / "outputs")
        assert config.video_settings.resolution == (1080, 1920)
        assert "test_profile" in config.video_profiles

    def test_load_video_config_invalid_file(self, temp_dir: Path):
        """Test loading config from non-existent file."""
        config_file = temp_dir / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            load_video_config(config_file)

    def test_load_video_config_invalid_yaml(self, temp_dir: Path):
        """Test loading config from invalid YAML file."""
        config_file = temp_dir / "invalid_config.yaml"
        config_file.write_text("invalid: yaml: content: [")

        with pytest.raises(ValueError):  # Function wraps YAML errors in ValueError
            load_video_config(config_file)
