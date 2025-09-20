"""Pytest configuration and shared fixtures for ContentEngineAI tests."""

import json
import logging
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
import yaml
from aioresponses import aioresponses

from src.scraper.amazon.scraper import ProductData
from src.scraper.base.models import Platform
from src.video.video_config import VideoConfig, VideoProfile, load_video_config


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_product_data() -> ProductData:
    """Sample product data for testing."""
    return ProductData(
        title="Test Product - Wireless Bluetooth Headphones",
        price="$99.99",
        description=(
            "High-quality wireless headphones with noise cancellation "
            "and 30-hour battery life."
        ),
        images=[
            "https://example.com/image1.jpg",
            "https://example.com/image2.jpg",
        ],
        videos=["https://example.com/video1.mp4"],
        affiliate_link="https://amazon.com/dp/B08TEST123",
        url="https://amazon.com/dp/B08TEST123",
        asin="B08TEST123",
        keyword="wireless headphones",
        serp_rating="4.5",
        serp_reviews_count="1,234",
        platform=Platform.AMAZON,
    )


@pytest.fixture
def sample_video_profile() -> VideoProfile:
    """Sample video profile for testing."""
    return VideoProfile(
        description="Test profile for unit testing",
        use_scraped_images=True,
        use_scraped_videos=False,
        use_stock_images=True,
        use_stock_videos=False,
        stock_image_count=2,
        stock_video_count=0,
        use_dynamic_image_count=False,
    )


@pytest.fixture
def mock_config(temp_dir: Path) -> VideoConfig:
    """Create a mock video configuration for testing."""
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
            "freesound_client_secret_env_var": "FREESOUND_CLIENT_SECRET",
            "freesound_refresh_token_env_var": "FREESOUND_REFRESH_TOKEN",
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
                    {"language_code": "en-US", "ssml_gender": "MALE"},
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
            "attribution_entry_template": "- {name} by {author} ({license}) - {url}",
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
        "api_settings": {
            "llm_model_fetch_timeout_sec": 15,
            "llm_retry_attempts": 3,
            "llm_retry_min_wait_sec": 2,
            "llm_retry_max_wait_sec": 30,
            "llm_retry_multiplier": 1,
            "stock_media_concurrent_downloads": 5,
            "stock_media_search_multiplier": 2,
            "stock_media_max_per_page": 80,
            "default_request_timeout_sec": 15,
            "default_retry_attempts": 3,
            "default_retry_delay_sec": 5,
        },
        "text_processing": {
            "script_chars_per_second_estimate": 15,
            "script_min_duration_sec": 0.05,
            "subtitle_text_similarity_min_confidence": 0.5,
            "subtitle_min_segment_duration_sec": 0.1,
            "subtitle_max_segment_duration_sec": 5.0,
            "subtitle_min_words_per_segment": 3,
            "subtitle_max_words_per_segment": 10,
            "subtitle_max_chars_per_line": 42,
            "subtitle_min_segment_gap_sec": 0.1,
        },
        "audio_processing": {
            "coqui_gpu_enabled": False,
            "google_tts_audio_encoding": "LINEAR16",
            "min_audio_file_size_bytes": 100,
            "audio_validation_timeout_sec": 30,
        },
        "video_processing": {
            "ffmpeg_probe_streams": "v:0",
            "ffmpeg_probe_entries": "stream=width,height",
            "ffmpeg_probe_format": "csv=s=x:p=0",
            "video_stream_check_timeout_sec": 30,
            "min_frame_count": 1,
            "visual_aspect_ratio_tolerance": 0.01,
            "visual_scaling_precision": 2,
        },
        "filesystem": {
            "temp_file_cleanup_delay_sec": 5,
            "file_operation_timeout_sec": 30,
            "max_filename_length": 255,
            "supported_image_extensions": [".jpg", ".jpeg", ".png", ".webp", ".bmp"],
            "supported_video_extensions": [".mp4", ".avi", ".mov", ".mkv", ".webm"],
            "supported_audio_extensions": [".wav", ".mp3", ".aac", ".flac"],
        },
        "debug_settings": {
            "max_log_line_length": 200,
            "debug_file_retention_days": 7,
            "intermediate_file_cleanup": True,
            "operation_timing_threshold_sec": 5.0,
            "memory_usage_warning_mb": 1000,
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
    }

    # Create config file
    config_file = temp_dir / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    return load_video_config(config_file)


@pytest.fixture
def mock_aiohttp_session() -> AsyncMock:
    """Mock aiohttp client session."""
    session = AsyncMock(spec=aiohttp.ClientSession)

    # Create a mock response object that will be returned by the async context manager
    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()  # raise_for_status is not async
    mock_response.json = AsyncMock(
        return_value={"choices": [{"message": {"content": "Mocked script content"}}]}
    )

    # Configure the session's get and post methods to return an awaitable context
    # manager
    # whose __aenter__ method returns our mock_response
    session.get.return_value.__aenter__.return_value = mock_response
    session.post.return_value.__aenter__.return_value = mock_response
    session.close = AsyncMock()
    return session


@pytest.fixture
def mock_aioresponses() -> Generator[aioresponses, None, None]:
    """Mock HTTP responses for testing."""
    with aioresponses() as m:
        yield m


@pytest.fixture
def sample_script() -> str:
    """Sample generated script for testing."""
    return (
        "Discover the amazing Test Product Wireless Bluetooth Headphones! "
        "With crystal clear sound quality and 30-hour battery life, these "
        "headphones are perfect for music lovers and professionals alike. "
        "Features include active noise cancellation, comfortable ear cushions, "
        "and seamless Bluetooth connectivity. Don't miss out on this incredible "
        "deal at just $99.99!"
    )


@pytest.fixture
def sample_srt_content() -> str:
    """Sample SRT subtitle content for testing."""
    return """1
00:00:00,000 --> 00:00:03,500
Discover the amazing Test Product

2
00:00:03,500 --> 00:00:07,000
Wireless Bluetooth Headphones!

3
00:00:07,000 --> 00:00:10,500
With crystal clear sound quality

4
00:00:10,500 --> 00:00:14,000
and 30-hour battery life
"""


@pytest.fixture
def mock_ffmpeg_path(temp_dir: Path) -> Path:
    """Create a mock FFmpeg executable for testing."""
    ffmpeg_path = temp_dir / "ffmpeg"
    ffmpeg_path.write_text("#!/bin/bash\necho 'FFmpeg mock'")
    ffmpeg_path.chmod(0o755)
    return ffmpeg_path


@pytest.fixture
def mock_google_cloud_credentials(temp_dir: Path) -> Path:
    """Create mock Google Cloud credentials file."""
    credentials_file = temp_dir / "google_credentials.json"
    credentials_data = {
        "type": "service_account",
        "project_id": "test-project",
        "private_key_id": "test-key-id",
        "private_key": (
            "-----BEGIN PRIVATE KEY-----\nMOCK_KEY\n-----END PRIVATE KEY-----\n"
        ),
        "client_email": "test@test-project.iam.gserviceaccount.com",
        "client_id": "123456789",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    credentials_file.write_text(json.dumps(credentials_data))
    return credentials_file


@pytest.fixture(autouse=True)
def setup_logging():
    """Set up logging for tests."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    env_vars = {
        "OPENROUTER_API_KEY": "test_openrouter_key",
        "PEXELS_API_KEY": "test_pexels_key",
        "FREESOUND_API_KEY": "test_freesound_key",
        "FREESOUND_CLIENT_ID": "test_client_id",
        "FREESOUND_CLIENT_SECRET": "test_client_secret",
        "FREESOUND_REFRESH_TOKEN": "test_refresh_token",
        "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/credentials.json",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars
