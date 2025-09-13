"""Integration tests for cleanup functionality using real configuration."""

import tempfile
import time
from pathlib import Path

import pytest

from src.video.video_config import load_video_config


class TestCleanupIntegration:
    """Integration tests for cleanup with real configuration."""

    @pytest.fixture
    def temp_config_and_outputs(self):
        """Create temporary config and outputs directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a test config file
            config_content = f"""
global_output_directory: "{temp_dir}/outputs"

directory_structure:
  videos_base: "videos"
  video_project_pattern: "{{videos_base}}/{{product_id}}/{{profile_name}}"
  working_files: "temp"
  final_outputs: ""
  audio_assets: "audio"
  visual_assets: "media"
  text_assets: "text"
  scraper_data_base: "data"
  scraper_run_pattern: "{{scraper_data_base}}/{{platform}}/{{run_id}}"
  product_data: "products"
  scraped_media: "media"
  scraper_debug: "debug"
  logs_base: "logs"
  video_producer_logs: "producer"
  scraper_logs: "scraper"
  general_logs: "app"
  temp_files: "tmp"
  cache_files: "cache"

file_patterns:
  final_video: "{{product_id}}_{{profile_name}}.mp4"
  pipeline_state: "pipeline_state.json"
  ffmpeg_log: "{{product_id}}_{{profile_name}}_ffmpeg.log"
  voiceover: "voiceover.wav"
  background_music: "music.{{ext}}"
  subtitles: "subtitles.{{ext}}"
  script: "script.txt"
  attribution: "attributions.txt"
  metadata: "{{name}}.json"
  timestamped_log: "{{component}}_{{timestamp}}.log"
  main_log: "{{component}}.log"

cleanup_settings:
  enabled: true
  dry_run: false
  max_age_days: 1
  preserve_patterns:
    - "*.md"
    - "*.txt"
    - "cache/**"
  force_cleanup_patterns:
    - "*.tmp"
    - "*.temp"
    - ".DS_Store"
  cleanup_empty_dirs: true
  create_report: true
  report_file: "cleanup_report.json"

pipeline_timeout_sec: 900

# Minimal required settings for testing
video_settings:
  resolution: [1080, 1920]
  frame_rate: 30
  output_codec: "libx264"

media_settings:
  stock_media_keywords: ["test"]
  stock_video_min_duration_sec: 5
  stock_video_max_duration_sec: 30

audio_settings:
  music_volume_db: -20.0
  voiceover_volume_db: 3.0
  background_music_paths: []
  freesound_api_key_env_var: "FREESOUND_API_KEY"
  freesound_search_query: "background music"
  freesound_filters: "duration:[10 TO *]"
  freesound_max_results: 10

tts_config:
  provider_order: ["test"]

llm_settings:
  provider: "test"
  models: ["test"]
  api_key_env_var: "LLM_API_KEY"
  prompt_template_path: "prompts/test.txt"

stock_media_settings:
  source: "test"
  pexels_api_key_env_var: "PEXELS_API_KEY"

ffmpeg_settings:
  executable_path: ""

attribution_settings:
  attribution_file_name: "ATTRIBUTIONS.txt"
  attribution_template: "Attribution Template"
  attribution_entry_template: "Entry Template"

subtitle_settings:
  enabled: true

video_profiles:
  test_profile:
    description: "Test profile"
    use_scraped_images: true

whisper_settings:
  enabled: false

google_cloud_stt_settings:
  enabled: false

api_settings:
  default_request_timeout_sec: 15

text_processing:
  script_chars_per_second_estimate: 15

audio_processing:
  min_audio_file_size_bytes: 100

video_processing:
  min_frame_count: 1

filesystem:
  max_filename_length: 255

debug_settings:
  max_log_line_length: 200
"""

            config_file = temp_path / "test_config.yaml"
            config_file.write_text(config_content)

            # Create outputs directory
            outputs_dir = temp_path / "outputs"
            outputs_dir.mkdir()

            yield config_file, outputs_dir

    @pytest.mark.integration
    def test_cleanup_with_real_config(self, temp_config_and_outputs):
        """Test cleanup using a real configuration file."""
        config_file, outputs_dir = temp_config_and_outputs

        # Create test file structure
        self._create_test_files(outputs_dir)

        # Load config and run cleanup
        config = load_video_config(config_file)
        result = config.cleanup_outputs_directory(dry_run=False)

        # Verify cleanup worked
        assert result["statistics"]["files_removed"] > 0

        # Check that force cleanup files were removed
        assert not (outputs_dir / "temp_file.tmp").exists()
        assert not (outputs_dir / ".DS_Store").exists()

        # Check that preserved files remain
        assert (outputs_dir / "README.md").exists()
        assert (outputs_dir / "config.txt").exists()

        # Check that expected structure remains
        assert (outputs_dir / "videos/TEST123/profile1/TEST123_profile1.mp4").exists()
        assert (outputs_dir / "data/amazon/run1/products/product.json").exists()

    @pytest.mark.integration
    def test_cleanup_dry_run_with_real_config(self, temp_config_and_outputs):
        """Test dry-run cleanup using real configuration."""
        config_file, outputs_dir = temp_config_and_outputs

        # Create test files
        self._create_test_files(outputs_dir)

        # Count files before cleanup
        files_before = list(outputs_dir.rglob("*"))
        files_before_count = len([f for f in files_before if f.is_file()])

        # Load config and run dry-run cleanup
        config = load_video_config(config_file)
        result = config.cleanup_outputs_directory(dry_run=True)

        # Verify no files were actually removed
        files_after = list(outputs_dir.rglob("*"))
        files_after_count = len([f for f in files_after if f.is_file()])

        assert files_after_count == files_before_count
        assert result["dry_run"] is True
        assert result["statistics"]["files_removed"] == 0

        # But should have identified files for cleanup
        assert len(result["actions"]) > 0

    @pytest.mark.integration
    def test_age_based_cleanup(self, temp_config_and_outputs):
        """Test that age-based cleanup works correctly."""
        config_file, outputs_dir = temp_config_and_outputs

        # Load config first to get the max_age_days setting
        config = load_video_config(config_file)
        max_age_days = config.cleanup_settings.max_age_days

        # Create old file
        old_file = outputs_dir / "old_file.xyz"
        old_file.write_text("old content")

        # Make it old
        import os

        old_time = time.time() - (max_age_days + 1) * 24 * 3600
        os.utime(old_file, (old_time, old_time))

        # Create new file
        new_file = outputs_dir / "new_file.xyz"
        new_file.write_text("new content")

        # Run cleanup
        config.cleanup_outputs_directory(dry_run=False)

        # Old file should be removed, new file should remain
        assert not old_file.exists()
        assert new_file.exists()

    @pytest.mark.integration
    def test_report_generation(self, temp_config_and_outputs):
        """Test that cleanup reports are generated."""
        config_file, outputs_dir = temp_config_and_outputs

        # Create test files
        self._create_test_files(outputs_dir)

        # Run cleanup
        config = load_video_config(config_file)
        result = config.cleanup_outputs_directory(dry_run=False)

        # Check that report file was created
        report_file = outputs_dir / "cleanup_report.json"
        assert report_file.exists()

        # Verify report content
        import json

        with report_file.open() as f:
            report_data = json.load(f)

        assert "timestamp" in report_data
        assert "statistics" in report_data
        assert "actions" in report_data
        assert report_data["statistics"] == result["statistics"]

    def _create_test_files(self, outputs_dir):
        """Helper to create test file structure."""
        # Expected structure
        (outputs_dir / "videos/TEST123/profile1").mkdir(parents=True)
        (outputs_dir / "videos/TEST123/profile1/TEST123_profile1.mp4").write_text(
            "video"
        )

        # Product-centric structure (current)
        (outputs_dir / "B0BTYCRJSS/output").mkdir(parents=True)
        (outputs_dir / "B0BTYCRJSS/output/data.json").write_text("{}")

        (outputs_dir / "logs/producer").mkdir(parents=True)
        (outputs_dir / "logs/producer/app.log").write_text("logs")

        # Files to preserve
        (outputs_dir / "README.md").write_text("# README")
        (outputs_dir / "config.txt").write_text("config")
        (outputs_dir / "cache/models").mkdir(parents=True)
        (outputs_dir / "cache/models/model.bin").write_text("model")

        # Files to force cleanup
        (outputs_dir / "temp_file.tmp").write_text("temp")
        (outputs_dir / "backup.temp").write_text("backup")
        (outputs_dir / ".DS_Store").write_text("")

        # Unexpected files
        (outputs_dir / "unexpected.xyz").write_text("unexpected")
        (outputs_dir / "random_dir").mkdir()
        (outputs_dir / "random_dir/file.dat").write_text("random")


if __name__ == "__main__":
    pytest.main([__file__])
