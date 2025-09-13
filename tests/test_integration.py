"""Integration tests for ContentEngineAI components."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.scraper.amazon.scraper import ProductData
from src.video.producer import (
    PipelineContext,
    _update_state_after_step,
    get_video_run_paths,
    load_visuals_info,
    save_visuals_info,
    step_assemble_video,
    step_create_voiceover,
    step_download_music,
    step_gather_visuals,
    step_generate_script,
    step_generate_subtitles,
)
from src.video.stock_media import StockMediaInfo
from src.video.video_config import VideoConfig, VideoProfile


class TestPipelineIntegration:
    """Test integration between pipeline components."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_pipeline_context_creation(
        self,
        sample_product_data: ProductData,
        sample_video_profile: VideoProfile,
        mock_config: VideoConfig,
        mock_aiohttp_session: AsyncMock,
    ):
        """Test pipeline context creation and initialization."""
        secrets = {"OPENROUTER_API_KEY": "test_key"}
        run_paths = get_video_run_paths(mock_config, "B08TEST123", "test_profile")

        context = PipelineContext(
            product=sample_product_data,
            profile=sample_video_profile,
            config=mock_config,
            secrets=secrets,
            session=mock_aiohttp_session,
            run_paths=run_paths,
            debug_mode=True,
        )

        assert context.product == sample_product_data
        assert context.profile == sample_video_profile
        assert context.config == mock_config
        assert context.secrets == secrets
        assert context.session == mock_aiohttp_session
        assert context.debug_mode is True
        assert context.visuals is None
        assert context.script is None
        assert context.voiceover_duration is None

    @pytest.mark.integration
    def test_video_run_paths_generation(self, mock_config: VideoConfig):
        """Test video run paths generation."""
        paths = get_video_run_paths(mock_config, "B08TEST123", "test_profile")

        assert "run_root" in paths
        assert "intermediate_base" in paths
        assert "assets_dir" in paths
        assert "info_dir" in paths
        assert "script_file" in paths
        assert "voiceover_file" in paths
        assert "subtitle_file" in paths
        assert "final_video_output" in paths
        assert "attribution_file" in paths

        # Check that paths are properly structured
        assert "B08TEST123" in str(paths["run_root"])
        assert paths["assets_dir"].parent == paths["intermediate_base"]
        assert paths["info_dir"].parent == paths["intermediate_base"]

    def test_visuals_info_save_load(self, temp_dir: Path):
        """Test saving and loading visuals information."""
        # Create test data
        scraped_imgs = [Path("/path/to/image1.jpg"), Path("/path/to/image2.jpg")]
        scraped_vids = [Path("/path/to/video1.mp4")]
        stock_media = [
            StockMediaInfo(
                url="https://example.com/stock1.jpg",
                path=Path("/path/to/stock1.jpg"),
                type="image",
                source="Pexels",
                author="test_author",
            )
        ]

        run_paths = {"gathered_visuals_file": temp_dir / "visuals.json"}

        # Save visuals info
        save_visuals_info(scraped_imgs, scraped_vids, stock_media, run_paths)

        # Verify file was created
        assert run_paths["gathered_visuals_file"].exists()

        # Load visuals info
        loaded_imgs, loaded_vids, loaded_stock = load_visuals_info(
            run_paths["gathered_visuals_file"]
        )

        # Verify data was preserved
        assert len(loaded_imgs) == 2
        assert len(loaded_vids) == 1
        assert len(loaded_stock) == 1

        assert str(loaded_imgs[0]) == "/path/to/image1.jpg"
        assert str(loaded_vids[0]) == "/path/to/video1.mp4"
        assert loaded_stock[0].source == "Pexels"

    @pytest.mark.asyncio
    async def test_step_gather_visuals_integration(
        self,
        sample_product_data: ProductData,
        sample_video_profile: VideoProfile,
        mock_config: VideoConfig,
        mock_aiohttp_session: AsyncMock,
        temp_dir: Path,
    ):
        """Test gather visuals step integration."""
        secrets = {"PEXELS_API_KEY": "test_key"}
        run_paths = get_video_run_paths(mock_config, "B08TEST123", "test_profile")

        context = PipelineContext(
            product=sample_product_data,
            profile=sample_video_profile,
            config=mock_config,
            secrets=secrets,
            session=mock_aiohttp_session,
            run_paths=run_paths,
            debug_mode=True,
        )

        # Mock stock media fetcher
        with patch("src.video.producer.StockMediaFetcher") as mock_fetcher_class:
            mock_fetcher = AsyncMock()
            mock_fetcher_class.return_value = mock_fetcher

            # Mock successful stock media fetch
            mock_stock_media = [
                StockMediaInfo(
                    url="https://example.com/stock1.jpg",
                    path=temp_dir / "stock1.jpg",
                    type="image",
                    source="Pexels",
                    author="test_author",
                )
            ]
            mock_fetcher.fetch_stock_media.return_value = mock_stock_media

            # Create some test image files
            (temp_dir / "image1.jpg").write_bytes(b"fake image data")
            (temp_dir / "image2.jpg").write_bytes(b"fake image data")
            (temp_dir / "stock1.jpg").write_bytes(b"fake stock image data")

            # Mock product images as local files
            sample_product_data.downloaded_images = [
                str(temp_dir / "image1.jpg"),
                str(temp_dir / "image2.jpg"),
            ]

            await step_gather_visuals(context)

            # Verify visuals were gathered
            assert context.visuals is not None
            assert len(context.visuals) > 0

            # Verify visuals info was saved
            assert run_paths["gathered_visuals_file"].exists()

    @pytest.mark.asyncio
    async def test_step_generate_script_integration(
        self,
        sample_product_data: ProductData,
        sample_video_profile: VideoProfile,
        mock_config: VideoConfig,
        mock_aiohttp_session: AsyncMock,
        temp_dir: Path,
    ):
        """Test generate script step integration."""
        secrets = {"OPENROUTER_API_KEY": "test_key"}
        run_paths = get_video_run_paths(mock_config, "B08TEST123", "test_profile")

        context = PipelineContext(
            product=sample_product_data,
            profile=sample_video_profile,
            config=mock_config,
            secrets=secrets,
            session=mock_aiohttp_session,
            run_paths=run_paths,
            debug_mode=True,
        )

        # Mock script generation
        with patch("src.video.producer.generate_ai_script") as mock_generate:
            mock_generate.return_value = "Generated promotional script content."

            await step_generate_script(context)

            # Verify script was generated
            assert context.script == "Generated promotional script content."

            # Verify script was saved
            assert run_paths["script_file"].exists()
            assert (
                run_paths["script_file"].read_text()
                == "Generated promotional script content."
            )

    @pytest.mark.asyncio
    async def test_step_create_voiceover_integration(
        self,
        sample_product_data: ProductData,
        sample_video_profile: VideoProfile,
        mock_config: VideoConfig,
        mock_aiohttp_session: AsyncMock,
        temp_dir: Path,
    ):
        """Test voiceover creation step integration."""
        secrets: dict[str, str] = {}
        run_paths = get_video_run_paths(mock_config, "B08TEST123", "test_profile")

        context = PipelineContext(
            product=sample_product_data,
            profile=sample_video_profile,
            config=mock_config,
            secrets=secrets,
            session=mock_aiohttp_session,
            run_paths=run_paths,
            debug_mode=True,
        )

        # Set up context with required data
        context.script = "Test script for voiceover generation."

        # Create mock script file
        run_paths["script_file"].parent.mkdir(parents=True, exist_ok=True)
        run_paths["script_file"].write_text(context.script)

        # Mock TTS manager
        with patch("src.video.producer.TTSManager") as mock_tts_class:
            mock_tts = AsyncMock()
            mock_tts_class.return_value = mock_tts

            # Mock successful voiceover generation
            mock_tts.generate_speech.return_value = run_paths["voiceover_file"]

            # Create mock voiceover file in the correct location
            run_paths["voiceover_file"].parent.mkdir(parents=True, exist_ok=True)
            run_paths["voiceover_file"].write_bytes(b"fake audio data")

            # Mock _get_video_duration to return expected duration
            with patch("src.video.producer._get_video_duration", return_value=5.5):
                await step_create_voiceover(context)

                # Verify voiceover was created
                assert context.voiceover_duration == 5.5
                assert run_paths["voiceover_file"].exists()
                assert run_paths["voiceover_duration_file"].exists()
                assert float(run_paths["voiceover_duration_file"].read_text()) == 5.5

    @pytest.mark.asyncio
    async def test_step_generate_subtitles_integration(
        self,
        sample_product_data: ProductData,
        sample_video_profile: VideoProfile,
        mock_config: VideoConfig,
        mock_aiohttp_session: AsyncMock,
        temp_dir: Path,
    ):
        """Test subtitle generation step integration."""
        secrets: dict[str, str] = {}
        run_paths = get_video_run_paths(mock_config, "B08TEST123", "test_profile")

        context = PipelineContext(
            product=sample_product_data,
            profile=sample_video_profile,
            config=mock_config,
            secrets=secrets,
            session=mock_aiohttp_session,
            run_paths=run_paths,
            debug_mode=True,
        )

        # Set up context with required data
        context.script = "Test script for subtitle generation."
        context.voiceover_duration = 5.5

        # Create required files
        run_paths["voiceover_file"].parent.mkdir(parents=True, exist_ok=True)
        run_paths["script_file"].parent.mkdir(parents=True, exist_ok=True)
        run_paths["voiceover_duration_file"].parent.mkdir(parents=True, exist_ok=True)

        run_paths["voiceover_file"].write_bytes(b"fake audio data")
        run_paths["script_file"].write_text(context.script)
        run_paths["voiceover_duration_file"].write_text(str(context.voiceover_duration))

        # Mock subtitle generation
        with patch(
            "src.video.producer.create_unified_subtitles"
        ) as mock_create_subtitles:
            mock_create_subtitles.return_value = run_paths["subtitle_file"]

            # Create mock subtitle file in the correct location
            run_paths["subtitle_file"].parent.mkdir(parents=True, exist_ok=True)
            run_paths["subtitle_file"].write_text(
                "1\n00:00:00,000 --> 00:00:03,500\nTest subtitle"
            )

            await step_generate_subtitles(context)

            # Verify subtitles were created
            assert run_paths["subtitle_file"].exists()

    @pytest.mark.asyncio
    async def test_step_download_music_integration(
        self,
        sample_product_data: ProductData,
        sample_video_profile: VideoProfile,
        mock_config: VideoConfig,
        mock_aiohttp_session: AsyncMock,
        temp_dir: Path,
    ):
        """Test music download step integration."""
        secrets = {"FREESOUND_API_KEY": "test_key"}
        run_paths = get_video_run_paths(mock_config, "B08TEST123", "test_profile")

        context = PipelineContext(
            product=sample_product_data,
            profile=sample_video_profile,
            config=mock_config,
            secrets=secrets,
            session=mock_aiohttp_session,
            run_paths=run_paths,
            debug_mode=True,
        )

        # Set up context with required data
        context.voiceover_duration = 5.5

        # Create required files
        run_paths["voiceover_duration_file"].parent.mkdir(parents=True, exist_ok=True)
        run_paths["voiceover_duration_file"].write_text(str(context.voiceover_duration))

        # Mock Freesound client
        with patch("src.video.producer.FreesoundClient") as mock_fs_class:
            mock_fs = MagicMock()  # Use MagicMock instead of AsyncMock for sync methods
            mock_fs_class.return_value = mock_fs

            # Mock successful music search and download
            mock_track = MagicMock()
            mock_track.name = "Test Music"
            mock_track.duration = 10.0
            mock_fs.search_music.return_value = [mock_track]
            mock_fs.download_full_sound_oauth2 = AsyncMock(
                return_value=(temp_dir / "music.mp3", {"name": "Test Music"})
            )

            # Create mock music file
            (temp_dir / "music.mp3").write_bytes(b"fake music data")

            await step_download_music(context)

            # Verify music info was saved
            assert run_paths["music_info_file"].exists()

    @pytest.mark.asyncio
    async def test_step_assemble_video_integration(
        self,
        sample_product_data: ProductData,
        sample_video_profile: VideoProfile,
        mock_config: VideoConfig,
        mock_aiohttp_session: AsyncMock,
        temp_dir: Path,
    ):
        """Test assemble video step integration."""
        secrets: dict[str, str] = {}
        run_paths = get_video_run_paths(mock_config, "B08TEST123", "test_profile")

        context = PipelineContext(
            product=sample_product_data,
            profile=sample_video_profile,
            config=mock_config,
            secrets=secrets,
            session=mock_aiohttp_session,
            run_paths=run_paths,
            debug_mode=True,
        )

        # Set up context with all required data
        context.visuals = [temp_dir / "image1.jpg", temp_dir / "image2.jpg"]
        context.script = "Test script for video assembly."
        context.voiceover_duration = 5.5

        # Create mock files
        (temp_dir / "image1.jpg").write_bytes(b"fake image data")
        (temp_dir / "image2.jpg").write_bytes(b"fake image data")
        run_paths["voiceover_file"].parent.mkdir(parents=True, exist_ok=True)
        run_paths["voiceover_file"].write_bytes(b"fake audio data")
        (temp_dir / "music.mp3").write_bytes(b"fake music data")
        (temp_dir / "captions.srt").write_text(
            "1\n00:00:00,000 --> 00:00:03,500\nTest subtitle"
        )

        # Mock video assembler
        with patch("src.video.producer.VideoAssembler") as mock_assembler_class:
            mock_assembler = (
                MagicMock()
            )  # Use MagicMock instead of AsyncMock for verify_video
            mock_assembler_class.return_value = mock_assembler

            # Mock successful video assembly
            mock_assembler.assemble_video = AsyncMock(
                return_value=run_paths["final_video_output"]
            )

            # Mock verify_video to return success (this is a sync method)
            mock_assembler.verify_video.return_value = {
                "success": True,
                "message": "Video verified successfully.",
            }

            # Create mock final video in the correct location
            run_paths["final_video_output"].parent.mkdir(parents=True, exist_ok=True)
            run_paths["final_video_output"].write_bytes(b"fake video data")

            await step_assemble_video(context)

            # Verify final video was created
            assert run_paths["final_video_output"].exists()


class TestPipelineStateManagement:
    """Test pipeline state management and persistence."""

    async def test_pipeline_state_save_load(self, temp_dir: Path):
        """Test pipeline state saving and loading."""
        from src.video.producer import _load_pipeline_state, _save_pipeline_state

        # Create mock context
        context = MagicMock()
        context.state = {
            "step1": {"status": "completed", "data": "test_data"},
            "step2": {"status": "pending"},
        }
        context.run_paths = {"state_file": temp_dir / "pipeline_state.json"}

        # Save state
        await _save_pipeline_state(context)

        # Verify state file was created
        assert context.run_paths["state_file"].exists()

        # Load state
        result = await _load_pipeline_state(context)

        # Verify state was loaded correctly
        assert result is True
        assert context.state["step1"]["status"] == "completed"
        assert context.state["step1"]["data"] == "test_data"
        assert context.state["step2"]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_pipeline_state_load_nonexistent(self, temp_dir: Path):
        """Test loading pipeline state from non-existent file."""
        from src.video.producer import _load_pipeline_state

        # Create mock context with non-existent state file
        context = MagicMock()
        context.run_paths = {"state_file": temp_dir / "nonexistent.json"}
        context.state = {}

        # Load state
        result = await _load_pipeline_state(context)

        # Should return False for non-existent file
        assert result is False

    @pytest.mark.asyncio
    async def test_pipeline_state_update_after_step(
        self,
        sample_product_data: ProductData,
        sample_video_profile: VideoProfile,
        mock_config: VideoConfig,
        mock_aiohttp_session: AsyncMock,
    ):
        """Test that pipeline state is updated after each step."""
        secrets: dict[str, str] = {}
        run_paths = get_video_run_paths(mock_config, "B08TEST123", "test_profile")

        context = PipelineContext(
            product=sample_product_data,
            profile=sample_video_profile,
            config=mock_config,
            secrets=secrets,
            session=mock_aiohttp_session,
            run_paths=run_paths,
            debug_mode=False,
        )

        # Test state update for gather_visuals step
        await _update_state_after_step(context, "gather_visuals")
        assert "gather_visuals" in context.state
        assert context.state["gather_visuals"]["status"] == "done"
        assert "gathered_visuals_file" in context.state["gather_visuals"]["artifacts"]

        # Test state update for create_voiceover step
        await _update_state_after_step(context, "create_voiceover")
        assert "create_voiceover" in context.state
        assert context.state["create_voiceover"]["status"] == "done"
        assert "voiceover_file" in context.state["create_voiceover"]["artifacts"]
        assert (
            "voiceover_duration_file" in context.state["create_voiceover"]["artifacts"]
        )


class TestErrorHandlingIntegration:
    """Test error handling across pipeline components."""

    @pytest.mark.asyncio
    async def test_pipeline_error_propagation(
        self,
        sample_product_data: ProductData,
        sample_video_profile: VideoProfile,
        mock_config: VideoConfig,
        mock_aiohttp_session: AsyncMock,
        temp_dir: Path,
    ):
        """Test that errors are properly propagated through the pipeline."""
        secrets = {"OPENROUTER_API_KEY": "test_key"}
        run_paths = get_video_run_paths(mock_config, "B08TEST123", "test_profile")

        context = PipelineContext(
            product=sample_product_data,
            profile=sample_video_profile,
            config=mock_config,
            secrets=secrets,
            session=mock_aiohttp_session,
            run_paths=run_paths,
            debug_mode=True,
        )

        # Mock script generation to raise exception
        with patch("src.video.producer.generate_ai_script") as mock_generate:
            mock_generate.side_effect = Exception("API Error")

            # Should raise exception
            with pytest.raises(Exception, match="API Error"):
                await step_generate_script(context)

    @pytest.mark.asyncio
    async def test_pipeline_graceful_failure(
        self,
        sample_product_data: ProductData,
        sample_video_profile: VideoProfile,
        mock_config: VideoConfig,
        mock_aiohttp_session: AsyncMock,
        temp_dir: Path,
    ):
        """Test pipeline handles failures gracefully."""
        secrets: dict[str, str] = {}
        run_paths = get_video_run_paths(mock_config, "B08TEST123", "test_profile")

        context = PipelineContext(
            product=sample_product_data,
            profile=sample_video_profile,
            config=mock_config,
            secrets=secrets,
            session=mock_aiohttp_session,
            run_paths=run_paths,
            debug_mode=True,
        )

        # Set up context with required data
        context.script = "Test script for failure handling."
        context.voiceover_duration = 5.5

        # Create required files
        run_paths["script_file"].parent.mkdir(parents=True, exist_ok=True)
        run_paths["voiceover_duration_file"].parent.mkdir(parents=True, exist_ok=True)
        run_paths["script_file"].write_text(context.script)
        run_paths["voiceover_duration_file"].write_text(str(context.voiceover_duration))

        # Mock TTS manager to raise an exception
        with patch("src.video.producer.TTSManager") as mock_tts_class:
            mock_tts = AsyncMock()
            mock_tts_class.return_value = mock_tts
            mock_tts.generate_speech.side_effect = Exception("TTS generation failed")

            # Test that the step raises the expected exception
            with pytest.raises(Exception, match="TTS generation failed"):
                await step_create_voiceover(context)


class TestConfigurationIntegration:
    """Test configuration integration across components."""

    def test_configuration_validation_integration(self, temp_dir: Path):
        """Test that configuration validation works across components."""
        from src.video.video_config import load_video_config

        # Create minimal valid config
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
                "positioning_mode": "static",
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
        }

        # Create config file
        config_file = temp_dir / "test_config.yaml"
        import yaml

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Load config
        config = load_video_config(config_file)

        # Verify config was loaded correctly
        assert config.global_output_directory == str(temp_dir / "outputs")
        assert config.video_settings.resolution == (1080, 1920)
        assert config.audio_settings.music_volume_db == -20.0
        assert "test_profile" in config.video_profiles

        # Test profile access
        profile = config.get_profile("test_profile")
        assert profile.description == "Test profile for unit testing"
        assert profile.use_scraped_images is True
