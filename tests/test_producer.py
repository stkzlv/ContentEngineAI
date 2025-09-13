"""Unit tests for the video producer pipeline orchestrator."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest

from src.scraper.amazon.scraper import ProductData
from src.video.producer import (
    PipelineContext,
    PipelineError,
    create_video_for_product,
    discover_products_for_batch,
    get_video_run_paths,
    step_assemble_video,
    step_create_voiceover,
    step_download_music,
    step_gather_visuals,
    step_generate_script,
    step_generate_subtitles,
)
from src.video.video_config import VideoConfig, VideoProfile


class TestVideoRunPaths:
    """Test video run path generation."""

    @pytest.mark.unit
    def test_get_video_run_paths_basic(self, mock_config: VideoConfig):
        """Test basic path generation."""
        paths = get_video_run_paths(mock_config, "B08TEST123", "test_profile")

        assert "run_root" in paths
        assert "intermediate_base" in paths
        assert "assets_dir" in paths
        assert "info_dir" in paths
        assert "script_file" in paths
        assert "voiceover_file" in paths
        assert "final_video_output" in paths

    @pytest.mark.unit
    def test_get_video_run_paths_sanitized(self, mock_config: VideoConfig):
        """Test path generation with sanitized filenames."""
        paths = get_video_run_paths(mock_config, "B08/TEST:123", "test/profile")

        # Check that special characters are sanitized
        assert "B08_TEST_123" in str(paths["run_root"])

    @pytest.mark.unit
    def test_get_video_run_paths_structure(self, mock_config: VideoConfig):
        """Test that all required paths are present and properly structured."""
        paths = get_video_run_paths(mock_config, "B08TEST123", "test_profile")

        # Check that all paths are Path objects
        for _key, path in paths.items():
            assert isinstance(path, Path)

        # Check that intermediate paths are under run_root
        assert paths["intermediate_base"].is_relative_to(paths["run_root"])
        assert paths["assets_dir"].is_relative_to(paths["intermediate_base"])
        assert paths["info_dir"].is_relative_to(paths["intermediate_base"])


class TestPipelineContext:
    """Test pipeline context management."""

    @pytest.mark.unit
    def test_pipeline_context_initialization(
        self,
        sample_product_data: ProductData,
        sample_video_profile: VideoProfile,
        mock_config: VideoConfig,
    ):
        """Test pipeline context initialization."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        run_paths = {"run_root": Path("/test/path")}
        secrets = {"test_key": "test_value"}

        ctx = PipelineContext(
            product=sample_product_data,
            profile=sample_video_profile,
            config=mock_config,
            secrets=secrets,
            session=session,
            run_paths=run_paths,
            debug_mode=False,
        )

        assert ctx.product == sample_product_data
        assert ctx.profile == sample_video_profile
        assert ctx.config == mock_config
        assert ctx.secrets == secrets
        assert ctx.session == session
        assert ctx.run_paths == run_paths
        assert ctx.debug_mode is False
        assert ctx.visuals is None
        assert ctx.script is None
        assert ctx.voiceover_duration is None
        assert ctx.state == {}


class TestStepGatherVisuals:
    """Test the gather visuals pipeline step."""

    @pytest.mark.asyncio
    async def test_step_gather_visuals_success(
        self,
        sample_product_data: ProductData,
        sample_video_profile: VideoProfile,
        mock_config: VideoConfig,
    ):
        """Test successful visual gathering."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        assert sample_product_data.asin is not None
        run_paths = get_video_run_paths(
            mock_config, sample_product_data.asin, "test_profile"
        )
        secrets = {"PEXELS_API_KEY": "test_key"}

        # Create necessary directories for the test
        from pathlib import Path

        for path_key, path_value in run_paths.items():
            if isinstance(path_value, Path):
                if path_key.endswith(("_dir", "_base", "_root")):
                    path_value.mkdir(parents=True, exist_ok=True)
                elif path_key.endswith("_file"):
                    path_value.parent.mkdir(parents=True, exist_ok=True)

        # Ensure gathered_visuals_file doesn't exist so test proceeds normally
        if run_paths.get("gathered_visuals_file"):
            gathered_visuals_file = run_paths["gathered_visuals_file"]
            if gathered_visuals_file.exists():
                gathered_visuals_file.unlink()

        ctx = PipelineContext(
            product=sample_product_data,
            profile=sample_video_profile,
            config=mock_config,
            secrets=secrets,
            session=session,
            run_paths=run_paths,
            debug_mode=False,
        )

        # Mock the stock media fetcher
        with patch("src.video.producer.StockMediaFetcher") as mock_fetcher_class:
            mock_fetcher = AsyncMock()
            mock_fetcher_class.return_value = mock_fetcher

            # Create mock stock media items instead of empty list
            from pathlib import Path

            from src.video.stock_media import StockMediaInfo

            mock_stock_items = [
                StockMediaInfo(
                    source="pexels",
                    type="image",
                    url="https://example.com/image1.jpg",
                    author="Test Author 1",
                    path=Path("/mock/path/image1.jpg"),
                ),
                StockMediaInfo(
                    source="pexels",
                    type="image",
                    url="https://example.com/image2.jpg",
                    author="Test Author 2",
                    path=Path("/mock/path/image2.jpg"),
                ),
                StockMediaInfo(
                    source="pexels",
                    type="image",
                    url="https://example.com/image3.jpg",
                    author="Test Author 3",
                    path=Path("/mock/path/image3.jpg"),
                ),
                StockMediaInfo(
                    source="pexels",
                    type="image",
                    url="https://example.com/image4.jpg",
                    author="Test Author 4",
                    path=Path("/mock/path/image4.jpg"),
                ),
                StockMediaInfo(
                    source="pexels",
                    type="image",
                    url="https://example.com/image5.jpg",
                    author="Test Author 5",
                    path=Path("/mock/path/image5.jpg"),
                ),
            ]
            mock_fetcher.fetch_and_download_stock.return_value = mock_stock_items

            await step_gather_visuals(ctx)

            # Verify stock media fetcher was called
            mock_fetcher_class.assert_called_once()
            mock_fetcher.fetch_and_download_stock.assert_called_once()

    @pytest.mark.asyncio
    async def test_step_gather_visuals_with_existing_visuals(
        self,
        sample_product_data: ProductData,
        sample_video_profile: VideoProfile,
        mock_config: VideoConfig,
    ):
        """Test visual gathering when visuals already exist."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        assert sample_product_data.asin is not None
        run_paths = get_video_run_paths(
            mock_config, sample_product_data.asin, "test_profile"
        )
        secrets = {"PEXELS_API_KEY": "test_key"}

        ctx = PipelineContext(
            product=sample_product_data,
            profile=sample_video_profile,
            config=mock_config,
            secrets=secrets,
            session=session,
            run_paths=run_paths,
            debug_mode=False,
        )

        # Create existing visuals file
        run_paths["gathered_visuals_file"].parent.mkdir(parents=True, exist_ok=True)
        existing_visuals = {
            "scraped_images": ["/path/to/image1.jpg"],
            "scraped_videos": [],
            "stock_media": [],
        }
        with open(run_paths["gathered_visuals_file"], "w") as f:
            json.dump(existing_visuals, f)

        await step_gather_visuals(ctx)

        # Should load existing visuals instead of fetching new ones
        assert ctx.visuals is not None

    @pytest.mark.asyncio
    async def test_step_gather_visuals_error_handling(
        self,
        sample_product_data: ProductData,
        sample_video_profile: VideoProfile,
        mock_config: VideoConfig,
    ):
        """Test error handling in visual gathering."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        assert sample_product_data.asin is not None
        run_paths = get_video_run_paths(
            mock_config, sample_product_data.asin, "test_profile"
        )
        secrets = {"PEXELS_API_KEY": "test_key"}

        ctx = PipelineContext(
            product=sample_product_data,
            profile=sample_video_profile,
            config=mock_config,
            secrets=secrets,
            session=session,
            run_paths=run_paths,
            debug_mode=False,
        )

        # Mock stock media fetcher to raise an exception
        with patch("src.video.producer.StockMediaFetcher") as mock_fetcher_class:
            mock_fetcher = AsyncMock()
            mock_fetcher_class.return_value = mock_fetcher
            mock_fetcher.fetch_stock_media.side_effect = Exception("API Error")

            with pytest.raises(PipelineError):
                await step_gather_visuals(ctx)


class TestStepGenerateScript:
    """Test the script generation pipeline step."""

    @pytest.mark.asyncio
    async def test_step_generate_script_success(
        self,
        sample_product_data: ProductData,
        sample_video_profile: VideoProfile,
        mock_config: VideoConfig,
    ):
        """Test successful script generation."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        assert sample_product_data.asin is not None
        run_paths = get_video_run_paths(
            mock_config, sample_product_data.asin, "test_profile"
        )
        secrets = {"OPENROUTER_API_KEY": "test_key"}

        ctx = PipelineContext(
            product=sample_product_data,
            profile=sample_video_profile,
            config=mock_config,
            secrets=secrets,
            session=session,
            run_paths=run_paths,
            debug_mode=False,
        )

        # Mock the script generator
        with patch("src.video.producer.generate_ai_script") as mock_generate:
            mock_generate.return_value = "This is a test script for the product."

            await step_generate_script(ctx)

            mock_generate.assert_called_once()
            assert ctx.script == "This is a test script for the product."

    @pytest.mark.asyncio
    async def test_step_generate_script_with_existing_script(
        self,
        sample_product_data: ProductData,
        sample_video_profile: VideoProfile,
        mock_config: VideoConfig,
    ):
        """Test script generation when script already exists."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        assert sample_product_data.asin is not None
        run_paths = get_video_run_paths(
            mock_config, sample_product_data.asin, "test_profile"
        )
        secrets = {"OPENROUTER_API_KEY": "test_key"}

        ctx = PipelineContext(
            product=sample_product_data,
            profile=sample_video_profile,
            config=mock_config,
            secrets=secrets,
            session=session,
            run_paths=run_paths,
            debug_mode=False,
        )

        # Create existing script file
        run_paths["script_file"].parent.mkdir(parents=True, exist_ok=True)
        with open(run_paths["script_file"], "w") as f:
            f.write("Existing test script")

        await step_generate_script(ctx)

        # Should load existing script instead of generating new one
        assert ctx.script == "Existing test script"

    @pytest.mark.asyncio
    async def test_step_generate_script_error_handling(
        self,
        sample_product_data: ProductData,
        sample_video_profile: VideoProfile,
        mock_config: VideoConfig,
    ):
        """Test error handling in script generation."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        assert sample_product_data.asin is not None
        run_paths = get_video_run_paths(
            mock_config, sample_product_data.asin, "test_profile"
        )
        secrets = {"OPENROUTER_API_KEY": "test_key"}

        ctx = PipelineContext(
            product=sample_product_data,
            profile=sample_video_profile,
            config=mock_config,
            secrets=secrets,
            session=session,
            run_paths=run_paths,
            debug_mode=False,
        )

        # Mock script generator to raise an exception
        with patch("src.video.producer.generate_ai_script") as mock_generate:
            mock_generate.side_effect = Exception("API Error")

            with pytest.raises(PipelineError):
                await step_generate_script(ctx)


class TestStepCreateVoiceover:
    """Test the voiceover creation pipeline step."""

    @pytest.mark.asyncio
    async def test_step_create_voiceover_success(
        self,
        sample_product_data: ProductData,
        sample_video_profile: VideoProfile,
        mock_config: VideoConfig,
    ):
        """Test successful voiceover creation."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        assert sample_product_data.asin is not None
        run_paths = get_video_run_paths(
            mock_config, sample_product_data.asin, "test_profile"
        )
        secrets = {"GOOGLE_APPLICATION_CREDENTIALS": "/path/to/creds.json"}

        ctx = PipelineContext(
            product=sample_product_data,
            profile=sample_video_profile,
            config=mock_config,
            secrets=secrets,
            session=session,
            run_paths=run_paths,
            debug_mode=False,
        )
        ctx.script = "Test script for voiceover"

        # Mock the TTS manager
        with patch("src.video.producer.TTSManager") as mock_tts_class:
            mock_tts = AsyncMock()
            mock_tts_class.return_value = mock_tts
            # Create a mock path that exists
            mock_path = Mock()
            mock_path.exists.return_value = True
            mock_path.name = "voiceover.wav"
            mock_tts.generate_speech.return_value = mock_path

            # Mock the duration calculation
            with patch("src.video.producer._get_video_duration", return_value=5.5):
                await step_create_voiceover(ctx)

            mock_tts.generate_speech.assert_called_once()
            assert ctx.voiceover_duration == 5.5

    @pytest.mark.asyncio
    async def test_step_create_voiceover_with_existing_voiceover(
        self,
        sample_product_data: ProductData,
        sample_video_profile: VideoProfile,
        mock_config: VideoConfig,
    ):
        """Test voiceover creation when voiceover already exists."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        assert sample_product_data.asin is not None
        run_paths = get_video_run_paths(
            mock_config, sample_product_data.asin, "test_profile"
        )
        secrets = {"GOOGLE_APPLICATION_CREDENTIALS": "/path/to/creds.json"}

        ctx = PipelineContext(
            product=sample_product_data,
            profile=sample_video_profile,
            config=mock_config,
            secrets=secrets,
            session=session,
            run_paths=run_paths,
            debug_mode=False,
        )

        # Create existing voiceover files
        run_paths["voiceover_file"].parent.mkdir(parents=True, exist_ok=True)
        run_paths["voiceover_file"].write_text("mock audio data")

        run_paths["voiceover_duration_file"].parent.mkdir(parents=True, exist_ok=True)
        run_paths["voiceover_duration_file"].write_text("6.2")

        await step_create_voiceover(ctx)

        # Should load existing voiceover instead of generating new one
        assert ctx.voiceover_duration == 6.2

    @pytest.mark.asyncio
    async def test_step_create_voiceover_error_handling(
        self,
        sample_product_data: ProductData,
        sample_video_profile: VideoProfile,
        mock_config: VideoConfig,
    ):
        """Test error handling in voiceover creation."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        assert sample_product_data.asin is not None
        run_paths = get_video_run_paths(
            mock_config, sample_product_data.asin, "test_profile"
        )
        secrets = {"GOOGLE_APPLICATION_CREDENTIALS": "/path/to/creds.json"}

        ctx = PipelineContext(
            product=sample_product_data,
            profile=sample_video_profile,
            config=mock_config,
            secrets=secrets,
            session=session,
            run_paths=run_paths,
            debug_mode=False,
        )
        ctx.script = "Test script"

        # Mock TTS manager to raise an exception
        with patch("src.video.producer.TTSManager") as mock_tts_class:
            mock_tts = AsyncMock()
            mock_tts_class.return_value = mock_tts
            mock_tts.generate_speech.side_effect = Exception("TTS Error")

            with pytest.raises(PipelineError):
                await step_create_voiceover(ctx)


class TestStepGenerateSubtitles:
    """Test the subtitle generation pipeline step."""

    @pytest.mark.asyncio
    async def test_step_generate_subtitles_success(
        self,
        sample_product_data: ProductData,
        sample_video_profile: VideoProfile,
        mock_config: VideoConfig,
    ):
        """Test successful subtitle generation."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        assert sample_product_data.asin is not None
        run_paths = get_video_run_paths(
            mock_config, sample_product_data.asin, "test_profile"
        )
        secrets = {"GOOGLE_APPLICATION_CREDENTIALS": "/path/to/creds.json"}

        ctx = PipelineContext(
            product=sample_product_data,
            profile=sample_video_profile,
            config=mock_config,
            secrets=secrets,
            session=session,
            run_paths=run_paths,
            debug_mode=False,
        )
        ctx.script = "Test script for subtitles"
        ctx.voiceover_duration = 5.5

        # Mock the subtitle generator
        with patch("src.video.producer.create_unified_subtitles") as mock_subtitles:
            mock_subtitles.return_value = True

            await step_generate_subtitles(ctx)

            mock_subtitles.assert_called_once()

    @pytest.mark.asyncio
    async def test_step_generate_subtitles_with_existing_subtitles(
        self,
        sample_product_data: ProductData,
        sample_video_profile: VideoProfile,
        mock_config: VideoConfig,
    ):
        """Test subtitle generation when subtitles already exist."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        assert sample_product_data.asin is not None
        run_paths = get_video_run_paths(
            mock_config, sample_product_data.asin, "test_profile"
        )
        secrets = {"GOOGLE_APPLICATION_CREDENTIALS": "/path/to/creds.json"}

        ctx = PipelineContext(
            product=sample_product_data,
            profile=sample_video_profile,
            config=mock_config,
            secrets=secrets,
            session=session,
            run_paths=run_paths,
            debug_mode=False,
        )

        # Create existing subtitle file
        run_paths["subtitle_file"].parent.mkdir(parents=True, exist_ok=True)
        with open(run_paths["subtitle_file"], "w") as f:
            f.write("1\n00:00:00,000 --> 00:00:03,000\nTest subtitle\n\n")

        await step_generate_subtitles(ctx)

        # Should use existing subtitles instead of generating new ones
        assert run_paths["subtitle_file"].exists()

    @pytest.mark.asyncio
    async def test_step_generate_subtitles_disabled(
        self,
        sample_product_data: ProductData,
        sample_video_profile: VideoProfile,
        mock_config: VideoConfig,
    ):
        """Test subtitle generation when disabled in config."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        assert sample_product_data.asin is not None
        run_paths = get_video_run_paths(
            mock_config, sample_product_data.asin, "test_profile"
        )
        secrets = {"GOOGLE_APPLICATION_CREDENTIALS": "/path/to/creds.json"}

        # Disable subtitles in config
        mock_config.subtitle_settings.enabled = False

        ctx = PipelineContext(
            product=sample_product_data,
            profile=sample_video_profile,
            config=mock_config,
            secrets=secrets,
            session=session,
            run_paths=run_paths,
            debug_mode=False,
        )

        await step_generate_subtitles(ctx)

        # Should skip subtitle generation when disabled
        assert not run_paths["subtitle_file"].exists()


class TestStepDownloadMusic:
    """Test the music download pipeline step."""

    @pytest.mark.asyncio
    async def test_step_download_music_success(
        self,
        sample_product_data: ProductData,
        sample_video_profile: VideoProfile,
        mock_config: VideoConfig,
    ):
        """Test successful music download."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        assert sample_product_data.asin is not None
        run_paths = get_video_run_paths(
            mock_config, sample_product_data.asin, "test_profile"
        )
        secrets = {"FREESOUND_API_KEY": "test_key"}

        ctx = PipelineContext(
            product=sample_product_data,
            profile=sample_video_profile,
            config=mock_config,
            secrets=secrets,
            session=session,
            run_paths=run_paths,
            debug_mode=False,
        )

        # Mock the Freesound client
        with patch("src.video.producer.FreesoundClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.download_background_music.return_value = "/path/to/music.wav"

            await step_download_music(ctx)

            mock_client.download_background_music.assert_called_once()

    @pytest.mark.asyncio
    async def test_step_download_music_with_existing_music(
        self,
        sample_product_data: ProductData,
        sample_video_profile: VideoProfile,
        mock_config: VideoConfig,
    ):
        """Test music download when music already exists."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        assert sample_product_data.asin is not None
        run_paths = get_video_run_paths(
            mock_config, sample_product_data.asin, "test_profile"
        )
        secrets = {"FREESOUND_API_KEY": "test_key"}

        ctx = PipelineContext(
            product=sample_product_data,
            profile=sample_video_profile,
            config=mock_config,
            secrets=secrets,
            session=session,
            run_paths=run_paths,
            debug_mode=False,
        )

        # Create existing music info file
        run_paths["music_info_file"].parent.mkdir(parents=True, exist_ok=True)
        music_info = {"music_path": "/path/to/existing/music.wav"}
        with open(run_paths["music_info_file"], "w") as f:
            json.dump(music_info, f)

        await step_download_music(ctx)

        # Should use existing music instead of downloading new one
        assert run_paths["music_info_file"].exists()

    @pytest.mark.asyncio
    async def test_step_download_music_error_handling(
        self,
        sample_product_data: ProductData,
        sample_video_profile: VideoProfile,
        mock_config: VideoConfig,
    ):
        """Test error handling in music download."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        assert sample_product_data.asin is not None
        run_paths = get_video_run_paths(
            mock_config, sample_product_data.asin, "test_profile"
        )
        secrets = {"FREESOUND_API_KEY": "test_key"}

        ctx = PipelineContext(
            product=sample_product_data,
            profile=sample_video_profile,
            config=mock_config,
            secrets=secrets,
            session=session,
            run_paths=run_paths,
            debug_mode=False,
        )

        # Mock Freesound client to raise an exception
        with patch("src.video.producer.FreesoundClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.download_background_music.side_effect = Exception(
                "Download Error"
            )

            with pytest.raises(PipelineError):
                await step_download_music(ctx)


class TestStepAssembleVideo:
    """Test the video assembly pipeline step."""

    @pytest.mark.asyncio
    async def test_step_assemble_video_success(
        self,
        sample_product_data: ProductData,
        sample_video_profile: VideoProfile,
        mock_config: VideoConfig,
    ):
        """Test successful video assembly."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        assert sample_product_data.asin is not None
        run_paths = get_video_run_paths(
            mock_config, sample_product_data.asin, "test_profile"
        )
        secrets: dict[str, str] = {}

        ctx = PipelineContext(
            product=sample_product_data,
            profile=sample_video_profile,
            config=mock_config,
            secrets=secrets,
            session=session,
            run_paths=run_paths,
            debug_mode=False,
        )
        ctx.visuals = [Path("/path/to/image1.jpg"), Path("/path/to/image2.jpg")]
        ctx.script = "Test script"
        ctx.voiceover_duration = 5.5

        # Mock the video assembler
        with patch("src.video.producer.VideoAssembler") as mock_assembler_class:
            mock_assembler = AsyncMock()
            mock_assembler_class.return_value = mock_assembler
            mock_assembler.assemble_video.return_value = True
            # Mock verify_video as a sync method using regular Mock
            from unittest.mock import Mock

            mock_assembler.verify_video = Mock(
                return_value={
                    "success": True,
                    "message": "Video verified successfully.",
                }
            )

            await step_assemble_video(ctx)

            mock_assembler.assemble_video.assert_called_once()

    @pytest.mark.asyncio
    async def test_step_assemble_video_error_handling(
        self,
        sample_product_data: ProductData,
        sample_video_profile: VideoProfile,
        mock_config: VideoConfig,
    ):
        """Test error handling in video assembly."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        assert sample_product_data.asin is not None
        run_paths = get_video_run_paths(
            mock_config, sample_product_data.asin, "test_profile"
        )
        secrets: dict[str, str] = {}

        ctx = PipelineContext(
            product=sample_product_data,
            profile=sample_video_profile,
            config=mock_config,
            secrets=secrets,
            session=session,
            run_paths=run_paths,
            debug_mode=False,
        )
        ctx.visuals = [Path("/path/to/image1.jpg")]
        ctx.script = "Test script"
        ctx.voiceover_duration = 5.5

        # Mock video assembler to raise an exception
        with patch("src.video.producer.VideoAssembler") as mock_assembler_class:
            mock_assembler = AsyncMock()
            mock_assembler_class.return_value = mock_assembler
            mock_assembler.assemble_video.side_effect = Exception("Assembly Error")
            # Mock verify_video as a sync method using regular Mock
            # (won't be called due to exception)
            from unittest.mock import Mock

            mock_assembler.verify_video = Mock(
                return_value={
                    "success": True,
                    "message": "Video verified successfully.",
                }
            )

            with pytest.raises(PipelineError):
                await step_assemble_video(ctx)


class TestCreateVideoForProduct:
    """Test the main video creation function."""

    @pytest.mark.asyncio
    async def test_create_video_for_product_success(
        self,
        sample_product_data: ProductData,
        mock_config: VideoConfig,
        mock_aiohttp_session: AsyncMock,
    ):
        """Test successful video creation for a product."""
        secrets = {
            "OPENROUTER_API_KEY": "test_key",
            "PEXELS_API_KEY": "test_key",
            "FREESOUND_API_KEY": "test_key",
        }

        # Mock all the pipeline steps
        with (
            patch("src.video.producer.step_gather_visuals") as mock_gather,
            patch("src.video.producer.step_generate_script") as mock_script,
            patch("src.video.producer.step_create_voiceover") as mock_voiceover,
            patch("src.video.producer.step_generate_subtitles") as mock_subtitles,
            patch("src.video.producer.step_download_music") as mock_music,
            patch("src.video.producer.step_assemble_video") as mock_assemble,
        ):
            await create_video_for_product(
                config=mock_config,
                product=sample_product_data,
                profile_name="test_profile",
                secrets=secrets,
                session=mock_aiohttp_session,
                debug_mode=False,
                clean_run=True,
                debug_step_target=None,
            )

            # Verify all steps were called
            mock_gather.assert_called_once()
            mock_script.assert_called_once()
            mock_voiceover.assert_called_once()
            mock_subtitles.assert_called_once()
            mock_music.assert_called_once()
            mock_assemble.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_video_for_product_single_step(
        self,
        sample_product_data: ProductData,
        mock_config: VideoConfig,
        mock_aiohttp_session: AsyncMock,
    ):
        """Test video creation with single step execution."""
        secrets = {"OPENROUTER_API_KEY": "test_key"}

        # Mock only the script generation step
        with patch("src.video.producer.step_generate_script") as mock_script:
            await create_video_for_product(
                config=mock_config,
                product=sample_product_data,
                profile_name="test_profile",
                secrets=secrets,
                session=mock_aiohttp_session,
                debug_mode=True,
                clean_run=False,
                debug_step_target="generate_script",
            )

            # Only script generation should be called
            mock_script.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_video_for_product_error_handling(
        self,
        sample_product_data: ProductData,
        mock_config: VideoConfig,
        mock_aiohttp_session: AsyncMock,
    ):
        """Test error handling in video creation."""
        secrets = {"OPENROUTER_API_KEY": "test_key"}

        # Mock script generation to raise an exception
        with patch("src.video.producer.step_generate_script") as mock_script:
            mock_script.side_effect = PipelineError("Script generation failed")

            with pytest.raises(PipelineError):
                await create_video_for_product(
                    config=mock_config,
                    product=sample_product_data,
                    profile_name="test_profile",
                    secrets=secrets,
                    session=mock_aiohttp_session,
                    debug_mode=False,
                    clean_run=True,
                    debug_step_target=None,
                )

    @pytest.mark.asyncio
    async def test_create_video_for_product_invalid_profile(
        self,
        sample_product_data: ProductData,
        mock_config: VideoConfig,
        mock_aiohttp_session: AsyncMock,
    ):
        """Test video creation with invalid profile name."""
        secrets = {"OPENROUTER_API_KEY": "test_key"}

        with pytest.raises(ValueError, match="Profile 'invalid_profile' not found"):
            await create_video_for_product(
                config=mock_config,
                product=sample_product_data,
                profile_name="invalid_profile",
                secrets=secrets,
                session=mock_aiohttp_session,
                debug_mode=False,
                clean_run=True,
                debug_step_target=None,
            )


class TestDiscoverProductsForBatch:
    """Test batch product discovery functionality."""

    @pytest.mark.unit
    def test_discover_products_for_batch_success(self, tmp_path: Path):
        """Test successful product discovery."""
        # Create test outputs directory structure
        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()

        # Create valid product directories with data.json
        product1_dir = outputs_dir / "B08TEST123"
        product1_dir.mkdir()
        product1_data = {
            "asin": "B08TEST123",
            "title": "Test Product 1",
            "price": "$29.99",
            "description": "A test product",
            "images": [],
            "videos": [],
        }
        (product1_dir / "data.json").write_text(json.dumps(product1_data))

        product2_dir = outputs_dir / "B08TEST456"
        product2_dir.mkdir()
        product2_data = {
            "asin": "B08TEST456",
            "title": "Test Product 2",
            "price": "$39.99",
            "description": "Another test product",
            "images": [],
            "videos": [],
        }
        (product2_dir / "data.json").write_text(json.dumps(product2_data))

        # Create system directories that should be ignored
        (outputs_dir / "cache").mkdir()
        (outputs_dir / "logs").mkdir()
        (outputs_dir / "coverage").mkdir()
        (outputs_dir / "reports").mkdir()

        # Import and test the function

        discovered_products = discover_products_for_batch(outputs_dir)

        # Verify results - function now returns tuples of (dir, product)
        assert len(discovered_products) == 2
        products = [product for (dir, product) in discovered_products]
        asins = [product.asin for product in products]
        assert "B08TEST123" in asins
        assert "B08TEST456" in asins

        # Verify product data is correct
        for _dir, product in discovered_products:
            if product.asin == "B08TEST123":
                assert product.title == "Test Product 1"
            elif product.asin == "B08TEST456":
                assert product.title == "Test Product 2"

    @pytest.mark.unit
    def test_discover_products_for_batch_with_list_format(self, tmp_path: Path):
        """Test product discovery with list format data.json."""
        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()

        # Create directory with list format data.json (multiple products)
        batch_dir = outputs_dir / "batch_products"
        batch_dir.mkdir()
        batch_data = [
            {
                "asin": "B08LIST001",
                "title": "List Product 1",
                "price": "$19.99",
                "description": "First product from list",
                "images": [],
                "videos": [],
            },
            {
                "asin": "B08LIST002",
                "title": "List Product 2",
                "price": "$24.99",
                "description": "Second product from list",
                "images": [],
                "videos": [],
            },
        ]
        (batch_dir / "data.json").write_text(json.dumps(batch_data))

        discovered_products = discover_products_for_batch(outputs_dir)

        # Should find both products from the list
        assert len(discovered_products) == 2
        products = [product for (dir, product) in discovered_products]
        asins = [product.asin for product in products]
        assert "B08LIST001" in asins
        assert "B08LIST002" in asins

    @pytest.mark.unit
    def test_discover_products_for_batch_skips_system_dirs(self, tmp_path: Path):
        """Test that system directories are properly ignored."""
        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()

        # Create system directories with data.json files
        for system_dir in ["cache", "logs", "coverage", "reports"]:
            dir_path = outputs_dir / system_dir
            dir_path.mkdir()
            # Add data.json to verify they're ignored
            fake_data = {
                "asin": f"FAKE{system_dir.upper()}",
                "title": "Should be ignored",
            }
            (dir_path / "data.json").write_text(json.dumps(fake_data))

        # Create one valid product directory
        valid_dir = outputs_dir / "B08VALID123"
        valid_dir.mkdir()
        valid_data = {
            "asin": "B08VALID123",
            "title": "Valid Product",
            "price": "$29.99",
            "description": "A valid product",
            "images": [],
            "videos": [],
        }
        (valid_dir / "data.json").write_text(json.dumps(valid_data))

        discovered_products = discover_products_for_batch(outputs_dir)

        # Should only find the valid product, not the system directories
        assert len(discovered_products) == 1
        dir, product = discovered_products[0]
        assert product.asin == "B08VALID123"

    @pytest.mark.unit
    def test_discover_products_for_batch_invalid_json(self, tmp_path: Path):
        """Test handling of directories with invalid JSON data."""
        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()

        # Create directory with invalid JSON
        invalid_dir = outputs_dir / "B08INVALID"
        invalid_dir.mkdir()
        (invalid_dir / "data.json").write_text("{ invalid json content")

        # Should handle invalid JSON gracefully and return empty list
        discovered_products = discover_products_for_batch(outputs_dir)
        assert len(discovered_products) == 0

    @pytest.mark.unit
    def test_discover_products_for_batch_empty_list(self, tmp_path: Path):
        """Test handling of directory with empty list in data.json."""
        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()

        empty_dir = outputs_dir / "B08EMPTY"
        empty_dir.mkdir()
        (empty_dir / "data.json").write_text("[]")

        discovered_products = discover_products_for_batch(outputs_dir)
        assert len(discovered_products) == 0

    @pytest.mark.unit
    def test_discover_products_for_batch_nonexistent_dir(self, tmp_path: Path):
        """Test handling of non-existent directory."""
        nonexistent_dir = tmp_path / "nonexistent"

        discovered_products = discover_products_for_batch(nonexistent_dir)
        assert len(discovered_products) == 0


class TestBatchProcessingArgparse:
    """Test batch processing command line argument parsing."""

    @pytest.mark.unit
    def test_batch_mode_validation_success(self):
        """Test successful batch mode argument validation."""
        import sys
        from unittest.mock import patch

        from src.video.producer import main

        # Test valid batch arguments
        test_args = ["--batch", "--batch-profile", "slideshow_images", "--debug"]

        with (
            patch.object(sys, "argv", ["producer.py"] + test_args),
            patch("src.video.producer.discover_products_for_batch", return_value=[]),
            patch("src.video.producer.load_video_config"),
            patch("src.video.producer.load_dotenv"),
            patch("src.video.producer.get_http_session"),
        ):
            try:
                # This should validate successfully but exit due to no products
                asyncio.run(main())
            except SystemExit as e:
                # Should exit with error due to no products found
                assert e.code == 1

    @pytest.mark.unit
    def test_batch_mode_missing_profile_error(self):
        """Test error when batch-profile is missing in batch mode."""
        import sys
        from unittest.mock import patch

        from src.video.producer import main

        test_args = ["--batch", "--debug"]

        with (
            patch.object(sys, "argv", ["producer.py"] + test_args),
            pytest.raises(SystemExit),
        ):
            asyncio.run(main())

    @pytest.mark.unit
    def test_single_mode_missing_required_args_error(self):
        """Test error when required args are missing in single mode."""
        import sys
        from unittest.mock import patch

        from src.video.producer import main

        test_args = ["--debug"]  # Missing products_file and profile

        with (
            patch.object(sys, "argv", ["producer.py"] + test_args),
            pytest.raises(SystemExit),
        ):
            asyncio.run(main())

    @pytest.mark.unit
    def test_mixed_mode_arguments_error(self):
        """Test error when batch and single mode arguments are mixed."""
        import sys
        from unittest.mock import patch

        from src.video.producer import main

        # Try to use both batch and single mode args
        test_args = [
            "--batch",
            "--batch-profile",
            "slideshow_images",
            "data.json",
            "profile",
        ]

        with (
            patch.object(sys, "argv", ["producer.py"] + test_args),
            pytest.raises(SystemExit),
        ):
            asyncio.run(main())

    @pytest.mark.unit
    def test_product_index_with_batch_error(self):
        """Test error when product-index is used with batch mode."""
        import sys
        from unittest.mock import patch

        from src.video.producer import main

        test_args = [
            "--batch",
            "--batch-profile",
            "slideshow_images",
            "--product-index",
            "0",
        ]

        with (
            patch.object(sys, "argv", ["producer.py"] + test_args),
            patch(
                "src.video.producer.discover_products_for_batch",
                return_value=[("dir", Mock())],
            ),
            patch("src.video.producer.load_video_config"),
            patch("src.video.producer.load_dotenv"),
        ):
            try:
                asyncio.run(main())
            except SystemExit as e:
                assert e.code == 1


class TestPipelineError:
    """Test custom pipeline error handling."""

    @pytest.mark.unit
    def test_pipeline_error_creation(self):
        """Test pipeline error creation and message."""
        error = PipelineError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    @pytest.mark.unit
    def test_pipeline_error_inheritance(self):
        """Test that PipelineError inherits from Exception."""
        error = PipelineError("Test")
        assert isinstance(error, Exception)
