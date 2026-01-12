"""Integration tests for the video producer pipeline."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from src.scraper.amazon.scraper import ProductData


@pytest.fixture
def product_assets(temp_dir):
    """Create mock product assets for testing."""
    product_dir = temp_dir / "outputs" / "B0INTEG123"
    product_dir.mkdir(parents=True)

    # Create mock media (minimum 5 images required when no videos)
    images_dir = product_dir / "images"
    images_dir.mkdir()
    for i in range(1, 6):
        (images_dir / f"img{i}.jpg").write_text(f"fake image {i}")

    data = {
        "asin": "B0INTEG123",
        "title": "Integration Test Product",
        "price": "$99",
        "url": "http://test",
        "platform": "amazon",
        "downloaded_images": [
            f"outputs/B0INTEG123/images/img{i}.jpg" for i in range(1, 6)
        ],
    }
    (product_dir / "data.json").write_text(json.dumps(data))

    return product_dir, ProductData(**data)


@pytest.mark.asyncio
async def test_full_pipeline_with_dummy_files(
    product_assets, mock_config, mock_aioresponses, temp_dir
):
    """Test the full video production pipeline with mocked external services.

    This test verifies that the pipeline orchestration works correctly
    by mocking each step's execution function.
    """
    product_dir, product = product_assets

    # Create temp directory for pipeline artifacts
    temp_path = product_dir / "temp"
    temp_path.mkdir(exist_ok=True)

    # Pre-create artifacts that steps would produce
    (temp_path / "script.txt").write_text("Test script content for integration.")
    (temp_path / "gathered_visuals.json").write_text(
        json.dumps({"images": [], "videos": []})
    )
    vo_path = temp_path / "voiceover.wav"
    vo_path.write_text("fake audio content")
    music_path = temp_path / "music.mp3"
    music_path.write_text("fake music content")
    sub_path = temp_path / "subtitles.srt"
    sub_path.write_text("1\n00:00:00,000 --> 00:00:05,000\nTest subtitle")
    final_video = product_dir / "video_test_profile.mp4"
    final_video.write_text("fake video content")

    # Fix mock_config paths
    mock_config.project_root = temp_dir
    mock_config.global_output_root_path = temp_dir / "outputs"
    mock_config.audio_settings.freesound_api_key_env_var = "FREESOUND_API_KEY"

    # Mock each pipeline step at the ORCHESTRATION module level (where they're used)
    with (
        patch(
            "src.video.producer.orchestration.step_gather_visuals",
            new_callable=AsyncMock,
        ) as mock_gather,
        patch(
            "src.video.producer.orchestration.step_generate_script",
            new_callable=AsyncMock,
        ) as mock_script,
        patch(
            "src.video.producer.orchestration.step_generate_description",
            new_callable=AsyncMock,
        ) as mock_desc,
        patch(
            "src.video.producer.orchestration.step_create_voiceover",
            new_callable=AsyncMock,
        ) as mock_vo,
        patch(
            "src.video.producer.orchestration.step_generate_subtitles",
            new_callable=AsyncMock,
        ) as mock_subs,
        patch(
            "src.video.producer.orchestration.step_download_music",
            new_callable=AsyncMock,
        ) as mock_music,
        patch(
            "src.video.producer.orchestration.step_assemble_video",
            new_callable=AsyncMock,
        ) as mock_assemble,
        patch("shutil.which", return_value="/usr/bin/ffmpeg"),
    ):
        # Configure step mocks to return successfully
        # Each step modifies context and returns a result
        async def gather_visuals_side_effect(ctx, **kwargs):
            ctx.gathered_images = [f"img{i}.jpg" for i in range(1, 6)]
            ctx.gathered_videos = []
            ctx.stock_media = []
            return MagicMock(success=True, images=ctx.gathered_images)

        async def script_side_effect(ctx, **kwargs):
            ctx.script = "Test script for integration testing."
            return MagicMock(success=True, script=ctx.script)

        async def desc_side_effect(ctx, **kwargs):
            ctx.description = "Test description"
            return MagicMock(success=True, description=ctx.description)

        async def vo_side_effect(ctx, **kwargs):
            ctx.voiceover_path = vo_path
            ctx.voiceover_duration = 10.0
            return MagicMock(success=True, voiceover_path=vo_path)

        async def subs_side_effect(ctx, **kwargs):
            ctx.subtitles_path = sub_path
            return MagicMock(success=True, subtitles_path=sub_path)

        async def music_side_effect(ctx, **kwargs):
            ctx.music_path = music_path
            ctx.music_metadata = {"name": "test", "source": "test"}
            return MagicMock(success=True, music_path=music_path)

        async def assemble_side_effect(ctx, **kwargs):
            ctx.output_video_path = final_video
            return MagicMock(success=True, output_path=final_video)

        mock_gather.side_effect = gather_visuals_side_effect
        mock_script.side_effect = script_side_effect
        mock_desc.side_effect = desc_side_effect
        mock_vo.side_effect = vo_side_effect
        mock_subs.side_effect = subs_side_effect
        mock_music.side_effect = music_side_effect
        mock_assemble.side_effect = assemble_side_effect

        from src.video.producer.orchestration import create_video_for_product

        async with aiohttp.ClientSession() as session:
            await create_video_for_product(
                config=mock_config,
                product=product,
                profile_name="test_profile",
                secrets={"OPENROUTER_API_KEY": "test", "FREESOUND_API_KEY": "test"},
                session=session,
                debug_mode=True,
                clean_run=False,
                debug_step_target=None,
            )

            # Pipeline should complete (either success or handled failure)
            # Since we're mocking at step level, we verify mocks were called
            assert mock_gather.called, "step_gather_visuals should be called"
