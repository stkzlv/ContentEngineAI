# src/video/producer/steps.py
"""Pipeline step implementations for video production."""

import asyncio
import json
import logging
import random
import subprocess
from pathlib import Path
from typing import Any, Literal

from src.ai.description_generator import generate_description as generate_ai_description
from src.ai.script_generator import generate_script as generate_ai_script
from src.audio.manager import AudioManager
from src.audio.registry import create_audio_provider
from src.utils import ensure_dirs_exist
from src.utils.performance import performance_monitor
from src.utils.script_sanitizer import sanitize_script
from src.video.assembler import VideoAssembler
from src.video.producer.artifact_registry import register_artifact_loader
from src.video.producer.constants import (
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_TOP_POSITION,
    DEFAULT_VIDEO_WIDTH,
    HASHTAG_SKIP_WORDS,
    SUPPORTED_PLATFORMS,
)
from src.video.producer.context import (
    InsufficientMediaError,
    PipelineContext,
    PipelineError,
)
from src.video.producer.state import (
    STEP_ASSEMBLE_VIDEO,
    STEP_CREATE_VOICEOVER,
    STEP_DOWNLOAD_MUSIC,
    STEP_GATHER_VISUALS,
    STEP_GENERATE_DESCRIPTION,
    STEP_GENERATE_SCRIPT,
    STEP_GENERATE_SUBTITLES,
    _get_video_duration,
    load_visuals_info,
    save_visuals_info,
)
from src.video.producer.utils import validate_media_requirements
from src.video.stock_media import StockMediaFetcher, StockMediaInfo
from src.video.subtitle_utils import create_unified_subtitles
from src.video.tts import TTSManager

logger = logging.getLogger(__name__)


@register_artifact_loader("gather_visuals")
def _load_artifacts_gather_visuals(ctx: PipelineContext) -> None:
    """Load artifacts from completed gather_visuals step."""
    visuals_file = ctx.run_paths["gathered_visuals_file"]
    if visuals_file.exists():
        ctx.scraped_images, ctx.scraped_videos, ctx.stock_media = load_visuals_info(
            visuals_file
        )


@register_artifact_loader("generate_script")
def _load_artifacts_generate_script(ctx: PipelineContext) -> None:
    """Load artifacts from completed generate_script step."""
    script_file = ctx.run_paths["script_file"]
    if script_file.exists():
        ctx.script = script_file.read_text(encoding="utf-8")


@register_artifact_loader("generate_description")
def _load_artifacts_generate_description(ctx: PipelineContext) -> None:
    """Load artifacts from completed generate_description step."""
    text_dir = ctx.run_paths["description_file"].parent

    # Try loading unified metadata.json first (in product root)
    unified_metadata = ctx.run_paths["run_root"] / "metadata.json"
    if unified_metadata.exists():
        meta = json.loads(unified_metadata.read_text(encoding="utf-8"))
        ctx.description = meta.get("description", "")
        return

    # Fallback to platform-specific metadata files
    for platform in SUPPORTED_PLATFORMS:
        metadata_file = text_dir / f"metadata_{platform}.json"
        if metadata_file.exists():
            return

    # Legacy fallback to description.txt
    description_file = ctx.run_paths["description_file"]
    if description_file.exists():
        ctx.description = description_file.read_text(encoding="utf-8")


@register_artifact_loader("create_voiceover")
def _load_artifacts_create_voiceover(ctx: PipelineContext) -> None:
    """Load artifacts from completed create_voiceover step."""
    duration_file = ctx.run_paths["voiceover_duration_file"]
    if duration_file.exists():
        ctx.voiceover_duration = float(duration_file.read_text())


@register_artifact_loader("generate_subtitles")
def _load_artifacts_generate_subtitles(ctx: PipelineContext) -> None:
    """Load artifacts from completed generate_subtitles step (no-op)."""
    pass


@register_artifact_loader("download_music")
def _load_artifacts_download_music(ctx: PipelineContext) -> None:
    """Load artifacts from completed download_music step (no-op)."""
    pass


async def step_gather_visuals(ctx: PipelineContext):
    async with performance_monitor.measure_step(
        "gather_visuals",
        profile=ctx.profile.description,
        scraped_images_enabled=ctx.profile.use_scraped_images,
        scraped_videos_enabled=ctx.profile.use_scraped_videos,
        stock_images_count=ctx.profile.stock_image_count,
        stock_videos_count=ctx.profile.stock_video_count,
    ):
        logger.info("Executing step: GATHER_VISUALS")

        # Check if visuals already exist from previous run
        visuals_file = ctx.run_paths["gathered_visuals_file"]
        if visuals_file.exists():
            logger.info("Loading existing visuals from previous run")
            ctx.scraped_images, ctx.scraped_videos, ctx.stock_media = load_visuals_info(
                visuals_file
            )
            # Build visuals list from loaded data
            all_visuals = ctx.scraped_images + ctx.scraped_videos
            all_visuals.extend(item.path for item in ctx.stock_media)
            ctx.visuals = all_visuals
            logger.info(
                f"Loaded visuals: Scraped Imgs: {len(ctx.scraped_images)}, "
                f"Vids: {len(ctx.scraped_videos)}, Stock: {len(ctx.stock_media)}."
            )
            return

        # Start resource pre-loading tasks (but not TTS warming yet)
        if ctx.resource_preloader:
            # Start resource pre-loading based on product data
            preload_task_ids = await ctx.resource_preloader.preload_for_product(
                ctx.product, ctx.config, ctx.profile
            )
            ctx.preload_task_ids.extend(preload_task_ids)
            logger.debug(
                f"Started {len(preload_task_ids)} background resource pre-loading tasks"
            )

        project_root = ctx.config.project_root
        scraped_images = []
        scraped_videos = []

        if ctx.profile.use_scraped_images:
            # First try using the downloaded_images array from product data
            scraped_images = [
                project_root / p
                for p in (ctx.product.downloaded_images or [])
                if (project_root / p).exists()
            ]

            # Fallback: scan the images directory if downloaded_images is empty
            if not scraped_images and hasattr(ctx.product, "asin") and ctx.product.asin:
                images_dir = project_root / "outputs" / ctx.product.asin / "images"
                if images_dir.exists():
                    scraped_images = [
                        img_path
                        for img_path in images_dir.glob("*.jpg")
                        if img_path.is_file()
                    ]
                    scraped_images.extend(
                        [
                            img_path
                            for img_path in images_dir.glob("*.png")
                            if img_path.is_file()
                        ]
                    )

        if ctx.profile.use_scraped_videos:
            # First try using the downloaded_videos array from product data
            scraped_videos = [
                project_root / p
                for p in (ctx.product.downloaded_videos or [])
                if (project_root / p).exists()
            ]

            # Fallback: scan the videos directory if downloaded_videos is empty
            if not scraped_videos and hasattr(ctx.product, "asin") and ctx.product.asin:
                videos_dir = project_root / "outputs" / ctx.product.asin / "videos"
                if videos_dir.exists():
                    scraped_videos = [
                        vid_path
                        for vid_path in videos_dir.glob("*.mp4")
                        if vid_path.is_file()
                    ]
        stock_media_fetched: list[Any] = []
        if (ctx.profile.use_stock_images and ctx.profile.stock_image_count > 0) or (
            ctx.profile.use_stock_videos and ctx.profile.stock_video_count > 0
        ):
            fetcher = StockMediaFetcher(
                ctx.config.stock_media_settings,
                ctx.secrets,
                ctx.config.media_settings,
                ctx.config.api_settings,
            )
            keywords = list(
                set(
                    ctx.config.media_settings.stock_media_keywords
                    + (
                        [
                            w
                            for w in (ctx.product.title or "").split()
                            if len(w)
                            >= ctx.config.media_settings.product_title_keyword_min_length  # noqa: E501
                        ]
                    )
                )
            )
            # Check for pre-loaded stock media first
            preloaded_media = None
            if ctx.resource_preloader:
                preloaded_media = ctx.resource_preloader.get_preloaded_stock_media(
                    keywords
                )
                if preloaded_media:
                    logger.debug(
                        "Using pre-loaded stock media from background processing"
                    )

            if preloaded_media:
                # Use pre-loaded media if available
                stock_media_fetched = []
                for media_type, media_list in preloaded_media.items():
                    for media_item in media_list:
                        stock_media_fetched.append(
                            StockMediaInfo(
                                source="Pexels",
                                type=media_type.rstrip("s"),  # 'images' -> 'image'
                                url=media_item.get("url", ""),
                                author=media_item.get("author", "Unknown"),
                                path=Path(media_item.get("path", "")),
                                duration=media_item.get("duration"),
                            )
                        )
                logger.info(
                    f"Using {len(stock_media_fetched)} pre-loaded stock media items"
                )
            else:
                # Fallback to regular fetch if no pre-loaded media
                stock_media_fetched = await fetcher.fetch_and_download_stock(
                    keywords,
                    ctx.profile.stock_image_count,
                    ctx.profile.stock_video_count,
                    ctx.run_paths["assets_dir"],
                    ctx.session,
                )

        all_visuals = scraped_images + scraped_videos
        all_visuals.extend(item.path for item in stock_media_fetched)
        ctx.visuals = all_visuals
        logger.info(
            f"Visuals gathered: Scraped Imgs: {len(scraped_images)}, "
            f"Vids: {len(scraped_videos)}, Stock: {len(stock_media_fetched)}."
        )
        if not ctx.visuals:
            raise PipelineError(
                "No visual inputs were found or gathered for this profile."
            )

        # Validate media requirements for quality video creation
        is_valid, reason = validate_media_requirements(
            scraped_images,
            scraped_videos,
            stock_media_fetched,
            ctx.profile,
            ctx.config,
        )
        logger.info("Media validation: %s", reason)
        if not is_valid:
            raise InsufficientMediaError(
                f"Product '{ctx.product.asin or 'unknown'}' skipped: {reason}"
            )

        # Now that validation passed, start TTS warming
        # (won't waste resources on skipped products)
        if ctx.tts_warmer:
            tts_task_ids = await ctx.tts_warmer.warm_tts_models(ctx.config)
            ctx.preload_task_ids.extend(tts_task_ids)
            logger.debug("Started %d TTS model warming tasks", len(tts_task_ids))

        # Save info file for both debug and resumability
        save_visuals_info(
            scraped_images,
            scraped_videos,
            stock_media_fetched,
            ctx.run_paths,
        )
        logger.info(
            f"Saved gathered visuals info to "
            f"{ctx.run_paths['gathered_visuals_file'].name}"
        )


async def step_generate_script(ctx: PipelineContext):
    async with performance_monitor.measure_step(
        "generate_script",
        product_title_length=len(ctx.product.title or ""),
        llm_model=(
            ctx.config.llm_settings.models[0]
            if ctx.config.llm_settings.models
            else "unknown"
        ),
        target_audience=ctx.config.llm_settings.target_audience,
    ):
        logger.info("Executing step: GENERATE_SCRIPT")

        # Check if script already exists from previous run
        script_file = ctx.run_paths["script_file"]
        if script_file.exists():
            logger.info("Loading existing script from previous run")
            ctx.script = script_file.read_text(encoding="utf-8")
            logger.info(
                f"Loaded existing script from {script_file.name} "
                f"({len(ctx.script or '')} characters)"
            )
            return

        try:
            pillar = ctx.state.get("pillar") or getattr(ctx.product, "pillar", None)
            script_text, template_name = await generate_ai_script(
                ctx.product,
                ctx.config.llm_settings,
                ctx.secrets,
                ctx.session,
                {
                    "script": ctx.run_paths["script_file"],
                    "formatted_prompt": ctx.run_paths["script_prompt"],
                },
                ctx.debug_mode,
                ctx.config.api_settings,
                product_id=ctx.product.asin,
                pillar=pillar,
            )
        except (RuntimeError, ValueError, OSError) as e:
            raise PipelineError(f"Script generation failed: {e}") from e

        if not script_text:
            raise PipelineError("Script generation failed to produce text.")
        ctx.script = sanitize_script(script_text)
        ensure_dirs_exist(ctx.run_paths["script_file"].parent)
        ctx.run_paths["script_file"].write_text(ctx.script, encoding="utf-8")
        if template_name:
            ctx.state["script_template"] = template_name
        logger.info(
            "Script generated (template=%s) and saved to %s",
            template_name,
            ctx.run_paths["script_file"].name,
        )


def _extract_hashtags_from_title(title: str) -> list[str]:
    """Extract hashtags from product title keywords.

    Args:
    ----
        title: Product title to extract hashtags from.

    Returns:
    -------
        List of hashtag strings (without # prefix), always includes 'ad'.

    """
    title_words = (title or "").split()
    hashtags = []
    for word in title_words:
        clean = "".join(c for c in word if c.isalnum())
        # Skip if: too short, all digits, common word, or looks like a year
        if (
            len(clean) < 4
            or clean.isdigit()
            or clean.lower() in HASHTAG_SKIP_WORDS
            or (len(clean) == 4 and clean.isdigit())  # Years like 2026
        ):
            continue
        hashtags.append(clean.capitalize())
        if len(hashtags) >= 3:
            break
    # Always include #ad for advertising disclosure
    hashtags.append("ad")
    return hashtags


def _check_existing_metadata(ctx: PipelineContext) -> bool:
    """Check for and load existing metadata from previous run.

    Args:
    ----
        ctx: Pipeline context to populate with loaded description.

    Returns:
    -------
        True if existing metadata was found and loaded, False otherwise.

    """
    description_file = ctx.run_paths["description_file"]
    product_root = ctx.run_paths["run_root"]
    unified_metadata_path = product_root / "metadata.json"
    platform_metadata_exists = any(
        (product_root / f"metadata_{platform}.json").exists()
        for platform in SUPPORTED_PLATFORMS
    )

    # Load from unified metadata.json first
    if unified_metadata_path.exists():
        logger.info("Loading existing unified metadata from previous run")
        meta = json.loads(unified_metadata_path.read_text(encoding="utf-8"))
        ctx.description = meta.get("description", "")
        logger.info(
            "Loaded existing description from metadata.json (%d characters)",
            len(ctx.description or ""),
        )
        return True

    # Fallback to platform-specific metadata or description.txt
    if platform_metadata_exists or description_file.exists():
        logger.info("Loading existing description/metadata from previous run")
        if description_file.exists():
            ctx.description = description_file.read_text(encoding="utf-8")
            logger.info(
                "Loaded existing description from %s (%d characters)",
                description_file.name,
                len(ctx.description or ""),
            )
        return True

    return False


async def _generate_optimized_metadata(ctx: PipelineContext) -> bool:
    """Generate platform-specific optimized metadata.

    Args:
    ----
        ctx: Pipeline context with product and configuration.

    Returns:
    -------
        True if generation succeeded, False to fall back to unified mode.

    """
    try:
        logger.info(
            "Platform metadata generation enabled, using PlatformMetadataFactory"
        )

        from src.ai.platform_metadata import (
            PlatformMetadataFactory,
            save_metadata_to_file,
        )
        from src.ai.platform_metadata.text_formatter import (
            format_upload_instructions,
        )

        # Extract platform settings from config
        pm_config = ctx.config.description_settings.platform_metadata
        if pm_config is None:
            logger.warning("Platform metadata is None, using unified mode")
            return False

        platform_settings: dict[str, dict] = {}
        if pm_config.youtube is not None:
            platform_settings["youtube"] = pm_config.youtube.model_dump()
        if pm_config.tiktok is not None:
            platform_settings["tiktok"] = pm_config.tiktok.model_dump()
        if pm_config.instagram is not None:
            platform_settings["instagram"] = pm_config.instagram.model_dump()

        if not platform_settings:
            logger.warning(
                "Platform metadata enabled but no platform "
                "configurations found, falling back to unified mode"
            )
            return False

        # Prepare intermediate paths for metadata files
        product_root = ctx.run_paths["run_root"]
        text_dir = ctx.run_paths["description_file"].parent
        intermediate_paths = {
            "description": text_dir / "description.txt",
            # Script path lets caption prompts mirror the closing
            # engagement-bait line into the platform caption (Phase 1.5).
            # Caption templates without the {VIDEO_SCRIPT} placeholder ignore it.
            "script": ctx.run_paths["script_file"],
            "metadata_youtube": product_root / "metadata_youtube.json",
            "metadata_tiktok": product_root / "metadata_tiktok.json",
            "metadata_instagram": product_root / "metadata_instagram.json",
        }

        # Resolve pillar from CLI override or product data
        active_pillar = ctx.state.get("pillar") or getattr(ctx.product, "pillar", None)
        script_cfg = ctx.config.llm_settings.script_templates

        # Generate metadata for all platforms in parallel
        metadata_results = await PlatformMetadataFactory.generate_multi_platform(
            product=ctx.product,
            settings=ctx.config.llm_settings,
            secrets=ctx.secrets,
            session=ctx.session,
            platform_settings=platform_settings,
            intermediate_paths=intermediate_paths,
            debug_mode=ctx.debug_mode,
            api_settings=ctx.config.api_settings,
            narrator_profile=script_cfg.narrator_profile,
            pillar=active_pillar,
            pillar_preambles=script_cfg.pillar_preambles,
        )

        # Save metadata to individual platform files
        saved_count = 0
        for platform, metadata in metadata_results.items():
            if metadata:
                metadata_file = product_root / f"metadata_{platform}.json"
                save_metadata_to_file(metadata, metadata_file)
                logger.info("Saved %s metadata to %s", platform, metadata_file.name)
                saved_count += 1

        if saved_count == 0:
            logger.warning(
                "All platform metadata generation failed, "
                "falling back to unified mode"
            )
            return False

        logger.info(
            "Platform metadata generation complete (%d/%d platforms succeeded)",
            saved_count,
            len(metadata_results),
        )

        # Generate upload instructions (non-critical)
        try:
            instructions_text = format_upload_instructions(
                metadata_results=metadata_results,
                product_id=ctx.product.asin or "unknown",
                video_filename=Path(ctx.run_paths["final_video_output"]).name,
                product_name=ctx.product.title or "Product",
                product_url=ctx.product.url,
            )
            instructions_file = text_dir / "UPLOAD_INSTRUCTIONS.txt"
            instructions_file.write_text(instructions_text, encoding="utf-8")
            logger.info("Generated upload instructions: %s", instructions_file.name)
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning("Failed to generate upload instructions: %s", e)

        # Set ctx.description for backward compatibility
        for platform in SUPPORTED_PLATFORMS:
            metadata = metadata_results.get(platform)
            if metadata is not None:
                ctx.description = metadata.description
                logger.debug("Using %s description for ctx.description", platform)
                break

        return True

    except (RuntimeError, ValueError, OSError) as e:
        logger.warning(
            "Platform metadata generation failed: %s, "
            "falling back to unified description mode",
            e,
            exc_info=ctx.debug_mode,
        )
        return False


async def _generate_unified_metadata(ctx: PipelineContext) -> None:
    """Generate unified metadata for all platforms.

    Args:
    ----
        ctx: Pipeline context with product and configuration.

    Raises:
    ------
        PipelineError: If description generation fails.

    """
    import re
    from datetime import UTC, datetime

    logger.info("Using unified description generation mode")

    try:
        description_text = await generate_ai_description(
            ctx.product,
            ctx.config.llm_settings,
            ctx.secrets,
            ctx.session,
            {"description": ctx.run_paths["description_file"]},
            ctx.debug_mode,
            ctx.config.api_settings,
        )
    except (RuntimeError, ValueError, OSError) as e:
        raise PipelineError(f"Description generation failed: {e}") from e

    if not description_text:
        raise PipelineError("Description generation failed to produce text.")

    # Strip any hashtags from description (LLM may still include them)
    description_clean = re.sub(r"\s*#\w+", "", description_text).strip()
    ctx.description = description_clean

    # Generate hashtags from product title
    hashtags = _extract_hashtags_from_title(ctx.product.title or "")

    # Generate unified metadata file
    product_root = ctx.run_paths["run_root"]
    metadata_dict = {
        "title": ctx.product.title,
        "description": ctx.description,
        "hashtags": hashtags,
        "keywords": [],
        "product_id": ctx.product.asin or "unknown",
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": "unified",
    }

    metadata_file = product_root / "metadata.json"
    with metadata_file.open("w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=2, ensure_ascii=False)

    logger.info("Saved unified metadata to %s", metadata_file.name)


async def step_generate_description(ctx: PipelineContext):
    """Generate AI-powered video description for social media platforms."""
    if not ctx.config.description_settings.enabled:
        logger.info("Description generation is disabled, skipping step")
        return

    async with performance_monitor.measure_step(
        "generate_description",
        product_title_length=len(ctx.product.title or ""),
        target_platforms=",".join(ctx.config.description_settings.target_platforms),
    ):
        logger.info("Executing step: GENERATE_DESCRIPTION")

        # Check for existing metadata from previous run
        if _check_existing_metadata(ctx):
            return

        # Try optimized (platform-specific) mode first
        use_optimized = ctx.config.description_settings.metadata_mode == "optimized"
        if use_optimized and await _generate_optimized_metadata(ctx):
            return

        # Fall back to unified mode
        await _generate_unified_metadata(ctx)


async def step_create_voiceover(ctx: PipelineContext):
    async with performance_monitor.measure_step(
        "create_voiceover",
        script_length=len(ctx.script or ""),
        tts_provider=(
            ctx.config.tts_config.provider_order[0]
            if ctx.config.tts_config.provider_order
            else "unknown"
        ),
    ):
        logger.info("Executing step: CREATE_VOICEOVER")

        # Check if voiceover already exists from previous run
        vo_file = ctx.run_paths["voiceover_file"]
        duration_file = ctx.run_paths["voiceover_duration_file"]
        if vo_file.exists() and duration_file.exists():
            logger.info("Loading existing voiceover from previous run")
            try:
                ctx.voiceover_duration = float(duration_file.read_text())
                logger.info(
                    f"Loaded existing voiceover ({vo_file.name}) "
                    f"with duration: {ctx.voiceover_duration:.2f}s"
                )
                return
            except (ValueError, FileNotFoundError):
                logger.warning("Failed to load voiceover duration, regenerating")

        if ctx.script is None:
            script_path = ctx.run_paths["script_file"]
            if not script_path.exists():
                raise FileNotFoundError(f"Missing required file: {script_path.name}.")
            ctx.script = script_path.read_text(encoding="utf-8")

        try:
            tts_manager = TTSManager(
                ctx.config.tts_config,
                ctx.secrets,
                product_id=ctx.product.asin,
                voice_profile_override=ctx.cli_overrides.get("voice_profile"),
            )
            vo_path = await tts_manager.generate_speech(
                ctx.script, ctx.run_paths["voiceover_file"]
            )
        except (RuntimeError, OSError) as e:
            raise PipelineError(f"TTS generation failed: {e}") from e

        if not vo_path or not vo_path.exists():
            raise PipelineError("TTS generation failed.")

        # Save TTS metadata for pipeline state tracking
        ctx.state["tts_metadata"] = {
            "voice_profile": tts_manager.selected_profile_name,
            "voice_name": tts_manager.selected_voice_name,
        }
        logger.debug(
            "TTS metadata: profile=%s, voice=%s",
            tts_manager.selected_profile_name,
            tts_manager.selected_voice_name,
        )

        # Trim leading/trailing silence from voiceover
        # Whisper normalizes timestamps to start at first speech
        audio_proc = ctx.config.audio_processing
        if audio_proc and audio_proc.silence_removal_enabled:
            ffmpeg_path = ctx.config.ffmpeg_settings.executable_path or "ffmpeg"
            trimmed_vo_path = vo_path.parent / f"{vo_path.stem}_trimmed.wav"

            try:
                import subprocess

                # Build silenceremove filter with config settings
                threshold_db = audio_proc.silence_threshold_db
                min_duration = audio_proc.silence_min_duration_sec
                silence_filter = (
                    f"silenceremove=start_periods=1:start_threshold={threshold_db}dB:start_duration={min_duration},"
                    f"areverse,"
                    f"silenceremove=start_periods=1:start_threshold={threshold_db}dB:start_duration={min_duration},"
                    f"areverse"
                )
                trim_cmd = [
                    ffmpeg_path,
                    "-i",
                    str(vo_path),
                    "-af",
                    silence_filter,
                    "-y",
                    str(trimmed_vo_path),
                ]
                subprocess.run(trim_cmd, check=True, capture_output=True)

                # Replace original with trimmed version
                trimmed_vo_path.replace(vo_path)
                logger.debug(
                    f"Trimmed silence from voiceover "
                    f"(threshold={threshold_db}dB, min_duration={min_duration}s)"
                )
            except (RuntimeError, OSError) as e:
                logger.warning(
                    "Failed to trim silence from voiceover: %s, using original", e
                )
        else:
            logger.debug("Silence removal disabled, skipping voiceover trimming")

        ctx.voiceover_duration = await _get_video_duration(vo_path, ffmpeg_path)
        ensure_dirs_exist(ctx.run_paths["voiceover_duration_file"].parent)
        ctx.run_paths["voiceover_duration_file"].write_text(str(ctx.voiceover_duration))
        logger.info(
            f"Voiceover created ({vo_path.name}) with duration: "
            f"{ctx.voiceover_duration:.2f}s"
        )


async def step_generate_subtitles(ctx: PipelineContext):
    # Handle both dict and object forms of subtitle_settings for performance tracking
    subtitle_enabled_value = (
        ctx.config.subtitle_settings.enabled
        if hasattr(ctx.config.subtitle_settings, "enabled")
        else ctx.config.subtitle_settings.get("enabled", True)
    )

    async with performance_monitor.measure_step(
        "generate_subtitles",
        subtitle_provider="whisper",  # Default subtitle provider
        voiceover_duration=ctx.voiceover_duration or 0.0,
        subtitle_enabled=subtitle_enabled_value,
    ):
        logger.info("Executing step: GENERATE_SUBTITLES")

        # Use the same value for early exit check
        subtitle_enabled = subtitle_enabled_value
        if not subtitle_enabled:
            logger.info("Subtitle generation is disabled in config. Skipping.")
            return

        voiceover_path = ctx.run_paths["voiceover_file"]
        if not voiceover_path.exists():
            raise FileNotFoundError(f"Missing voiceover file at {voiceover_path}.")
        if ctx.script is None:
            script_path = ctx.run_paths["script_file"]
            if not script_path.exists():
                raise FileNotFoundError(f"Missing script file at {script_path}.")
            ctx.script = script_path.read_text(encoding="utf-8")
        if ctx.voiceover_duration is None:
            duration_path = ctx.run_paths["voiceover_duration_file"]
            if not duration_path.exists():
                raise FileNotFoundError(f"Missing duration file at {duration_path}.")
            ctx.voiceover_duration = float(duration_path.read_text())

        # Get merged profile settings (similar to assembler approach)
        merged_profile_settings = ctx.config.get_profile_merged_settings(
            ctx.profile_name, ctx.cli_overrides
        )
        subtitle_settings = merged_profile_settings.subtitle_settings

        # Derive product_id for randomization
        from src.utils import sanitize_filename

        product_id = ctx.product.asin or sanitize_filename(ctx.product.title[:30])

        # Subtitle engine dispatch: "pycaps" path skips SRT/ASS emission, saves
        # a raw Whisper transcript for the downstream burn step, and disables
        # two-part (upper+lower) which is FFmpeg-only in this iteration.
        subtitle_engine = subtitle_settings.subtitle_engine
        two_part_enabled = subtitle_settings.two_part_subtitles.enabled

        # Early availability check: if pycaps is requested but not installed,
        # apply fallback_policy before committing to the pycaps-only path.
        if subtitle_engine == "pycaps":
            from src.video.pycaps_engine import is_pycaps_available

            if not is_pycaps_available():
                pycaps_cfg = subtitle_settings.pycaps
                if pycaps_cfg is None:
                    from src.video.config.subtitle_models import PycapsSettings

                    pycaps_cfg = PycapsSettings()  # type: ignore[call-arg]
                policy = pycaps_cfg.fallback_policy
                if policy == "fallback_ffmpeg":
                    logger.warning(
                        "pycaps is not installed, falling back to ffmpeg "
                        "subtitle engine (fallback_policy='fallback_ffmpeg'). "
                        "Install with `poetry install --with pycaps`."
                    )
                    subtitle_engine = "ffmpeg"
                elif policy == "warn_and_skip":
                    logger.warning(
                        "pycaps is not installed and fallback_policy='warn_and_skip'. "
                        "No subtitles will be generated for this run."
                    )
                    return
                else:
                    raise PipelineError(
                        "subtitle_engine is 'pycaps' but pycaps is not installed. "
                        "Install with `poetry install --with pycaps` or set "
                        "pycaps.fallback_policy to 'fallback_ffmpeg'."
                    )

        if subtitle_engine == "pycaps":
            if two_part_enabled:
                logger.debug(
                    "Two-part subtitles are not supported in pycaps mode. "
                    "Disabling two-part for this run; re-enable by switching "
                    "subtitle_engine back to 'ffmpeg'."
                )
                two_part_enabled = False

            subtitle_dict = subtitle_settings.model_dump()
            transcript_path = ctx.run_paths["whisper_transcript_file"]
            result_path = await create_unified_subtitles(
                voiceover_path,
                ctx.run_paths["subtitle_file"],
                subtitle_dict,
                ctx.config.whisper_settings,
                ctx.config.google_cloud_stt_settings,
                ctx.secrets,
                ctx.script,
                ctx.voiceover_duration,
                ctx.debug_mode,
                ctx.config,
                Path(ctx.run_paths["run_root"])
                / ctx.config.output_structure.product_subdirs.temp,
                product_id,
                transcript_out_path=transcript_path,
            )
            if not result_path or not result_path.exists():
                raise PipelineError(
                    "Pycaps transcript generation failed (no whisper_json " "produced)."
                )
            logger.info("Pycaps transcript ready: %s", result_path.name)
            ctx.state.setdefault("generate_subtitles", {})["engine"] = "pycaps"
            return

        logger.debug(
            "two_part_subtitles_enabled=%s (now two_part_subtitles.enabled)",
            two_part_enabled,
        )

        if two_part_enabled:
            logger.info("Two-part subtitle system enabled, generating dual subtitles")

            from src.video.producer.two_part_subtitles import TwoPartSubtitleHandler

            handler = TwoPartSubtitleHandler(
                ctx=ctx,
                merged_profile_settings=merged_profile_settings,
            )

            lower_path, upper_path = await handler.generate(voiceover_path, product_id)

            lower_failed = not lower_path or not lower_path.exists()
            if handler.config.lower_line.enabled and lower_failed:
                raise PipelineError("Lower subtitle generation failed.")

            if upper_path:
                ctx.run_paths["subtitle_upper_file"] = upper_path

        else:
            # Standard single-line subtitle generation
            subtitle_dict = subtitle_settings.model_dump()
            srt_path = await create_unified_subtitles(
                voiceover_path,
                ctx.run_paths["subtitle_file"],
                subtitle_dict,
                ctx.config.whisper_settings,
                ctx.config.google_cloud_stt_settings,
                ctx.secrets,
                ctx.script,
                ctx.voiceover_duration,
                ctx.debug_mode,
                ctx.config,
                Path(ctx.run_paths["run_root"])
                / ctx.config.output_structure.product_subdirs.temp,
                product_id,
            )
            if not srt_path or not srt_path.exists():
                raise PipelineError("Subtitle generation process failed.")
            logger.info("Subtitles file created: %s", srt_path.name)


async def step_download_music(ctx: PipelineContext):
    async with performance_monitor.measure_step(
        "download_music",
        required_duration=ctx.voiceover_duration or 0.0,
        freesound_enabled=bool(
            ctx.secrets.get(ctx.config.audio_settings.freesound_api_key_env_var)
        ),
        search_query=ctx.config.audio_settings.freesound_search_query,
    ):
        logger.info("Executing step: DOWNLOAD_MUSIC")
        if ctx.voiceover_duration is None:
            duration_file = ctx.run_paths["voiceover_duration_file"]
            if not duration_file.exists():
                raise FileNotFoundError(f"Missing required file: {duration_file.name}.")
            ctx.voiceover_duration = float(duration_file.read_text())

        vo_duration = ctx.voiceover_duration
        logger.info("Required music duration is at least %.2f seconds.", vo_duration)

        providers = _build_audio_providers(ctx.config, ctx.secrets)
        manager = AudioManager(
            providers=providers,
            local_paths=[
                p
                for p in ctx.config.audio_settings.background_music_paths
                if p.exists()
            ],
        )

        music_info = await manager.find_music(
            query=ctx.config.audio_settings.freesound_search_query,
            min_duration=vo_duration,
            max_duration=ctx.config.audio_settings.freesound_max_search_duration_sec,
            max_results=ctx.config.audio_settings.freesound_max_results,
            output_dir=ctx.run_paths["assets_dir"],
            session=ctx.session,
        )

        if music_info:
            if isinstance(music_info.get("path"), Path):
                music_info["path"] = str(music_info["path"])
            ensure_dirs_exist(ctx.run_paths["music_info_file"].parent)
            ctx.run_paths["music_info_file"].write_text(
                json.dumps(music_info, indent=2), encoding="utf-8"
            )
            logger.info(
                "Music info saved. Selected track: %s",
                music_info.get("name", "N/A"),
            )
        else:
            logger.warning("No background music could be found from any source.")


def _build_audio_providers(config: Any, secrets: dict[str, str]) -> list:
    """Build audio provider instances from config.

    If audio_providers is configured, uses that list. Otherwise falls back
    to creating a single FreesoundProvider from legacy freesound_* fields.
    """
    from src.audio.base import BaseAudioProvider

    providers: list[BaseAudioProvider] = []
    audio_settings = config.audio_settings
    provider_configs = getattr(audio_settings, "audio_providers", [])

    if provider_configs:
        for pc in provider_configs:
            if not pc.enabled:
                continue
            try:
                provider = create_audio_provider(
                    pc.name,
                    config=config,
                    secrets=secrets,
                    settings=pc.settings,
                )
                providers.append(provider)
                logger.info("Audio provider loaded: %s", pc.name)
            except ValueError as exc:
                logger.warning("Skipping audio provider '%s': %s", pc.name, exc)
    else:
        # Legacy mode: auto-create FreesoundProvider from freesound_* fields
        try:
            provider = create_audio_provider(
                "freesound",
                config=config,
                secrets=secrets,
            )
            providers.append(provider)
            logger.info("Audio provider loaded: freesound (legacy config)")
        except ValueError:
            pass

    return providers


async def step_assemble_video(ctx: PipelineContext):
    # Handle both dict and object forms of subtitle_settings for performance tracking
    subtitle_enabled_value = (
        ctx.config.subtitle_settings.enabled
        if hasattr(ctx.config.subtitle_settings, "enabled")
        else ctx.config.subtitle_settings.get("enabled", True)
    )

    async with performance_monitor.measure_step(
        "assemble_video",
        visual_count=len(ctx.visuals) if ctx.visuals else 0,
        target_duration=ctx.voiceover_duration or 0.0,
        has_music=ctx.run_paths["music_info_file"].exists(),
        has_subtitles=ctx.run_paths["subtitle_file"].exists()
        and subtitle_enabled_value,
    ):
        logger.info("Executing step: ASSEMBLE_VIDEO")
        if ctx.voiceover_duration is None:
            path = ctx.run_paths["voiceover_duration_file"]
            if not path.exists():
                raise FileNotFoundError(f"Missing required file: {path.name}.")
            ctx.voiceover_duration = float(path.read_text())

        if ctx.visuals is None:
            path = ctx.run_paths["gathered_visuals_file"]
            scraped_imgs, scraped_vids, stock_media = load_visuals_info(path)
            ctx.visuals = scraped_imgs + scraped_vids
            ctx.visuals.extend([item.path for item in stock_media])

        if not ctx.visuals:
            raise PipelineError(
                "No visual files available for assembly after selection process."
            )

        random.shuffle(ctx.visuals)
        logger.info("Final timeline contains %d visual elements.", len(ctx.visuals))
        music_path_str = None
        music_info_path = ctx.run_paths["music_info_file"]
        if music_info_path.exists():
            music_path_str = json.loads(music_info_path.read_text())["path"]
        music_path = (
            Path(music_path_str)
            if music_path_str and Path(music_path_str).exists()
            else None
        )

        # Use the same value for subtitle check
        subtitle_path = (
            ctx.run_paths["subtitle_file"]
            if ctx.run_paths["subtitle_file"].exists() and subtitle_enabled_value
            else None
        )

        assembler = VideoAssembler(ctx.config, debug_mode=ctx.debug_mode)
        assembler.set_profile_settings(
            ctx.profile_name, ctx.cli_overrides
        )  # Apply profile settings with CLI overrides

        # Set product_id for randomization (derive from product data)
        from src.utils import sanitize_filename

        product_id = ctx.product.asin or sanitize_filename(ctx.product.title[:30])
        assembler.set_product_id(product_id)
        # Hook overlay text source: the rendered spoken script. extract_hook_line
        # in overlay_builder pulls the first sentence and caps to max_words.
        # When the script file doesn't exist (rare), the assembler treats the
        # overlay as disabled — same behaviour as `hook_overlay.enabled: false`.
        hook_text: str | None = None
        script_path = ctx.run_paths.get("script_file")
        if script_path and script_path.exists():
            try:
                hook_text = script_path.read_text(encoding="utf-8")
            except OSError as e:
                logger.warning("Could not read script for hook overlay: %s", e)

        # Phase 1.2e: pick a cold-open variant deterministically and persist
        # it for downstream analytics. Variant name lands in pipeline_state.json
        # via _update_state_for_completed_step (state.py).
        from src.video.cold_open_selector import select_cold_open_variant

        cold_open_variant = select_cold_open_variant(
            product_id, ctx.config.video_settings.cold_open_variant_pool
        )
        ctx.state["cold_open_variant"] = cold_open_variant
        logger.info("Cold-open variant for %s: %s", product_id, cold_open_variant)

        try:
            final_video_path = await assembler.assemble_video(
                visual_inputs=ctx.visuals,
                voiceover_audio_path=ctx.run_paths["voiceover_file"],
                music_track_path=music_path,
                output_path=ctx.run_paths["final_video_output"],
                subtitle_path=subtitle_path,
                total_video_duration=ctx.voiceover_duration
                + ctx.config.outro_duration_sec,  # Extra time for music fade-out
                temp_dir=ctx.run_paths["intermediate_base"],
                debug_mode=ctx.debug_mode,
                subtitle_upper_path=ctx.run_paths.get("subtitle_upper_file"),
                hook_text=hook_text,
            )
            if not final_video_path:
                raise PipelineError("Video assembly process failed.")
        except PipelineError:
            raise
        except (RuntimeError, OSError, subprocess.CalledProcessError) as e:
            raise PipelineError(f"Video assembly process failed: {e}") from e

        if ctx.script is None and ctx.run_paths["script_file"].exists():
            ctx.script = ctx.run_paths["script_file"].read_text(encoding="utf-8")

        results = assembler.verify_video(
            video_path=final_video_path,
            expected_duration=ctx.voiceover_duration,
            should_have_subtitles=(subtitle_path is not None),
            script=ctx.script,
            subtitle_path=subtitle_path,
        )
    logger.info("Verification results: %s", results["message"])
    if not results["success"]:
        logger.warning("Verification for %s reported issues.", final_video_path.name)
    logger.info("Video successfully created: %s", final_video_path)


def _build_gemini_adapter_for_pycaps(ctx: PipelineContext, pycaps_settings: Any) -> Any:
    """Construct a GeminiLlm adapter when AI tagging is enabled and the key is present.

    Returns ``None`` when AI tagging is off or the Gemini key is missing.
    Caller registers the adapter via ``LlmProvider.set()`` only when non-None.
    """
    if not getattr(pycaps_settings, "enable_ai_tagging", False):
        return None
    api_key_var = ctx.config.llm_settings.api_key_env_var
    api_key = ctx.secrets.get(api_key_var)
    if not api_key:
        logger.warning(
            "pycaps enable_ai_tagging=true but %s is not set; AI tagger rules "
            "will silently no-op for this run.",
            api_key_var,
        )
        return None

    from src.video.pycaps_engine import GeminiLlm

    return GeminiLlm(
        api_key=api_key,
        model=pycaps_settings.llm_model,
        on_error=pycaps_settings.ai_tagging_on_error,
    )


@register_artifact_loader("burn_pycaps_subtitles")
def _load_artifacts_burn_pycaps_subtitles(ctx: PipelineContext) -> None:
    """Load artifacts from completed burn_pycaps_subtitles step (no-op)."""
    pass


def _handle_pycaps_burn_failure(
    fallback_policy: Literal["raise", "fallback_ffmpeg", "warn_and_skip"],
    msg: str,
) -> None:
    """Resolve any pycaps burn-step failure per ``fallback_policy``.

    Covers every failure in ``step_burn_pycaps_subtitles`` that leaves the video
    without captions: a missing transcript, a missing assembled video, or a
    runtime render failure. ``warn_and_skip`` logs and returns, so the caller
    keeps the caption-less video. ``raise`` and ``fallback_ffmpeg`` both raise
    ``PipelineError``: a burn-step failure must not silently ship a caption-less
    video reported as success. ``fallback_ffmpeg`` only re-routes to the ffmpeg
    subtitle engine when pycaps is *unavailable* (handled earlier in
    ``step_generate_subtitles``), not after the assembler has already run
    without captions, so there is nothing to fall back to here.

    Call sites use ``return _handle_pycaps_burn_failure(...)`` so the caller
    can't accidentally continue past a skipped burn.
    """
    if fallback_policy == "warn_and_skip":
        logger.warning(
            "%s Keeping caption-less video (fallback_policy='warn_and_skip').", msg
        )
        return
    raise PipelineError(msg)


async def step_burn_pycaps_subtitles(ctx: PipelineContext):
    """Burn pycaps animated captions onto the assembled video.

    Runs after ``assemble_video``. Short-circuits when the profile's
    ``subtitle_engine`` is not ``"pycaps"``. Any failure that would leave the
    video without captions (missing transcript, missing assembled video, or a
    runtime render failure) goes through ``_handle_pycaps_burn_failure``:
    ``warn_and_skip`` keeps the caption-less video, while ``raise`` and
    ``fallback_ffmpeg`` both abort. ``fallback_ffmpeg`` cannot re-burn here (the
    assembler already ran without captions), so it fails loudly rather than ship
    a caption-less video. The pycaps-unavailable case still degrades to ffmpeg
    earlier, in ``step_generate_subtitles``.
    """
    merged_profile_settings = ctx.config.get_profile_merged_settings(
        ctx.profile_name, ctx.cli_overrides
    )
    subtitle_settings = merged_profile_settings.subtitle_settings

    if subtitle_settings.subtitle_engine != "pycaps":
        logger.debug(
            "Skipping burn_pycaps_subtitles (subtitle_engine=%s)",
            subtitle_settings.subtitle_engine,
        )
        return

    pycaps_settings = subtitle_settings.pycaps
    if pycaps_settings is None:
        # Engine is pycaps but no sub-settings object — construct defaults.
        # All PycapsSettings fields have defaults; mypy warns because the
        # pydantic plugin isn't configured for this project.
        from src.video.config.subtitle_models import PycapsSettings

        pycaps_settings = PycapsSettings()  # type: ignore[call-arg]

    async with performance_monitor.measure_step(
        "burn_pycaps_subtitles",
        pycaps_renderer=pycaps_settings.renderer,
        pycaps_template_pool_size=len(pycaps_settings.template_pool or []),
    ):
        logger.info("Executing step: BURN_PYCAPS_SUBTITLES")

        transcript_path = ctx.run_paths.get("whisper_transcript_file")
        final_video_path = ctx.run_paths["final_video_output"]

        if transcript_path is None or not transcript_path.exists():
            msg = (
                f"Pycaps mode requested but whisper transcript is missing at "
                f"{transcript_path}. Did generate_subtitles run in pycaps mode?"
            )
            return _handle_pycaps_burn_failure(pycaps_settings.fallback_policy, msg)

        if not final_video_path.exists():
            msg = (
                f"Assembled video not found at {final_video_path}, cannot "
                f"burn pycaps captions."
            )
            return _handle_pycaps_burn_failure(pycaps_settings.fallback_policy, msg)

        # Reuse the two-part helper's visual bounds calculation even though
        # two-part is disabled in pycaps mode. The helper is standalone.
        from src.video.producer.two_part_subtitles import TwoPartSubtitleHandler

        bounds_handler = TwoPartSubtitleHandler(
            ctx=ctx, merged_profile_settings=merged_profile_settings
        )
        visual_bounds = bounds_handler.calculate_visual_bounds()

        # Derive product id and output target.
        from src.utils import sanitize_filename

        product_id = ctx.product.asin or sanitize_filename(ctx.product.title[:30])
        burned_output = final_video_path.with_name(
            final_video_path.stem + "_pycaps.mp4"
        )

        # Run the renderer in a worker thread — the library itself is sync.
        from src.video.pycaps_engine import (
            PycapsRenderer,
            PycapsUnavailableError,
        )

        # Extract safe zone from subtitle settings (flows via extra="allow"
        # from _build_subtitle_base). Clamps pycaps max_width_ratio so
        # captions stay inside platform UI overlay boundaries.
        safe_zone = getattr(subtitle_settings, "safe_zone", None)

        # Wire Gemini into pycaps' LlmProvider when AI tagging is enabled and
        # the key is present. LlmProvider is a process-wide singleton, so set
        # it once per render. Skipped silently when the key is missing —
        # pycaps' tagger then falls back to the default Gpt provider, which
        # is itself disabled without PYCAPS_OPENAI_API_KEY, so AI rules just
        # no-op (matches today's behavior).
        gemini_adapter = _build_gemini_adapter_for_pycaps(ctx, pycaps_settings)
        if gemini_adapter is not None:
            from pycaps.ai import LlmProvider

            LlmProvider.set(gemini_adapter)
            logger.info(
                "pycaps AI word tagging enabled (model=%s, on_error=%s)",
                pycaps_settings.llm_model,
                pycaps_settings.ai_tagging_on_error,
            )

        renderer = PycapsRenderer()
        try:
            result = await asyncio.to_thread(
                renderer.render,
                final_video_path,
                transcript_path,
                burned_output,
                product_id,
                visual_bounds,
                pycaps_settings,
                safe_zone=safe_zone,
            )
        except PycapsUnavailableError as e:
            msg = (
                f"pycaps library is not installed: {e}. Install with "
                f"`poetry install --with pycaps`."
            )
            if pycaps_settings.fallback_policy in ("raise", "fallback_ffmpeg"):
                # fallback_ffmpeg should have been caught earlier in
                # step_generate_subtitles; if we got here, something is wrong.
                raise PipelineError(msg) from e
            logger.warning(msg + " Skipping burn; keeping FFmpeg output.")
            return

        if not result.success:
            msg = (
                f"pycaps render failed: {result.error}. "
                f"template={result.template_used}, renderer={result.renderer_used}"
            )
            return _handle_pycaps_burn_failure(pycaps_settings.fallback_policy, msg)

        # Swap the burned output over the original final video atomically.
        # Path.replace is atomic on POSIX when source and dest are on the
        # same filesystem.
        burned_output.replace(final_video_path)
        ai_call_count = gemini_adapter.call_count if gemini_adapter is not None else 0
        logger.info(
            "Replaced %s with pycaps-burned video "
            "(template=%s, renderer=%s, wall=%.2fs, peak=%.0f MB, ai_calls=%d)",
            final_video_path.name,
            result.template_used,
            result.renderer_used,
            result.wall_time_sec,
            result.peak_rss_mb,
            ai_call_count,
        )

        # Save per-run metadata for audit / pipeline_state.json.
        metadata_path = ctx.run_paths.get("pycaps_metadata_file")
        if metadata_path is not None:
            metadata = {
                "engine": "pycaps",
                "template": result.template_used,
                "renderer": result.renderer_used,
                "wall_time_sec": round(result.wall_time_sec, 3),
                "peak_rss_mb": round(result.peak_rss_mb, 1),
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            ctx.state["pycaps_metadata"] = metadata
