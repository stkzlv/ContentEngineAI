# src/video/producer/steps.py
"""Pipeline step implementations for video production."""

import asyncio
import json
import logging
import random
import shutil
from pathlib import Path
from typing import Any

from src.ai.description_generator import generate_description as generate_ai_description
from src.ai.script_generator import generate_script as generate_ai_script
from src.audio.freesound_client import FreesoundClient
from src.utils import ensure_dirs_exist
from src.utils.memory_mapped_io import copy_file_mmap, is_file_suitable_for_mmap
from src.utils.performance import performance_monitor
from src.utils.script_sanitizer import sanitize_script
from src.video.assembler import VideoAssembler
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


def _load_artifacts_gather_visuals(ctx: PipelineContext):
    """Load artifacts from completed gather_visuals step."""
    try:
        visuals_file = ctx.run_paths["gathered_visuals_file"]
        if visuals_file.exists():
            ctx.scraped_images, ctx.scraped_videos, ctx.stock_media = load_visuals_info(
                visuals_file
            )
            logger.debug("Loaded artifacts for skipped step 'gather_visuals'")
    except Exception as e:
        logger.warning(f"Error loading gather_visuals artifacts: {e}")


def _load_artifacts_generate_script(ctx: PipelineContext):
    """Load artifacts from completed generate_script step."""
    try:
        script_file = ctx.run_paths["script_file"]
        if script_file.exists():
            ctx.script = script_file.read_text(encoding="utf-8")
            logger.debug("Loaded artifacts for skipped step 'generate_script'")
    except Exception as e:
        logger.warning(f"Error loading generate_script artifacts: {e}")


def _load_artifacts_generate_description(ctx: PipelineContext):
    """Load artifacts from completed generate_description step."""
    try:
        description_file = ctx.run_paths["description_file"]
        if description_file.exists():
            ctx.description = description_file.read_text(encoding="utf-8")
            logger.debug("Loaded artifacts for skipped step 'generate_description'")
    except Exception as e:
        logger.warning(f"Error loading generate_description artifacts: {e}")


def _load_artifacts_create_voiceover(ctx: PipelineContext):
    """Load artifacts from completed create_voiceover step."""
    try:
        duration_file = ctx.run_paths["voiceover_duration_file"]
        if duration_file.exists():
            ctx.voiceover_duration = float(duration_file.read_text())
            logger.debug("Loaded artifacts for skipped step 'create_voiceover'")
    except Exception as e:
        logger.warning(f"Error loading create_voiceover artifacts: {e}")


def _load_artifacts_generate_subtitles(ctx: PipelineContext):
    """Load artifacts from completed generate_subtitles step."""
    logger.debug("Loaded artifacts for skipped step 'generate_subtitles'")


def _load_artifacts_download_music(ctx: PipelineContext):
    """Load artifacts from completed download_music step."""
    logger.debug("Loaded artifacts for skipped step 'download_music'")


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
        logger.info(f"Media validation: {reason}")
        if not is_valid:
            raise InsufficientMediaError(
                f"Product '{ctx.product.asin or 'unknown'}' skipped: {reason}"
            )

        # Now that validation passed, start TTS warming
        # (won't waste resources on skipped products)
        if ctx.tts_warmer:
            tts_task_ids = await ctx.tts_warmer.warm_tts_models(ctx.config)
            ctx.preload_task_ids.extend(tts_task_ids)
            logger.debug(f"Started {len(tts_task_ids)} TTS model warming tasks")

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
            script_text = await generate_ai_script(
                ctx.product,
                ctx.config.llm_settings,
                ctx.secrets,
                ctx.session,
                {"script": ctx.run_paths["script_file"]},
                ctx.debug_mode,
                ctx.config.api_settings,
            )
        except Exception as e:
            raise PipelineError(f"Script generation failed: {e}") from e

        if not script_text:
            raise PipelineError("Script generation failed to produce text.")
        ctx.script = sanitize_script(script_text)
        ensure_dirs_exist(ctx.run_paths["script_file"].parent)
        ctx.run_paths["script_file"].write_text(ctx.script, encoding="utf-8")
        logger.info(
            f"Script generated and saved to {ctx.run_paths['script_file'].name}"
        )


async def step_generate_description(ctx: PipelineContext):
    """Generate AI-powered video description for social media platforms."""
    # Check if description generation is enabled
    if not ctx.config.description_settings.enabled:
        logger.info("Description generation is disabled, skipping step")
        return

    async with performance_monitor.measure_step(
        "generate_description",
        product_title_length=len(ctx.product.title or ""),
        target_platforms=",".join(ctx.config.description_settings.target_platforms),
    ):
        logger.info("Executing step: GENERATE_DESCRIPTION")

        # Check if description already exists from previous run
        description_file = ctx.run_paths["description_file"]
        if description_file.exists():
            logger.info("Loading existing description from previous run")
            ctx.description = description_file.read_text(encoding="utf-8")
            logger.info(
                f"Loaded existing description from {description_file.name} "
                f"({len(ctx.description or '')} characters)"
            )
            return

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
        except Exception as e:
            raise PipelineError(f"Description generation failed: {e}") from e

        if not description_text:
            raise PipelineError("Description generation failed to produce text.")

        ctx.description = description_text.strip()
        ensure_dirs_exist(ctx.run_paths["description_file"].parent)
        ctx.run_paths["description_file"].write_text(ctx.description, encoding="utf-8")
        logger.info(
            f"Description generated and saved to "
            f"{ctx.run_paths['description_file'].name}"
        )


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
            tts_manager = TTSManager(ctx.config.tts_config, ctx.secrets)
            vo_path = await tts_manager.generate_speech(
                ctx.script, ctx.run_paths["voiceover_file"]
            )
        except Exception as e:
            raise PipelineError(f"TTS generation failed: {e}") from e

        if not vo_path or not vo_path.exists():
            raise PipelineError("TTS generation failed.")

        ctx.voiceover_duration = await _get_video_duration(
            vo_path, ctx.config.ffmpeg_settings.executable_path or "ffmpeg"
        )
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
        profile_subtitle_settings = merged_profile_settings["subtitle_settings"]

        # Derive product_id for randomization
        from src.utils import sanitize_filename

        product_id = ctx.product.asin or sanitize_filename(ctx.product.title[:30])

        # Check if two-part subtitle system is enabled
        # Handle both nested dict and flat key structures
        two_part_config = profile_subtitle_settings.get("two_part_subtitles", {})
        logger.debug(f"DEBUG: two_part_config = {two_part_config}")
        logger.debug(
            "DEBUG: profile_subtitle_settings keys = "
            f"{list(profile_subtitle_settings.keys())}"
        )

        if isinstance(two_part_config, dict) and "enabled" in two_part_config:
            two_part_enabled = two_part_config.get("enabled", False)
            logger.debug(
                f"DEBUG: Using nested structure, two_part_enabled = {two_part_enabled}"
            )
        else:
            # Fallback to flat structure
            two_part_enabled = profile_subtitle_settings.get(
                "two_part_subtitles_enabled", False
            )
            logger.debug(
                "DEBUG: Using flat structure, two_part_subtitles_enabled key = "
                f"{profile_subtitle_settings.get('two_part_subtitles_enabled')}"
            )
            logger.debug(f"DEBUG: Final two_part_enabled = {two_part_enabled}")

        if two_part_enabled:
            logger.info("Two-part subtitle system enabled, generating dual subtitles")

            # Import static subtitle generator
            from src.video.subtitle_utils import create_static_upper_subtitle

            # Generate upper line (static product info)
            # Handle both nested dict and flat key structures
            if isinstance(two_part_config, dict) and "upper_line" in two_part_config:
                upper_config = two_part_config.get("upper_line", {})
                upper_enabled = upper_config.get("enabled", True)
            else:
                # Fallback to flat structure (settings at profile level,
                # not in subtitle_settings)
                upper_enabled = merged_profile_settings.get(
                    "two_part_subtitles_upper_enabled", True
                )
                upper_config = {
                    "enabled": upper_enabled,
                    "source_field": merged_profile_settings.get(
                        "two_part_subtitles_upper_source_field",
                        "shortened_affiliate_link",
                    ),
                    "anchor": merged_profile_settings.get(
                        "two_part_subtitles_upper_anchor", "above_content"
                    ),
                    "margin": merged_profile_settings.get(
                        "two_part_subtitles_upper_margin", 0.03
                    ),
                    "font_size_scale": merged_profile_settings.get(
                        "two_part_subtitles_upper_font_size_scale", 0.75
                    ),
                    "style_preset": merged_profile_settings.get(
                        "two_part_subtitles_upper_style_preset", "minimal"
                    ),
                    "use_full_duration": merged_profile_settings.get(
                        "two_part_subtitles_upper_use_full_duration", False
                    ),
                    "randomize_effects": merged_profile_settings.get(
                        "two_part_subtitles_upper_randomize_effects", False
                    ),
                }

            # Generate lower line (voiceover subtitles) first - needed for CTA detection
            lower_path = None
            # Handle both nested dict and flat key structures
            if isinstance(two_part_config, dict) and "lower_line" in two_part_config:
                lower_config = two_part_config.get("lower_line", {})
                lower_enabled = lower_config.get("enabled", True)
            else:
                # Fallback to flat structure
                lower_enabled = profile_subtitle_settings.get(
                    "two_part_subtitles_lower_enabled", True
                )
                lower_config = {
                    "enabled": lower_enabled,
                    "anchor": profile_subtitle_settings.get(
                        "two_part_subtitles_lower_anchor", "below_content"
                    ),
                    "margin": profile_subtitle_settings.get(
                        "two_part_subtitles_lower_margin", 0.05
                    ),
                }

            # Calculate visual bounds for content-aware positioning
            from src.video.subtitle_positioning import VisualBounds

            video_top = (
                ctx.profile.video_top_position_percent
                or ctx.profile.image_top_position_percent
                or 0.07
            )
            video_height = ctx.profile.video_content_height_percent or 0.8
            video_width = ctx.profile.image_width_percent or 0.9

            visual_bounds = VisualBounds(
                x=(1.0 - video_width) / 2,
                y=video_top,
                width=video_width,
                height=video_height,
            )

            logger.debug(
                f"Visual bounds for subtitles: "
                f"y={video_top:.2%}, height={video_height:.2%}"
            )

            if lower_enabled:
                # Update subtitle settings for lower line positioning
                lower_subtitle_settings = profile_subtitle_settings.copy()
                lower_subtitle_settings["anchor"] = lower_config.get(
                    "anchor", "below_content"
                )
                lower_subtitle_settings["margin"] = lower_config.get("margin", 0.05)

                # Override with custom style if provided
                if lower_config.get("custom_style"):
                    lower_subtitle_settings.update(lower_config["custom_style"])

                lower_path = await create_unified_subtitles(
                    voiceover_path,
                    ctx.run_paths["subtitle_file"],
                    lower_subtitle_settings,
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
                    visual_bounds,
                )

                if not lower_path or not lower_path.exists():
                    raise PipelineError("Lower subtitle generation failed.")
                logger.info(f"Lower subtitle created: {lower_path.name}")

            # Generate upper line (static URL) - after lower subtitle for CTA detection
            logger.debug(f"DEBUG: upper_enabled={upper_enabled}")
            if upper_enabled:
                # Check for custom URL first (overrides product URL)
                custom_url = profile_subtitle_settings.get(
                    "two_part_subtitles_upper_custom_url"
                )

                if custom_url:
                    upper_text = custom_url
                    logger.info(f"Using custom URL for upper subtitle: {custom_url}")
                else:
                    # Get product URL from data
                    source_field = upper_config.get(
                        "source_field", "shortened_affiliate_link"
                    )
                    product_data_dict = ctx.product.__dict__
                    upper_text = product_data_dict.get(source_field, "")

                    if not upper_text:
                        # Fallback to other URL fields
                        for fallback_field in [
                            "shortened_affiliate_link",
                            "affiliate_link",
                            "url",
                        ]:
                            upper_text = product_data_dict.get(fallback_field, "")
                            if upper_text:
                                logger.info(
                                    f"Using fallback field '{fallback_field}' "
                                    "for upper subtitle"
                                )
                                break

                if upper_text:
                    # Apply URL prefix replacement if configured
                    prefix_replace = merged_profile_settings.get(
                        "two_part_subtitles_upper_prefix_replace"
                    )
                    if prefix_replace:
                        # Replace "https://" with the configured prefix
                        if upper_text.startswith("https://"):
                            upper_text = (
                                prefix_replace + upper_text[8:]
                            )  # Remove "https://"
                        elif upper_text.startswith("http://"):
                            upper_text = (
                                prefix_replace + upper_text[7:]
                            )  # Remove "http://"

                    # Determine output format
                    subtitle_format = profile_subtitle_settings.get(
                        "subtitle_format", "srt"
                    )
                    upper_output_path = ctx.run_paths["subtitle_file"].with_name(
                        f"subtitle_upper.{subtitle_format}"
                    )

                    upper_path = create_static_upper_subtitle(
                        text=upper_text,
                        output_path=upper_output_path,
                        subtitle_settings=profile_subtitle_settings,
                        video_config=ctx.config,
                        format_type=subtitle_format,
                        product_id=product_id,
                        voiceover_duration=ctx.voiceover_duration,
                        visual_bounds=visual_bounds,
                        # Pass lower subtitle for CTA detection
                        lower_subtitle_path=lower_path,
                    )

                    if upper_path and upper_path.exists():
                        logger.info(f"Upper subtitle created: {upper_path.name}")
                        # Store upper subtitle path for assembler
                        ctx.run_paths["subtitle_upper_file"] = upper_path
                    else:
                        logger.warning(
                            "Failed to generate upper subtitle, "
                            "continuing with lower only"
                        )
                else:
                    logger.warning(
                        f"No data found for upper subtitle field '{source_field}'"
                    )

        else:
            # Standard single-line subtitle generation
            srt_path = await create_unified_subtitles(
                voiceover_path,
                ctx.run_paths["subtitle_file"],
                profile_subtitle_settings,
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
            logger.info(f"Subtitles file created: {srt_path.name}")


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
        logger.info(f"Required music duration is at least {vo_duration:.2f} seconds.")
        music_info = None

        if ctx.secrets.get(ctx.config.audio_settings.freesound_api_key_env_var):
            fs_client = FreesoundClient(**ctx.secrets)
            duration_filter = (
                f"duration:[{int(vo_duration)} TO "
                f"{ctx.config.audio_settings.freesound_max_search_duration_sec}]"
            )
            tracks = await fs_client.search_music(
                query=ctx.config.audio_settings.freesound_search_query,
                filters=duration_filter,
                max_results=ctx.config.audio_settings.freesound_max_results,
                timeout_sec=ctx.config.audio_settings.freesound_api_timeout_sec,
            )
            if not tracks:
                logger.warning(
                    "Dynamic duration search yielded no results. "
                    "Falling back to general search."
                )
                tracks = await fs_client.search_music(
                    query=ctx.config.audio_settings.freesound_search_query,
                    filters=ctx.config.audio_settings.freesound_filters,
                    max_results=ctx.config.audio_settings.freesound_max_results,
                    timeout_sec=ctx.config.audio_settings.freesound_api_timeout_sec,
                )
            if tracks:
                for track in sorted(tracks, key=lambda t: t.duration):
                    if track.duration >= vo_duration:
                        logger.info(
                            f"Found suitable track: '{track.name}' "
                            f"(Duration: {track.duration}s)"
                        )
                        try:
                            _, music_info = await fs_client.download_full_sound_oauth2(
                                track.id, ctx.run_paths["assets_dir"], ctx.session
                            ) or (None, None)
                            if not music_info:
                                (
                                    _,
                                    music_info,
                                ) = await fs_client.download_sound_preview_with_api_key(
                                    track, ctx.run_paths["assets_dir"], ctx.session
                                ) or (
                                    None,
                                    None,
                                )
                            if music_info:
                                break
                        except Exception as e:
                            logger.warning(f"Failed to download from Freesound: {e}")
                            # Continue to try next track, will fall back to local if
                            # all fail

        if not music_info and ctx.config.audio_settings.background_music_paths:
            local_path = random.choice(  # noqa: S311
                [
                    p
                    for p in ctx.config.audio_settings.background_music_paths
                    if p.exists()
                ]
            )
            if local_path:
                ensure_dirs_exist(ctx.run_paths["assets_dir"])
                dest_path = ctx.run_paths["assets_dir"] / local_path.name

                # Use memory-mapped I/O for large files, fallback to shutil.copy
                if is_file_suitable_for_mmap(
                    local_path, min_size=1024 * 1024
                ):  # 1MB threshold
                    logger.debug(
                        f"Using memory-mapped copy for large file: {local_path.name}"
                    )
                    copy_success = copy_file_mmap(local_path, dest_path)
                    if not copy_success:
                        logger.warning(
                            "Memory-mapped copy failed, falling back to standard copy"
                        )
                        shutil.copy(local_path, dest_path)
                else:
                    logger.debug(
                        f"Using standard copy for small file: {local_path.name}"
                    )
                    shutil.copy(local_path, dest_path)
                # Generate complete attribution metadata per R6 (Requirement 6)
                music_info = {
                    "source": "Local",
                    "type": "Music",
                    "path": str(dest_path),
                    "name": local_path.stem,
                    "author": "Unknown",
                    "license": "Local File",
                    "url": "",
                    "id": "",
                }

        if music_info:
            if isinstance(music_info.get("path"), Path):
                music_info["path"] = str(music_info["path"])
            ensure_dirs_exist(ctx.run_paths["music_info_file"].parent)
            ctx.run_paths["music_info_file"].write_text(
                json.dumps(music_info, indent=2), encoding="utf-8"
            )
            logger.info(
                f"Music info saved. Selected track: {music_info.get('name', 'N/A')}"
            )
        else:
            logger.warning("No background music could be found from any source.")


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
        logger.info(f"Final timeline contains {len(ctx.visuals)} visual elements.")
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
        try:
            final_video_path = await assembler.assemble_video(
                visual_inputs=ctx.visuals,
                voiceover_audio_path=ctx.run_paths["voiceover_file"],
                music_track_path=music_path,
                output_path=ctx.run_paths["final_video_output"],
                subtitle_path=subtitle_path,
                total_video_duration=ctx.voiceover_duration
                + ctx.config.duration_padding_sec,  # Add padding to prevent cutoff
                temp_dir=ctx.run_paths["intermediate_base"],
                debug_mode=ctx.debug_mode,
                subtitle_upper_path=ctx.run_paths.get("subtitle_upper_file"),
            )
            if not final_video_path:
                raise PipelineError("Video assembly process failed.")
        except Exception as e:
            if isinstance(e, PipelineError):
                raise
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
    logger.info(f"Verification results: {results['message']}")
    if not results["success"]:
        logger.warning(f"Verification for {final_video_path.name} reported issues.")
    logger.info(f"Video successfully created: {final_video_path}")
