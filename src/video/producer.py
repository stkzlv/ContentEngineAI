# src/video/producer.py
import argparse
import asyncio
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import aiohttp
from dotenv import load_dotenv

from src.ai.script_generator import generate_script as generate_ai_script
from src.audio.freesound_client import FreesoundClient
from src.scraper.amazon.scraper import ProductData
from src.utils import cleanup_temp_dirs, ensure_dirs_exist, sanitize_filename
from src.utils.background_processing import (
    ResourcePreloader,
    TTSWarmer,
    cleanup_global_background_processor,
    get_background_processor,
)
from src.utils.connection_pool import get_http_session
from src.utils.memory_mapped_io import copy_file_mmap, is_file_suitable_for_mmap
from src.utils.performance import PerformanceHistoryManager, performance_monitor
from src.utils.script_sanitizer import sanitize_script
from src.video.assembler import VideoAssembler
from src.video.config_validator import validate_config_and_exit_on_error
from src.video.pipeline_graph import PipelineGraph, StepStatus
from src.video.stock_media import StockMediaFetcher, StockMediaInfo
from src.video.subtitle_utils import create_unified_subtitles
from src.video.tts import TTSManager
from src.video.video_config import VideoConfig, VideoProfile, load_video_config

logger = logging.getLogger(__name__)


def setup_logging(config: VideoConfig, debug_mode: bool = False) -> Path:
    """Set up logging to both console and file.

    Args:
    ----
        config: Video configuration containing log directory path
        debug_mode: Whether to enable debug logging

    Returns:
    -------
        Path to the log file

    """
    log_level = logging.DEBUG if debug_mode else logging.INFO

    # Create log directory
    log_dir = config.general_video_producer_log_dir_path
    ensure_dirs_exist(log_dir)

    # Use fixed log filename that gets overwritten on each run
    log_file = log_dir / "producer.log"

    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)

    # Set up file handler (overwrite mode)
    file_handler = logging.FileHandler(
        log_file,
        mode="w",  # Overwrite file on each run
        encoding="utf-8",
    )
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(log_level)

    # Configure root logger
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("numba").setLevel(logging.WARNING)
    if not debug_mode:
        for lib in ["httpx", "google", "aiohttp", "urllib3", "asyncio", "hpack"]:
            logging.getLogger(lib).setLevel(logging.WARNING)

    logger.info(
        f"Logging configured - Level: {logging.getLevelName(log_level)}, "
        f"File: {log_file}"
    )
    return log_file


STEP_GATHER_VISUALS = "gather_visuals"
STEP_GENERATE_SCRIPT = "generate_script"
STEP_CREATE_VOICEOVER = "create_voiceover"
STEP_GENERATE_SUBTITLES = "generate_subtitles"
STEP_DOWNLOAD_MUSIC = "download_music"
STEP_ASSEMBLE_VIDEO = "assemble_video"

VALID_STEPS = [
    STEP_GATHER_VISUALS,
    STEP_GENERATE_SCRIPT,
    STEP_CREATE_VOICEOVER,
    STEP_GENERATE_SUBTITLES,
    STEP_DOWNLOAD_MUSIC,
    STEP_ASSEMBLE_VIDEO,
]


class PipelineError(Exception):
    """Custom exception for pipeline failures."""

    pass


class InsufficientMediaError(PipelineError):
    """Exception raised when product has insufficient media for video creation."""

    pass


def validate_media_requirements(
    scraped_images: list, scraped_videos: list, stock_media: list, profile
) -> tuple[bool, str]:
    """Validate if gathered media meets minimum requirements for video creation.

    Args:
    ----
        scraped_images: List of scraped image paths
        scraped_videos: List of scraped video paths
        stock_media: List of stock media items
        profile: Video profile configuration object

    Returns:
    -------
        Tuple of (is_valid: bool, reason: str)

    """
    scraped_image_count = len(scraped_images)
    scraped_video_count = len(scraped_videos)

    # Count stock media by type
    stock_image_count = sum(
        1 for item in stock_media if getattr(item, "type", "image") == "image"
    )
    stock_video_count = sum(
        1 for item in stock_media if getattr(item, "type", "image") == "video"
    )

    # Total counts including stock media
    total_image_count = scraped_image_count + stock_image_count
    total_video_count = scraped_video_count + stock_video_count
    total_media = total_image_count + total_video_count

    uses_scraped_videos = getattr(profile, "use_scraped_videos", True)

    # Minimum requirements for quality video creation
    MIN_TOTAL_MEDIA = 3
    MIN_IMAGES_IF_NO_VIDEO = 5
    MIN_IMAGES_WITH_VIDEO = 2

    # Check basic minimum
    if total_media < MIN_TOTAL_MEDIA:
        msg = (
            f"Insufficient total media: {total_media} items "
            f"(minimum {MIN_TOTAL_MEDIA})"
        )
        return (False, msg)

    # If profile doesn't use videos, need more images for slideshow
    if (
        not uses_scraped_videos or total_video_count == 0
    ) and total_image_count < MIN_IMAGES_IF_NO_VIDEO:
        if not uses_scraped_videos:
            msg = (
                f"Profile excludes videos, need {MIN_IMAGES_IF_NO_VIDEO} images "
                f"but only have {total_image_count}"
            )
        else:
            msg = (
                f"No videos found, need at least {MIN_IMAGES_IF_NO_VIDEO} images "
                f"but only have {total_image_count}"
            )
        return (False, msg)

    # If we have videos, we can work with fewer images
    if total_video_count > 0 and total_image_count < MIN_IMAGES_WITH_VIDEO:
        msg = (
            f"Have {total_video_count} video(s) but only {total_image_count} image(s), "
            f"need at least {MIN_IMAGES_WITH_VIDEO}"
        )
        return (False, msg)

    # Warn for borderline cases but allow processing
    if total_media == MIN_TOTAL_MEDIA:
        logger.warning(
            f"Minimal media count ({total_media}) - video quality may be limited"
        )

    msg = (
        f"Media validation passed: {total_image_count} images, "
        f"{total_video_count} videos, "
        f"{len(stock_media)} stock items"
    )
    return (True, msg)


class PipelineContext:
    def __init__(
        self,
        product: ProductData,
        profile: VideoProfile,
        profile_name: str,
        config: VideoConfig,
        secrets: dict,
        session: aiohttp.ClientSession,
        run_paths: dict,
        debug_mode: bool,
    ):
        self.product = product
        self.profile = profile
        self.profile_name = profile_name
        self.config = config
        self.secrets = secrets
        self.session = session
        self.run_paths = run_paths
        self.debug_mode = debug_mode
        self.visuals: list[Path] | None = None
        self.script: str | None = None
        self.voiceover_duration: float | None = None
        self.state: dict[str, Any] = {}

        # Background processing support
        self.background_processor: Any | None = None
        self.resource_preloader: Any | None = None
        self.tts_warmer: Any | None = None
        self.preload_task_ids: list[str] = []

        # Pipeline state synchronization
        self._state_lock = asyncio.Lock()

        # Additional attributes for pipeline steps
        self.scraped_images: list[Path] = []
        self.scraped_videos: list[Path] = []
        self.stock_media: list[Any] = []


def _clean_producer_files(
    run_paths: dict[str, Path], config: VideoConfig, product_id: str, profile_name: str
) -> None:
    """Clean only producer-generated files, preserving scraper input files."""
    from src.utils import sanitize_filename

    logger = logging.getLogger(__name__)
    product_root = run_paths["run_root"]

    # Get the product files configuration to know what to clean
    safe_profile_name = sanitize_filename(profile_name)
    files = config.output_structure.product_files

    # Producer-generated files to remove (preserve scraper inputs like data.json,
    # images/, videos/)
    producer_files_to_remove = [
        product_root / files.script,  # script.txt
        product_root / files.voiceover,  # voiceover.wav
        product_root / files.subtitles,  # subtitles.srt
        product_root / "subtitles_content_aware.ass",  # content-aware subtitle file
        product_root
        / files.final_video.format(
            product_id=product_id, profile=safe_profile_name
        ),  # video_{product_id}_{profile}.mp4
        product_root / f"video_{safe_profile_name}.mp4",  # old naming pattern
        product_root / files.metadata,  # metadata.json
        product_root
        / files.ffmpeg_log.format(
            profile=safe_profile_name
        ),  # ffmpeg_command.log with profile
        product_root / files.performance,  # performance.json
        product_root / config.path_config.temp_dir,  # temp/ directory
        product_root / config.path_config.music_dir,  # music/ directory
        product_root / config.path_config.gathered_visuals,  # internal producer file
        product_root / files.attribution,  # attributions file
    ]

    # Add debug files using configurable patterns
    debug_patterns = config.path_config.cleanup.debug_file_patterns

    for pattern in debug_patterns:
        for file_path in product_root.glob(pattern):
            producer_files_to_remove.append(file_path)

    removed_count = 0
    for file_path in producer_files_to_remove:
        if file_path.exists():
            try:
                if file_path.is_file():
                    file_path.unlink()
                    logger.debug(f"Removed file: {file_path.name}")
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
                    logger.debug(f"Removed directory: {file_path.name}")
                removed_count += 1
            except OSError as e:
                logger.warning(f"Could not remove {file_path}: {e}")

    logger.info(f"Cleaned {removed_count} producer-generated files/directories")


def get_video_run_paths(
    config: VideoConfig, product_identifier: str, profile_name: str
) -> dict[str, Path]:
    """Generate video production paths using the configurable directory structure."""
    # Use the new path generation method from VideoConfig
    paths = config.get_video_project_paths(product_identifier, profile_name)

    # Map new paths to legacy path names for backward compatibility
    legacy_paths = {
        "run_root": paths["project_root"],
        "intermediate_base": paths["working_dir"],
        # Keep audio assets as "assets" for compatibility
        "assets_dir": paths["audio_dir"],
        # Keep text assets as "info" for compatibility
        "info_dir": paths["text_dir"],
        "script_file": paths["script"],
        "voiceover_file": paths["voiceover"],
        "voiceover_duration_file": paths["text_dir"] / "voiceover_duration.txt",
        "gathered_visuals_file": paths["text_dir"]
        / config.path_config.gathered_visuals,
        "music_info_file": paths["text_dir"] / "music_choice.json",
        "subtitle_file": paths["subtitles"],
        "final_video_output": paths["final_video"],
        "attribution_file": paths["attribution"],
        "state_file": paths["pipeline_state"],
    }

    # Add additional paths that may be needed
    legacy_paths.update(
        {
            "visual_dir": paths["visual_dir"],  # New visual assets directory
            "ffmpeg_log": paths["ffmpeg_log"],  # FFmpeg command log
        }
    )

    return legacy_paths


async def _save_pipeline_state(ctx: PipelineContext):
    """Saves the current pipeline state to a JSON file."""
    # Check if pipeline metadata should be created
    create_metadata = True
    try:
        create_metadata = (
            getattr(ctx.config.debug_settings, "create_pipeline_metadata", True)
            if hasattr(ctx.config, "debug_settings") and ctx.config.debug_settings
            else True
        )
    except Exception:
        create_metadata = True

    if not create_metadata:
        logger.debug("Pipeline metadata creation disabled")
        return

    state_file = ctx.run_paths["state_file"]
    try:
        ensure_dirs_exist(state_file.parent)
        # Use default=str to handle Path objects during serialization
        state_file.write_text(
            json.dumps(ctx.state, indent=2, default=str), encoding="utf-8"
        )
        logger.debug(f"Saved pipeline state to {state_file.name}")
    except Exception as e:
        logger.error(f"Failed to save pipeline state: {e}")


async def _load_pipeline_state(ctx: PipelineContext) -> bool:
    """Loads and verifies an existing pipeline state file."""
    state_file = ctx.run_paths["state_file"]
    if not state_file.exists():
        logger.info("No existing state file found. Starting a new run.")
        ctx.state = {}
        return False

    try:
        logger.info(f"Loading existing state from {state_file.name}")
        state_data = json.loads(state_file.read_text(encoding="utf-8"))

        # Verify that all artifacts for completed steps still exist
        for step, data in state_data.items():
            if data.get("status") == "done":
                for key, path_str in data.get("artifacts", {}).items():
                    if not Path(path_str).exists():
                        logger.warning(
                            f"State is invalid. Artifact '{key}' for step '{step}' "
                            f"not found at '{path_str}'. "
                            f"Restarting from step '{step}'."
                        )
                        # Truncate state up to the failed step
                        valid_steps = VALID_STEPS[: VALID_STEPS.index(step)]
                        ctx.state = {
                            k: v for k, v in state_data.items() if k in valid_steps
                        }
                        async with ctx._state_lock:
                            await _save_pipeline_state(ctx)  # Save the truncated state
                        return True  # State was loaded, but it's partial

        ctx.state = state_data
        logger.info("Successfully loaded and verified existing pipeline state.")
        return True
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(
            f"Could not parse state file {state_file.name}, starting fresh: {e}"
        )
        ctx.state = {}
        return False


async def _update_state_after_step(ctx: PipelineContext, step_name: str):
    """Updates the state dictionary with the artifacts of a completed step."""
    artifacts = {}
    if step_name == STEP_GATHER_VISUALS:
        artifacts["gathered_visuals_file"] = ctx.run_paths["gathered_visuals_file"]
    elif step_name == STEP_GENERATE_SCRIPT:
        artifacts["script_file"] = ctx.run_paths["script_file"]
    elif step_name == STEP_CREATE_VOICEOVER:
        artifacts["voiceover_file"] = ctx.run_paths["voiceover_file"]
        artifacts["voiceover_duration_file"] = ctx.run_paths["voiceover_duration_file"]
    elif step_name == STEP_GENERATE_SUBTITLES:
        artifacts["subtitle_file"] = ctx.run_paths["subtitle_file"]
    elif step_name == STEP_DOWNLOAD_MUSIC:
        artifacts["music_info_file"] = ctx.run_paths["music_info_file"]
    elif step_name == STEP_ASSEMBLE_VIDEO:
        artifacts["final_video_output"] = ctx.run_paths["final_video_output"]

    ctx.state[step_name] = {
        "status": "done",
        "artifacts": {k: str(v) for k, v in artifacts.items()},
    }
    logger.debug(f"Updated state for completed step: {step_name}")


def _load_artifacts_from_state(ctx: PipelineContext, step_name: str) -> bool:
    """Loads artifacts from a completed step's state into the pipeline context."""
    state_entry = ctx.state.get(step_name, {})
    if state_entry.get("status") != "done":
        return False

    logger.debug(f"Loading artifacts for skipped step '{step_name}' into context.")
    try:
        if step_name == STEP_GATHER_VISUALS:
            path = Path(state_entry["artifacts"]["gathered_visuals_file"])
            scraped_imgs, scraped_vids, stock_media = load_visuals_info(path)
            ctx.visuals = scraped_imgs + scraped_vids
            ctx.visuals.extend([item.path for item in stock_media])
        elif step_name == STEP_GENERATE_SCRIPT:
            path = Path(state_entry["artifacts"]["script_file"])
            ctx.script = path.read_text(encoding="utf-8")
        elif step_name == STEP_CREATE_VOICEOVER:
            path = Path(state_entry["artifacts"]["voiceover_duration_file"])
            ctx.voiceover_duration = float(path.read_text())
        # Other steps don't load data into context; subsequent steps use their
        # files directly.
    except (KeyError, FileNotFoundError) as e:
        logger.error(
            f"Failed to load artifact for step '{step_name}': {e}. "
            f"This may cause downstream failures."
        )
        return False
    return True


def save_visuals_info(
    scraped_imgs: list[Path],
    scraped_vids: list[Path],
    stock_media: list[StockMediaInfo],
    run_paths: dict,
):
    data = {
        "scraped_images": [str(p) for p in scraped_imgs],
        "scraped_videos": [str(p) for p in scraped_vids],
        "stock_media": [
            {**item.__dict__, "path": str(item.path)} for item in stock_media
        ],
    }
    ensure_dirs_exist(run_paths["gathered_visuals_file"].parent)
    run_paths["gathered_visuals_file"].write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )


def load_visuals_info(
    path: Path,
) -> tuple[list[Path], list[Path], list[StockMediaInfo]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing gathered visuals info file: {path}. "
            f"Please run the 'gather_visuals' step first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))

    def resolve_path(rel_path: str) -> Path:
        return Path(rel_path)

    scraped_imgs = [resolve_path(p) for p in data.get("scraped_images", []) if p]
    scraped_vids = [resolve_path(p) for p in data.get("scraped_videos", []) if p]
    stock_media = []
    for item_dict in data.get("stock_media", []):
        try:
            if item_dict.get("path"):
                item_dict["path"] = resolve_path(item_dict["path"])
                stock_media.append(StockMediaInfo(**item_dict))
        except TypeError as e:
            logger.warning(
                f"Skipping stock media item due to unexpected keyword argument: "
                f"{e}. Item: {item_dict}"
            )
    return scraped_imgs, scraped_vids, stock_media


async def _get_video_duration(video_path: Path, ffmpeg_path: str) -> float:
    ffprobe_path = ffmpeg_path.replace("ffmpeg", "ffprobe")
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.warning(f"ffprobe failed for {video_path.name}: {stderr.decode()}")
            return 0.0
        return float(stdout.strip())
    except Exception as e:
        logger.error(f"Error getting duration for {video_path.name}: {e}")
        return 0.0


async def execute_pipeline_parallel(ctx: PipelineContext) -> bool:
    """Execute pipeline steps using parallel execution framework.

    Args:
    ----
        ctx: Pipeline context with all necessary data

    Returns:
    -------
        True if pipeline completed successfully, False otherwise

    """
    logger.info("Using parallel pipeline execution framework")

    # Check which steps are already completed
    completed_steps = set()
    if ctx.run_paths["state_file"].exists():
        try:
            state_data = json.loads(ctx.run_paths["state_file"].read_text())
            completed_steps = {
                step_name
                for step_name, step_info in state_data.items()
                if step_info.get("status") == "done"
            }
            logger.info(
                f"Found {len(completed_steps)} already completed steps: "
                f"{completed_steps}"
            )
        except Exception as e:
            logger.warning(f"Could not load existing pipeline state: {e}")
            completed_steps = set()

    # Create pipeline graph with dependencies
    pipeline = PipelineGraph()

    # Add steps with proper dependencies
    pipeline.add_step("gather_visuals", lambda ctx: step_gather_visuals(ctx), set())

    pipeline.add_step(
        "generate_script", lambda ctx: step_generate_script(ctx), {"gather_visuals"}
    )

    pipeline.add_step(
        "create_voiceover", lambda ctx: step_create_voiceover(ctx), {"generate_script"}
    )

    # These two steps can run in parallel - they only depend on voiceover
    pipeline.add_step(
        "generate_subtitles",
        lambda ctx: step_generate_subtitles(ctx),
        {"create_voiceover"},
    )

    pipeline.add_step(
        "download_music", lambda ctx: step_download_music(ctx), {"create_voiceover"}
    )

    pipeline.add_step(
        "assemble_video",
        lambda ctx: step_assemble_video(ctx),
        {"generate_subtitles", "download_music"},
    )

    # Skip already completed steps
    if completed_steps:
        for step_name in completed_steps:
            if pipeline.has_step(step_name):
                pipeline.skip_step(step_name)
                logger.info(f"Skipping already completed step: {step_name}")

                # Load artifacts for skipped steps
                if step_name == "gather_visuals":
                    _load_artifacts_gather_visuals(ctx)
                elif step_name == "generate_script":
                    _load_artifacts_generate_script(ctx)
                elif step_name == "create_voiceover":
                    _load_artifacts_create_voiceover(ctx)
                elif step_name == "generate_subtitles":
                    _load_artifacts_generate_subtitles(ctx)
                elif step_name == "download_music":
                    _load_artifacts_download_music(ctx)

    # Execute pipeline with parallel execution
    try:
        results = await pipeline.execute_pipeline(context=ctx)

        # Check results and update pipeline state
        failed_steps = [r for r in results if r.status == StepStatus.FAILED]
        if failed_steps:
            for failed_result in failed_steps:
                logger.error(
                    f"Step '{failed_result.step_name}' failed: {failed_result.error}"
                )
            return False

        # Update state for newly completed steps with synchronization
        async with ctx._state_lock:
            for result in results:
                if result.status == StepStatus.COMPLETED:
                    step_name = result.step_name
                    await _update_state_after_step(ctx, step_name)
                    logger.info(f"Step '{step_name}' completed successfully")

            await _save_pipeline_state(ctx)
        return True

    except InsufficientMediaError:
        # Re-raise InsufficientMediaError so main handler can process it cleanly
        raise
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        return False


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
            scraped_images, scraped_videos, stock_media_fetched, ctx.profile
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
    async with performance_monitor.measure_step(
        "generate_subtitles",
        subtitle_provider="whisper",  # Default subtitle provider
        voiceover_duration=ctx.voiceover_duration or 0.0,
        subtitle_enabled=ctx.config.subtitle_settings.enabled,
    ):
        logger.info("Executing step: GENERATE_SUBTITLES")
        if not ctx.config.subtitle_settings.enabled:
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

        # Create unified subtitle configuration from legacy settings
        ctx.config.subtitle_settings.model_dump()

        srt_path = await create_unified_subtitles(
            voiceover_path,
            ctx.run_paths["subtitle_file"],
            ctx.config.subtitle_settings,
            ctx.config.whisper_settings,
            ctx.config.google_cloud_stt_settings,
            ctx.secrets,
            ctx.script,
            ctx.voiceover_duration,
            ctx.debug_mode,
            ctx.config,  # Pass video config for ASS generation
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
                music_info = {
                    "path": str(dest_path),
                    "author": "Local File",
                    "source": "Local",
                    "name": local_path.stem,
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
    async with performance_monitor.measure_step(
        "assemble_video",
        visual_count=len(ctx.visuals) if ctx.visuals else 0,
        target_duration=ctx.voiceover_duration or 0.0,
        has_music=ctx.run_paths["music_info_file"].exists(),
        has_subtitles=ctx.run_paths["subtitle_file"].exists()
        and ctx.config.subtitle_settings.enabled,
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

        subtitle_path = (
            ctx.run_paths["subtitle_file"]
            if ctx.run_paths["subtitle_file"].exists()
            and ctx.config.subtitle_settings.enabled
            else None
        )

        assembler = VideoAssembler(ctx.config, debug_mode=ctx.debug_mode)
        assembler.set_profile_settings(ctx.profile_name)  # Apply profile settings
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


async def create_video_for_product(
    config: VideoConfig,
    product: ProductData,
    profile_name: str,
    secrets: dict,
    session: aiohttp.ClientSession,
    debug_mode: bool,
    clean_run: bool,
    debug_step_target: str | None,
):
    product_id = product.asin or sanitize_filename(product.title[:30])
    logger.info(f"--- Starting video for '{product_id}' profile '{profile_name}' ---")

    # Initialize performance history manager
    history_manager = PerformanceHistoryManager(
        history_dir=config.global_output_root_path / "performance_history",
        max_runs=100,  # Configurable retention
    )

    # Set history manager on the global performance monitor
    performance_monitor.history_manager = history_manager

    # Generate run ID for this pipeline execution
    import uuid

    run_id = str(uuid.uuid4())[:8]  # Short ID for readability

    # Start performance monitoring for the entire pipeline
    performance_monitor.start_pipeline(
        run_id=run_id, product_id=product_id, profile_name=profile_name
    )

    step = ""
    run_paths = get_video_run_paths(config, product_id, profile_name)
    successful_run = False

    if clean_run and run_paths["run_root"].exists():
        logger.info(
            f"--clean flag set. Removing producer-generated files from: "
            f"{run_paths['run_root']}"
        )
        try:
            _clean_producer_files(run_paths, config, product_id, profile_name)
        except OSError as e:
            logger.error(f"Error cleaning producer files: {e}")
            raise PipelineError("Could not clean producer files for fresh run.") from e

    try:
        profile = config.get_profile(profile_name)
        ensure_dirs_exist(run_paths["run_root"])

        ctx = PipelineContext(
            product,
            profile,
            profile_name,
            config,
            secrets,
            session,
            run_paths,
            debug_mode,
        )

        # Initialize background processing with configuration
        opt_settings = config.optimization_settings
        bg_processor_params = {}
        if opt_settings:
            bg_processor_params = {
                "max_concurrent_tasks": opt_settings.background_max_concurrent_tasks,
                "thread_pool_workers": opt_settings.background_thread_pool_workers,
                "max_recent_completed": opt_settings.background_max_recent_completed,
            }

        async with get_background_processor(**bg_processor_params) as bg_processor:
            ctx.background_processor = bg_processor
            ctx.resource_preloader = ResourcePreloader(bg_processor)
            ctx.tts_warmer = TTSWarmer(bg_processor)

            await _load_pipeline_state(ctx)

        if debug_step_target:
            target_index = VALID_STEPS.index(debug_step_target)
            for i in range(target_index):
                step_to_load = VALID_STEPS[i]
                if ctx.state.get(step_to_load, {}).get("status") == "done":
                    logger.info(
                        f"Loading prerequisites for '{debug_step_target}': "
                        f"Loading artifacts from '{step_to_load}'."
                    )
                    if not _load_artifacts_from_state(ctx, step_to_load):
                        raise PipelineError(
                            f"Cannot run step '{debug_step_target}': failed to load "
                            f"required artifacts from preceding step '{step_to_load}'."
                        )
                else:
                    raise PipelineError(
                        f"Cannot run step '{debug_step_target}': preceding step "
                        f"'{step_to_load}' "
                        f"is not complete. Run it first."
                    )
            steps_to_run = [debug_step_target]
        else:
            steps_to_run = VALID_STEPS

        # Use parallel pipeline execution unless debugging specific step
        if debug_step_target:
            # For debugging specific steps, use sequential execution
            for current_step in steps_to_run:
                step = current_step

                if (
                    debug_step_target is None
                    and ctx.state.get(current_step, {}).get("status") == "done"  # type: ignore[unreachable]
                ):
                    logger.info(f"Skipping step '{current_step}': Already completed.")  # type: ignore[unreachable]
                    _load_artifacts_from_state(ctx, current_step)
                    continue

                # Ensure directories for the step's outputs exist
                for path in run_paths.values():
                    if isinstance(path, Path):
                        ensure_dirs_exist(path.parent)

                if step == STEP_GATHER_VISUALS:
                    await step_gather_visuals(ctx)
                elif step == STEP_GENERATE_SCRIPT:
                    await step_generate_script(ctx)
                elif step == STEP_CREATE_VOICEOVER:
                    await step_create_voiceover(ctx)
                elif step == STEP_DOWNLOAD_MUSIC:
                    await step_download_music(ctx)
                elif step == STEP_GENERATE_SUBTITLES:
                    await step_generate_subtitles(ctx)
                elif step == STEP_ASSEMBLE_VIDEO:
                    await step_assemble_video(ctx)

                async with ctx._state_lock:
                    await _update_state_after_step(ctx, step)
                    await _save_pipeline_state(ctx)
        else:
            # Use parallel pipeline execution for normal runs
            successful_run = await execute_pipeline_parallel(ctx)
            if not successful_run:
                raise PipelineError("Parallel pipeline execution failed")

        successful_run = True
        logger.info(
            f"<<< SUCCESS: Video for '{product_id}': "
            f"{run_paths.get('final_video_output', 'N/A')}"
        )

        # Save performance metrics for successful runs
        if debug_mode:
            # Check if performance metrics should be created
            create_metrics = True
            try:
                create_metrics = (
                    getattr(config.debug_settings, "create_performance_metrics", True)
                    if hasattr(config, "debug_settings") and config.debug_settings
                    else True
                )
            except Exception:
                create_metrics = True

            if create_metrics:
                performance_metrics_path = (
                    run_paths["run_root"] / "performance_metrics.json"
                )
                performance_monitor.save_metrics(performance_metrics_path)

        # Mark pipeline as successful for history tracking
        performance_monitor.finish_pipeline(success=True)

        # Clean up background processing
        if ctx.background_processor:
            summary = ctx.background_processor.get_summary()
            logger.debug(f"Background processing summary: {summary}")
            await cleanup_global_background_processor()

        return run_paths.get("final_video_output")

    except InsufficientMediaError as e:
        logger.warning(f"Product skipped due to insufficient media: {e}")
        # Mark as skipped, not failed - this is expected for some products
        performance_monitor.finish_pipeline(success=False, error_message=str(e))
        # Clean up background processing on skip
        await cleanup_global_background_processor()
        # Return special value to indicate skip
        return "SKIPPED"
    except (FileNotFoundError, PipelineError, KeyError) as e:
        logger.error(f"Pipeline stopped at step '{step}': {e}", exc_info=debug_mode)
        # Mark pipeline as failed for history tracking
        performance_monitor.finish_pipeline(success=False, error_message=str(e))
        # Clean up background processing on failure
        await cleanup_global_background_processor()
        return None
    except Exception as e:
        logger.error(
            f"An unexpected error occurred in pipeline for '{product_id}': {e}",
            exc_info=True,
        )
        # Mark pipeline as failed for history tracking
        performance_monitor.finish_pipeline(success=False, error_message=str(e))
        # Clean up background processing on failure
        await cleanup_global_background_processor()
        return None
    finally:
        # Log performance summary
        summary = performance_monitor.get_pipeline_summary()
        if summary:
            logger.info(
                f"Pipeline performance: "
                f"{summary.get('total_duration', 0):.2f}s total, "
                f"{summary.get('steps_completed', 0)} steps, "
                f"Memory: {summary.get('total_memory_delta_mb', 0):+.1f}MB"
            )

        if (
            successful_run
            and not debug_mode
            and run_paths
            and run_paths["intermediate_base"].exists()
        ):
            logger.info("Successful run; cleaning up intermediate files.")
            cleanup_temp_dirs(run_paths["intermediate_base"])
        elif debug_mode:
            logger.info(
                f"Debug mode: Intermediate files preserved in "
                f"{run_paths.get('run_root')}"
            )
        elif not successful_run:
            logger.warning(
                f"Run failed. Files preserved in "
                f"{run_paths.get('run_root')} for resume."
            )


def discover_products_for_batch(outputs_dir: Path) -> list[tuple[Path, ProductData]]:
    """Discover products in the outputs directory for batch processing.

    Args:
    ----
        outputs_dir: Directory to scan for product subdirectories

    Returns:
    -------
        List of (product_dir_path, ProductData) tuples for valid products

    """
    products: list[tuple[Path, ProductData]] = []

    if not outputs_dir.exists():
        logger.warning(f"Outputs directory does not exist: {outputs_dir}")
        return products

    for product_dir in outputs_dir.iterdir():
        if not product_dir.is_dir():
            continue

        # Skip global directories (cache, logs, reports, etc.)
        if product_dir.name in {
            "cache",
            "logs",
            "reports",
            "coverage",
            "error_logs",
            "output",
            "outputs",
            "performance_history",
            "unknown_product",
        }:
            continue

        data_file = product_dir / "data.json"
        if not data_file.exists():
            logger.debug(f"Skipping {product_dir.name}: no data.json found")
            continue

        try:
            product_data = json.loads(data_file.read_text(encoding="utf-8"))
            if isinstance(product_data, list):
                # Handle list format - take first product
                if product_data:
                    product = ProductData(**product_data[0])
                else:
                    logger.warning(f"Empty product list in {data_file}")
                    continue
            else:
                product = ProductData(**product_data)

            products.append((product_dir, product))
            logger.debug(f"Found valid product: {product_dir.name}")

        except Exception as e:
            logger.warning(f"Failed to load product data from {data_file}: {e}")
            continue

    logger.info(f"Discovered {len(products)} valid products for batch processing")
    return products


async def main():
    parser = argparse.ArgumentParser(
        description="Generate promotional videos for e-commerce products."
    )
    parser.add_argument(
        "products_file",
        type=Path,
        nargs="?",
        help="Path to JSON file with product data (not required with --batch).",
    )
    parser.add_argument(
        "profile",
        type=str,
        nargs="?",
        help="Video profile name from config (not required with --batch).",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all products found in outputs directory.",
    )
    parser.add_argument(
        "--batch-profile",
        type=str,
        help="Video profile to use for batch processing (required with --batch).",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to scan for products (default: outputs).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop batch processing on first failure.",
    )
    parser.add_argument(
        "--product-index", type=int, help="0-based index of product in JSON list."
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument(
        "--step",
        type=str,
        choices=VALID_STEPS,
        help="Run a single, specific pipeline step.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Force a clean run by deleting the existing output directory "
            "before starting."
        ),
    )
    parser.add_argument(
        "--subtitle-format",
        choices=["srt", "ass"],
        help="Subtitle format: srt (default) or ass (with animations).",
    )
    parser.add_argument(
        "--ass-karaoke",
        action="store_true",
        help="Enable karaoke word highlighting (ASS format only).",
    )
    parser.add_argument(
        "--ass-fade",
        action="store_true",
        help="Enable fade-in/out effects (ASS format only).",
    )
    args = parser.parse_args()

    # Validate argument combinations
    if args.batch:
        if not args.batch_profile:
            parser.error("--batch-profile is required when using --batch")
        if args.products_file or args.profile:
            parser.error(
                "products_file and profile arguments cannot be used with --batch"
            )
    else:
        if not args.products_file or not args.profile:
            parser.error(
                "products_file and profile are required when not using --batch"
            )
        if args.batch_profile or args.fail_fast:
            parser.error(
                "--batch-profile and --fail-fast can only be used with --batch"
            )

    project_root = Path(__file__).resolve().parent.parent.parent
    load_dotenv(project_root / ".env")

    # Load config first to get log directory path
    try:
        config = load_video_config(project_root / "config" / "video_producer.yaml")
    except Exception as e:
        # Fallback logging setup if config fails
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logger.critical(f"Config loading failed, using fallback logging: {e}")
        sys.exit(1)

    # Set up logging to both console and file
    log_file = setup_logging(config, args.debug)
    logger.info(f"Video producer started - Log file: {log_file}")

    # Validate configuration early to catch errors before processing
    logger.info("Validating configuration and runtime dependencies...")
    try:
        validate_config_and_exit_on_error(config)
    except SystemExit:
        logger.critical(f"Complete log saved to: {log_file}")
        raise

    # Apply CLI arguments to config
    if args.subtitle_format:
        config.subtitle_settings.subtitle_format = args.subtitle_format
        logger.info(f"CLI override: subtitle_format = {args.subtitle_format}")

    if args.ass_karaoke:
        config.subtitle_settings.ass_enable_karaoke = True
        logger.info("CLI override: ass_enable_karaoke = True")

    if args.ass_fade:
        config.subtitle_settings.ass_enable_fade = True
        logger.info("CLI override: ass_enable_fade = True")

    try:
        secrets = {
            name: os.getenv(name)
            for name in [
                config.llm_settings.api_key_env_var,
                config.stock_media_settings.pexels_api_key_env_var,
                config.audio_settings.freesound_api_key_env_var,
                "GOOGLE_APPLICATION_CREDENTIALS",
                config.audio_settings.freesound_client_id_env_var,
                config.audio_settings.freesound_client_secret_env_var,
                config.audio_settings.freesound_refresh_token_env_var,
            ]
            if name and os.getenv(name)
        }
    except Exception as e:
        logger.critical(f"Config/Secrets Error: {e}", exc_info=True)
        logger.critical(f"Complete log saved to: {log_file}")
        sys.exit(1)

    if not shutil.which(config.ffmpeg_settings.executable_path or "ffmpeg"):
        logger.error("FFmpeg not found in PATH or at specified executable_path.")
        logger.error(f"Complete log saved to: {log_file}")
        sys.exit(1)
    try:
        if args.batch:
            # Batch mode: discover products from outputs directory
            # Resolve outputs_dir relative to project root to handle working
            # directory changes
            if args.outputs_dir.is_absolute():
                outputs_path = args.outputs_dir
            else:
                outputs_path = project_root / args.outputs_dir
            discovered_products = discover_products_for_batch(outputs_path)
            if not discovered_products:
                logger.error(f"No valid products found in {outputs_path}")
                sys.exit(1)

            # Create products list with directory info for batch processing
            products_list = list(discovered_products)
            profile_name = args.batch_profile
        else:
            # Single product mode: load from file
            # Fix path resolution: resolve relative to project root, not current
            # working
            # directory
            # This handles cases where Botasaurus changes the working directory to
            # outputs/
            if args.products_file.is_absolute():
                products_file_path = args.products_file
            else:
                # Resolve relative paths against the original project root
                products_file_path = project_root / args.products_file
            product_data = json.loads(products_file_path.read_text(encoding="utf-8"))
            raw_products = [
                ProductData(**p)
                for p in (
                    product_data if isinstance(product_data, list) else [product_data]
                )
            ]
            # For single mode, we don't have directory info, so use a placeholder path
            placeholder_path = Path(".")  # Use current directory as placeholder
            products_list = [(placeholder_path, product) for product in raw_products]
            profile_name = args.profile
    except Exception as e:
        error_msg = f"Failed to load products: {e}"
        if not args.batch:
            error_msg = (
                f"Failed to load or validate products from {products_file_path}: {e}"
            )
        logger.critical(error_msg, exc_info=True)
        logger.critical(f"Complete log saved to: {log_file}")
        sys.exit(1)

    # Handle product index for single product mode only
    if args.batch and args.product_index is not None:
        logger.error("--product-index cannot be used with --batch mode")
        sys.exit(1)

    indices = (
        [args.product_index]
        if args.product_index is not None
        and 0 <= args.product_index < len(products_list)
        else range(len(products_list))
    )
    if args.product_index is not None and not indices:
        logger.error(
            f"Product index {args.product_index} out of range for file with "
            f"{len(products_list)} products."
        )
        sys.exit(1)

    succeeded, failed, skipped = 0, 0, 0
    skipped_products = []
    failed_products = []
    session = await get_http_session()  # Use global connection pool

    # Enhanced progress reporting for batch mode
    total_products = len(indices)
    if args.batch:
        logger.info(
            f"Starting batch processing of {total_products} products with "
            f"profile '{profile_name}'"
        )

    for i, idx in enumerate(indices):
        product_dir, product = products_list[idx]
        product_id = product.asin or product.title or f"product_{idx}"

        # Enhanced progress reporting
        if args.batch:
            logger.info(f"[{i+1}/{total_products}] Processing product: {product_id}")

        try:
            result_path = await asyncio.wait_for(
                create_video_for_product(
                    config,
                    product,
                    profile_name,
                    secrets,
                    session,
                    args.debug,
                    args.clean,
                    args.step,
                ),
                timeout=config.pipeline_timeout_sec,
            )
        except TimeoutError:
            logger.error(
                f"Pipeline timed out after {config.pipeline_timeout_sec} seconds "
                f"for product {product_id}"
            )
            result_path = None
        except Exception as e:
            logger.error(
                f"Unexpected error processing product {product_id}: {e}", exc_info=True
            )
            result_path = None

        if result_path == "SKIPPED":
            skipped += 1
            skipped_products.append(product_id)
            if args.batch:
                logger.info(
                    f"[{i+1}/{total_products}] Skipped {product_id} "
                    f"(insufficient media)"
                )
        elif result_path:
            succeeded += 1
            if args.batch:
                logger.info(
                    f"[{i+1}/{total_products}] Successfully completed {product_id}"
                )
        elif not args.step:
            failed += 1
            failed_products.append(product_id)
            if args.batch:
                logger.error(f"[{i+1}/{total_products}] Failed to process {product_id}")
                if args.fail_fast:
                    logger.error(
                        f"Stopping batch processing due to --fail-fast "
                        f"(failed on product {product_id})"
                    )
                    break

        if i < len(indices) - 1:
            delay = random.uniform(  # noqa: S311
                config.video_settings.inter_product_delay_min_sec,
                config.video_settings.inter_product_delay_max_sec,
            )
            await asyncio.sleep(delay)

    logger.info("\n--- Run Summary ---")
    if args.batch:
        logger.info(f"Batch Processing Summary (Profile: {profile_name})")
    logger.info(f"Total Products Processed: {len(indices)}")
    logger.info(f"Succeeded: {succeeded}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped: {skipped}")
    if skipped_products:
        logger.info(
            f"Skipped products (insufficient media): {', '.join(skipped_products)}"
        )
    if failed_products:
        logger.info(f"Failed products: {', '.join(failed_products)}")
    if args.step:
        logger.info(f"NOTE: Run was limited to debug step '{args.step}'.")
    if args.batch and args.fail_fast and failed > 0:
        logger.info("NOTE: Batch processing stopped early due to --fail-fast.")

    logger.info("Video producer completed successfully")
    logger.info(f"Complete log saved to: {log_file}")

    # Ensure all log messages are flushed
    for handler in logging.getLogger().handlers:
        handler.flush()


if __name__ == "__main__":
    asyncio.run(main())
