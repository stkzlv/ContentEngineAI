# src/video/producer/orchestration.py
"""Pipeline orchestration and high-level execution logic."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import aiohttp

from src.scraper.amazon.scraper import ProductData
from src.utils import cleanup_temp_dirs, ensure_dirs_exist, sanitize_filename
from src.utils.background_processing import (
    ResourcePreloader,
    TTSWarmer,
    cleanup_global_background_processor,
    get_background_processor,
)
from src.utils.connection_pool import get_http_session
from src.utils.logging_setup import setup_debug_logging
from src.utils.performance import PerformanceHistoryManager, performance_monitor
from src.video.config import VideoConfig, VideoProfile
from src.video.config_adapter import load_video_config_modular
from src.video.config_validator import validate_config_and_exit_on_error
from src.video.pipeline_graph import PipelineGraph, StepStatus
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
    VALID_STEPS,
    _clean_producer_files,
    _load_artifacts_from_state,
    _load_pipeline_state,
    _save_pipeline_state,
    _update_state_after_step,
    get_video_run_paths,
)
from src.video.producer.steps import (
    _load_artifacts_create_voiceover,
    _load_artifacts_download_music,
    _load_artifacts_gather_visuals,
    _load_artifacts_generate_description,
    _load_artifacts_generate_script,
    _load_artifacts_generate_subtitles,
    step_assemble_video,
    step_create_voiceover,
    step_download_music,
    step_gather_visuals,
    step_generate_description,
    step_generate_script,
    step_generate_subtitles,
)
from src.video.producer.utils import setup_logging, validate_media_requirements

logger = logging.getLogger(__name__)


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
        "generate_description",
        lambda ctx: step_generate_description(ctx),
        {"generate_script"},
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
                elif step_name == "generate_description":
                    _load_artifacts_generate_description(ctx)
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


async def create_video_for_product(
    config: VideoConfig,
    product: ProductData,
    profile_name: str,
    secrets: dict,
    session: aiohttp.ClientSession,
    debug_mode: bool,
    clean_run: bool,
    debug_step_target: str | None,
    cli_overrides: dict[str, Any] | None = None,
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
            cli_overrides,
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
                elif step == STEP_GENERATE_DESCRIPTION:
                    await step_generate_description(ctx)
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
                performance_monitor.save_metrics(run_paths["performance"])

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
