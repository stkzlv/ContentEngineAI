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
from src.video.producer.artifact_registry import load_artifacts_for_step
from src.video.producer.context import (
    InsufficientMediaError,
    PipelineContext,
    PipelineError,
)
from src.video.producer.state import (
    STEP_ASSEMBLE_VIDEO,
    STEP_BURN_PYCAPS_SUBTITLES,
    STEP_CREATE_VOICEOVER,
    STEP_DOWNLOAD_MUSIC,
    STEP_GATHER_VISUALS,
    STEP_GENERATE_DESCRIPTION,
    STEP_GENERATE_SCRIPT,
    STEP_GENERATE_SUBTITLES,
    _clean_producer_files,
    _load_artifacts_from_state,
    _load_pipeline_state,
    _save_pipeline_state,
    _update_state_after_step,
    get_video_run_paths,
    resolved_step_order,
)
from src.video.producer.steps import (
    step_assemble_video,
    step_burn_pycaps_subtitles,
    step_create_voiceover,
    step_download_music,
    step_gather_visuals,
    step_generate_description,
    step_generate_script,
    step_generate_subtitles,
)
from src.video.producer.utils import (
    draws_visuals_from_script,
    setup_logging,
    validate_media_requirements,
)

logger = logging.getLogger(__name__)

FAILED_PREFIX = "FAILED:"


def step_runners() -> dict[str, Any]:
    """Every step name mapped to the coroutine that runs it.

    One table for both execution paths. The parallel graph and the
    sequential ``--step`` path used to name their steps separately, so a step
    added to the graph alone was accepted on the command line, ran nothing,
    and was still recorded as done.
    """
    return {
        STEP_GATHER_VISUALS: step_gather_visuals,
        STEP_GENERATE_SCRIPT: step_generate_script,
        STEP_GENERATE_DESCRIPTION: step_generate_description,
        STEP_CREATE_VOICEOVER: step_create_voiceover,
        STEP_GENERATE_SUBTITLES: step_generate_subtitles,
        STEP_DOWNLOAD_MUSIC: step_download_music,
        STEP_ASSEMBLE_VIDEO: step_assemble_video,
        STEP_BURN_PYCAPS_SUBTITLES: step_burn_pycaps_subtitles,
    }


def completed_steps_from_state(state_data: dict[str, Any]) -> set[str]:
    """Names of the steps a saved state file records as done.

    The file is not only step entries: it also holds top-level scalars
    (`script_template`, `hook_headline`, `subtitle_engine_resolved`, ...).
    Calling ``.get`` on those raises, and the caller's broad handler then
    reports a corrupt state and re-runs a completed pipeline from scratch.
    """
    return {
        name
        for name, info in state_data.items()
        if isinstance(info, dict) and info.get("status") == "done"
    }


def step_dependencies(profile: VideoProfile) -> dict[str, set[str]]:
    """Each step's declared prerequisites under this profile's step order.

    The single source of truth for the pipeline DAG. Both the parallel
    executor and the ``--step`` prerequisite check read it, so a partial run
    can never be refused for a step the graph does not actually require.

    Which of the first two runs first depends on where the visuals come from.
    A scraped product has its photography before anything is written, and
    gathering first also rejects a product with too few images before an LLM
    call is paid for. A profile rendering entirely from stock has no such
    imagery: its search terms are the whole visual layer, so the script has
    to exist first for the footage to match the narration.

    ``generate_script`` reads nothing ``gather_visuals`` writes, which is what
    makes the edge safe to reverse.
    """
    script_first = draws_visuals_from_script(profile)

    # Both of the paid steps spend money: captions call the LLM once per
    # platform and the voiceover synthesises audio. On the product path
    # `gather_visuals` has already run and rejected a product with too few
    # images before either starts. Under the script-first order it would
    # otherwise run *alongside* them, and `fail_fast` only stops between
    # levels, so a render destined to be skipped would pay for both first.
    # Naming it here restores the ordering property at the cost of one
    # serialised level.
    paid_step_deps = {STEP_GENERATE_SCRIPT}
    if script_first:
        paid_step_deps = {STEP_GENERATE_SCRIPT, STEP_GATHER_VISUALS}

    return {
        STEP_GATHER_VISUALS: {STEP_GENERATE_SCRIPT} if script_first else set(),
        STEP_GENERATE_SCRIPT: set() if script_first else {STEP_GATHER_VISUALS},
        STEP_GENERATE_DESCRIPTION: set(paid_step_deps),
        STEP_CREATE_VOICEOVER: set(paid_step_deps),
        STEP_GENERATE_SUBTITLES: {STEP_CREATE_VOICEOVER},
        STEP_DOWNLOAD_MUSIC: {STEP_CREATE_VOICEOVER},
        # `gather_visuals` is named explicitly because it is no longer always
        # an ancestor: on the script-first order it is a leaf, and without
        # this edge the assembler could start before the footage is
        # downloaded.
        STEP_ASSEMBLE_VIDEO: {
            STEP_GENERATE_SUBTITLES,
            STEP_DOWNLOAD_MUSIC,
            STEP_GATHER_VISUALS,
        },
        # Pycaps engine: post-assembly burn-in. Short-circuits at runtime when
        # the resolved subtitle engine is not "pycaps", so depending on it
        # unconditionally is safe -- no extra cost in the default ffmpeg path.
        STEP_BURN_PYCAPS_SUBTITLES: {STEP_ASSEMBLE_VIDEO},
    }


def data_dependencies(profile: VideoProfile) -> dict[str, set[str]]:
    """The subset of the graph that carries data, not just ordering.

    Two of the graph's edges exist to order execution rather than to move
    anything. On the scraped-product order `generate_script` is placed after
    `gather_visuals` so a product with too few images is rejected before an
    LLM call is paid for, and on the script-first order the same reasoning
    names `gather_visuals` as a prerequisite of the two paid steps. Neither
    means the later step has to be redone when the earlier one runs again.

    Anything reasoning about correctness -- which steps a `--step` run really
    needs, and which recorded steps a re-run invalidates -- reads this.
    Scheduling reads `step_dependencies`. Treating an ordering edge as data
    deletes a script and a voiceover because the footage was re-fetched.
    """
    dependencies = step_dependencies(profile)
    if draws_visuals_from_script(profile):
        for paid in (STEP_GENERATE_DESCRIPTION, STEP_CREATE_VOICEOVER):
            dependencies[paid] = dependencies[paid] - {STEP_GATHER_VISUALS}
    else:
        dependencies[STEP_GENERATE_SCRIPT] = dependencies[STEP_GENERATE_SCRIPT] - {
            STEP_GATHER_VISUALS
        }
    return dependencies


def transitive_prereqs(dependencies: dict[str, set[str]], target: str) -> set[str]:
    """Every step ``target`` transitively depends on, itself excluded.

    Walks the declared DAG rather than the positional step order. A step
    sitting earlier in the order is not a prerequisite unless an edge says so:
    `create_voiceover` reads the script and nothing else, so a `--step` run of
    it must not be blocked on `generate_description`.
    """
    seen: set[str] = set()
    pending = list(dependencies.get(target, set()))
    while pending:
        step = pending.pop()
        if step in seen:
            continue
        seen.add(step)
        pending.extend(dependencies.get(step, set()))
    return seen


async def execute_pipeline_parallel(
    ctx: PipelineContext,
) -> tuple[bool, str | None]:
    """Execute pipeline steps using parallel execution framework.

    Args:
    ----
        ctx: Pipeline context with all necessary data

    Returns:
    -------
        Tuple of (success, failed_step_name). ``failed_step_name`` is the
        first failed step when success is False, or None when the failure
        happened outside any step.

    """
    logger.info("Using parallel pipeline execution framework")

    # Check which steps are already completed
    completed_steps = set()
    if ctx.run_paths["state_file"].exists():
        try:
            state_data = json.loads(ctx.run_paths["state_file"].read_text())
            completed_steps = completed_steps_from_state(state_data)
            logger.info(
                "Found %s already completed steps: %s",
                len(completed_steps),
                completed_steps,
            )
        except Exception as e:
            logger.warning("Could not load existing pipeline state: %s", e)
            completed_steps = set()

    # Create pipeline graph from the shared dependency map, walked in run
    # order so a step is always added after the steps it depends on.
    # A media rejection is a skip, not a step failure: it must reach
    # `create_video_for_product`, which reports the product SKIPPED.
    pipeline = PipelineGraph(propagate=(InsufficientMediaError,))
    dependencies = step_dependencies(ctx.profile)
    runners = step_runners()
    for step_name in resolved_step_order(ctx.profile):
        pipeline.add_step(step_name, runners[step_name], dependencies[step_name])

    # Skip already completed steps
    if completed_steps:
        for step_name in completed_steps:
            if pipeline.has_step(step_name):
                pipeline.skip_step(step_name)
                logger.info("Skipping already completed step: %s", step_name)

                # Load artifacts for skipped steps via registry
                load_artifacts_for_step(step_name, ctx)

    # Execute pipeline with parallel execution
    try:
        results = await pipeline.execute_pipeline(context=ctx)

        # Check results and update pipeline state
        failed_steps = [r for r in results if r.status == StepStatus.FAILED]
        if failed_steps:
            for failed_result in failed_steps:
                logger.error(
                    "Step '%s' failed: %s", failed_result.step_name, failed_result.error
                )
            return False, failed_steps[0].step_name

        # Update state for newly completed steps with synchronization
        async with ctx._state_lock:
            for result in results:
                if result.status == StepStatus.COMPLETED:
                    step_name = result.step_name
                    await _update_state_after_step(ctx, step_name)
                    logger.info("Step '%s' completed successfully", step_name)

            await _save_pipeline_state(ctx)
        return True, None

    except InsufficientMediaError:
        # Re-raise InsufficientMediaError so main handler can process it cleanly
        raise
    except Exception as e:
        logger.error("Pipeline execution failed: %s", e, exc_info=True)
        return False, None


def failed_step_from_result(result: str | Path | None) -> str | None:
    """Return the failing step name if a producer result signals a step failure.

    ``create_video_for_product`` returns the final video path on success,
    ``"SKIPPED"`` for insufficient media, or ``"FAILED:<step>"`` when a
    pipeline step raised. Callers may also hold ``None`` when their own
    exception handling (timeout, unexpected error) produced no result. This
    parses the failure sentinel so callers count it as a failure and name the
    step, instead of mistaking it for a skip or a success.
    """
    if isinstance(result, str) and result.startswith(FAILED_PREFIX):
        return result[len(FAILED_PREFIX) :] or "unknown"
    return None


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
    logger.info(
        "--- Starting video for '%s' profile '%s' ---", product_id, profile_name
    )

    # Initialize performance history manager with configurable retention
    max_runs = 100
    if config.optimization_settings:
        max_runs = config.optimization_settings.performance_history_max_runs

    history_manager = PerformanceHistoryManager(
        history_dir=config.global_output_root_path / "performance_history",
        max_runs=max_runs,
    )

    # Reset the global performance monitor with fresh state
    monitor_interval = 0.1
    if config.optimization_settings:
        opt = config.optimization_settings
        monitor_interval = opt.performance_monitoring_interval_sec
    performance_monitor.reset(
        history_manager=history_manager, memory_monitor_interval=monitor_interval
    )

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
    skipped_run = False

    if clean_run and run_paths["run_root"].exists():
        logger.info(
            "--clean flag set. Removing producer-generated files from: %s",
            run_paths["run_root"],
        )
        try:
            _clean_producer_files(run_paths, config, product_id, profile_name)
        except OSError as e:
            logger.error("Error cleaning producer files: %s", e)
            raise PipelineError("Could not clean producer files for fresh run.") from e

    try:
        profile = config.get_profile(profile_name)
        ensure_dirs_exist(run_paths["run_root"])

        # Apply script template override to LLM settings
        if cli_overrides and cli_overrides.get("script_template"):
            config.llm_settings.script_templates.fixed_template = cli_overrides[
                "script_template"
            ]

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

            # Resolve the pillar once, on every run, and record it here.
            # CLI --pillar wins; otherwise the product's own value, which the
            # scraper writes into `data.json`.
            #
            # This has to sit after the state load rather than inside
            # `step_generate_script`: a resume that truncates the state drops
            # every non-step key and then skips the steps it kept, so a
            # product-level pillar recorded inside the step would be lost on
            # exactly the runs that reload it. A repeat render then draws
            # from a different template pool, preamble and audience than the
            # script already on disk was written for.
            # CLI, then what a previous run recorded, then the product's own
            # value. The middle term matters on a resume: without it the
            # product record overwrites a `--pillar` the earlier run resolved
            # and already wrote the script under, and the flag is not repeated
            # on the rerun.
            resolved_pillar = (
                (cli_overrides or {}).get("pillar")
                or ctx.state.get("pillar")
                or getattr(product, "pillar", None)
            )
            if resolved_pillar:
                ctx.state["pillar"] = resolved_pillar

        if debug_step_target:
            # This profile's real order, not the storage order: on a
            # script-first render `gather_visuals` runs after the script, so
            # demanding it as a prerequisite of `generate_script` would refuse
            # a run that is in a perfectly good state.
            step_order = resolved_step_order(ctx.profile)
            # Only the requested step's transitive prerequisites are
            # required. The positional walk this replaces blocked
            # `--step create_voiceover` on `generate_description`, which
            # feeds it nothing. Iterated in run order so artifacts load in
            # the order they were produced.
            required = transitive_prereqs(
                data_dependencies(ctx.profile), debug_step_target
            )
            for step_to_load in step_order[: step_order.index(debug_step_target)]:
                if step_to_load not in required:
                    continue
                if ctx.state.get(step_to_load, {}).get("status") == "done":
                    logger.info(
                        "Loading prerequisites for '%s': Loading artifacts from '%s'.",
                        debug_step_target,
                        step_to_load,
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
            steps_to_run = resolved_step_order(ctx.profile)

        # Use parallel pipeline execution unless debugging specific step
        if debug_step_target:
            # For debugging specific steps, use sequential execution
            for current_step in steps_to_run:
                step = current_step

                if (
                    debug_step_target is None
                    and ctx.state.get(current_step, {}).get("status") == "done"  # type: ignore[unreachable]
                ):
                    logger.info("Skipping step '%s': Already completed.", current_step)  # type: ignore[unreachable]
                    _load_artifacts_from_state(ctx, current_step)
                    continue

                # Ensure directories for the step's outputs exist
                for path in run_paths.values():
                    if isinstance(path, Path):
                        ensure_dirs_exist(path.parent)

                runner = step_runners().get(step)
                if runner is None:
                    raise PipelineError(f"No handler for step '{step}'.")
                await runner(ctx)

                async with ctx._state_lock:
                    await _update_state_after_step(ctx, step)
                    await _save_pipeline_state(ctx)
        else:
            # Use parallel pipeline execution for normal runs
            successful_run, parallel_failed_step = await execute_pipeline_parallel(ctx)
            if not successful_run:
                # Record the failing step so the FAILED:<step> sentinel names
                # it instead of reporting 'unknown'.
                step = parallel_failed_step or ""
                raise PipelineError("Parallel pipeline execution failed")

        successful_run = True
        logger.info(
            "<<< SUCCESS: Video for '%s': %s",
            product_id,
            run_paths.get("final_video_output", "N/A"),
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

        # Check performance thresholds and log warnings
        if config.debug_settings:
            timing_threshold = config.debug_settings.operation_timing_threshold_sec
            memory_warning = config.debug_settings.memory_usage_warning_mb
        else:
            timing_threshold = 180.0
            memory_warning = 3000
        threshold_warnings = performance_monitor.check_thresholds(
            timing_threshold_sec=timing_threshold,
            memory_warning_mb=memory_warning,
        )
        for warning in threshold_warnings:
            logger.warning("Performance threshold exceeded: %s", warning)

        # Clean up background processing
        if ctx.background_processor:
            summary = ctx.background_processor.get_summary()
            logger.debug("Background processing summary: %s", summary)
            await cleanup_global_background_processor()

        return run_paths.get("final_video_output")

    except InsufficientMediaError as e:
        skipped_run = True
        logger.warning("Product skipped due to insufficient media: %s", e)
        # Mark as skipped, not failed - this is expected for some products
        performance_monitor.finish_pipeline(success=False, error_message=str(e))
        # Clean up background processing on skip
        await cleanup_global_background_processor()
        # Return special value to indicate skip
        return "SKIPPED"
    except (FileNotFoundError, PipelineError, KeyError) as e:
        logger.error("Pipeline stopped at step '%s': %s", step, e, exc_info=debug_mode)
        # Mark pipeline as failed for history tracking
        performance_monitor.finish_pipeline(success=False, error_message=str(e))
        # Clean up background processing on failure
        await cleanup_global_background_processor()
        # Signal a step failure, distinct from "SKIPPED" and from a partial
        # None return, naming the step so callers can report it.
        return f"{FAILED_PREFIX}{step or 'unknown'}"
    except Exception as e:
        logger.error(
            "An unexpected error occurred in pipeline for '%s': %s",
            product_id,
            e,
            exc_info=True,
        )
        # Mark pipeline as failed for history tracking
        performance_monitor.finish_pipeline(success=False, error_message=str(e))
        # Clean up background processing on failure
        await cleanup_global_background_processor()
        return f"{FAILED_PREFIX}{step or 'unknown'}"
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
                "Debug mode: Intermediate files preserved in %s",
                run_paths.get("run_root"),
            )
        elif skipped_run:
            # Not a failure: nothing broke, the product just had too little
            # media. Saying otherwise sends an operator looking for a step
            # that never went wrong.
            logger.info(
                "Product skipped. Files preserved in %s.", run_paths.get("run_root")
            )
        elif not successful_run:
            logger.warning(
                "Run failed. Files preserved in %s for resume.",
                run_paths.get("run_root"),
            )
