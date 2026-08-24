# src/video/producer/state.py
"""Pipeline state management and path utilities."""

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from src.utils import ensure_dirs_exist
from src.video.config import VideoConfig
from src.video.producer.context import PipelineContext
from src.video.stock_media import StockMediaInfo

logger = logging.getLogger(__name__)

# Step constants
STEP_GATHER_VISUALS = "gather_visuals"
STEP_GENERATE_SCRIPT = "generate_script"
STEP_GENERATE_DESCRIPTION = "generate_description"
STEP_CREATE_VOICEOVER = "create_voiceover"
STEP_GENERATE_SUBTITLES = "generate_subtitles"
STEP_DOWNLOAD_MUSIC = "download_music"
STEP_ASSEMBLE_VIDEO = "assemble_video"
STEP_BURN_PYCAPS_SUBTITLES = "burn_pycaps_subtitles"

VALID_STEPS = [
    STEP_GATHER_VISUALS,
    STEP_GENERATE_SCRIPT,
    STEP_GENERATE_DESCRIPTION,
    STEP_CREATE_VOICEOVER,
    STEP_GENERATE_SUBTITLES,
    STEP_DOWNLOAD_MUSIC,
    STEP_ASSEMBLE_VIDEO,
    STEP_BURN_PYCAPS_SUBTITLES,
]


def resolved_step_order(profile: Any) -> list[str]:
    """The order this profile's steps actually run in.

    ``VALID_STEPS`` is the default order and the one a scraped product uses.
    A profile whose visuals all come from stock gathers them after the script,
    so the search terms can be taken from the narration; everything downstream
    keeps its place. Callers that reason about "the steps before this one",
    including resume truncation and the ``--step`` prerequisite check, have to
    use this rather than ``VALID_STEPS`` or they will treat a step that has not
    run yet as a completed prerequisite.
    """
    from src.video.producer.utils import draws_visuals_from_script

    if not draws_visuals_from_script(profile):
        return list(VALID_STEPS)
    order = list(VALID_STEPS)
    order.remove(STEP_GATHER_VISUALS)
    order.insert(order.index(STEP_GENERATE_SCRIPT) + 1, STEP_GATHER_VISUALS)
    return order


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
    temp_files = config.output_structure.product_temp_files
    temp_dir = product_root / config.output_structure.product_subdirs.temp

    # Producer-generated files to remove (preserve scraper inputs like data.json,
    # images/, videos/)
    producer_files_to_remove = [
        # Intermediate files (now in temp/)
        temp_dir / files.script,  # script.txt
        temp_dir / files.description,  # description.txt
        temp_dir / files.voiceover,  # voiceover.wav
        temp_dir / files.subtitles,  # subtitles.srt
        temp_dir / "subtitles.ass",  # ASS subtitle file
        temp_dir / "subtitles_content_aware.ass",  # content-aware subtitle file
        temp_dir / "subtitle_upper.ass",  # Upper subtitle (two-part system)
        temp_dir / files.attribution,  # attributions file
        # Legacy paths (product_root) for backwards compatibility cleanup
        product_root / files.script,
        product_root / files.description,
        product_root / files.voiceover,
        product_root / files.subtitles,
        product_root / "subtitles.ass",
        product_root / "subtitles_content_aware.ass",
        product_root / "subtitle_upper.ass",
        product_root / files.attribution,
        # Final video (stays in product_root)
        product_root
        / files.final_video.format(
            product_id=product_id, profile=safe_profile_name
        ),  # video_{product_id}_{profile}.mp4
        product_root / f"video_{safe_profile_name}.mp4",  # old naming pattern
        # Debug/pipeline files (in temp/)
        temp_dir / temp_files.pipeline_state,  # pipeline_state.json
        temp_dir / temp_files.ffmpeg_log,  # ffmpeg_command.log
        temp_dir / temp_files.performance,  # performance.json
        product_root / config.path_config.temp_dir,  # temp/ directory
        product_root / config.path_config.music_dir,  # music/ directory
        temp_dir / config.path_config.gathered_visuals,  # internal producer file
        product_root / "~",  # Erroneous home directory from unescaped paths
        product_root / "outputs",  # Erroneous nested outputs directory
        # Metadata files (unified and platform-specific)
        product_root / "metadata.json",  # unified mode
        product_root / "metadata_youtube.json",  # optimized mode
        product_root / "metadata_tiktok.json",  # optimized mode
        product_root / "metadata_instagram.json",  # optimized mode
        product_root / "UPLOAD_INSTRUCTIONS.txt",  # optimized mode instructions
        # Pycaps engine artifacts
        temp_dir / "whisper_transcript.json",
        temp_dir / "pycaps_metadata.json",
    ]

    # Clean all video files with any profile name (video_*_{product_id}_*.mp4)
    for video_file in product_root.glob(f"video_{product_id}_*.mp4"):
        producer_files_to_remove.append(video_file)

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
        "description_file": paths["description"],
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
            "performance": paths["performance"],  # Performance metrics file
            "script_prompt": paths["script_prompt"],  # Rendered LLM prompt
            # Pycaps engine artifacts
            "whisper_transcript_file": paths["working_dir"] / "whisper_transcript.json",
            "pycaps_metadata_file": paths["working_dir"] / "pycaps_metadata.json",
            "pycaps_burn_marker_file": paths["working_dir"] / "pycaps_burned.json",
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


# Artifacts whose presence makes a step short-circuit and skip its work. Only
# these are removed when a step is dropped: deleting anything else destroys
# output for no benefit, and `final_video_output` in particular is the
# deliverable, rebuilt unconditionally by a step that never short-circuits.
_RERUN_BLOCKING_ARTIFACTS = frozenset(
    {
        "gathered_visuals_file",
        "script_file",
        "description_file",
        "voiceover_file",
        "voiceover_duration_file",
    }
)

# `generate_description` short-circuits on the platform metadata rather than on
# `description_file`, and those keys are generated per platform, so they are
# matched by prefix instead of listed.
_RERUN_BLOCKING_PREFIXES = ("unified_metadata_file", "platform_metadata_")


def _discard_stale_artifacts(
    state_data: dict[str, Any], valid_steps: list[str]
) -> None:
    """Delete the outputs that would stop a dropped step from re-running.

    Dropping a step from the state is not enough on its own, because a step
    can short-circuit on its own artifact file rather than on the state:
    ``step_gather_visuals`` returns the previous run's media whenever
    ``gathered_visuals.json`` is on disk. Left in place, a lost script would be
    replaced by fresh narration and then paired with the footage searched from
    the old one, which is exactly the mismatch the reordering exists to
    prevent.

    Two things are deliberately not deleted. Anything outside
    ``_RERUN_BLOCKING_ARTIFACTS``, because a step that re-runs regardless loses
    nothing by keeping its previous output and everything by having it removed:
    ``pipeline_state.json`` is shared by every profile of a product while the
    rendered video is per-profile, so a blanket delete takes another profile's
    finished render with it. And any path a surviving step still claims, since
    ``assemble_video`` and ``burn_pycaps_subtitles`` record the same video and
    only one of them may be dropped.
    """
    kept_paths = {
        path_str
        for name, data in state_data.items()
        if name in valid_steps and isinstance(data, dict)
        for path_str in (data.get("artifacts") or {}).values()
    }
    for name, data in state_data.items():
        if name in valid_steps or not isinstance(data, dict):
            continue
        for key, path_str in (data.get("artifacts") or {}).items():
            blocks_rerun = key in _RERUN_BLOCKING_ARTIFACTS or key.startswith(
                _RERUN_BLOCKING_PREFIXES
            )
            if not blocks_rerun or path_str in kept_paths:
                continue
            path = Path(path_str)
            try:
                if path.exists():
                    path.unlink()
                    logger.info(
                        "Discarded stale artifact '%s' from dropped step '%s': %s",
                        key,
                        name,
                        path.name,
                    )
            except OSError as e:
                # A leftover file is recoverable (the step overwrites it);
                # failing the whole run over one unlink is not.
                logger.warning("Could not remove stale artifact %s: %s", path, e)


def _artifact_invalid_reason(
    ctx: PipelineContext, key: str, path_str: str
) -> str | None:
    """Why a recorded artifact cannot be reused, or None when it can.

    Two ways it fails. The file may be gone. Or it may belong to a different
    run: ``pipeline_state.json`` is product-level while several artifacts are
    profile-level, so rendering the same product under a second profile finds
    the first profile's video recorded, present, and completely wrong. Left
    unchecked the whole pipeline is skipped and the run reports success with
    a video path nothing wrote.
    """
    if not Path(path_str).exists():
        return f"not found at '{path_str}'"
    expected = ctx.run_paths.get(key)
    if isinstance(expected, Path) and Path(path_str) != expected:
        return f"belongs to another run ('{path_str}', this run uses '{expected}')"
    return None


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

        # Verify that all artifacts for completed steps still exist.
        # Top-level scalars (pillar, script_template) live alongside step dicts;
        # skip non-dict entries instead of calling .get() on them.
        for step, data in state_data.items():
            if not isinstance(data, dict):
                continue
            if data.get("status") == "done":
                for key, path_str in data.get("artifacts", {}).items():
                    reason = _artifact_invalid_reason(ctx, key, path_str)
                    if reason is not None:
                        logger.warning(
                            f"State is invalid. Artifact '{key}' for step "
                            f"'{step}' {reason}. Restarting from step '{step}'."
                        )
                        # Truncate state up to the failed step. Ordered by
                        # this profile's real order: on a script-first render
                        # the visuals are searched on terms taken from the
                        # script, so a lost script has to drop them too rather
                        # than pair new narration with the old footage.
                        step_order = resolved_step_order(ctx.profile)
                        valid_steps = step_order[: step_order.index(step)]
                        _discard_stale_artifacts(state_data, valid_steps)
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


def _drop_dependents(ctx: PipelineContext, step_name: str) -> None:
    """Forget the recorded steps that read what ``step_name`` just rewrote.

    Running a step on its own does not invalidate what comes after it, so
    ``--step assemble_video`` left ``burn_pycaps_subtitles`` marked done over
    a video whose captions had just been re-rendered away; the next full run
    skipped the burn and shipped the uncaptioned video as a success.

    Imported here rather than at module scope: ``orchestration`` imports this
    module, so the dependency map can only be reached lazily.
    """
    from src.video.producer.orchestration import step_dependencies, transitive_prereqs

    dependencies = step_dependencies(ctx.profile)
    stale = [
        name
        for name in dependencies
        if name != step_name
        and name in ctx.state
        and step_name in transitive_prereqs(dependencies, name)
    ]
    for name in stale:
        del ctx.state[name]
    if stale:
        logger.info(
            "Step '%s' ran again; dropping the steps that read its output: %s",
            step_name,
            ", ".join(sorted(stale)),
        )


async def _update_state_after_step(ctx: PipelineContext, step_name: str):
    """Updates the state dictionary with the artifacts of a completed step."""
    _drop_dependents(ctx, step_name)
    artifacts = {}
    if step_name == STEP_GATHER_VISUALS:
        artifacts["gathered_visuals_file"] = ctx.run_paths["gathered_visuals_file"]
    elif step_name == STEP_GENERATE_SCRIPT:
        artifacts["script_file"] = ctx.run_paths["script_file"]
    elif step_name == STEP_GENERATE_DESCRIPTION:
        # Only when it exists, which today is never: the step writes
        # `metadata.json` unified or `metadata_<platform>.json` per platform,
        # and nothing writes `description.txt`. Recording an absent file fails
        # state verification on the next run: the step is dropped, and with it
        # everything after it, so a resume re-ran the description, the
        # voiceover, the subtitles, the music, the assembly and the burn on a
        # render that had already finished all six.
        description_file = ctx.run_paths["description_file"]
        if description_file.exists():
            artifacts["description_file"] = description_file
        # `_check_existing_metadata` short-circuits on these, not on
        # `description_file`, so they have to be recorded or a dropped step
        # keeps returning captions written for a script that no longer exists.
        unified = ctx.run_paths["run_root"] / "metadata.json"
        if unified.exists():
            artifacts["unified_metadata_file"] = unified
        text_dir = ctx.run_paths["description_file"].parent
        for platform_meta in sorted(text_dir.glob("metadata_*.json")):
            artifacts[f"platform_metadata_{platform_meta.stem}"] = platform_meta
    elif step_name == STEP_CREATE_VOICEOVER:
        artifacts["voiceover_file"] = ctx.run_paths["voiceover_file"]
        artifacts["voiceover_duration_file"] = ctx.run_paths["voiceover_duration_file"]
    elif step_name == STEP_GENERATE_SUBTITLES:
        # Either subtitle_file (ffmpeg engine) or whisper_transcript_file (pycaps).
        # Both are recorded when present so state verification succeeds on rerun.
        subtitle_file = ctx.run_paths["subtitle_file"]
        if subtitle_file.exists():
            artifacts["subtitle_file"] = subtitle_file
        transcript_file = ctx.run_paths.get("whisper_transcript_file")
        if transcript_file is not None and transcript_file.exists():
            artifacts["whisper_transcript_file"] = transcript_file
    elif step_name == STEP_DOWNLOAD_MUSIC:
        artifacts["music_info_file"] = ctx.run_paths["music_info_file"]
    elif step_name == STEP_ASSEMBLE_VIDEO:
        artifacts["final_video_output"] = ctx.run_paths["final_video_output"]
    elif step_name == STEP_BURN_PYCAPS_SUBTITLES:
        artifacts["final_video_output"] = ctx.run_paths["final_video_output"]
        pycaps_meta = ctx.run_paths.get("pycaps_metadata_file")
        if pycaps_meta is not None and pycaps_meta.exists():
            artifacts["pycaps_metadata_file"] = pycaps_meta

    step_state: dict[str, Any] = {
        "status": "done",
        "artifacts": {k: str(v) for k, v in artifacts.items()},
    }
    # Include script template metadata (saved by step_generate_script)
    if step_name == STEP_GENERATE_SCRIPT and ctx.state.get("script_template"):
        step_state["script_template"] = ctx.state["script_template"]
    # Include TTS metadata if available (saved by step_create_voiceover)
    if step_name == STEP_CREATE_VOICEOVER and ctx.state.get("tts_metadata"):
        step_state["tts_metadata"] = ctx.state["tts_metadata"]
    # Include pycaps metadata (saved by step_burn_pycaps_subtitles)
    if step_name == STEP_BURN_PYCAPS_SUBTITLES and ctx.state.get("pycaps_metadata"):
        step_state["pycaps_metadata"] = ctx.state["pycaps_metadata"]
    # Phase 1.2e: cold-open variant chosen by select_cold_open_variant in
    # step_assemble_video. Persisted so the analytics layer can segment
    # retention by variant when comparing renders.
    if step_name == STEP_ASSEMBLE_VIDEO and ctx.state.get("cold_open_variant"):
        step_state["cold_open_variant"] = ctx.state["cold_open_variant"]

    ctx.state[step_name] = step_state
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
        elif step_name == STEP_GENERATE_DESCRIPTION:
            path = Path(state_entry["artifacts"]["description_file"])
            ctx.description = path.read_text(encoding="utf-8")
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
