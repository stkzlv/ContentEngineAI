# src/video/producer/__init__.py
"""Video producer module - Modular structure with backward compatibility.

This module has been refactored into smaller, focused modules:
- context.py: PipelineContext and exception classes
- state.py: State management and path utilities
- utils.py: Logging and validation utilities
- steps.py: Pipeline step implementations
- orchestration.py: Pipeline execution and high-level orchestration
- cli.py: CLI argument parsing and batch processing

All classes and functions are re-exported from this module to maintain
backward compatibility with existing imports like:
    from src.video.producer import PipelineContext, validate_media_requirements
"""

# Re-export artifact registry
from src.video.producer.artifact_registry import (  # noqa: F401
    load_artifacts_for_step,
    register_artifact_loader,
)

# Re-export context classes
from src.video.producer.context import (  # noqa: F401
    InsufficientMediaError,
    PipelineContext,
    PipelineError,
)

# Re-export orchestration functions
from src.video.producer.orchestration import (  # noqa: F401
    create_video_for_product,
    execute_pipeline_parallel,
)

# Re-export state management functions and constants
from src.video.producer.state import (  # noqa: F401
    STEP_ASSEMBLE_VIDEO,
    STEP_CREATE_VOICEOVER,
    STEP_DOWNLOAD_MUSIC,
    STEP_GATHER_VISUALS,
    STEP_GENERATE_DESCRIPTION,
    STEP_GENERATE_SCRIPT,
    STEP_GENERATE_SUBTITLES,
    VALID_STEPS,
    _clean_producer_files,
    _get_video_duration,
    _load_artifacts_from_state,
    _load_pipeline_state,
    _save_pipeline_state,
    _update_state_after_step,
    get_video_run_paths,
    load_visuals_info,
    save_visuals_info,
)

# Re-export pipeline step functions
from src.video.producer.steps import (  # noqa: F401
    step_assemble_video,
    step_create_voiceover,
    step_download_music,
    step_gather_visuals,
    step_generate_description,
    step_generate_script,
    step_generate_subtitles,
)

# Re-export utility functions
from src.video.producer.utils import (  # noqa: F401
    setup_logging,
    validate_media_requirements,
)

__all__ = [
    # Context classes
    "PipelineContext",
    "PipelineError",
    "InsufficientMediaError",
    # State constants
    "STEP_GATHER_VISUALS",
    "STEP_GENERATE_SCRIPT",
    "STEP_GENERATE_DESCRIPTION",
    "STEP_CREATE_VOICEOVER",
    "STEP_GENERATE_SUBTITLES",
    "STEP_DOWNLOAD_MUSIC",
    "STEP_ASSEMBLE_VIDEO",
    "VALID_STEPS",
    # State functions
    "_clean_producer_files",
    "get_video_run_paths",
    "_save_pipeline_state",
    "_load_pipeline_state",
    "_update_state_after_step",
    "_load_artifacts_from_state",
    "save_visuals_info",
    "load_visuals_info",
    "_get_video_duration",
    # Utility functions
    "setup_logging",
    "validate_media_requirements",
    # Pipeline step functions
    "step_gather_visuals",
    "step_generate_script",
    "step_generate_description",
    "step_create_voiceover",
    "step_generate_subtitles",
    "step_download_music",
    "step_assemble_video",
    # Orchestration functions
    "execute_pipeline_parallel",
    "create_video_for_product",
]
