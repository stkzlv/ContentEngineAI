# src/video/producer/__init__.py
"""Video producer module - Modular structure with backward compatibility.

This module has been refactored into smaller, focused modules:
- context.py: PipelineContext and exception classes
- state.py: State management and path utilities
- utils.py: Logging and validation utilities
- steps.py: Pipeline step implementations
- orchestration.py: Pipeline execution and high-level orchestration
- cli.py: CLI argument parsing and batch processing

All names below are re-exported for backward compatibility with imports like:
    from src.video.producer import PipelineContext, validate_media_requirements

**The re-exports resolve on first access, not at import.** A package's
`__init__` runs whenever any submodule is imported, so eager re-exports here
pulled `orchestration` and `steps`, and with them Whisper and torch, into every
process that touched a leaf module -- `src/pipeline/config.py` importing
`topic_input` was enough. That is also why function-local imports inside those
leaves deferred nothing: the stack was already resident.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # For type checkers and IDEs only. A module `__getattr__` returns `Any`,
    # so without these every re-exported name would type as `Any` and mypy
    # would stop reporting mistakes at the call sites that use them.
    from src.video.producer.artifact_registry import (
        load_artifacts_for_step,
        register_artifact_loader,
    )
    from src.video.producer.context import (
        InsufficientMediaError,
        PipelineContext,
        PipelineError,
    )
    from src.video.producer.orchestration import (
        create_video_for_product,
        execute_pipeline_parallel,
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
        _get_video_duration,
        _load_artifacts_from_state,
        _load_pipeline_state,
        _save_pipeline_state,
        _update_state_after_step,
        get_video_run_paths,
        load_visuals_info,
        save_visuals_info,
    )
    from src.video.producer.steps import (
        step_assemble_video,
        step_create_voiceover,
        step_download_music,
        step_gather_visuals,
        step_generate_description,
        step_generate_script,
        step_generate_subtitles,
    )
    from src.video.producer.utils import (
        setup_logging,
        validate_media_requirements,
    )

_EXPORTS: dict[str, str] = {
    "load_artifacts_for_step": "artifact_registry",
    "register_artifact_loader": "artifact_registry",
    "InsufficientMediaError": "context",
    "PipelineContext": "context",
    "PipelineError": "context",
    "create_video_for_product": "orchestration",
    "execute_pipeline_parallel": "orchestration",
    "STEP_ASSEMBLE_VIDEO": "state",
    "STEP_CREATE_VOICEOVER": "state",
    "STEP_DOWNLOAD_MUSIC": "state",
    "STEP_GATHER_VISUALS": "state",
    "STEP_GENERATE_DESCRIPTION": "state",
    "STEP_GENERATE_SCRIPT": "state",
    "STEP_GENERATE_SUBTITLES": "state",
    "VALID_STEPS": "state",
    "_clean_producer_files": "state",
    "_get_video_duration": "state",
    "_load_artifacts_from_state": "state",
    "_load_pipeline_state": "state",
    "_save_pipeline_state": "state",
    "_update_state_after_step": "state",
    "get_video_run_paths": "state",
    "load_visuals_info": "state",
    "save_visuals_info": "state",
    "step_assemble_video": "steps",
    "step_create_voiceover": "steps",
    "step_download_music": "steps",
    "step_gather_visuals": "steps",
    "step_generate_description": "steps",
    "step_generate_script": "steps",
    "step_generate_subtitles": "steps",
    "setup_logging": "utils",
    "validate_media_requirements": "utils",
}


def __getattr__(name: str) -> Any:
    """Resolve a re-export on first access (PEP 562)."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value  # cached, so the lookup happens once
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


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
    # Artifact registry
    "load_artifacts_for_step",
    "register_artifact_loader",
]
