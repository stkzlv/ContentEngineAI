# src/video/producer/artifact_registry.py
"""Registry pattern for artifact loading with centralized error handling.

This module provides a decorator-based registry for artifact loaders,
consolidating the previously scattered artifact loading functions into
a single, maintainable pattern.
"""

import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.video.producer.context import PipelineContext

logger = logging.getLogger(__name__)

# Registry mapping step names to their artifact loader functions
_ARTIFACT_LOADERS: dict[str, Callable[["PipelineContext"], None]] = {}


def register_artifact_loader(step_name: str):
    """Decorator to register an artifact loader for a pipeline step.

    Args:
    ----
        step_name: The name of the pipeline step this loader handles.

    Returns:
    -------
        Decorator function that registers the loader.

    Example:
    -------
        @register_artifact_loader("gather_visuals")
        def _load_gather_visuals_artifacts(ctx: PipelineContext) -> None:
            # Load artifacts...

    """

    def decorator(func: Callable[["PipelineContext"], None]):
        _ARTIFACT_LOADERS[step_name] = func
        return func

    return decorator


def load_artifacts_for_step(step_name: str, ctx: "PipelineContext") -> None:
    """Load artifacts for a skipped pipeline step with consistent error handling.

    This function provides centralized error handling for all artifact loaders,
    distinguishing between "not found" (debug) and "invalid" (warning) cases.

    Args:
    ----
        step_name: Name of the pipeline step to load artifacts for.
        ctx: Pipeline context to populate with loaded artifacts.

    """
    loader = _ARTIFACT_LOADERS.get(step_name)
    if loader is None:
        logger.debug("No artifact loader registered for step '%s'", step_name)
        return

    try:
        loader(ctx)
        logger.debug("Loaded artifacts for skipped step '%s'", step_name)
    except FileNotFoundError:
        logger.debug("Artifacts not found for step '%s' (may not exist yet)", step_name)
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as e:
        logger.warning("Invalid artifacts for step '%s': %s", step_name, e)
    except OSError as e:
        logger.warning("Error reading artifacts for step '%s': %s", step_name, e)


def get_registered_loaders() -> list[str]:
    """Return list of step names with registered artifact loaders."""
    return list(_ARTIFACT_LOADERS.keys())
