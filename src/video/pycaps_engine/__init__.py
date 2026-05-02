"""pycaps subtitle engine integration for ContentEngineAI.

Thin wrapper that plugs the `pycaps` library into the ContentEngineAI pipeline
as an alternative to the FFmpeg + SRT/ASS subtitle path. Selected per-profile
or per-run via ``subtitle_settings.subtitle_engine == "pycaps"``.

The real work happens in :mod:`renderer` and :mod:`transcript_adapter`. This
package intentionally does NOT import ``pycaps`` at module load time so the
main app keeps working when the optional Poetry group is not installed.
Downstream code that uses these helpers must handle ``ImportError`` gracefully
(see :class:`PycapsUnavailableError` below).
"""

from src.video.pycaps_engine.gemini_llm import GeminiLlm
from src.video.pycaps_engine.renderer import (
    PycapsRenderer,
    PycapsRenderResult,
    PycapsUnavailableError,
    layout_from_visual_bounds,
    merge_layout_with_template,
    select_template_for_product,
)
from src.video.pycaps_engine.transcript_adapter import (
    load_whisper_transcript,
    save_whisper_transcript,
)

__all__ = [
    "GeminiLlm",
    "PycapsRenderer",
    "PycapsRenderResult",
    "PycapsUnavailableError",
    "is_pycaps_available",
    "layout_from_visual_bounds",
    "load_whisper_transcript",
    "merge_layout_with_template",
    "save_whisper_transcript",
    "select_template_for_product",
]


def is_pycaps_available() -> bool:
    """Check whether the pycaps library can be imported."""
    try:
        import pycaps  # noqa: F401

        return True
    except ImportError:
        return False
