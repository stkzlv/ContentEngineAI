"""Whisper transcript adapter for pycaps.

pycaps accepts several transcript formats, including ``whisper_json`` which is
precisely the shape of OpenAI Whisper's raw result dict. ContentEngineAI
already produces that dict inside
:func:`src.video.stt_functions.generate_subtitles_with_whisper`, so the
"adapter" is really a typed save/load shim that makes the lifecycle explicit
and testable.

The dict we save looks like:

.. code-block:: json

    {
      "language": "en",
      "text": "...",
      "segments": [
        {"id": 0, "start": 0.0, "end": 1.2, "text": "Hello there",
         "words": [{"word": "Hello", "start": 0.0, "end": 0.6},
                   {"word": "there", "start": 0.6, "end": 1.2}]}
      ]
    }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.utils import ensure_dirs_exist

logger = logging.getLogger(__name__)


def save_whisper_transcript(whisper_result: dict[str, Any], out_path: Path) -> Path:
    """Serialise a raw Whisper result dict to disk in ``whisper_json`` format.

    Args:
    ----
        whisper_result: Raw dict returned by ``whisper.Model.transcribe``.
            Must contain a ``"segments"`` list with per-word timings.
        out_path: Destination file path. Parent directory is created if
            missing.

    Returns:
    -------
        The same path on success.

    Raises:
    ------
        ValueError: If the dict does not contain a ``"segments"`` list —
            pycaps rejects transcripts without it.

    """
    if not isinstance(whisper_result, dict) or "segments" not in whisper_result:
        raise ValueError(
            "Whisper result is missing the 'segments' key; cannot save as "
            "whisper_json for pycaps consumption"
        )
    ensure_dirs_exist(out_path.parent)
    out_path.write_text(
        json.dumps(whisper_result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    segment_count = len(whisper_result.get("segments") or [])
    logger.info(
        "Saved Whisper transcript for pycaps (%d segments): %s",
        segment_count,
        out_path.name,
    )
    return out_path


def load_whisper_transcript(path: Path) -> dict[str, Any]:
    """Load a saved ``whisper_json`` transcript back from disk.

    Used when ``burn_pycaps_subtitles`` runs after a reload (e.g. pipeline
    resume), picking up the artifact produced by a previous
    ``generate_subtitles`` run.
    """
    if not path.exists():
        raise FileNotFoundError(f"Whisper transcript file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "segments" not in data:
        raise ValueError(
            f"Transcript file {path} is not in the expected whisper_json format "
            f"(missing 'segments' key)"
        )
    return data
