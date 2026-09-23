"""The author sign-off, read back for the readers that quote the script's end.

The sign-off is spoken between the closing beat and the call to action, which
is exactly where the first-comment extractor and the platform caption prompts
look for the closing line. The script step records the drawn sign-off in the
pipeline state beside the script; both readers remove it before reading the
end, so neither quotes the sign-off in place of the closing line.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

STATE_FILE = "pipeline_state.json"


def recorded_signoff(temp_dir: Path) -> str | None:
    """The sign-off the script step drew for this render, if any."""
    try:
        state = json.loads((temp_dir / STATE_FILE).read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(state, dict):
        return None
    signoff = state.get("signoff")
    if not signoff:
        step = state.get("generate_script")
        signoff = step.get("signoff") if isinstance(step, dict) else None
    return signoff if isinstance(signoff, str) and signoff.strip() else None


def normalise(text: str) -> str:
    """Lower-case words only, for comparing a sentence with the sign-off."""
    return " ".join(re.sub(r"[^a-z0-9\s]+", "", text.lower()).split())


def drop_signoff(script: str, signoff: str | None) -> str:
    """The script without its last occurrence of the sign-off sentence."""
    if not signoff:
        return script
    index = script.lower().rfind(signoff.strip().lower())
    if index < 0:
        return script
    before = script[:index].rstrip(" \t")
    after = script[index + len(signoff.strip()) :].lstrip(" \t")
    if before.endswith("\n") and after.startswith("\n"):
        # The sign-off had its own line; drop the line, not just its text.
        after = after[1:]
    if before.endswith("\n") or after.startswith("\n") or not before:
        return (before + after).strip()
    return f"{before} {after}".strip()
