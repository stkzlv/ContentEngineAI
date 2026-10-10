"""The author sign-off, read back for the readers that quote the script's end.

The sign-off is spoken between the closing beat and the call to action, which
is exactly where the first-comment extractor and the platform caption prompts
look for the closing line. The script step records the drawn sign-off in the
pipeline state beside the script; both readers remove it before reading the
end, so neither quotes the sign-off in place of the closing line.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from src.utils.script_sanitizer import split_sentences

STATE_FILE = "pipeline_state.json"
SCRIPT_FILE = "script.txt"
# Kept in the product directory, which outlives the intermediate files: a
# successful render deletes `temp/`, and the publisher runs after it
# (REQ-PUB-153).
SCRIPT_RECORD = "script.json"

logger = logging.getLogger(__name__)


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


def keep_script_record(product_dir: Path, temp_dir: Path) -> None:
    """Copy the script and its sign-off out of `temp_dir` before it is removed.

    Best effort: a render that could not keep the record still succeeded, and
    the first comment then skips with its usual warning.
    """
    try:
        script = (temp_dir / SCRIPT_FILE).read_text("utf-8")
        (product_dir / SCRIPT_RECORD).write_text(
            json.dumps({"script": script, "signoff": recorded_signoff(temp_dir)}),
            encoding="utf-8",
        )
    except OSError as e:
        logger.warning("Could not keep the script record in %s: %s", product_dir, e)


def read_script(product_dir: Path) -> tuple[str, str | None] | None:
    """The render's script and sign-off, from `temp/` or the kept record."""
    temp_dir = product_dir / "temp"
    try:
        return (temp_dir / SCRIPT_FILE).read_text("utf-8"), recorded_signoff(temp_dir)
    except OSError:
        pass
    try:
        record = json.loads((product_dir / SCRIPT_RECORD).read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict) or not isinstance(record.get("script"), str):
        return None
    signoff = record.get("signoff")
    return record["script"], signoff if isinstance(signoff, str) else None


def normalise(text: str) -> str:
    """Lower-case words only, for comparing a sentence with the sign-off."""
    return " ".join(re.sub(r"[^a-z0-9\s]+", "", text.lower()).split())


def signature_in_script(line: str, script: str) -> bool:
    """Whether a signature line is spoken in the script, on words only.

    The model often folds an opener into its first sentence ("Quick find for
    you, if your..."), so the configured line's closing punctuation and case
    are ignored; the words must still appear whole and in order.
    """
    target = normalise(line)
    return bool(target) and f" {target} " in f" {normalise(script)} "


def drop_signoff(script: str, signoff: str | None) -> str:
    """The script without its last sentence that matches the sign-off.

    Matched sentence by sentence on words only, the same comparison the
    first-comment extractor makes, so a sign-off the model ended with "!"
    or a pool entry with no full stop is removed on both paths alike.
    """
    if not signoff:
        return script
    target = normalise(signoff)
    match = None
    cursor = 0
    for sentence in split_sentences(script):
        start = script.find(sentence, cursor)
        if start < 0:
            continue
        cursor = start + len(sentence)
        if normalise(sentence) == target:
            match = (start, cursor)
    if match is None:
        return script
    before = script[: match[0]].rstrip(" \t")
    after = script[match[1] :].lstrip(" \t")
    if before.endswith("\n") and after.startswith("\n"):
        # The sign-off had its own line; drop the line, not just its text.
        after = after[1:]
    if before.endswith("\n") or after.startswith("\n") or not before:
        return (before + after).strip()
    return f"{before} {after}".strip()


def place_signoff(script: str, signoff: str | None, cta: str | None) -> str:
    """The script with its sign-off moved to directly before the CTA.

    The prompt asks for that place, and the model sometimes says the sign-off
    a beat early, before its closing line (REQ-CNT-045). A script without the
    sign-off, or not ending on the CTA, is returned unchanged.
    """
    if not signoff or not cta:
        return script
    sentences = split_sentences(script)
    if len(sentences) < 3 or normalise(sentences[-1]) != normalise(cta):
        return script
    target = normalise(signoff)
    if normalise(sentences[-2]) == target:
        return script
    if not any(normalise(s) == target for s in sentences[:-2]):
        return script
    spoken = next(s for s in sentences[:-2] if normalise(s) == target)
    trimmed = drop_signoff(script, signoff)
    at = trimmed.rfind(sentences[-1])
    if at < 0:
        return script
    head = trimmed[:at].rstrip(" \t")
    joiner = "" if head.endswith("\n") or not head else " "
    return f"{head}{joiner}{spoken} {trimmed[at:]}"
