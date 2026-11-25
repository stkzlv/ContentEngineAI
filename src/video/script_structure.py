"""Script-shape targets: the rules asked for, the beats named, their times.

REQ-CNT-189 asks the script for a framed setup and no recap, names the turn
and first-payoff sentences with a small labelling call, and estimates when
each starts from the words before it. REQ-CNT-190 measures the same starts
from the voiceover's word timings. REQ-CNT-191 draws one named variant of the
targets per product so a retention read can compare variants.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

from src.utils.script_signoff import split_sentences
from src.video.config.structure_models import ScriptStructure, StructureTargets

logger = logging.getLogger(__name__)

_SALT = "structure_variant"

MISCONCEPTION_RULE = (
    "After the opening claim, give the background in one or two sentences, "
    "framed as what most people believe, not as a history."
)
NO_RECAP_RULE = "Don't summarise or recap before the closing question."

LABEL_PROMPT = (
    "Below is a short video script, one numbered sentence per line. Reply "
    'with JSON only, {{"turn": N, "payoff": M}}: N is the number of the '
    "sentence where the script turns (the surprise, the 'but', the reveal), "
    "and M is the number of the first sentence that delivers the checked "
    "fact or the answer. Use null for a beat the script doesn't have.\n\n"
    "{sentences}"
)


def prompt_rules(structure: ScriptStructure) -> str:
    """The rules the script prompt gains for this kind, or an empty string."""
    rules = []
    if structure.setup == "misconception":
        rules.append(MISCONCEPTION_RULE)
    if structure.no_recap:
        rules.append(NO_RECAP_RULE)
    return " ".join(rules)


def draw_variant(structure: ScriptStructure, product_id: str) -> str | None:
    """One variant name per product, the same on every run, or None."""
    names = sorted(structure.variants)
    if not names:
        return None
    digest = hashlib.md5(f"{product_id}:{_SALT}".encode()).hexdigest()  # noqa: S324
    return names[int(digest[0:8], 16) % len(names)]


def targets_for(structure: ScriptStructure, variant: str | None) -> StructureTargets:
    """The kind's targets with the variant's overrides applied."""
    base = StructureTargets(
        turn_by_sec=structure.turn_by_sec,
        payoff_by_sec=structure.payoff_by_sec,
        signoff_max_sec=structure.signoff_max_sec,
    )
    if variant is None:
        return base
    override = structure.variants[variant].model_dump(exclude_none=True)
    return base.model_copy(update=override)


def parse_labels(reply: str, count: int) -> dict[str, int | None]:
    """The 0-based turn and payoff indices from the labelling reply."""
    match = re.search(r"\{.*\}", reply or "", re.DOTALL)
    out: dict[str, int | None] = {"turn": None, "payoff": None}
    if not match:
        return out
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return out
    for key in out:
        value = data.get(key) if isinstance(data, dict) else None
        if isinstance(value, int) and 1 <= value <= count:
            out[key] = value - 1
    return out


def label_prompt(sentences: list[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
    return LABEL_PROMPT.format(sentences=numbered)


def _start(word: dict[str, Any]) -> float | None:
    """A transcript word's start: `start_time` from the pipeline's extractor,
    `start` from a raw Whisper word.
    """
    value = word.get("start_time", word.get("start"))
    return float(value) if isinstance(value, int | float) else None


def _end(word: dict[str, Any]) -> float | None:
    value = word.get("end_time", word.get("end"))
    return float(value) if isinstance(value, int | float) else None


def _words(text: str) -> int:
    return len(re.findall(r"\b\w[\w'%-]*\b", text))


def estimate_starts(
    sentences: list[str], labels: dict[str, int | None], wpm: float
) -> dict[str, float | None]:
    """Seconds into the voice each named sentence starts, from the words
    before it at `wpm`.
    """
    out: dict[str, float | None] = {}
    for key, index in labels.items():
        if index is None:
            out[key] = None
            continue
        before = sum(_words(s) for s in sentences[:index])
        out[key] = round(before / wpm * 60, 2)
    return out


def measure_starts(
    sentences: list[str],
    labels: dict[str, int | None],
    words: list[dict[str, Any]],
) -> dict[str, float | None]:
    """Seconds each named sentence starts, from the transcript's word timings.

    The transcript's words are matched to the script by count: the named
    sentence starts at the word whose index is the number of script words
    before it, clamped to the transcript. A transcript that splits or merges
    words shifts this by a word or two.
    """
    out: dict[str, float | None] = {}
    starts = [t for t in (_start(w) for w in words) if t is not None]
    for key, index in labels.items():
        if index is None or not starts:
            out[key] = None
            continue
        before = sum(_words(s) for s in sentences[:index])
        out[key] = round(starts[min(before, len(starts) - 1)], 2)
    return out


def signoff_length(words: list[dict[str, Any]], signoff: str | None) -> float | None:
    """Seconds the sign-off takes at the end of the transcript, or None."""
    n = _words(signoff or "")
    timed = [w for w in words if _start(w) is not None]
    if not n or len(timed) < n:
        return None
    first = _start(timed[-n]) or 0.0
    end = _end(timed[-1]) or _start(timed[-1]) or first
    return round(end - first, 2)


def misses(times: dict[str, float | None], targets: StructureTargets) -> list[str]:
    """Each beat that starts after its target, as `beat at X s, target Y s`."""
    found = []
    for key, target in (
        ("turn", targets.turn_by_sec),
        ("payoff", targets.payoff_by_sec),
    ):
        value = times.get(key)
        if target is not None and value is not None and value > target:
            found.append(f"{key} at {value} s, target {target} s")
    return found


def sentences_of(script: str) -> list[str]:
    return split_sentences(script.strip())
