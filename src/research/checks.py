"""Measured checks over the sampled scripts (REQ-OPS-110).

Pure functions over the sample records, so a recorded sample can be checked
again after a rule changes without a model call.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from typing import Any

from src.ai.llm_settings import ScriptLintConfig
from src.ai.script_lint import lint_script
from src.ai.step_list import word_range
from src.research.sample import is_task
from src.utils.script_sanitizer import split_sentences
from src.video.search_phrase import contains

# Templates that open on a symptom or a mistake, which a task topic lacks.
PROBLEM_TEMPLATES = frozenset({"topic_symptom_cause", "topic_mistake_fix"})
# An opening used this often across one variant's sample reads as a formula.
REPEATED_OPENING = 3


def words(text: str) -> int:
    return len(re.findall(r"[\w'’-]+", text))


def opening(script: str) -> str:
    sentences = split_sentences(script)
    return " ".join(sentences[0].split()[:3]).lower() if sentences else ""


def check_one(record: dict[str, Any], band: tuple[int, int]) -> dict[str, Any]:
    """The checks for one sampled script; an errored sample has none."""
    if record.get("error") or not record.get("script"):
        return {"error": record.get("error") or "no script"}
    script = record["script"]
    count = words(script)
    low, high = word_range(record["steps"]) if record.get("steps") else band
    sentences = split_sentences(script)
    first = sentences[0] if sentences else ""
    # What a viewer types: a topic's `search` where it has one, else its
    # title; a product's scraper keyword.
    if record["kind"] == "topic":
        phrase = str(record.get("search") or record["title"])
    else:
        phrase = str(record.get("keyword") or "")
    fact = record.get("fact_check") or {}
    flagged = fact.get("flagged") or []
    revision = fact.get("revision") or {}
    tell = lint_script(script, ScriptLintConfig(enabled=True), word_cap=False)
    return {
        "words": count,
        "band": [low, high],
        "in_band": low <= count <= high,
        "phrase_in_first_sentence": bool(phrase)
        and contains(first, phrase, question=record["kind"] == "topic"),
        "cta_last": bool(record.get("cta"))
        and bool(sentences)
        and sentences[-1].strip() == str(record["cta"]).strip(),
        "lint": tell,
        "flagged": len(flagged),
        "rewritten": bool(revision.get("accepted")),
        "template_misfit": record["kind"] == "topic"
        and is_task(record["title"])
        and record.get("template") in PROBLEM_TEMPLATES,
        "opening": opening(script),
    }


def summarise(checked: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-variant totals for the report's comparison table."""
    ok = [c for c in checked if "error" not in c]
    n = len(ok)

    def share(key: str) -> float | None:
        return round(sum(bool(c[key]) for c in ok) / n, 2) if n else None

    openings = Counter(c["opening"] for c in ok if c["opening"])
    return {
        "samples": len(checked),
        "errors": len(checked) - n,
        "median_words": statistics.median(c["words"] for c in ok) if n else None,
        "in_band": share("in_band"),
        "phrase_in_first_sentence": share("phrase_in_first_sentence"),
        "cta_last": share("cta_last"),
        "with_lint_tell": share("lint"),
        "template_misfit": share("template_misfit"),
        "flagged_claims": sum(c["flagged"] for c in ok),
        "rewritten": sum(c["rewritten"] for c in ok),
        "repeated_openings": {
            o: k for o, k in openings.most_common() if k >= REPEATED_OPENING
        },
    }


def run_checks(records: list[dict[str, Any]], band: tuple[int, int]) -> dict[str, Any]:
    """Check every sample and summarise per variant and kind."""
    rows = [{**r, "checks": check_one(r, band)} for r in records]
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(f"{r['variant']}/{r['kind']}", []).append(r["checks"])
    return {"samples": rows, "summary": {k: summarise(v) for k, v in groups.items()}}
