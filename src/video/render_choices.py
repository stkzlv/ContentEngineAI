"""What shaped each render, and how varied recent renders are.

Platforms demote templated output with little variation between posts, and a
daily automated pipeline is the profile those rules look for. Each finished
render appends one row to `state/render_choices.jsonl` under the outputs root:
the template, hook, voice, caption template, music, motion and transition it
used, and its script. The report reads the most recent rows and flags a
dimension where one value dominates, and pairs of scripts that read nearly the
same. It only measures; nothing here changes what a render chooses.

Usage:
  python -m src.video.render_choices [--last N] [--dominance 0.6]
      [--similarity 0.5] [--outputs-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.utils.outputs_paths import resolve_outputs_dir
from src.utils.render_choices_store import (
    choices_path,
    latest_per_product,
    load_recent,
)

logger = logging.getLogger(__name__)


# The dimensions the report counts. `script` is compared, not counted.
DIMENSIONS = (
    "profile",
    "script_template",
    "pillar",
    "cta",
    "voice_profile",
    "voice_name",
    "caption_engine",
    "caption_template",
    "music",
    "cold_open_variant",
    "assembly_mode",
    "pre_motion",
    "transition_sec",
    "ending",
)

DEFAULT_LAST = 14
DEFAULT_DOMINANCE = 0.6
DEFAULT_SIMILARITY = 0.5
# Character n-gram size for the script similarity check.
SHINGLE = 5
# Below this many renders a share means little: two of two is 100%.
MIN_RENDERS_FOR_DOMINANCE = 5


def _setting(profile: Any, global_settings: Any, name: str) -> Any:
    """A profile's value for a video setting, else the global one."""
    value = getattr(profile, name, None)
    return value if value is not None else getattr(global_settings, name, None)


def _music_name(music_info_file: Path | None) -> str | None:
    if music_info_file is None or not music_info_file.exists():
        return None
    try:
        info = json.loads(music_info_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    name = (info.get("name") or info.get("path")) if isinstance(info, dict) else None
    return str(name) if name else None


def choices_from_context(ctx: Any) -> dict[str, Any]:
    """The row for one finished render, read off its pipeline context."""
    state = ctx.state
    tts = state.get("tts_metadata") or {}
    pycaps = state.get("pycaps_metadata") or {}
    settings = ctx.config.video_settings
    return {
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        # The product directory's name: the id the run logs under, which falls
        # back to the title when a record has no ASIN.
        "product_id": Path(ctx.run_paths["run_root"]).name,
        "profile": ctx.profile_name,
        "script_template": state.get("script_template"),
        "pillar": state.get("pillar"),
        "cta": state.get("cta"),
        "hook_headline": state.get("hook_headline"),
        "voice_profile": tts.get("voice_profile"),
        "voice_name": tts.get("voice_name"),
        "caption_engine": state.get("subtitle_engine_resolved"),
        "caption_template": pycaps.get("template"),
        "music": _music_name(ctx.run_paths.get("music_info_file")),
        "cold_open_variant": state.get("cold_open_variant"),
        "assembly_mode": ctx.profile.video_assembly_mode,
        "pre_motion": _setting(ctx.profile, settings, "first_frame_pre_motion"),
        "transition_sec": _setting(ctx.profile, settings, "video_transition_duration"),
        "ending": _setting(ctx.profile, settings, "ending"),
        "script": ctx.script,
    }


def record_render_choices(outputs_dir: Path, row: dict[str, Any]) -> None:
    """Append one render's row. A failed write is logged, not raised: the
    video is already made, and a measurement must not undo it.
    """
    path = choices_path(outputs_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # A write cut off before its newline would swallow the next row.
        lead = ""
        if path.exists() and path.stat().st_size:
            with path.open("rb") as fh:
                fh.seek(-1, 2)
                lead = "" if fh.read(1) == b"\n" else "\n"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(lead + json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("Could not record render choices in %s: %s", path, exc)


def distribution(rows: Sequence[dict[str, Any]]) -> dict[str, Counter]:
    return {
        dim: Counter(str(row.get(dim)) for row in rows if row.get(dim) is not None)
        for dim in DIMENSIONS
    }


def dominance_alerts(
    rows: Sequence[dict[str, Any]], threshold: float
) -> list[tuple[str, str, float]]:
    """(dimension, value, share) where one value holds more than `threshold`.

    A dimension with one value in the whole window is a fixed setting (a
    pinned voice, a single profile), not drift, so it is listed in the
    distribution but does not alert.
    """
    if len(rows) < MIN_RENDERS_FOR_DOMINANCE:
        return []
    alerts = []
    for dim, counts in distribution(rows).items():
        total = sum(counts.values())
        if total < MIN_RENDERS_FOR_DOMINANCE or len(counts) < 2:
            continue
        value, count = counts.most_common(1)[0]
        share = count / total
        if share > threshold:
            alerts.append((dim, value, share))
    return alerts


def _shingles(text: str) -> set[str]:
    norm = " ".join(text.lower().split())
    return {norm[i : i + SHINGLE] for i in range(max(len(norm) - SHINGLE + 1, 1))}


def script_similarity(a: str, b: str) -> float:
    """Jaccard similarity of the two scripts' character 5-grams."""
    sa, sb = _shingles(a), _shingles(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def similar_scripts(
    rows: Sequence[dict[str, Any]], threshold: float
) -> list[tuple[str, str, float]]:
    """(product_id, product_id, similarity) for pairs at or above `threshold`."""
    scripts = [
        (str(row.get("product_id")), str(row["script"]))
        for row in rows
        if row.get("script")
    ]
    pairs = []
    for i, (id_a, text_a) in enumerate(scripts):
        for id_b, text_b in scripts[i + 1 :]:
            if id_a == id_b:
                continue
            ratio = script_similarity(text_a, text_b)
            if ratio >= threshold:
                pairs.append((id_a, id_b, ratio))
    return pairs


def warn_if_similar(
    outputs_dir: Path,
    product_id: str,
    script: str,
    last: int = DEFAULT_LAST,
    threshold: float = DEFAULT_SIMILARITY,
) -> list[tuple[str, float]]:
    """Log a warning for each recent script this one closely repeats.

    Warns only; the script is kept. Returns the matches for the caller.
    """
    matches = []
    for row in load_recent(outputs_dir, last):
        other = row.get("script")
        if not other or row.get("product_id") == product_id:
            continue
        ratio = script_similarity(script, str(other))
        if ratio >= threshold:
            matches.append((str(row.get("product_id")), ratio))
            logger.warning(
                "Script is %.0f%% alike the script of %s (variety check)",
                ratio * 100,
                row.get("product_id"),
            )
    return matches


def report(
    rows: Sequence[dict[str, Any]], dominance: float, similarity: float
) -> list[str]:
    rows = latest_per_product(rows)
    lines = [f"Render variety over the last {len(rows)} product(s)"]
    for dim, counts in distribution(rows).items():
        if not counts:
            continue
        shown = ", ".join(f"{value} {count}" for value, count in counts.most_common())
        lines.append(f"  {dim}: {shown}")
    alerts = dominance_alerts(rows, dominance)
    pairs = similar_scripts(rows, similarity)
    for dim, value, share in alerts:
        lines.append(f"ALERT {dim}: {value} in {share:.0%} of renders")
    for id_a, id_b, ratio in pairs:
        lines.append(f"ALERT scripts {id_a} and {id_b} are {ratio:.0%} alike")
    if not alerts and not pairs:
        lines.append("No alerts.")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--last", type=int, default=DEFAULT_LAST)
    parser.add_argument("--dominance", type=float, default=DEFAULT_DOMINANCE)
    parser.add_argument("--similarity", type=float, default=DEFAULT_SIMILARITY)
    parser.add_argument("--outputs-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    from dotenv import load_dotenv

    from src.utils.outputs_paths import get_project_root

    load_dotenv(get_project_root() / ".env")
    rows = load_recent(resolve_outputs_dir(args.outputs_dir), args.last)
    if not rows:
        print("No renders recorded yet.")
        return 0
    print("\n".join(report(rows, args.dominance, args.similarity)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
