"""The recorded author sign-off is removed before the script's end is read."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ai.platform_metadata import _read_video_script
from src.publisher.first_comment import extract_closing_line
from src.utils.script_signoff import drop_signoff, recorded_signoff

SIGNOFF = "That's the fix for today."
SCRIPT = (
    "Quick one, your wifi drops at night.\n"
    "Change the channel to 1, 6 or 11.\n"
    "You'll know it worked when it stays steady.\n"
    f"{SIGNOFF}\n"
    "Drop a comment if this worked."
)


def test_the_signoff_line_is_removed_and_nothing_else() -> None:
    out = drop_signoff(SCRIPT, SIGNOFF)
    assert SIGNOFF not in out
    assert out.endswith(
        "You'll know it worked when it stays steady.\nDrop a comment if this worked."
    )


def test_an_inline_signoff_keeps_the_spacing() -> None:
    script = f"It stays steady. {SIGNOFF} Drop a comment."
    assert drop_signoff(script, SIGNOFF) == "It stays steady. Drop a comment."


@pytest.mark.parametrize(
    ("spoken", "configured"),
    [
        ("That's the fix for today!", SIGNOFF),
        (SIGNOFF, "That's the fix for today"),
    ],
)
def test_punctuation_drift_matches_like_the_first_comment(
    spoken: str, configured: str
) -> None:
    script = f"It stays steady. Team A or B? {spoken} Drop a comment."
    out = drop_signoff(script, configured)
    assert out == "It stays steady. Team A or B? Drop a comment."
    assert extract_closing_line(script, signoff=configured) == "Team A or B?"


def test_no_signoff_or_no_match_is_unchanged() -> None:
    assert drop_signoff(SCRIPT, None) == SCRIPT
    assert drop_signoff(SCRIPT, "Something never said.") == SCRIPT


def test_it_is_read_from_the_state_or_the_step_entry(tmp_path: Path) -> None:
    assert recorded_signoff(tmp_path) is None
    (tmp_path / "pipeline_state.json").write_text(json.dumps({"signoff": SIGNOFF}))
    assert recorded_signoff(tmp_path) == SIGNOFF
    (tmp_path / "pipeline_state.json").write_text(
        json.dumps({"generate_script": {"signoff": SIGNOFF}})
    )
    assert recorded_signoff(tmp_path) == SIGNOFF
    (tmp_path / "pipeline_state.json").write_text("not json")
    assert recorded_signoff(tmp_path) is None


def test_the_caption_prompts_get_the_script_without_the_signoff(
    tmp_path: Path,
) -> None:
    """The caption prompts mirror the line right before the CTA, which is
    the sign-off when one was drawn.
    """
    script = tmp_path / "script.txt"
    script.write_text(SCRIPT)
    (tmp_path / "pipeline_state.json").write_text(json.dumps({"signoff": SIGNOFF}))
    text = _read_video_script({"script": script})
    assert text is not None
    assert SIGNOFF not in text
    assert "stays steady.\nDrop a comment" in text
