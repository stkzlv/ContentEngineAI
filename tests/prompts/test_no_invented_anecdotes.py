"""No shipped prompt asks for, or models, an invented personal moment.

The narrator profile asked for "a real moment" and quoted "Charged it Sunday,
forgot about it until Friday"; live smartwatch scripts copied the anecdote in
every attempt, so product videos claimed a week of use nobody had.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.video.config import config

TEMPLATES = Path(__file__).resolve().parents[2] / "src" / "ai" / "prompts" / "scripts"
WEEKDAY = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I
)


@pytest.mark.req("REQ-CNT-156")
def test_the_narrator_profile_takes_detail_from_the_description() -> None:
    profile = config.llm_settings.script_templates.narrator_profile

    assert "taken from the product description" in profile
    assert "Never invent a personal moment" in profile
    assert not WEEKDAY.search(profile)


@pytest.mark.req("REQ-CNT-156")
@pytest.mark.parametrize("path", sorted(TEMPLATES.glob("*.md")), ids=lambda p: p.stem)
def test_no_template_models_a_day_spent_with_the_product(path: Path) -> None:
    assert not WEEKDAY.search(path.read_text(encoding="utf-8")), path.name
