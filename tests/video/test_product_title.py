"""A product video's short written title (design 0009)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.video.config import load_video_config_modular
from src.video.product_title import short_title

LISTING = (
    'Smart Watch for Men Women, 1.83" HD Touch Screen with Bluetooth Call, '
    "IP68 Waterproof, 140+ Sport Modes, Fitness Tracker"
)


@pytest.mark.req("REQ-PUB-008", "REQ-CNT-149")
@pytest.mark.parametrize(
    ("keyword", "headline", "expected"),
    [
        # The headline names the keyword: it is the title.
        (
            "smart watch",
            "Smartwatch that takes calls from your wrist",
            "Smartwatch that takes calls from your wrist",
        ),
        # It doesn't: the keyword leads.
        (
            "bluetooth tracker",
            "Never lose your keys again",
            "Bluetooth tracker: Never lose your keys again",
        ),
        # No headline: the listing title's first clause.
        ("smart watch", None, "Smart Watch for Men Women"),
    ],
)
def test_the_title_leads_with_the_keyword_and_makes_the_hooks_promise(
    keyword: str, headline: str | None, expected: str
) -> None:
    assert short_title(LISTING, keyword, headline, 60) == expected


@pytest.mark.req("REQ-PUB-008")
def test_the_title_stays_within_the_maximum() -> None:
    headline = "This tiny tracker finds your keys anywhere in the whole house"
    title = short_title(LISTING, "bluetooth tracker", headline, 40)

    assert len(title) <= 40
    assert not title.endswith((" ", ","))
    # The keyword form did not fit and neither did the headline: the clause.
    assert title == "Smart Watch for Men Women"


@pytest.mark.req("REQ-PUB-008")
def test_a_listing_with_no_clause_break_is_cut_on_a_word() -> None:
    title = short_title("word " * 30, None, None, 22)

    assert title == "word word word word"


def _ctx(tmp_path: Path, on: bool, topic: str | None = None) -> MagicMock:
    config = load_video_config_modular()
    data = config.model_dump()
    data["description_settings"]["short_product_titles"] = on
    ctx = MagicMock()
    ctx.config = type(config).model_validate(data)
    ctx.product.title = LISTING
    ctx.product.keyword = "smart watch"
    ctx.product.topic = topic
    ctx.product.asin = "B0WATCH001"
    ctx.description = None
    ctx.run_paths = {"run_root": tmp_path, "description_file": tmp_path / "d.txt"}
    ctx.state = {"hook_headline": "Smartwatch that takes calls from your wrist"}
    return ctx


def _written_title(ctx: MagicMock, tmp_path: Path) -> str:
    from src.video.producer.steps import _generate_unified_metadata

    with patch(
        "src.video.producer.steps.generate_ai_description",
        new=AsyncMock(return_value="A watch that takes calls."),
    ):
        asyncio.run(_generate_unified_metadata(ctx))
    return str(json.loads((tmp_path / "metadata.json").read_text())["title"])


@pytest.mark.req("REQ-PUB-008")
def test_on_the_published_title_is_the_short_one(tmp_path: Path) -> None:
    title = _written_title(_ctx(tmp_path, on=True), tmp_path)

    assert title == "Smartwatch that takes calls from your wrist"


@pytest.mark.req("REQ-PUB-008")
def test_off_the_title_is_the_listing_title(tmp_path: Path) -> None:
    assert _written_title(_ctx(tmp_path, on=False), tmp_path) == LISTING


@pytest.mark.req("REQ-PUB-008")
def test_a_topic_keeps_its_own_title(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, on=True, topic="How to stop battery drain")
    ctx.product.title = "How to stop battery drain"

    assert _written_title(ctx, tmp_path) == "How to stop battery drain"
