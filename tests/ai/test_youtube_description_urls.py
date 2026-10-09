"""A YouTube Shorts description carries no URL (REQ-PUB-148, #590)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ai.platform_metadata.youtube import drop_urls

PROMPT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "ai"
    / "prompts"
    / "youtube_metadata.md"
)


@pytest.mark.req("REQ-PUB-148")
def test_a_lead_in_and_url_line_goes_whole() -> None:
    text = "Great tripod.\n\nTeam A or team B?\n\nShop now: https://example.com/p"

    assert drop_urls(text) == "Great tripod.\n\nTeam A or team B?"


@pytest.mark.req("REQ-PUB-148")
def test_a_url_inside_a_sentence_goes_alone() -> None:
    text = "See www.example.com for the tripod and its mount specs.\nLink in bio."

    assert drop_urls(text) == "See for the tripod and its mount specs.\nLink in bio."


@pytest.mark.req("REQ-PUB-148")
def test_the_prompt_asks_for_the_profile_pointer_not_a_url() -> None:
    text = PROMPT.read_text(encoding="utf-8")

    assert "https://" not in text and "product URL" not in text
    assert '"Link in bio." on the final line' in text


@pytest.mark.req("REQ-PUB-148")
def test_the_parser_drops_a_url_the_model_wrote() -> None:
    from src.ai.platform_metadata.youtube import YouTubeMetadataGenerator

    response = (
        "TITLE: Tripod that reaches 62 inches for desk and travel\n"
        "DESCRIPTION: A tall tripod.\n\nShop now: https://example.com/p\n"
        "HASHTAGS: #Shorts #ad\nKEYWORDS: tripod"
    )
    parsed = YouTubeMetadataGenerator._parse_llm_response(None, response)

    assert parsed is not None and "http" not in parsed[1]
