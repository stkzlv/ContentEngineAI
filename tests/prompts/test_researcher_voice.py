"""Product scripts speak as the researcher, never the owner (design 0024)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.ai.script_generator import (
    RATING_TEMPLATES,
    format_prompt,
    listing_reviews,
    select_script_template,
)
from src.research.checks import OWNERSHIP, PRICE, check_one
from src.scraper.amazon.models import ProductData
from src.video.config import config

SCRIPTS = Path(__file__).resolve().parents[2] / "src" / "ai" / "prompts" / "scripts"
PRODUCT_TEMPLATES = sorted(
    p for p in SCRIPTS.glob("*.md") if not p.stem.startswith("topic_")
)


def _product(rating: str | None = "4.6", reviews: str | None = "(298)") -> ProductData:
    product = ProductData(
        title="Magnetic phone tripod", price="", url="", platform="amazon"
    )
    product.rating = rating
    product.reviews_count = reviews
    return product


@pytest.mark.req("REQ-CNT-157", "REQ-CNT-158")
@pytest.mark.parametrize("path", PRODUCT_TEMPLATES, ids=lambda p: p.stem)
def test_no_template_models_a_claim_of_use_or_a_price(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    quoted = " ".join(re.findall(r'"([^"]+)"', text))

    assert not OWNERSHIP.search(quoted), OWNERSHIP.search(quoted)
    assert not PRICE.search(text)
    assert "price-first" not in text and "price band" not in text


@pytest.mark.req("REQ-CNT-156", "REQ-CNT-157", "REQ-CNT-158")
def test_the_narrator_profile_speaks_as_the_researcher() -> None:
    profile = config.llm_settings.script_templates.narrator_profile

    assert "never owned, bought, received, wore, used or tested" in profile
    assert "Never speak a price" in profile
    assert not OWNERSHIP.search(profile.split("Voice example")[1])


@pytest.mark.req("REQ-CNT-159")
@pytest.mark.parametrize("path", PRODUCT_TEMPLATES, ids=lambda p: p.stem)
def test_closing_examples_are_shapes(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for rule in re.findall(r"^- \*\*Close with.*$", text, re.M):
        for example in re.findall(r'"([^"]{12,})"', rule):
            # A unit or a word the model is told about is fine; a sentence
            # it could reuse carries no placeholder.
            sentence = example[:1].isalpha() and example.rstrip()[-1:] in ".?!"
            if sentence and example.count(" ") >= 3:
                assert "[" in example, example


@pytest.mark.req("REQ-CNT-160")
@pytest.mark.parametrize("path", PRODUCT_TEMPLATES, ids=lambda p: p.stem)
def test_each_template_asks_who_it_suits(path: Path) -> None:
    assert "who this suits or who should skip it" in path.read_text(encoding="utf-8")


@pytest.mark.req("REQ-CNT-157")
@pytest.mark.parametrize(
    ("rating", "reviews", "found"),
    [
        ("4.6", "(298)", ("4.6", "298")),
        ("4.5", "1,234 ratings", ("4.5", "1,234")),
        ("4.4", "(2.1K)", ("4.4", "2.1K")),
        (None, "(298)", None),
        ("4.6", "", None),
    ],
)
def test_listing_reviews_need_both_figures(rating, reviews, found) -> None:
    assert listing_reviews(_product(rating, reviews)) == found


@pytest.mark.req("REQ-CNT-157")
def test_a_product_without_reviews_never_draws_the_rating_template() -> None:
    settings = config.llm_settings.model_copy(deep=True)
    settings.script_templates.fixed_template = None
    settings.script_templates.template_pool = sorted(RATING_TEMPLATES) + ["rapid_fire"]

    drawn = {
        select_script_template(settings, f"B0{n:08d}", has_reviews=False).stem
        for n in range(30)
    }
    with_reviews = {
        select_script_template(settings, f"B0{n:08d}", has_reviews=True).stem
        for n in range(30)
    }

    assert drawn == {"rapid_fire"}
    assert with_reviews >= RATING_TEMPLATES


@pytest.mark.req("REQ-CNT-157")
def test_the_rating_template_quotes_the_listing_figures() -> None:
    template = (SCRIPTS / "social_proof.md").read_text(encoding="utf-8")

    prompt = format_prompt(template, _product(), "anyone")

    assert "a rating of 4.6 from 298 ratings" in prompt


@pytest.mark.req("REQ-CNT-157", "REQ-CNT-158")
def test_the_research_checks_count_claims_and_prices() -> None:
    record = {
        "kind": "product",
        "title": "Tripod",
        "keyword": "tripod",
        "cta": "Follow for more finds like this.",
        "script": (
            "I picked this up last month. Spent $40 on it. It's heavier than I "
            "expected. Follow for more finds like this."
        ),
    }

    checked = check_one(record, (10, 100))

    assert checked["ownership"] == 3 and checked["price"] is True
    record["script"] = (
        "I went through the listing for this tripod. It reaches 62 inches. "
        "Follow for more finds like this."
    )
    clean = check_one(record, (10, 100))
    assert clean["ownership"] == 0 and clean["price"] is False


@pytest.mark.req("REQ-CNT-157")
@pytest.mark.parametrize(
    "claim",
    [
        "I use it every day.",
        "I've had mine for a month.",
        "I have one on my desk.",
        "I ended up buying it.",
        "Mine came with two straps.",
        "My kids love it.",
        "I picked this up last month.",
        "Just got this cable.",
        "It's heavier than I expected.",
        "I tried five tripods.",
        "I regret buying the cheap one.",
        "Three friends recommended this.",
    ],
)
def test_a_claim_of_use_is_counted(claim: str) -> None:
    assert OWNERSHIP.search(claim)


@pytest.mark.req("REQ-CNT-157")
@pytest.mark.parametrize(
    "line",
    [
        "I tried to find a catch.",
        "I used to think these were gimmicks.",
        "I've got to say the specs hold up.",
        "I got curious about this one.",
        "I'd skip it if you travel light.",
        "If you've used one before, you'll get it.",
        "I went through the listing.",
        "I have to say it holds up.",
    ],
)
def test_researcher_phrasing_is_not_counted(line: str) -> None:
    assert not OWNERSHIP.search(line)


@pytest.mark.req("REQ-CNT-157")
def test_a_fixed_rating_template_is_ignored_without_ratings() -> None:
    settings = config.llm_settings.model_copy(deep=True)
    settings.script_templates.fixed_template = "social_proof"

    assert select_script_template(settings, "B0X", has_reviews=True).stem == (
        "social_proof"
    )
    assert select_script_template(settings, "B0X", has_reviews=False).stem != (
        "social_proof"
    )


@pytest.mark.req("REQ-CNT-157")
@pytest.mark.asyncio
async def test_the_script_step_passes_whether_the_listing_has_ratings() -> None:
    from unittest.mock import AsyncMock, patch

    from src.ai import script_generator

    settings = config.llm_settings.model_copy(deep=True)
    settings.script_templates.fixed_template = None
    settings.script_templates.template_pool = ["social_proof", "rapid_fire"]
    settings.script_validation.lint.enabled = False
    seen = []
    for n in range(12):
        call = AsyncMock(return_value="")
        with patch.object(script_generator, "_call_llm_api_with_retry", call):
            await script_generator.generate_script(
                _product(None, None),
                settings,
                {settings.api_key_env_var: "k"},
                AsyncMock(),
                {},
                False,
                product_id=f"B0{n:08d}",
            )
        prompt = call.await_args_list[0].args[0]
        seen.append("Social Proof" in prompt)

    assert not any(seen)
