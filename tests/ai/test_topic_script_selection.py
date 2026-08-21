"""Tests for topic scripts selecting problem-first templates.

Split the same way as the other config-field tests in this repo: one group
proves the setting is read, a separate group proves the value reaches the
prompt. A template that is selected but whose placeholders never fill looks
identical to one that works.
"""

import re

import pytest

from src.ai.script_generator import (
    _short_product_name,
    format_prompt,
    select_script_template,
)
from src.scraper.amazon.models import ProductData
from src.scraper.base.models import Platform
from src.video.producer.topic_input import TopicSpec, build_topic_product


def _product() -> ProductData:
    return ProductData(
        title="Anker 65W USB-C Charger, GaN, Foldable",
        price="$39.99",
        url="https://example.test/dp/B0",
        platform=Platform.AMAZON,
        description="A compact charger.",
    )


@pytest.mark.unit
class TestTopicTemplateSelection:
    """Which template a topic gets.

    The pool is replaced rather than narrowed: every non-topic template is
    written to pitch a product, so one left reachable means a topic sometimes
    renders as an advertisement for a subject.
    """

    def test_topic_selects_only_from_the_topic_pool(self):
        from src.video.config import config

        st = config.llm_settings.script_templates
        assert st.topic_templates, "no topic templates configured"
        chosen = {
            select_script_template(
                config.llm_settings, product_id=f"topic-{i}", is_topic=True
            ).stem
            for i in range(40)
        }
        assert chosen <= set(st.topic_templates), chosen

    def test_a_product_never_draws_a_topic_template(self):
        from src.video.config import config

        st = config.llm_settings.script_templates
        chosen = {
            select_script_template(config.llm_settings, product_id=f"B0{i}").stem
            for i in range(40)
        }
        assert not chosen & set(st.topic_templates), chosen

    def test_selection_is_deterministic(self):
        from src.video.config import config

        a = select_script_template(config.llm_settings, "topic-x", is_topic=True)
        b = select_script_template(config.llm_settings, "topic-x", is_topic=True)
        assert a == b


@pytest.mark.unit
class TestTopicPromptContent:
    """What actually reaches the model."""

    def test_topic_placeholders_fill(self):
        from src.video.config import config

        spec = TopicSpec(
            title="Why your laptop fan is always loud",
            description="Dust, thermal paste, background CPU load.",
        )
        product = build_topic_product(spec)
        path = select_script_template(
            config.llm_settings, product_id=product.asin, is_topic=True
        )
        prompt = format_prompt(path.read_text(encoding="utf-8"), product, "anyone")
        assert "Why your laptop fan is always loud" in prompt
        assert "Dust, thermal paste" in prompt
        assert "{TOPIC_TITLE}" not in prompt

    def test_a_topic_title_is_not_trimmed_to_a_product_alias(self):
        """`_short_product_name` keeps three words and cuts at SEO separators.

        Right for a listing title, wrong for a question: the template then
        instructs the model to speak the fragment as the thing's name.
        """
        title = "Why your laptop fan is always loud"
        assert _short_product_name(title) == "Why your laptop"

        product = build_topic_product(TopicSpec(title=title))
        prompt = format_prompt(
            "{SHORT_PRODUCT_NAME}|{FULL_PRODUCT_NAME}", product, "anyone"
        )
        assert prompt.split("|")[0] == title

    def test_a_product_title_is_still_trimmed(self):
        prompt = format_prompt("{SHORT_PRODUCT_NAME}", _product(), "anyone")
        assert prompt == "Anker 65W USB-C"


@pytest.mark.unit
class TestTopicNarratorProfile:
    """The affiliate call to action lives in the narrator profile.

    Swapping templates alone leaves it in place, so a topic script still ends by
    telling the viewer where to buy something that does not exist.
    """

    def test_a_topic_narrator_is_configured(self):
        from src.video.config import config

        assert config.llm_settings.script_templates.narrator_profile_topic

    def test_its_cta_options_offer_nothing_to_buy(self):
        from src.video.config import config

        topic_narrator = config.llm_settings.script_templates.narrator_profile_topic
        cta_line = next(
            ln for ln in topic_narrator.splitlines() if ln.strip().startswith("CTA:")
        )
        # Only the quoted options; the surrounding prose says the word "buy"
        # precisely to forbid it.
        options = re.findall(r'"([^"]+)"', cta_line)
        assert options, cta_line
        for option in options:
            lowered = option.lower()
            for buy_phrase in ("link in bio", "want one", "grab", "buy", "get yours"):
                assert buy_phrase not in lowered, option

    def test_the_product_narrator_still_does(self):
        """Guards against the two profiles being conflated later."""
        from src.video.config import config

        assert "Link in bio" in config.llm_settings.script_templates.narrator_profile


@pytest.mark.unit
class TestNarratorResolver:
    """One resolver, because three call sites need the same answer.

    The hook overlay and the per-platform caption prompts take a narrator
    profile too, and the hook overlay is on by default. Choosing at each call
    site left a topic render's burned-in headline carrying the purchase voice
    while only the spoken script changed.
    """

    def test_a_topic_gets_the_topic_profile(self):
        from src.video.config import config

        st = config.llm_settings.script_templates
        assert st.narrator_for(True) == st.narrator_profile_topic

    def test_a_product_gets_the_product_profile(self):
        from src.video.config import config

        st = config.llm_settings.script_templates
        assert st.narrator_for(False) == st.narrator_profile

    def test_every_consumer_resolves_rather_than_reading_the_field(self):
        """A new consumer that reads the field directly reintroduces the bug.

        This is the shape of the defect the resolver exists to prevent, so it is
        cheaper to assert than to rediscover.
        """
        from pathlib import Path

        consumers = [
            Path("src/video/producer/steps.py"),
            Path("src/ai/script_generator.py"),
        ]
        for path in consumers:
            text = path.read_text(encoding="utf-8")
            assert "narrator_profile=script_cfg.narrator_profile," not in text, path
            assert "script_templates.narrator_profile," not in text, path


@pytest.mark.unit
class TestPillarPreamblesAreProductShaped:
    """Why `--pillar` is refused with `--topic`.

    The preambles and audience hints are written about a product, so combining
    them with a topic template produces a prompt that argues with itself: one
    half says never invent a product, the other assumes there is one.
    """

    def test_preambles_still_talk_about_a_product(self):
        from src.video.config import config

        preambles = config.llm_settings.script_templates.pillar_preambles
        assert preambles, "no pillar preambles configured"
        assert any("product" in text.lower() for text in preambles.values())

    def test_audiences_still_describe_buyers(self):
        from src.video.config import config

        audiences = config.llm_settings.script_templates.pillar_audiences
        assert audiences, "no pillar audiences configured"
        assert any(
            "buyer" in text.lower() or "shopper" in text.lower()
            for text in audiences.values()
        )
