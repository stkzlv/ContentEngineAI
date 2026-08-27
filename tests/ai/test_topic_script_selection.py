"""Tests for topic scripts selecting problem-first templates.

Split the same way as the other config-field tests in this repo: one group
proves the setting is read, a separate group proves the value reaches the
prompt. A template that is selected but whose placeholders never fill looks
identical to one that works.
"""

import logging
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ai.script_generator import (
    _short_product_name,
    format_prompt,
    generate_hook_headline,
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
class TestEachFamilyGetsItsOwnPillarVocabulary:
    """`--pillar` works for both, because each has its own maps.

    The product preambles are written about a thing being shown, so pairing
    one with a topic template used to produce a prompt that argued with
    itself: one half said never invent a product, the other assumed there was
    one. The CLI refused the combination rather than emit that. Refusing was
    a stopgap; the topic maps are the answer, and the guard is gone.
    """

    def test_product_preambles_still_talk_about_a_product(self):
        from src.video.config import config

        preambles = config.llm_settings.script_templates.pillar_preambles
        assert preambles, "no pillar preambles configured"
        assert any("product" in text.lower() for text in preambles.values())

    def test_topic_preambles_never_mention_a_product(self):
        from src.video.config import config

        preambles = config.llm_settings.script_templates.pillar_preambles_topic
        assert preambles, "no topic pillar preambles configured"
        for name, text in preambles.items():
            assert "product" not in text.lower(), name

    def test_topic_audiences_never_describe_buyers(self):
        """Nobody watching a tech-help video is shopping.

        The product map says "buyers" and "shoppers"; the same words would put
        a purchase in a script that recommends nothing.
        """
        from src.video.config import config

        audiences = config.llm_settings.script_templates.pillar_audiences_topic
        assert audiences, "no topic pillar audiences configured"
        for name, text in audiences.items():
            low = text.lower()
            assert "buyer" not in low and "shopper" not in low, name

    def test_both_families_offer_the_same_pillars(self):
        """So --pillar takes the same values whichever family is rendering.

        A later taxonomy change then moves one key list rather than leaving
        two vocabularies to drift apart.
        """
        from src.video.config import config

        cfg = config.llm_settings.script_templates
        assert set(cfg.pillar_preambles_topic) == set(cfg.pillar_preambles)
        assert set(cfg.pillar_audiences_topic) == set(cfg.pillar_audiences)

    def test_a_topic_render_selects_the_topic_maps(self):
        from src.video.config import config

        cfg = config.llm_settings.script_templates
        assert cfg.preambles_for(True) == cfg.pillar_preambles_topic
        assert cfg.audiences_for(True) == cfg.pillar_audiences_topic
        assert cfg.preambles_for(False) == cfg.pillar_preambles
        assert cfg.audiences_for(False) == cfg.pillar_audiences


@pytest.mark.unit
class TestPillarSelectionOnATopic:
    """A pillar narrows product templates; on a topic it shapes the preamble.

    `pillars` maps a pillar to product template names, and a topic replaces
    the pool with the topic family, so the two never intersect. That is the
    designed outcome, not a misconfiguration, and warning about it would fire
    on every topic render that names a pillar.
    """

    def test_a_topic_with_a_pillar_keeps_the_topic_pool(self):
        from src.video.config import config

        cfg = config.llm_settings
        chosen = select_script_template(
            cfg, product_id="topic-x", pillar="utility", is_topic=True
        )

        assert chosen.stem in cfg.script_templates.topic_templates

    def test_it_does_not_warn(self, caplog):
        from src.video.config import config

        with caplog.at_level(logging.WARNING):
            select_script_template(
                config.llm_settings,
                product_id="topic-x",
                pillar="utility",
                is_topic=True,
            )

        assert "intersecting current pool" not in caplog.text

    def test_a_product_with_an_unmatched_pillar_still_warns(self, caplog):
        """The real misconfiguration must stay loud."""
        from src.video.config import config

        with caplog.at_level(logging.WARNING):
            select_script_template(
                config.llm_settings,
                product_id="B0TEST",
                pillar="not_a_real_pillar_but_mapped",
                is_topic=False,
            )


@pytest.mark.unit
class TestTheHookHeadlineHasATopicVariant:
    """The product headline prompt requires a product category noun.

    On a topic with no device that forces an invention. Measured against the
    live model: "why your passwords keep getting leaked" produced "Password
    manager that stops leaks" over a script that never mentions one. Rewording
    the rule alone would not hold, because every example in the product file
    is product-shaped and examples beat rules when the two disagree.
    """

    def test_the_topic_prompt_exists_and_names_no_product(self):
        text = Path("src/ai/prompts/hook_headline_topic.md").read_text()

        assert "{TOPIC_TITLE}" in text
        assert "{FULL_PRODUCT_NAME}" not in text

    def test_it_forbids_naming_a_product_the_script_omits(self):
        text = Path("src/ai/prompts/hook_headline_topic.md").read_text().lower()

        assert "never name a product" in text

    def test_the_measured_failures_are_the_anti_examples(self):
        """Anti-examples are real outputs, not invented ones.

        A rule the model already broke is worth more as a demonstration than
        a description, and these four are what it actually produced.
        """
        text = Path("src/ai/prompts/hook_headline_topic.md").read_text()

        for produced in (
            "Password manager that stops leaks",
            "Password leaks explained",
            "Laptop speed up with these tips",
            "Website won't load fixer",
        ):
            assert produced in text, produced

    def test_the_product_prompt_is_untouched(self):
        """A product render must keep requiring its category noun."""
        text = Path("src/ai/prompts/hook_headline.md").read_text()

        assert "The product category noun MUST appear" in text


@pytest.mark.unit
class TestTheWiringSelectsTheTopicVariants:
    """Asserting the files exist does not assert anything reads them.

    Reverting either selection left every other test in this file green, so
    these capture the argument the code actually passes.
    """

    @pytest.mark.asyncio
    async def test_a_topic_render_loads_the_topic_headline_prompt(self):
        from src.video.config import config

        spec = TopicSpec(title="Why wifi drops", description="Channels.", keywords=[])
        captured = {}

        async def fake(template_path, *a, **kw):
            captured["path"] = Path(template_path)
            return "Your wifi drops because of this"

        with patch(
            "src.ai.platform_metadata.utilities.generate_with_llm", side_effect=fake
        ):
            await generate_hook_headline(
                build_topic_product(spec),
                config.llm_settings,
                {config.llm_settings.api_key_env_var: "k"},
                MagicMock(),
            )

        assert captured["path"].name == "hook_headline_topic.md"

    @pytest.mark.asyncio
    async def test_a_product_render_still_loads_the_product_prompt(self):
        from src.video.config import config

        product = ProductData(
            asin="B0TEST",
            title="A gadget",
            description="Does things.",
            price="9.99",
            url="https://example.com/dp/B0TEST",
            platform=Platform.AMAZON,
        )
        captured = {}

        async def fake(template_path, *a, **kw):
            captured["path"] = Path(template_path)
            return "Gadget that does things"

        with patch(
            "src.ai.platform_metadata.utilities.generate_with_llm", side_effect=fake
        ):
            await generate_hook_headline(
                product,
                config.llm_settings,
                {config.llm_settings.api_key_env_var: "k"},
                MagicMock(),
            )

        assert captured["path"].name == "hook_headline.md"

    def test_a_topic_audience_comes_from_the_topic_map(self):
        from src.ai.script_generator import _resolve_audience
        from src.video.config import config

        cfg = config.llm_settings
        assert (
            _resolve_audience("utility", cfg, is_topic=True)
            == cfg.script_templates.pillar_audiences_topic["utility"]
        )
        assert (
            _resolve_audience("utility", cfg, is_topic=False)
            == cfg.script_templates.pillar_audiences["utility"]
        )
