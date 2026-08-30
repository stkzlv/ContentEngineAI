"""A topic render must not be described by a product-shaped prompt.

Measured on a real topic render with the generators production uses. The
worked examples in `youtube_metadata.md` all end `Shop now:
https://example.com/product`, and the model copied that line verbatim into the
YouTube description -- on the one surface where a viewer could click, a topic
render offered `example.com`. The same measurement produced an Instagram
caption ending "Most people only need two ports, but three is usually
better.", which is the literal example string from the Closing-line Mirror
rule in `instagram_caption.md`, carried onto a phone-battery topic.

Both are the project's recorded failure mode: an example teaches its subject,
not just its shape, and when an example contradicts a rule the example wins.
So the topic path gets its own files rather than reworded rules.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest

from src.ai.prompt_selection import (
    _TOPIC_VARIANTS,
    is_topic_record,
    prompt_path_for,
)
from src.scraper.amazon.models import ProductData
from src.scraper.base.models import Platform

PROMPT_DIR = Path("src/ai/prompts")


def topic_record() -> ProductData:
    return ProductData(
        title="Why your wifi keeps dropping",
        price="",
        url="",
        platform=Platform.AMAZON,
        description="Router placement and channel congestion.",
        asin="topic-why-wifi-drops",
        topic="Why your wifi keeps dropping",
    )


def product_record() -> ProductData:
    return ProductData(
        title="Wireless earbuds",
        price="19.99",
        url="https://www.amazon.com/dp/B0TEST1234",
        platform=Platform.AMAZON,
        description="Bluetooth earbuds.",
        asin="B0TEST1234",
        affiliate_link="https://www.amazon.com/dp/B0TEST1234",
    )


class TestSelection:
    @pytest.mark.parametrize(("product_prompt", "variant"), _TOPIC_VARIANTS.items())
    def test_a_topic_gets_the_variant(self, product_prompt, variant):
        chosen = prompt_path_for(topic_record(), PROMPT_DIR / product_prompt)

        assert chosen.name == variant

    @pytest.mark.parametrize("product_prompt", _TOPIC_VARIANTS)
    def test_a_product_gets_the_product_prompt(self, product_prompt):
        """The product path must be untouched; it is the one that works."""
        chosen = prompt_path_for(product_record(), PROMPT_DIR / product_prompt)

        assert chosen.name == product_prompt

    def test_a_prompt_with_no_variant_is_unchanged(self):
        chosen = prompt_path_for(topic_record(), PROMPT_DIR / "hook_headline.md")

        assert chosen.name == "hook_headline.md"

    def test_a_missing_variant_falls_back(self, tmp_path):
        """A caller wired up before its variant exists still renders.

        Raising would take out the whole description step for a defect that
        degrades to the previous behaviour.
        """
        stub = tmp_path / "youtube_metadata.md"
        stub.write_text("x", encoding="utf-8")

        assert prompt_path_for(topic_record(), stub) == stub

    def test_topic_detection_matches_the_disclosure_rule(self):
        """Selection and `carries_affiliate_content` must agree.

        They key off the same property. If they diverge, a render can get
        product framing while its disclosure says there is no product, or the
        reverse.
        """
        from src.scraper.base.models import carries_affiliate_content

        assert is_topic_record(topic_record()) is True
        assert carries_affiliate_content(topic_record()) is False
        assert is_topic_record(product_record()) is False
        assert carries_affiliate_content(product_record()) is True


class TestEveryVariantExists:
    @pytest.mark.parametrize(("product_prompt", "variant"), _TOPIC_VARIANTS.items())
    def test_both_files_are_present(self, product_prompt, variant):
        assert (PROMPT_DIR / product_prompt).exists()
        assert (PROMPT_DIR / variant).exists(), (
            f"{variant} is mapped but absent, so a topic silently falls back "
            "to the product prompt"
        )


class TestTheVariantsCarryNoProductArtefacts:
    """What the measurement found, asserted against the files."""

    @pytest.mark.parametrize("variant", _TOPIC_VARIANTS.values())
    def test_no_placeholder_url(self, variant):
        """`https://example.com/product` shipped in a real description."""
        text = (PROMPT_DIR / variant).read_text(encoding="utf-8")

        assert "example.com" not in text
        assert "Shop now" not in text

    @pytest.mark.parametrize("variant", _TOPIC_VARIANTS.values())
    def test_no_product_placeholder(self, variant):
        """The topic files must not reference the product slots at all.

        `format_prompt` supplies both sets, so a leftover
        `{FULL_PRODUCT_NAME}` renders the topic's title under a product label
        and reintroduces the framing this exists to remove.
        """
        text = (PROMPT_DIR / variant).read_text(encoding="utf-8")

        assert "{FULL_PRODUCT_NAME}" not in text
        assert "{PRODUCT_DESCRIPTION}" not in text

    @pytest.mark.parametrize("variant", _TOPIC_VARIANTS.values())
    def test_the_topic_placeholders_are_used(self, variant):
        text = (PROMPT_DIR / variant).read_text(encoding="utf-8")

        assert "{TOPIC_TITLE}" in text
        assert "{TOPIC_DETAIL}" in text

    @pytest.mark.parametrize("variant", _TOPIC_VARIANTS.values())
    def test_no_example_asks_for_a_disclosure(self, variant):
        """A topic has no material connection, and the record says so.

        The tag is stripped downstream either way, so this is about the file
        not contradicting its own sibling -- and about not spending one of the
        platform's hashtag slots on a tag that is about to be removed.
        """
        text = (PROMPT_DIR / variant).read_text(encoding="utf-8")
        lines = [line for line in text.splitlines() if "#ad" in line]

        # The only permitted mention is the instruction not to emit one.
        assert all(
            "Do NOT include #ad" in line for line in lines
        ), f"{variant} still demonstrates or requests #ad: {lines}"

    @pytest.mark.parametrize("variant", _TOPIC_VARIANTS.values())
    def test_no_carried_over_product_example(self, variant):
        """The exact string that leaked into a phone-battery caption."""
        text = (PROMPT_DIR / variant).read_text(encoding="utf-8")

        assert "two ports" not in text
        assert "wireless earbuds" not in text.lower()
        assert "power bank" not in text.lower()


class TestTheVariantsRender:
    """A prompt that raises takes the whole step with it.

    `format_prompt` raises on an unsupplied placeholder, and the description
    caller turns that into a `DescriptionGenerationError`. A topic variant
    naming a placeholder its call site does not supply would fail every topic
    render rather than degrading.
    """

    @pytest.mark.parametrize("variant", _TOPIC_VARIANTS.values())
    def test_the_unified_placeholders_are_enough(self, variant):
        # `src.video.config` first: importing `description_generator` as the
        # first project import trips a pre-existing cycle, the same one that
        # kept `make clean-outputs` from ever running.
        import src.video.config  # noqa: F401
        from src.ai.description_generator import format_prompt

        text = (PROMPT_DIR / variant).read_text(encoding="utf-8")
        if "{VIDEO_SCRIPT}" in text:
            pytest.skip("needs a script; covered by the platform call sites")

        rendered = format_prompt(text, topic_record())

        assert "Why your wifi keeps dropping" in rendered
        assert "{" not in rendered.replace(
            "{}", ""
        ), "an unsubstituted placeholder survived into the rendered prompt"


class TestTheCallSitesUseTheSelector:
    """The selector having tests is not the guard; the four call sites are.

    Unwiring them leaves every assertion above green while a topic render goes
    back to the product prompt -- which is the whole defect.
    """

    @pytest.mark.parametrize(
        ("module", "expected"),
        [
            ("src.ai.platform_metadata.youtube", "youtube_metadata_topic.md"),
            ("src.ai.platform_metadata.tiktok", "tiktok_caption_topic.md"),
            ("src.ai.platform_metadata.instagram", "instagram_caption_topic.md"),
        ],
    )
    @pytest.mark.asyncio
    async def test_a_platform_generator_loads_the_variant(
        self, module, expected, monkeypatch, tmp_path
    ):
        """Both entry points are captured and both abort.

        The three generators do not agree on how the template reaches the LLM:
        two pass the path into `generate_with_llm`, one loads it itself. Patch
        only the first and the third makes a live API call from a test.
        """
        import importlib
        from unittest.mock import AsyncMock

        import src.video.config as vconf
        from src.ai.platform_metadata.models import PlatformMetadataSettings

        mod = importlib.import_module(module)
        seen: list = []

        class _StopError(Exception):
            pass

        def capture(template_path, *args, **kwargs):
            seen.append(Path(template_path).name)
            raise _StopError

        for name in ("generate_with_llm", "load_prompt_template"):
            if hasattr(mod, name):
                monkeypatch.setattr(mod, name, capture)

        settings = PlatformMetadataSettings()
        platform = module.rsplit(".", 1)[1]
        generator_cls = next(
            obj
            for name, obj in vars(mod).items()
            if name.endswith("MetadataGenerator")
            and name != "BasePlatformMetadataGenerator"
        )
        generator = generator_cls(getattr(settings, platform).model_dump())
        llm = vconf.config.llm_settings

        with contextlib.suppress(Exception):
            await generator.generate(
                product=topic_record(),
                settings=llm,
                secrets={llm.api_key_env_var: "test-key"},
                session=AsyncMock(),
                intermediate_paths={"temp_dir": tmp_path},
                debug_mode=False,
            )

        assert seen and seen[0] == expected, (
            f"{module} loaded {seen or 'nothing'}; a topic render is being "
            "described by the product prompt"
        )

    @pytest.mark.asyncio
    async def test_the_unified_generator_loads_the_variant(self, monkeypatch, tmp_path):
        import src.video.config  # noqa: F401
        from src.ai import description_generator as dg

        seen: list = []
        real = dg.load_prompt_template

        def capture(path):
            seen.append(Path(path).name)
            return real(path)

        monkeypatch.setattr(dg, "load_prompt_template", capture)
        # Stop right after the template is chosen; the API call is not the
        # subject and would need a live key.
        monkeypatch.setattr(
            dg,
            "format_prompt",
            lambda *a, **k: (_ for _ in ()).throw(ValueError("stop")),
        )

        llm = src.video.config.config.llm_settings

        with pytest.raises(dg.DescriptionGenerationError):
            await dg.generate_description(
                product=topic_record(),
                settings=llm,
                secrets={llm.api_key_env_var: "test-key"},
                session=None,
                intermediate_paths={},
                debug_mode=False,
            )

        assert seen == ["video_description_topic.md"], (
            f"the unified path loaded {seen or 'nothing'}; the default mode "
            "still describes a topic with the product prompt"
        )
