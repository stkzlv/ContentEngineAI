"""Every script ends on a configured call to action.

Five of five scheduled renders had none. The four CTAs lived as prose in the
narrator profile, forty lines from the task, while fifteen of eighteen
templates owned the closing beat with an imperative -- "Close with a debatable
claim right before the CTA" -- that named the CTA only as a position. The
nearer imperative won every time.

The fix puts the rule where it binds, refuses a script that ignores it, and
keeps the first-comment extractor able to strip it. Each half is pinned here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from src.ai.script_generator import (
    NO_CTA_REASON,
    ends_with_cta,
    format_prompt,
    render_cta_rule,
    validate_script_completeness,
)
from src.scraper.amazon.models import ProductData

REPO = Path(__file__).resolve().parents[2]
TEMPLATES = sorted((REPO / "src" / "ai" / "prompts" / "scripts").glob("*.md"))
PRODUCT_CTAS = [
    "Link in bio if you want one.",
    "Follow for more finds like this.",
    "Drop a comment if you've tried it.",
    "Share with someone who needs this.",
]

BODY = (
    "So I picked this up last month and figured I'd share. It is a small "
    "thing, fits in your jacket pocket, but the magnetic mount actually "
    "grips. Took it on a hike and never lost signal. The battery is fine, "
    "not great, about six hours under load. Charged it Sunday, forgot "
    "about it until Friday. Team magnetic or team plug-in?"
)


def _product() -> ProductData:
    return ProductData(title="Magnetic phone mount", price="", url="", platform="test")


@pytest.fixture(scope="module")
def shipped_ctas() -> dict[str, list[str]]:
    raw = yaml.safe_load((REPO / "config" / "ai_services.yaml").read_text())
    block = raw["llm_settings"]["script_templates"]
    return {"product": block["cta_options"], "topic": block["cta_options_topic"]}


@pytest.mark.unit
class TestTheRuleSitsNextToTheBeat:
    @pytest.mark.parametrize("template", TEMPLATES, ids=lambda p: p.stem)
    def test_every_template_carries_the_placeholder(self, template: Path) -> None:
        assert template.read_text().count("{CTA_RULE}") == 1

    @pytest.mark.parametrize(
        "template",
        [t for t in TEMPLATES if not t.stem.startswith("topic_")],
        ids=lambda p: p.stem,
    )
    def test_it_follows_the_closing_beat_rule(self, template: Path) -> None:
        """Adjacency is the fix. A placeholder at the end of the file would
        reproduce the distance that let the profile's version lose.
        """
        lines = template.read_text().split("\n")
        beat = next(i for i, line in enumerate(lines) if "right before the CTA" in line)

        assert lines[beat + 1] == "{CTA_RULE}"

    def test_the_rule_quotes_every_option_verbatim(self) -> None:
        rule = render_cta_rule(PRODUCT_CTAS)

        for cta in PRODUCT_CTAS:
            assert f'"{cta}"' in rule
        assert "very last sentence" in rule

    def test_the_topic_tail_does_not_point_at_a_beat_rule(self) -> None:
        """Topic templates have no closing-beat rule above the placeholder;
        the line above is an honest-limit rule, and "the closing beat above"
        would point the model at that.
        """
        assert "closing beat above" in render_cta_rule(PRODUCT_CTAS)
        assert "closing beat above" not in render_cta_rule(PRODUCT_CTAS, is_topic=True)
        assert "closing line the template asks for" in render_cta_rule(
            PRODUCT_CTAS, is_topic=True
        )

    def test_no_options_renders_nothing(self) -> None:
        assert render_cta_rule([]) == ""

    def test_the_prompt_renders_it(self) -> None:
        template = (REPO / "src/ai/prompts/scripts/curiosity_hook.md").read_text()

        prompt = format_prompt(
            template, _product(), "buyers", cta_rule=render_cta_rule(PRODUCT_CTAS)
        )

        assert "{CTA_RULE}" not in prompt
        assert '"Link in bio if you want one."' in prompt


@pytest.mark.unit
class TestTheValidatorRefusesAScriptWithoutOne:
    @pytest.mark.parametrize("cta", PRODUCT_CTAS)
    def test_each_option_is_accepted(self, cta: str) -> None:
        assert ends_with_cta(f"{BODY} {cta}", PRODUCT_CTAS)

    @pytest.mark.parametrize("ending", ["", "!", "..."])
    def test_punctuation_drift_is_tolerated(self, ending: str) -> None:
        """A dropped full stop is the same CTA."""
        script = f"{BODY} Link in bio if you want one{ending}"

        assert ends_with_cta(script, PRODUCT_CTAS)

    @pytest.mark.parametrize(
        "cta", ["Link in bio\nif you want one.", "Link in bio  if you want one."]
    )
    def test_whitespace_drift_is_tolerated(self, cta: str) -> None:
        """A wrapped or double-spaced CTA is the same CTA, not a paid retry."""
        assert ends_with_cta(f"{BODY} {cta}", PRODUCT_CTAS)

    def test_an_option_with_an_internal_full_stop_validates(self) -> None:
        options = ["Link in bio. Seriously."]

        assert ends_with_cta(f"{BODY} Link in bio. Seriously.", options)

    def test_a_suffix_of_a_cta_is_not_a_cta(self) -> None:
        assert not ends_with_cta(f"{BODY} Unlink in bio if you want one.", PRODUCT_CTAS)

    def test_an_empty_option_is_refused_at_load(self) -> None:
        from src.video.config.llm_settings import ScriptTemplateConfig

        with pytest.raises(ValueError, match="no words"):
            ScriptTemplateConfig(cta_options=["Link in bio.", "..."])

    def test_a_spec_claim_ending_is_refused(self) -> None:
        """The shipped failure: the closing beat with nothing after it."""
        script = f"{BODY} These bulbs have a 25,000-hour lifespan."

        assert not ends_with_cta(script, PRODUCT_CTAS)

    def test_a_cta_in_the_middle_does_not_count(self) -> None:
        script = f"Link in bio if you want one. {BODY}"

        assert not ends_with_cta(script, PRODUCT_CTAS)

    def test_a_paraphrase_is_refused(self) -> None:
        """Verbatim, so the extractor and the markers keep recognising it."""
        script = f"{BODY} The link is in my bio if you want it."

        assert not ends_with_cta(script, PRODUCT_CTAS)

    def test_validation_fails_with_the_named_reason(self) -> None:
        ok, reason = validate_script_completeness(
            f"{BODY} These bulbs have a 25,000-hour lifespan.",
            min_chars=50,
            min_words=10,
            cta_options=PRODUCT_CTAS,
        )

        assert not ok
        assert reason == NO_CTA_REASON

    def test_validation_passes_with_one(self) -> None:
        ok, _ = validate_script_completeness(
            f"{BODY} {PRODUCT_CTAS[0]}",
            min_chars=50,
            min_words=10,
            cta_options=PRODUCT_CTAS,
        )

        assert ok

    def test_no_options_means_no_check(self) -> None:
        """Programmatic construction without config keeps the old contract."""
        ok, _ = validate_script_completeness(BODY, min_chars=50, min_words=10)

        assert ok


@pytest.mark.unit
class TestTheGeneratorAppliesItEverywhere:
    def test_all_four_attempt_paths_validate_through_one_closure(self) -> None:
        """Primary, fallback provider, discovered model: one site skipped is
        the per-site defect this repo has shipped before.
        """
        source = (REPO / "src" / "ai" / "script_generator.py").read_text()
        body = source[source.index("async def generate_script(") :]
        body = body[: body.index("\nasync def ", 10)]

        direct = re.findall(r"validate_script_completeness\(", body)
        wrapped = re.findall(r"_validate\(clean_script\)", body)

        assert len(direct) == 1, "only the closure may call the validator"
        assert len(wrapped) == 4

    @staticmethod
    async def _run(monkeypatch, responses: list[str]) -> tuple[str | None, int]:
        """Drive the real generator with only the LLM call patched."""
        from unittest.mock import AsyncMock

        from src.ai import script_generator
        from src.video.config import load_video_config_modular

        settings = load_video_config_modular().llm_settings
        calls = AsyncMock(side_effect=responses)
        monkeypatch.setattr(script_generator, "_call_llm_api_with_retry", calls)
        monkeypatch.setattr(
            script_generator, "_fetch_and_select_model", AsyncMock(return_value=[])
        )
        script, _ = await script_generator.generate_script(
            _product(),
            settings,
            {settings.api_key_env_var: "k"},
            None,
            {},
            False,
            product_id="B0TEST0001",
        )
        return script, calls.await_count

    @pytest.mark.asyncio
    async def test_a_response_ending_on_a_cta_is_taken_first_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        good = f"{BODY} {PRODUCT_CTAS[1]}"

        script, calls = await self._run(monkeypatch, [good] * 4)

        assert script == good
        assert calls == 1

    @pytest.mark.asyncio
    async def test_a_missing_cta_is_retried_then_appended(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The retry is real; the append is the last resort after it."""
        bare = f"{BODY} These bulbs have a 25,000-hour lifespan."

        script, calls = await self._run(monkeypatch, [bare] * 4)

        assert calls >= 2, "no retry happened"
        assert script is not None
        assert script.endswith(PRODUCT_CTAS[0])
        assert "25,000-hour lifespan." in script

    @pytest.mark.asyncio
    async def test_a_paraphrased_cta_is_replaced_not_doubled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two calls to action back to back read wrong, and the paraphrase
        would become the YouTube first comment.
        """
        from src.publisher.first_comment import extract_closing_line

        para = f"{BODY} The link is in my bio if you want it."

        script, _ = await self._run(monkeypatch, [para] * 4)

        assert script is not None
        assert script.endswith(PRODUCT_CTAS[0])
        assert "in my bio if you want it" not in script
        assert extract_closing_line(script) == "Team magnetic or team plug-in?"


@pytest.mark.unit
class TestTheFirstCommentStillFindsTheBeat:
    """The YouTube first comment is the beat *before* the CTA.

    The extractor strips trailing CTA sentences by marker. A configured CTA
    the markers miss is not stripped, and the comment becomes the CTA.
    """

    def test_every_configured_cta_matches_a_marker(self, shipped_ctas) -> None:
        from src.publisher.first_comment import _CTA_MARKERS

        for kind, options in shipped_ctas.items():
            for cta in options:
                assert any(
                    cta.lower().startswith(m) for m in _CTA_MARKERS
                ), f"{kind} CTA {cta!r} does not open with an extractor marker"

    def test_a_beat_that_contains_a_marker_word_survives(self) -> None:
        """Matched at the start of the sentence, not anywhere in it.

        A substring match popped a two-option beat that merely contained a
        marker phrase, and the sentence before it became the first comment.
        A beat that *opens* with one -- "Save this or skip it?" -- is not a
        shape the templates ask for, and is the one case anchoring cannot
        separate from the CTA.
        """
        from src.publisher.first_comment import extract_closing_line

        beat = "Would you share it with a friend or keep it?"
        script = f"{BODY} {beat} Follow for more finds like this."

        assert extract_closing_line(script) == beat

    @pytest.mark.parametrize("kind", ["product", "topic"])
    def test_the_beat_survives_stripping(self, shipped_ctas, kind: str) -> None:
        from src.publisher.first_comment import extract_closing_line

        for cta in shipped_ctas[kind]:
            closing = extract_closing_line(f"{BODY} {cta}")

            assert closing == "Team magnetic or team plug-in?", (cta, closing)


@pytest.mark.unit
class TestTheShippedConfig:
    def test_both_lists_are_present_and_distinct(self, shipped_ctas) -> None:
        assert len(shipped_ctas["product"]) >= 3
        assert len(shipped_ctas["topic"]) >= 3
        assert not set(shipped_ctas["product"]) & set(shipped_ctas["topic"])

    def test_topic_ctas_imply_nothing_to_buy(self, shipped_ctas) -> None:
        for cta in shipped_ctas["topic"]:
            assert "bio" not in cta.lower()
            assert "want one" not in cta.lower()

    def test_the_model_chooses_by_kind(self, shipped_ctas) -> None:
        from src.video.config.llm_settings import ScriptTemplateConfig

        cfg = ScriptTemplateConfig(
            cta_options=shipped_ctas["product"],
            cta_options_topic=shipped_ctas["topic"],
        )

        assert cfg.cta_options_for(is_topic=False) == shipped_ctas["product"]
        assert cfg.cta_options_for(is_topic=True) == shipped_ctas["topic"]

    def test_no_topic_list_falls_back_to_product(self) -> None:
        from src.video.config.llm_settings import ScriptTemplateConfig

        cfg = ScriptTemplateConfig(cta_options=PRODUCT_CTAS)

        assert cfg.cta_options_for(is_topic=True) == PRODUCT_CTAS

    def test_the_narrator_profiles_no_longer_carry_the_lists(self) -> None:
        """One source. A second copy in prose is the drift that started this.

        The voice examples may still *end* on a CTA -- an example that agrees
        with the rule reinforces it -- so this checks for the list, the
        `Options: "..." / "..."` shape, not for the phrases themselves.
        """
        raw = (REPO / "config" / "ai_services.yaml").read_text()
        profiles = re.findall(
            r"narrator_profile(?:_topic)?: \|-\n(.*?)\n    [a-z_]+:", raw, re.S
        )

        assert len(profiles) == 2
        for text in profiles:
            assert "Options:" not in text
            assert "the list the template gives you" in text
