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
    select_cta,
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
TOPIC_CTAS = [
    "Save this for the next time it happens.",
    "Follow for more fixes like this.",
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

    def test_the_rule_quotes_the_one_chosen_line(self) -> None:
        """One line, not the pool. The rule used to quote all four and leave
        the choice to the model, which took the first every time: five of
        five product scripts on one day and all four on the next day's
        batches closed on the same line. `select_cta` decides instead, and
        one imperative binds better than a menu.
        """
        rule = render_cta_rule(PRODUCT_CTAS[1])

        assert f'"{PRODUCT_CTAS[1]}"' in rule
        assert "very last sentence" in rule
        for other in PRODUCT_CTAS[0], PRODUCT_CTAS[2]:
            assert f'"{other}"' not in rule

    def test_the_topic_tail_does_not_point_at_a_beat_rule(self) -> None:
        """Topic templates have no closing-beat rule above the placeholder;
        the line above is an honest-limit rule, and "the closing beat above"
        would point the model at that.
        """
        assert "closing beat above" in render_cta_rule(PRODUCT_CTAS[0])
        assert "closing beat above" not in render_cta_rule(
            PRODUCT_CTAS[0], is_topic=True
        )
        assert "closing line the template asks for" in render_cta_rule(
            PRODUCT_CTAS[0], is_topic=True
        )

    def test_no_line_renders_nothing(self) -> None:
        assert render_cta_rule("") == ""

    def test_the_prompt_renders_it(self) -> None:
        template = (REPO / "src/ai/prompts/scripts/curiosity_hook.md").read_text()

        prompt = format_prompt(
            template,
            _product(),
            "buyers",
            cta_rule=render_cta_rule(PRODUCT_CTAS[0]),
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

    @pytest.mark.parametrize(
        "sentence",
        [
            "So, link in bio if you want one.",
            "That's my take, and link in bio if you want one.",
            "Anyway, follow for more finds like this.",
        ],
    )
    def test_a_cta_with_words_in_front_is_not_a_cta(self, sentence: str) -> None:
        """Sentence-level, not word-level.

        A tail match accepted these, and the extractor -- which strips a CTA
        only when the sentence opens with it -- then made this sentence the
        YouTube first comment, where the substring match on main had
        stripped it. Every doc states the final-sentence rule; the code has
        to enforce it.
        """
        assert not ends_with_cta(f"{BODY} {sentence}", PRODUCT_CTAS)

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
class TestBothEntryPointsCarryTheOverride:
    """The Module/Batch Alignment Rule, as a test.

    The producer CLI and `global_batch` re-implement the same argument
    surface, so a flag added to one and not the other is silently absent on
    the path `make batch-lowpri` runs.
    """

    def test_the_producer_carries_the_flag_into_its_overrides(self) -> None:
        """The first version of these tests asserted the *strings*
        `"--cta"`, `script_templates.fixed_cta` and `overrides["cta"]` were
        present in the two files. All three were, and the flag did nothing:
        `_build_cli_overrides` never put `cta` in the dict, so the apply
        branch reading it could not fire. A test that greps for a symbol
        passes on a flag parsed and never read, which is the shape it was
        written to catch.
        """
        from src.video.producer.cli import (
            _build_cli_overrides,
            create_argument_parser,
        )

        args = create_argument_parser().parse_args(
            ["outputs/B0X/data.json", "slideshow_images1", "--cta", PRODUCT_CTAS[2]]
        )

        assert _build_cli_overrides(args)["cta"] == PRODUCT_CTAS[2]

    def test_the_batch_carries_the_flag_the_whole_way(self) -> None:
        """The batch chain is parser, then `load_global_batch_config`, then
        `GlobalBatchConfig.cta`, then its own override dict. Pinning only the
        two ends leaves the middle free to drop the key, which is where the
        inert flag lived.
        """
        from src.pipeline import global_batch
        from src.pipeline.config import load_global_batch_config

        args = global_batch.create_argument_parser().parse_args(
            [
                "--product-ids",
                "B0X",
                "--profile",
                "slideshow_images1",
                "--cta",
                PRODUCT_CTAS[1],
            ]
        )
        cfg = load_global_batch_config(cli_args=args)
        assert cfg.cta == PRODUCT_CTAS[1]

        orch = global_batch.GlobalPipelineOrchestrator.__new__(
            global_batch.GlobalPipelineOrchestrator
        )
        orch.config = cfg
        overrides = orch._build_cli_overrides()
        assert overrides is not None
        assert overrides["cta"] == PRODUCT_CTAS[1]

    def test_the_render_path_calls_the_apply(self) -> None:
        """A helper nothing calls is the same inert flag one layer down, and
        no behavioural test of the helper can see it. Read the call site, the
        way the publish hooks are pinned.
        """
        import ast
        import inspect

        from src.video.producer import orchestration

        tree = ast.parse(inspect.getsource(orchestration.create_video_for_product))
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }

        assert "apply_script_template_overrides" in called

    def test_the_apply_changes_what_gets_selected(self) -> None:
        """The end of the chain, driving the real apply and asserting what it
        selects. A grep for `fixed_cta` in the file passes on an apply that
        writes a neighbouring attribute, and a test that sets `fixed_cta`
        itself never runs the apply at all -- both of which is how this
        shipped parsed, forwarded and inert.
        """
        from src.ai.script_generator import select_cta
        from src.video.config import load_video_config_modular
        from src.video.producer.orchestration import (
            apply_script_template_overrides,
        )

        config = load_video_config_modular()
        st = config.llm_settings.script_templates
        pool = st.cta_options_for(False)
        unforced = select_cta(pool, "B0AAAAAAAA", st.fixed_cta)
        forced = next(c for c in pool if c != unforced)

        apply_script_template_overrides(config, {"cta": forced})

        assert st.fixed_cta == forced
        assert select_cta(pool, "B0AAAAAAAA", st.fixed_cta) == forced

    def test_the_apply_leaves_the_sibling_override_alone(self) -> None:
        """The two overrides share a settings object, so writing the wrong
        attribute is the mistake this helper exists to make visible.
        """
        from src.video.config import load_video_config_modular
        from src.video.producer.orchestration import (
            apply_script_template_overrides,
        )

        config = load_video_config_modular()
        st = config.llm_settings.script_templates
        before = st.fixed_template

        apply_script_template_overrides(config, {"cta": PRODUCT_CTAS[1]})

        assert st.fixed_template == before
        assert st.fixed_cta == PRODUCT_CTAS[1]

    def test_the_chosen_line_is_the_one_rendered(self) -> None:
        """End of the chain: an override that reaches `fixed_cta` has to come
        out in the prompt the model is given.
        """
        from src.ai.script_generator import select_cta

        chosen = select_cta(PRODUCT_CTAS, "B0TEST0001", fixed_cta=PRODUCT_CTAS[3])
        rule = render_cta_rule(chosen)

        assert f'"{PRODUCT_CTAS[3]}"' in rule


@pytest.mark.unit
class TestTheLineIsChosenPerProduct:
    """The pool always yielded its first entry.

    `render_cta_rule` quoted all four options and left the choice to the
    model, which took the first every time: five of five product scripts on
    one day, and all four on the next day's batches, closed on `Link in bio
    if you want one.` A pool that always yields its first entry is one CTA
    and three unused strings, and every render sharing a closing line is the
    templated-sameness signal the platforms throttle on.
    """

    def test_the_same_product_gets_the_same_line(self) -> None:
        first = select_cta(PRODUCT_CTAS, "B0AAAAAAAA")
        assert all(select_cta(PRODUCT_CTAS, "B0AAAAAAAA") == first for _ in range(5))

    def test_a_batch_does_not_land_on_one_line(self) -> None:
        """The defect, stated as a test. Not a distribution claim -- just
        that selection reads the product id at all.
        """
        ids = [f"B0TEST{n:04d}" for n in range(40)]
        chosen = {select_cta(PRODUCT_CTAS, i) for i in ids}
        assert len(chosen) > 1
        assert chosen <= set(PRODUCT_CTAS)

    def test_topic_and_product_pools_are_separate(self) -> None:
        assert select_cta(TOPIC_CTAS, "topic-x") in TOPIC_CTAS

    def test_no_product_id_takes_the_first(self) -> None:
        """A caller with nothing to hash still needs a line, and the first is
        the one the pool has always produced.
        """
        assert select_cta(PRODUCT_CTAS) == PRODUCT_CTAS[0]

    def test_an_empty_pool_yields_nothing(self) -> None:
        assert select_cta([], "B0AAAAAAAA") == ""

    def test_the_override_wins(self) -> None:
        assert (
            select_cta(PRODUCT_CTAS, "B0AAAAAAAA", fixed_cta=PRODUCT_CTAS[3])
            == PRODUCT_CTAS[3]
        )

    def test_an_override_outside_the_pool_falls_through(self) -> None:
        """Rendering a rule the validator will then refuse costs the render a
        retry loop and ends on an appended line the rule never asked for.
        """
        chosen = select_cta(PRODUCT_CTAS, "B0AAAAAAAA", fixed_cta="Buy it now.")
        assert chosen == select_cta(PRODUCT_CTAS, "B0AAAAAAAA")


def _chosen_cta(product_id: str = "B0TEST0001") -> str:
    """The line `select_cta` picks for the product the generator tests use.

    Derived rather than written down, so the test says "the line the rule
    asked for" instead of pinning today's hash output.
    """
    from src.ai.script_generator import select_cta

    return select_cta(PRODUCT_CTAS, product_id)


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
        script, _, _ = await script_generator.generate_script(
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
        # The line the rule asked for, not the pool's first entry. Appending
        # a different one would make the recorded choice a lie about what
        # shipped, and would put every fallback render back on one CTA.
        assert script.endswith(_chosen_cta())
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
        assert script.endswith(_chosen_cta())
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
