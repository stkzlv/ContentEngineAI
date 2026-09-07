"""The fact check catches wrong values, and cannot lose a render (#380).

The 0.103.1 prompt fix removed the demand that made topic scripts invent
navigation paths. What a prompt rule cannot reach is a value that is simply
wrong: "go to Devices, then Power & battery" names its platform, satisfies
every rule in the template, and is still the wrong page.

Measured precision is 14 flags right in 16, so roughly one flag in eight is
wrong. That number is why the guards below exist: a false positive must be
able to damage the sentence it named and nothing else, and must never be able
to cost the render. Every test here is offline.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ai.script_fact_check import (
    FactCheckClaim,
    accept_revision,
    check_script,
    fact_check_and_revise,
    parse_check_answer,
    sentences,
)
from src.video.config import config
from src.video.config.llm_settings import ScriptFactCheckConfig

PROMPTS = Path(__file__).resolve().parents[2] / "src" / "ai" / "prompts"

CTA = "Save this for the next time it happens."
GOOD = (
    "Your laptop charges slower on one port because the ports differ. "
    "On Windows, go to Devices, then select Power and battery. "
    "Look for the lightning bolt symbol next to the port. "
    "If no port has one, the laptop may not support fast charging at all. "
    f"{CTA}"
)
FLAG = [
    FactCheckClaim(
        claim="On Windows, go to Devices, then select Power and battery.",
        reason="That page sits under System, not Devices.",
        fix="On Windows 11 it is Settings, then System, then Power & battery.",
    )
]
GUARDS = {
    "cta_options": [CTA],
    "min_chars": 50,
    "min_words": 10,
    "max_length_drift": 0.25,
}


def swap(script: str, old: str, new: str) -> str:
    assert old in script
    return script.replace(old, new)


@pytest.mark.unit
class TestParsingTheAnswer:
    def test_a_labelled_flag(self) -> None:
        r = parse_check_answer(
            "VERDICT: FLAGGED\nCLAIM: The sky is green.\n"
            "REASON: It is blue.\nFIX: The sky is blue."
        )
        assert r.ran and len(r.flagged) == 1
        assert r.flagged[0].claim == "The sky is green."
        assert r.flagged[0].fix == "The sky is blue."

    def test_several_flags_split_on_the_rule(self) -> None:
        r = parse_check_answer(
            "VERDICT: FLAGGED\nCLAIM: One.\nREASON: a\nFIX: b\n---\n"
            "CLAIM: Two.\nREASON: c\nFIX: d"
        )
        assert [c.claim for c in r.flagged] == ["One.", "Two."]

    def test_a_clean_verdict_flags_nothing(self) -> None:
        r = parse_check_answer("VERDICT: OK")
        assert r.ran and r.flagged == [] and r.error is None

    def test_a_preamble_before_the_labels_is_ignored(self) -> None:
        r = parse_check_answer(
            "Here is my assessment.\n\nVERDICT: FLAGGED\n"
            "CLAIM: X is wrong.\nREASON: y\nFIX: z"
        )
        assert len(r.flagged) == 1

    def test_json_parses_too(self) -> None:
        """Forward compatibility: a tier that supports schema plus grounding
        becomes a config edit, not a code change.
        """
        r = parse_check_answer(
            '{"claims": [{"claim": "A.", "verdict": "wrong", "reason": "r", '
            '"fix": "f"}, {"claim": "B.", "verdict": "correct"}]}'
        )
        assert [c.claim for c in r.flagged] == ["A."]

    def test_a_claim_without_a_fix_is_dropped(self) -> None:
        """The revision is built from the fix; without one the model would be
        asked to invent the correction, which is the defect being caught.
        """
        r = parse_check_answer("VERDICT: FLAGGED\nCLAIM: X.\nREASON: y")
        assert r.flagged == []

    def test_a_claim_ruled_correct_is_dropped(self) -> None:
        """Measured live, twice. Asked for wrong claims only, the checker
        listed a sentence it agreed was right. That costs a repair slot and
        pushes the genuinely wrong sentences past the length guard, so the
        whole revision is thrown away and the wrong script ships. The per-claim
        ruling is what makes "list only what you rule wrong" enforceable.
        """
        r = parse_check_answer(
            "VERDICT: FLAGGED\n"
            "CLAIM: Bad one.\nRULING: wrong\nREASON: r\nFIX: The right fact.\n"
            "---\n"
            "CLAIM: Good one.\nRULING: correct\nREASON: r\nFIX: Leave it.\n"
        )
        assert [c.claim for c in r.flagged] == ["Bad one."]

    def test_a_block_with_no_ruling_is_still_kept(self) -> None:
        """It was listed under a FLAGGED verdict, so a missing line is a
        formatting slip, not a ruling. Dropping it would lose a real flag.
        """
        r = parse_check_answer(
            "VERDICT: FLAGGED\nCLAIM: Bad one.\nREASON: r\nFIX: The right fact."
        )
        assert len(r.flagged) == 1

    @pytest.mark.parametrize(
        "fix",
        [
            "Correct.",
            "This claim is correct.",
            "The statement as written is accurate.",
            "No change needed.",
            "correct",
        ],
    )
    def test_a_fix_that_only_says_the_claim_was_right_is_dropped(self, fix) -> None:
        """The second net under the ruling: the same non-correction arrived as
        `Correct.` and then as `This claim is correct.`, so matching the first
        word alone caught one of the two.
        """
        r = parse_check_answer(
            f"VERDICT: FLAGGED\nCLAIM: Some sentence.\nREASON: r\nFIX: {fix}"
        )
        assert r.flagged == []

    @pytest.mark.parametrize(
        "fix",
        [
            "The correct path is Settings, then System.",
            "It is accurate only on Windows 11, not on Windows 10.",
            "None of the ports carry more than 15 watts.",
        ],
    )
    def test_a_real_fix_mentioning_correctness_is_kept(self, fix) -> None:
        """The net must not eat a fix that happens to use one of its words."""
        r = parse_check_answer(
            f"VERDICT: FLAGGED\nCLAIM: Some sentence.\nREASON: r\nFIX: {fix}"
        )
        assert len(r.flagged) == 1

    @pytest.mark.parametrize("text", ["", None, "   ", "I could not check this."])
    def test_nothing_usable_is_not_an_error_worth_failing_on(self, text) -> None:
        r = parse_check_answer(text)
        assert r.ran is True and r.flagged == [] and r.error


@pytest.mark.unit
class TestAcceptingARevision:
    def test_a_contained_fix_is_accepted(self) -> None:
        revised = swap(
            GOOD,
            "go to Devices, then select Power and battery",
            "go to System, then select Power and battery",
        )
        accepted, reason = accept_revision(GOOD, revised, FLAG, **GUARDS)
        assert accepted is not None and reason == "accepted"
        assert "System" in accepted and accepted.endswith(CTA)

    def test_the_successor_sentence_may_move_too(self) -> None:
        """A fix often has to carry into the step that referenced the wrong
        thing, so the flagged sentence plus its successor are touchable.
        """
        revised = swap(
            GOOD,
            "go to Devices, then select Power and battery. "
            "Look for the lightning bolt symbol next to the port.",
            "go to System, then Power and battery. "
            "The wattage per port is listed there.",
        )
        accepted, _ = accept_revision(GOOD, revised, FLAG, **GUARDS)
        assert accepted is not None

    def test_a_rewritten_unflagged_sentence_is_refused(self) -> None:
        """The containment guard. One flag in eight is wrong, so a bad
        correction must not be able to reach the rest of the script.
        """
        revised = swap(
            GOOD,
            "Your laptop charges slower on one port because " "the ports differ.",
            "Laptops are complicated machines with many ports.",
        )
        accepted, reason = accept_revision(GOOD, revised, FLAG, **GUARDS)
        assert accepted is None and "not asked to" in reason

    def test_losing_the_cta_is_refused(self) -> None:
        revised = swap(GOOD, CTA, "Thanks for watching.")
        accepted, reason = accept_revision(GOOD, revised, FLAG, **GUARDS)
        assert accepted is None and "validation" in reason

    def test_too_short_is_refused(self) -> None:
        accepted, reason = accept_revision(
            GOOD,
            f"Short. {CTA}",
            FLAG,
            **{**GUARDS, "min_chars": 500, "min_words": 100},
        )
        assert accepted is None and "validation" in reason

    def test_drift_is_refused(self) -> None:
        revised = swap(
            GOOD,
            "the ports differ",
            "the ports differ " + "in many interesting ways " * 12,
        )
        accepted, reason = accept_revision(GOOD, revised, FLAG, **GUARDS)
        assert accepted is None and "length" in reason

    def test_an_inserted_sentence_is_refused(self) -> None:
        """The other half of containment. Keeping every unflagged sentence
        still lets the model add ones nobody checked, in a module whose whole
        purpose is that claims get checked. The assertion below is the point:
        this padding is inside the length allowance, so drift does not catch
        it.
        """
        padded = swap(
            GOOD,
            "Look for the lightning bolt symbol next to the port.",
            "Look for the lightning bolt symbol next to the port. "
            "Cables differ too. "
            "Try another one.",
        )
        assert abs(len(padded.split()) - len(GOOD.split())) / len(GOOD.split()) < 0.25
        accepted, reason = accept_revision(GOOD, padded, FLAG, **GUARDS)
        assert accepted is None and "beyond the repair" in reason

    def test_splitting_a_flagged_sentence_in_two_is_allowed(self) -> None:
        """A correction often needs two sentences where there was one, so the
        allowance is one extra per claim rather than none.
        """
        split = swap(
            GOOD,
            "On Windows, go to Devices, then select Power and battery.",
            "On Windows, open Settings, then System. "
            "Power and battery is the page you want.",
        )
        accepted, reason = accept_revision(GOOD, split, FLAG, **GUARDS)
        assert accepted is not None, reason

    def test_a_claim_split_by_inner_punctuation_still_matches(self) -> None:
        """`sentences()` used to split on any sentence-final punctuation
        followed by a space, so an ellipsis or an abbreviation cut one spoken
        sentence into two entries -- while the checker copies the whole
        sentence, as the prompt demands. Matching on equality found neither
        fragment, so nothing was touchable and a correct repair was refused
        for altering sentences it never touched, with the record blaming the
        reviser. The splitter now requires something that starts a sentence
        after the space; `_covers` absorbs what is left.
        """
        original = (
            "Your fan gets loud for one reason. "
            "Open Device Manager, expand Components... then read the Power page. "
            "Blow the dust out from the outside. "
            f"{CTA}"
        )
        flagged = [
            FactCheckClaim(
                claim=(
                    "Open Device Manager, expand Components... "
                    "then read the Power page."
                ),
                reason="No Power page exists there.",
                fix="Device Manager has no wattage page.",
            )
        ]
        revised = original.replace(
            "Open Device Manager, expand Components... then read the Power page.",
            "Device Manager will not show you the wattage.",
        )
        accepted, reason = accept_revision(original, revised, flagged, **GUARDS)
        assert accepted is not None, reason

    def test_a_claim_quoting_the_whole_script_is_refused(self) -> None:
        """`_covers` matches a claim against a sentence in either direction,
        so a claim that quotes several sentences makes every one of them
        touchable -- and a claim quoting the whole script leaves nothing
        protected at all, which is the guard switched off by one answer. The
        prompt asks for one sentence; a span this wide means the answer is
        unusable, not that the whole script was flagged.
        """
        whole = [FactCheckClaim(claim=GOOD, reason="r", fix="f")]
        rewrite = (
            "Nothing here resembles what was written before at all. "
            "Every port on every laptop delivers the same power. "
            "Buy a different charger and stop worrying about it. "
            f"{CTA}"
        )
        accepted, reason = accept_revision(GOOD, rewrite, whole, **GUARDS)
        assert accepted is None and "spans" in reason

    def test_a_claim_spanning_two_sentences_is_still_allowed(self) -> None:
        """The bound has to leave room for the mismatch `_covers` exists for,
        or it re-opens the lost-repair defect it was added to close.
        """
        pair = [
            FactCheckClaim(
                claim=(
                    "On Windows, go to Devices, then select Power and battery. "
                    "Look for the lightning bolt symbol next to the port."
                ),
                reason="Neither step is right.",
                fix="It is Settings, then System, then Power & battery.",
            )
        ]
        revised = swap(
            GOOD,
            "On Windows, go to Devices, then select Power and battery. "
            "Look for the lightning bolt symbol next to the port.",
            "On Windows, open Settings, then System, then Power and battery. "
            "The wattage per port is listed there.",
        )
        accepted, reason = accept_revision(GOOD, revised, pair, **GUARDS)
        assert accepted is not None, reason

    def test_a_repeated_sentence_does_not_read_as_re_ordered(self) -> None:
        """`list.index` returns the first occurrence, so a script that says
        the same line twice -- or twice after case and punctuation are
        normalised away -- read as re-ordered when nothing had moved, throwing
        away a correct repair and blaming the reviser in the record.
        """
        original = (
            "That is it. "
            "Open the settings page now. "
            "That is it. "
            "Check the box marked fast charging on the second tab. "
            f"{CTA}"
        )
        flagged = [
            FactCheckClaim(claim="Open the settings page now.", reason="r", fix="f")
        ]
        revised = original.replace(
            "Open the settings page now.", "Open the power page instead."
        )
        accepted, reason = accept_revision(original, revised, flagged, **GUARDS)
        assert accepted is not None, reason

    def test_a_pure_reordering_is_refused(self) -> None:
        """Membership alone let this through: every sentence unchanged,
        nothing added, the length identical, and the steps of a how-to
        swapped. The revise prompt forbids re-ordering; this module's own
        position is that a prompt rule is not a guard.
        """
        old = sentences(GOOD)
        swapped = " ".join([old[0], old[2], old[1], old[3], old[4]])
        accepted, reason = accept_revision(GOOD, swapped, [], **GUARDS)
        assert accepted is None and "re-ordered" in reason

    def test_a_no_op_revision_is_refused(self) -> None:
        accepted, reason = accept_revision(GOOD, GOOD, FLAG, **GUARDS)
        assert accepted is None and "changed nothing" in reason

    @pytest.mark.parametrize("revised", [None, "", "   "])
    def test_nothing_back_is_refused(self, revised) -> None:
        accepted, reason = accept_revision(GOOD, revised, FLAG, **GUARDS)
        assert accepted is None and reason

    def test_every_refusal_carries_a_reason(self) -> None:
        for revised in [None, GOOD, swap(GOOD, CTA, "Bye.")]:
            accepted, reason = accept_revision(GOOD, revised, FLAG, **GUARDS)
            assert accepted is None and reason.strip()


def fake_client(text: str = "VERDICT: OK") -> MagicMock:
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=MagicMock(text=text))
    client.aio.aclose = AsyncMock()
    return client


def checker_settings(**over) -> ScriptFactCheckConfig:
    return ScriptFactCheckConfig(**{"enabled": True, **over})


@pytest.mark.unit
class TestTheCheckerCall:
    @pytest.mark.asyncio
    async def test_the_search_tool_is_actually_passed(self) -> None:
        """Without the tool this is an ungrounded model asserting from memory,
        which is the failure mode being caught rather than a check of it.
        """
        client = fake_client()
        with patch("google.genai.Client", return_value=client):
            await check_script(
                GOOD, "usb charging", api_key="k", settings=checker_settings()
            )
        cfg = client.aio.models.generate_content.await_args.kwargs["config"]
        assert cfg.tools and cfg.tools[0].google_search is not None

    @pytest.mark.asyncio
    async def test_the_script_and_subject_reach_the_prompt(self) -> None:
        client = fake_client()
        with patch("google.genai.Client", return_value=client):
            await check_script(
                GOOD, "usb charging", api_key="k", settings=checker_settings()
            )
        prompt = client.aio.models.generate_content.await_args.kwargs["contents"]
        assert GOOD in prompt and "usb charging" in prompt

    @pytest.mark.asyncio
    async def test_the_configured_model_is_used(self) -> None:
        client = fake_client()
        with patch("google.genai.Client", return_value=client):
            await check_script(
                GOOD, "s", api_key="k", settings=checker_settings(model="gemini-3-pro")
            )
        assert (
            client.aio.models.generate_content.await_args.kwargs["model"]
            == "gemini-3-pro"
        )

    @pytest.mark.asyncio
    async def test_the_client_is_closed(self) -> None:
        client = fake_client()
        with patch("google.genai.Client", return_value=client):
            await check_script(GOOD, "s", api_key="k", settings=checker_settings())
        client.aio.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_the_client_is_closed_even_when_the_call_fails(self) -> None:
        client = fake_client()
        client.aio.models.generate_content = AsyncMock(side_effect=OSError("down"))
        with patch("google.genai.Client", return_value=client):
            r = await check_script(GOOD, "s", api_key="k", settings=checker_settings())
        assert r.ran is False
        client.aio.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "boom",
        [
            OSError("network"),
            ValueError("bad"),
            RuntimeError("loop"),
            TimeoutError("slow"),
        ],
    )
    async def test_a_failing_call_is_not_an_exception(self, boom) -> None:
        client = fake_client()
        client.aio.models.generate_content = AsyncMock(side_effect=boom)
        with patch("google.genai.Client", return_value=client):
            r = await check_script(GOOD, "s", api_key="k", settings=checker_settings())
        assert r.ran is False and r.flagged == []

    @pytest.mark.asyncio
    async def test_a_missing_library_is_not_an_exception(self) -> None:
        with patch("google.genai.Client", side_effect=ImportError("no genai")):
            r = await check_script(GOOD, "s", api_key="k", settings=checker_settings())
        assert r.ran is False

    @pytest.mark.asyncio
    async def test_a_timeout_is_bounded_by_the_configured_seconds(self) -> None:
        """A grounded call has a search round trip inside it, so it can hang
        past any generation timeout the pipeline is used to.
        """
        import asyncio

        async def never(*_a, **_k):
            await asyncio.sleep(30)

        client = fake_client()
        client.aio.models.generate_content = never
        with patch("google.genai.Client", return_value=client):
            r = await asyncio.wait_for(
                check_script(
                    GOOD, "s", api_key="k", settings=checker_settings(timeout_seconds=1)
                ),
                timeout=5,
            )
        assert r.ran is False


def pipeline_settings(**over):
    """The real LLM settings, with the check switched on and thresholds the
    sample script clears.
    """
    base = config.llm_settings
    return base.model_copy(
        update={
            "script_fact_check": checker_settings(**over),
            "script_validation": base.script_validation.model_copy(
                update={"min_chars": 50, "min_words": 10}
            ),
        }
    )


async def run(script=GOOD, secrets=None, **over):
    return await fact_check_and_revise(
        script,
        "usb charging",
        MagicMock(topic="Why one port charges faster", product_id="t"),
        pipeline_settings(**over),
        {"GEMINI_API_KEY": "k"} if secrets is None else secrets,
        MagicMock(),
    )


FLAGGED_ANSWER = (
    "VERDICT: FLAGGED\n"
    "CLAIM: On Windows, go to Devices, then select Power and battery.\n"
    "REASON: That page is under System.\n"
    "FIX: It is Settings, then System, then Power & battery."
)
FIXED = GOOD.replace("go to Devices, then select", "go to System, then select")


@pytest.mark.unit
class TestNothingCanLoseTheRender:
    """The governing rule. A checker is a net under the generator, and a net
    that can drop the render is worse than no net: the fabrication it prevents
    costs one wrong sentence, and a lost render costs the whole video and
    every paid step before it. `FactCheckOutcome.script` is non-optional so
    the type cannot express the bad outcome; these are the paths to it.
    """

    @pytest.mark.asyncio
    async def test_disabled_ships_the_original_and_calls_nothing(self) -> None:
        with patch("google.genai.Client") as client:
            out = await run(enabled=False)
        assert out.script == GOOD and client.call_count == 0
        assert out.record["revision"]["reason"] == "disabled"

    @pytest.mark.asyncio
    async def test_no_api_key_ships_the_original(self) -> None:
        with patch("google.genai.Client") as client:
            out = await run(secrets={})
        assert out.script == GOOD and client.call_count == 0
        assert out.record["revision"]["reason"] == "no api key"

    @pytest.mark.asyncio
    async def test_a_checker_that_raises_outright_ships_the_original(self) -> None:
        """Not the handled failures -- this is the one nobody predicted."""
        with patch(
            "src.ai.script_fact_check.check_script",
            AsyncMock(side_effect=RuntimeError("boom")),
        ):
            out = await run()
        assert out.script == GOOD
        assert "checker raised" in out.record["revision"]["reason"]

    @pytest.mark.asyncio
    async def test_a_dead_checker_ships_the_original_and_records_it(self) -> None:
        with patch("google.genai.Client", side_effect=ImportError("gone")):
            out = await run()
        assert out.script == GOOD
        assert out.record["ran"] is False
        assert out.record["revision"]["reason"] == "checker unavailable"

    @pytest.mark.asyncio
    async def test_a_clean_verdict_is_recorded_as_having_run(self) -> None:
        """Ran-and-found-nothing must be distinguishable from never-ran,
        or the record cannot be read back to re-measure precision.
        """
        with patch("google.genai.Client", return_value=fake_client("VERDICT: OK")):
            out = await run()
        assert out.script == GOOD and out.record["ran"] is True
        assert out.record["revision"]["reason"] == "nothing flagged"

    @pytest.mark.asyncio
    async def test_a_reviser_returning_nothing_ships_the_original(self) -> None:
        with (
            patch("google.genai.Client", return_value=fake_client(FLAGGED_ANSWER)),
            patch(
                "src.ai.platform_metadata.utilities.generate_with_llm",
                AsyncMock(return_value=None),
            ),
        ):
            out = await run()
        assert out.script == GOOD
        assert out.record["revision"]["attempted"] is True
        assert out.record["revision"]["accepted"] is False

    @pytest.mark.asyncio
    async def test_a_reviser_that_raises_ships_the_original(self) -> None:
        with (
            patch("google.genai.Client", return_value=fake_client(FLAGGED_ANSWER)),
            patch(
                "src.ai.platform_metadata.utilities.generate_with_llm",
                AsyncMock(side_effect=RuntimeError("boom")),
            ),
        ):
            out = await run()
        assert out.script == GOOD
        assert "reviser raised" in out.record["revision"]["reason"]

    @pytest.mark.asyncio
    async def test_a_revision_failing_the_guards_ships_the_original(self) -> None:
        wrecked = GOOD.replace(CTA, "Thanks for watching.")
        with (
            patch("google.genai.Client", return_value=fake_client(FLAGGED_ANSWER)),
            patch(
                "src.ai.platform_metadata.utilities.generate_with_llm",
                AsyncMock(return_value=wrecked),
            ),
        ):
            out = await run()
        assert out.script == GOOD and out.record["revision"]["accepted"] is False

    @pytest.mark.asyncio
    async def test_a_good_revision_ships_and_is_recorded(self) -> None:
        with (
            patch("google.genai.Client", return_value=fake_client(FLAGGED_ANSWER)),
            patch(
                "src.ai.platform_metadata.utilities.generate_with_llm",
                AsyncMock(return_value=FIXED),
            ),
        ):
            out = await run()
        assert out.script == FIXED
        assert out.record["revision"]["accepted"] is True
        assert out.record["flagged"][0]["fix"].startswith("It is Settings")

    @pytest.mark.asyncio
    async def test_the_grounded_call_happens_once_per_script(self) -> None:
        """One grounded query is roughly $0.035, and grounded queries bill
        separately from tokens, so the query count is the cost. There is no
        round count to configure: the module contains no loop, which is what
        makes this assertion the guarantee.
        """
        client = fake_client(FLAGGED_ANSWER)
        with (
            patch("google.genai.Client", return_value=client),
            patch(
                "src.ai.platform_metadata.utilities.generate_with_llm",
                AsyncMock(return_value=FIXED),
            ),
        ):
            await run()
        assert client.aio.models.generate_content.await_count == 1

    @pytest.mark.asyncio
    async def test_the_reviser_never_enters_the_grounded_path(self) -> None:
        """The reviser is handed the correct fact; a second search would pay
        the grounded rate to answer a question already answered.
        """
        client = fake_client(FLAGGED_ANSWER)
        gen = AsyncMock(return_value=FIXED)
        with (
            patch("google.genai.Client", return_value=client),
            patch("src.ai.platform_metadata.utilities.generate_with_llm", gen),
        ):
            await run()
        assert gen.await_count == 1
        assert client.aio.models.generate_content.await_count == 1

    @pytest.mark.asyncio
    async def test_the_reviser_can_reach_the_fallback_provider(self) -> None:
        """The fallback provider looks its own key up in `secrets`, not in the
        `api_key` it is handed. Passing only the key made a rate-limited
        primary lose the repair and ship the wrong claim while a funded
        fallback sat unused, logging that the key was not found.
        """
        gen = AsyncMock(return_value=FIXED)
        with (
            patch("google.genai.Client", return_value=fake_client(FLAGGED_ANSWER)),
            patch("src.ai.platform_metadata.utilities.generate_with_llm", gen),
        ):
            await run()
        assert gen.await_args is not None
        assert gen.await_args.kwargs["secrets"] == {"GEMINI_API_KEY": "k"}

    @pytest.mark.asyncio
    async def test_no_more_flags_are_revised_than_configured(self) -> None:
        many = "VERDICT: FLAGGED\n" + "\n---\n".join(
            f"CLAIM: c{i}.\nREASON: r\nFIX: f" for i in range(6)
        )
        gen = AsyncMock(return_value=None)
        with (
            patch("google.genai.Client", return_value=fake_client(many)),
            patch("src.ai.platform_metadata.utilities.generate_with_llm", gen),
        ):
            out = await run(max_flags_to_revise=2)
        assert len(out.record["flagged"]) == 6
        assert gen.await_args is not None
        listing = gen.await_args.kwargs["extra_placeholders"]["FLAGGED_CLAIMS"]
        assert listing.count("SENTENCE:") == 2


@pytest.mark.unit
class TestConfigAndPrompts:
    def test_the_code_default_is_off(self) -> None:
        """A fork that ships no `pycaps:`-style block must not start paying a
        grounded rate it never asked for.
        """
        assert ScriptFactCheckConfig().enabled is False

    def test_the_bundled_config_turns_it_on(self) -> None:
        assert config.llm_settings.script_fact_check.enabled is True

    def test_there_is_no_round_count_to_configure(self) -> None:
        """A `max_rounds` field lived here briefly. It could not raise the
        query count above what the code does, so it promised a second round
        that never happened while duplicating `enabled` at zero. The cap is
        structural, asserted by the call-count test above.
        """
        assert not hasattr(ScriptFactCheckConfig(), "max_rounds")

    def test_the_checker_prompt_demands_the_sentence_verbatim(self) -> None:
        """The containment guard matches the flagged sentence against the
        script, so a paraphrase leaves the guard nothing to anchor on.
        """
        text = (PROMPTS / "script_fact_check.md").read_text(encoding="utf-8")
        assert "EXACTLY" in text and "{SCRIPT}" in text and "{SUBJECT}" in text

    def test_the_checker_prompt_excludes_a_claim_being_argued_against(self) -> None:
        """Measured false positive: the checker extracted the myth a script was
        debunking and reported the script as asserting it.
        """
        text = (PROMPTS / "script_fact_check.md").read_text(encoding="utf-8")
        assert "arguing against" in text

    def test_the_revise_prompt_confines_the_rewrite(self) -> None:
        text = (PROMPTS / "script_revise.md").read_text(encoding="utf-8")
        assert "ONLY" in text
        assert "{FLAGGED_CLAIMS}" in text and "{VIDEO_SCRIPT}" in text

    def test_the_revise_prompt_protects_the_final_call_to_action(self) -> None:
        """Validation refuses a script whose last sentence is not a configured
        call to action verbatim, so a helpfully reworded one loses the render's
        revision and, before this rule, was the commonest rejection.
        """
        text = (PROMPTS / "script_revise.md").read_text(encoding="utf-8")
        assert "final sentence" in text.lower()
