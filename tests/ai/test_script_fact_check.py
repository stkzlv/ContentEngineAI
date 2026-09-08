"""The fact check catches wrong values, and cannot lose a render (#380).

The 0.103.1 prompt fix removed the demand that made topic scripts invent
navigation paths. What a prompt rule cannot reach is a value that is simply
wrong: "go to Devices, then Power & battery" names its platform, satisfies
every rule in the template, and is still the wrong page.

Measured precision is 14 flags right in 16, so roughly one flag in eight is
wrong. That number is why the guards below exist: a false positive must be
able to damage the sentences it named and the one following each, and nothing
beyond that, and must never be able to cost the render. Every test here is offline.
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

    def test_a_repeated_claim_is_listed_once(self) -> None:
        """Observed on a real topic render: four claims that were two, each
        listed twice, so one of the three repair slots went to a sentence
        already named. The repair was good there, but on a script with three
        genuinely wrong claims a duplicate pushes a real one out of the
        window with nothing to show it happened.
        """
        r = parse_check_answer(
            "VERDICT: FLAGGED\n"
            "CLAIM: Blast air into the vents for ten seconds.\n"
            "RULING: wrong\nREASON: r\nFIX: Use short bursts.\n---\n"
            "CLAIM: This should clear most of the dust.\n"
            "RULING: wrong\nREASON: r\nFIX: It clears the loose dust only.\n---\n"
            "CLAIM: Blast air into the vents for ten seconds.\n"
            "RULING: wrong\nREASON: r\nFIX: Use short bursts.\n---\n"
            "CLAIM: This should clear most of the dust.\n"
            "RULING: wrong\nREASON: r\nFIX: It clears the loose dust only.\n"
        )
        assert len(r.flagged) == 2
        assert r.flagged[0].claim.startswith("Blast air")
        assert r.flagged[1].claim.startswith("This should")

    def test_two_errors_in_one_sentence_keep_both_fixes(self) -> None:
        """The claim alone reads as the obvious key -- it is what containment
        matches on -- but the prompt asks for one block per *claim*, and the
        checker does decompose a sentence: the one real answer available
        rules a single sentence in three parts. Keying on the claim alone
        drops the second correction before the reviser sees it.
        """
        r = parse_check_answer(
            "VERDICT: FLAGGED\n"
            "CLAIM: Open Settings then Power.\nRULING: wrong\nREASON: path\n"
            "FIX: It is System then Power.\n---\n"
            "CLAIM: Open Settings then Power.\nRULING: wrong\nREASON: timeout\n"
            "FIX: The timeout is 15 minutes.\n"
        )
        assert [c.fix for c in r.flagged] == [
            "It is System then Power.",
            "The timeout is 15 minutes.",
        ]

    @pytest.mark.parametrize(
        "answer",
        [
            pytest.param(
                '{"claims": [{"claim": "", "verdict": "wrong", '
                '"fix": "the real one"}, '
                '{"claim": "---", "verdict": "wrong", '
                '"fix": "the real one"}]}',
                id="json",
            ),
            pytest.param(
                "VERDICT: FLAGGED\n"
                'CLAIM: "\nRULING: wrong\nREASON: r\nFIX: the real one\n'
                "---\n"
                "CLAIM: .\nRULING: wrong\nREASON: r\nFIX: the real one\n",
                id="labelled",
            ),
        ],
    )
    def test_an_empty_claim_does_not_shadow_a_real_one(self, answer: str) -> None:
        """The dedup runs after the discard filters, and that ordering is
        load-bearing. Deduping first -- the obvious simplification, one call
        instead of two -- lets a copy the filters would have dropped take the
        key, so the real correction vanishes and the sentence ships
        unrepaired with "nothing flagged" recorded.

        The filter that can shadow is the raw `c.claim` truthiness check,
        because the key normalises the claim and the check does not: an empty
        claim and one that normalises to empty share a key, and only the
        second survives. Both entries carry the same fix on purpose, or the
        keys differ, the dedup never fires, and the test passes against the
        reordered code it is meant to catch.

        The other two filters cannot be pinned this way and do not need to
        be. `_is_not_a_fix` is a pure function of the normalised fix, which
        is half the key, so equal keys always get the same answer from it.
        The JSON path's verdict check sits inside the comprehension that
        builds the list, so it is not reorderable against the dedup at all --
        a test written against it pins nothing, which is what an earlier
        version of this test did.
        """
        r = parse_check_answer(answer)
        assert [c.fix for c in r.flagged] == ["the real one"]
        assert r.flagged[0].claim

    def test_a_run_on_verdict_header_does_not_land_in_the_fix(self) -> None:
        """A real answer repeated its whole block set with the second
        `VERDICT: FLAGGED` running on from the previous `FIX:` with no
        newline. Without the verdict alternative in the lookahead the first
        copy's fix absorbed the header, and that contaminated string was
        handed to the reviser as the correct fact.
        """
        r = parse_check_answer(
            "VERDICT: FLAGGED\n"
            "CLAIM: One sentence.\nRULING: wrong\nREASON: r\n"
            "FIX: Use short bursts instead.VERDICT: FLAGGED\n"
            "CLAIM: One sentence.\nRULING: wrong\nREASON: r\n"
            "FIX: Use short bursts instead.\n"
        )
        assert [c.fix for c in r.flagged] == ["Use short bursts instead."]

    def test_a_fix_containing_the_word_verdict_is_kept_whole(self) -> None:
        """The lookahead stops at the header's own grammar -- the word, then
        FLAGGED or OK -- not at the bare word. Matching the bare word reads a
        correction that happens to say "verdict:" as the start of the next
        block, truncates the fix there, and hands the reviser a fragment as
        the correct fact. The fix is the only thing the revise prompt is
        built from, so a truncated one is a wrong repair, not a missing one.
        """
        r = parse_check_answer(
            "VERDICT: FLAGGED\n"
            "CLAIM: The ruling takes a week.\nRULING: wrong\nREASON: r\n"
            "FIX: The court records it as a verdict: usually within two "
            "days.\n"
        )
        assert [c.fix for c in r.flagged] == [
            "The court records it as a verdict: usually within two days."
        ]

    def test_a_run_on_ok_header_does_not_land_in_the_fix(self) -> None:
        """`OK` is the other half of the alternation, and the prompt makes it
        reachable: it specifies `VERDICT: OK` as a real header value, so a
        repeated answer set can begin with one. Dropping that half from the
        lookahead left every other test in this file green.
        """
        r = parse_check_answer(
            "VERDICT: FLAGGED\n"
            "CLAIM: One sentence.\nRULING: wrong\nREASON: r\n"
            "FIX: Paste is a separate job.VERDICT: OK\n"
        )
        assert [c.fix for c in r.flagged] == ["Paste is a separate job."]

    def test_a_fix_saying_verdict_before_a_word_starting_ok_is_kept(
        self,
    ) -> None:
        """The word boundary after the alternation, not just the alternation.
        Without it, `OK` matches the start of `okay` and the fix is cut at
        "the verdict:" -- the same truncation the anchoring exists to stop,
        reached through the other half. Dropping that boundary also left
        every other test in this file green.
        """
        r = parse_check_answer(
            "VERDICT: FLAGGED\n"
            "CLAIM: The step is optional.\nRULING: wrong\nREASON: r\n"
            "FIX: The manual calls the verdict: okay to proceed.\n"
        )
        assert [c.fix for c in r.flagged] == [
            "The manual calls the verdict: okay to proceed."
        ]

    def test_a_fix_quoting_a_whole_verdict_header_is_kept(self) -> None:
        """The anchor matches the header's whole grammar, ending the line
        included. Stopping at the verdict value alone truncates a correction
        that merely quotes a header mid-sentence, and the fix is the only
        thing the revise prompt is built from, so the reviser is handed a
        fragment as the fact -- the same defect the anchor exists to stop,
        reached from the other side. Every run-on header observed ends its
        line, so requiring that costs nothing.
        """
        r = parse_check_answer(
            "VERDICT: FLAGGED\n"
            "CLAIM: The log is silent.\nRULING: wrong\nREASON: r\n"
            "FIX: The scanner writes verdict: flagged into the log.\n"
        )
        assert [c.fix for c in r.flagged] == [
            "The scanner writes verdict: flagged into the log."
        ]

    def test_claims_differing_only_in_case_or_punctuation_collapse(self) -> None:
        """The normalisation is the guard's own, so two spellings of one
        sentence collapse here exactly as they would there. Both halves of
        the key are normalised, or a fix differing only in case would keep a
        duplicate the guard treats as one sentence.
        """
        r = parse_check_answer(
            "VERDICT: FLAGGED\n"
            "CLAIM: Open the settings page.\nRULING: wrong\nREASON: r\n"
            "FIX: It is the power page.\n---\n"
            "CLAIM: open the settings page\nRULING: wrong\nREASON: r\n"
            "FIX: it is the POWER page\n"
        )
        assert len(r.flagged) == 1

    def test_the_json_form_dedupes_too(self) -> None:
        """Both parse paths, or the behaviour depends on which shape the
        model happened to answer in.
        """
        r = parse_check_answer(
            '{"claims": [{"claim": "A.", "verdict": "wrong", "fix": "x"}, '
            '{"claim": "A.", "verdict": "wrong", "fix": "x"}, '
            '{"claim": "B.", "verdict": "wrong", "fix": "z"}]}'
        )
        assert [c.claim for c in r.flagged] == ["A.", "B."]

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

    def test_alternate_flags_cannot_between_them_cover_the_script(self) -> None:
        """The per-claim span bound is not an aggregate one. Two claims naming
        alternate sentences, each well inside its own span, between them make
        every sentence touchable -- and then the checks below have nothing
        left to compare, so a wholly fabricated rewrite is accepted. Every
        containment test before this one used a single claim, which is why it
        went unseen.
        """
        old = sentences(GOOD)
        alternate = [
            FactCheckClaim(claim=old[0], reason="r", fix="f"),
            FactCheckClaim(claim=old[2], reason="r", fix="f"),
        ]
        fabricated = (
            "Charging speed is identical across every port on a modern laptop. "
            "The manufacturer prints a red ring beside the fast charging port. "
            "Windows lists the wattage under Devices, then Power and battery. "
            "Swap the cable before you blame the port on any machine at all. "
            f"{CTA}"
        )
        accepted, reason = accept_revision(GOOD, fabricated, alternate, **GUARDS)
        assert accepted is None and "no sentence of the original" in reason

    def test_most_of_a_script_may_still_be_repaired(self) -> None:
        """The floor has to leave the repair that motivated the feature. Three
        adjoining sentences of five were flagged on a live run and the repair
        was good; refusing that would ship the wrong claims to protect against
        a rewrite that did not happen.
        """
        old = sentences(GOOD)
        three = [FactCheckClaim(claim=old[i], reason="r", fix="f") for i in (1, 2, 3)]
        repaired = (
            f"{old[0]} "
            "On Windows, open Settings, then System, then Power and battery. "
            "The wattage per port is listed on that page for you. "
            "If none is listed, the laptop does not fast charge at all. "
            f"{CTA}"
        )
        accepted, reason = accept_revision(GOOD, repaired, three, **GUARDS)
        assert accepted is not None, reason

    def test_the_first_three_of_five_may_still_be_repaired(self) -> None:
        """The floor counts what survived, not what the claims permitted.
        Counting permission refused this: sentence 3 was touchable merely for
        following a flagged one and sentence 4 is the CTA, so nothing was
        left even though two sentences came back verbatim, and the recorded
        reason said the claims covered the script when they covered three of
        five.
        """
        old = sentences(GOOD)
        three = [FactCheckClaim(claim=old[i], reason="r", fix="f") for i in (0, 1, 2)]
        repaired = (
            "Ports on one laptop differ in the power they can deliver. "
            "On Windows, open Settings, then System, then Power and battery. "
            "The wattage for each port is listed on that page. "
            f"{old[3]} {CTA}"
        )
        accepted, reason = accept_revision(GOOD, repaired, three, **GUARDS)
        assert accepted is not None, reason

    def test_a_sentence_ending_in_a_quoted_label_still_splits(self) -> None:
        """A how-to quotes interface labels, so its sentences end `."` -- and
        a lookbehind demanding the punctuation last merged that sentence into
        the next one. One flagged claim then reached a sentence the checker
        had never examined.
        """
        split = sentences(
            'Look under "Components," then "Power." '
            "You will see a list of controllers. "
            "Find the right one."
        )
        assert len(split) == 3

    def test_a_repair_ending_on_a_quoted_label_passes_validation(self) -> None:
        """The containment guard and the CTA validator have to split
        sentences the same way. They did not: this module widened its own
        splitter while `validate_script_completeness` kept the narrow one, so
        a rewritten sentence closing on a quoted interface label merged into
        the CTA and the repair was refused for not ending on one -- the shape
        the revise prompt asks for, on scripts that quote labels constantly.
        """
        flagged = [
            FactCheckClaim(
                claim="Look for the lightning bolt symbol next to the port.",
                reason="Vendors mark it differently.",
                fix='The marking may read "Power Delivery" instead.',
            )
        ]
        revised = swap(
            GOOD,
            "Look for the lightning bolt symbol next to the port.",
            'Look for a bolt or the words "Power Delivery."',
        )
        accepted, reason = accept_revision(GOOD, revised, flagged, **GUARDS)
        assert accepted is not None, reason

    def test_a_sentence_carrying_two_inner_breaks_is_still_repairable(self) -> None:
        """The span bound was sized for a splitter that broke only before a
        capital. Widening it to admit a lowercase start made over-splits
        common, and one spoken sentence can carry two of them -- an
        abbreviation and an ellipsis. At a bound of two the checker quoting
        that sentence exactly as the prompt demands lost every repair in the
        call, to a reason that blamed it for quoting too much.
        """
        original = (
            "Your fan gets loud on a schedule for one reason. "
            "Check the 5 p.m. reading, and then... nothing happens at all. "
            "Blow the dust out from the outside of the vent. "
            f"{CTA}"
        )
        flagged = [
            FactCheckClaim(
                claim="Check the 5 p.m. reading, and then... nothing happens at all.",
                reason="There is no scheduled reading to check.",
                fix="Fan speed follows temperature, not a clock.",
            )
        ]
        revised = original.replace(
            "Check the 5 p.m. reading, and then... nothing happens at all.",
            "Fan speed follows the temperature rather than any clock.",
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
