"""Check a tutorial script's falsifiable claims, and repair only what is wrong.

A shipped render told viewers to open Windows System Information, go to
Components then Power, and read the USB controllers. No such node exists.
Measured on 17 topic scripts (#380): 83 falsifiable claims, 16 flagged, 14 of
the 16 correct on hand review, four distinct invented methods.

The prompt fix in 0.103.1 removed the demand that produced most of them. What
it cannot reach is a value that is simply wrong: "go to Devices, then Power &
battery" names its platform, satisfies every prompt rule, and is still the
wrong page. Only a grounded lookup catches that.

**Nothing here can lose a render.** That is the same rule the stock judge
follows, and it is enforced by the type rather than by discipline:
`FactCheckOutcome.script` is not optional, so "the checker ate the script" is
unrepresentable. A dead API, an unparseable answer, a missing key, a revision
that fails validation -- every one of them ships the original.

Two calls of two different kinds. The check is grounded and builds its own
client, because `call_llm` carries no tools and should not grow them for this.
The revision is *not* grounded: the research already happened, rewriting a
sentence is not a search problem, and holding it to one grounded query per
script is what holds the bill down.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.video.config.llm_settings import ScriptFactCheckConfig

from src.utils.script_sanitizer import split_sentences

logger = logging.getLogger(__name__)

_CLAIM_BLOCK = re.compile(
    r"CLAIM:\s*(?P<claim>.+?)\s*"
    r"(?:RULING:\s*(?P<ruling>.+?)\s*)?"
    r"REASON:\s*(?P<reason>.+?)\s*"
    r"FIX:\s*(?P<fix>.+?)\s*(?=(?:\n\s*-{3,})|(?:\n\s*CLAIM:)|\Z)",
    re.S | re.I,
)

# A ruling of "correct" arriving in the shape of a fix. Measured live, twice:
# asked for wrong claims only, the checker listed a sentence, explained in
# REASON that it was right, and wrote `FIX: Correct.` and then `FIX: This
# claim is correct.` -- which the reviser reads as an instruction to rewrite a
# good sentence, spending one of the `max_flags_to_revise` slots on it and
# pushing the whole revision past the drift guard, so the genuinely wrong
# sentences ship unrepaired. The prompt forbids it in two places; a prompt
# rule is not a guard, which is why the per-claim RULING line exists and why
# this second net does too.
_AFFIRMATIONS = frozenset(
    {"correct", "accurate", "true", "right", "fine", "none", "na", "change"}
)
_FILLER = frozenset(
    {
        "this",
        "that",
        "the",
        "claim",
        "statement",
        "sentence",
        "is",
        "was",
        "are",
        "were",
        "it",
        "as",
        "written",
        "a",
        "an",
        "and",
        "no",
        "not",
        "needed",
        "required",
        "fix",
        "already",
    }
)


def _is_not_a_fix(fix: str) -> bool:
    """Whether a FIX is a ruling of "correct" wearing a fix's clothes.

    Filler is stripped before the test because the shape varies: "Correct.",
    "This claim is correct.", "The statement as written is accurate." all say
    the same nothing, and matching on the first word alone caught only the
    first of them.
    """
    words = [w for w in _normalise(fix).split() if w not in _FILLER]
    return not words or (len(words) <= 2 and set(words) <= _AFFIRMATIONS)


@dataclass(frozen=True)
class FactCheckClaim:
    """One sentence the checker ruled wrong, with its correction."""

    claim: str
    reason: str
    fix: str


@dataclass(frozen=True)
class FactCheckResult:
    """What the checker said, or why it said nothing.

    `ran` is False when the checker could not be reached at all. An answer that
    arrived but could not be parsed is `ran=True` with no flags and an `error`,
    because the script was checked and nothing actionable came back -- the
    caller treats both as "ship it".
    """

    ran: bool
    flagged: list[FactCheckClaim] = field(default_factory=list)
    raw: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class FactCheckOutcome:
    """The script to ship, and what happened on the way to it.

    `script` is never None and never empty. The whole safety property of this
    module is that the caller cannot end up holding nothing.
    """

    script: str
    record: dict[str, Any]


def _normalise(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9\s]+", "", text.lower()).split())


def sentences(text: str) -> list[str]:
    """The shared splitter, named here for this module's readers.

    It has to be the one `validate_script_completeness` uses, or a repair
    confined correctly here is refused there for not ending on a CTA.
    """
    return split_sentences(text)


def _ruling_is_wrong(ruling: str | None) -> bool:
    """Whether a per-claim ruling says the claim is wrong."""
    if ruling is None:
        return True
    return "wrong" in ruling.strip().lower()


# How many split sentences one flagged claim may cover. The prompt asks for
# one; two absorbs a residual mismatch between the checker's copy and how the
# text splits. Beyond that the claim is quoting the script rather than naming
# a sentence in it.
_MAX_SPAN_PER_CLAIM = 2


def _covers(claim: str, sentence: str) -> bool:
    """Whether a flagged claim and a split sentence are the same material.

    Not equality. `sentences()` splits where sentence-final punctuation is
    followed by whitespace and something that starts a sentence, which is
    right for an ellipsis mid-sentence but still wrong for an abbreviation
    before a capital (`Wi-Fi 5 vs. Wi-Fi 6`) -- while the checker copies the
    whole sentence, as the prompt demands. Equality then matched neither
    fragment, nothing was marked touchable, and a revision that changed only
    the flagged sentence was refused for altering sentences it was not asked
    to: the repair thrown away and the wrong claim published, with the record
    blaming the reviser for it.

    Containment either way covers both that case and a checker that quoted
    only part of a sentence. Padded, so the match is on whole words and a
    short sentence cannot match mid-word inside an unrelated claim.
    """
    a, b = _normalise(claim), _normalise(sentence)
    if not a or not b:
        return False
    return f" {b} " in f" {a} " or f" {a} " in f" {b} "


def parse_check_answer(text: str | None) -> FactCheckResult:
    """Read the checker's answer, in either shape it may arrive in.

    JSON first, then the labelled-line format the prompt asks for. The labels
    exist because grounding and a response schema cannot combine on the 2.5
    tier -- verified, the API returns 400 `Tool use with a response mime type:
    'application/json' is unsupported` -- so the answer is free text and the
    parser has to be the guarantee. Trying JSON first means a later move to a
    tier that supports both is a config edit rather than a code change.

    An answer that parses to nothing actionable is not an error worth failing
    on: the script ships as it would have anyway.
    """
    if not text or not text.strip():
        return FactCheckResult(ran=True, raw=text, error="empty answer")

    body = text.strip().strip("`").removeprefix("json").strip()
    try:
        loaded = json.loads(body)
    except (ValueError, TypeError):
        loaded = None
    if isinstance(loaded, dict):
        claims = [
            FactCheckClaim(
                claim=str(c.get("claim", "")).strip(),
                reason=str(c.get("reason", "")).strip(),
                fix=str(c.get("fix", "")).strip(),
            )
            for c in loaded.get("claims", [])
            if isinstance(c, dict) and str(c.get("verdict", "wrong")).lower() == "wrong"
        ]
        return FactCheckResult(
            ran=True,
            flagged=[c for c in claims if c.claim and not _is_not_a_fix(c.fix)],
            raw=text,
        )

    flagged = [
        FactCheckClaim(
            claim=m.group("claim").strip().strip('"'),
            reason=m.group("reason").strip(),
            fix=m.group("fix").strip(),
        )
        for m in _CLAIM_BLOCK.finditer(body)
        # The per-claim ruling is what makes "list only what you rule wrong"
        # enforceable rather than requested. Absent, the block is kept: it was
        # listed under a FLAGGED verdict, and dropping it would lose a real
        # flag to a formatting slip.
        if _ruling_is_wrong(m.group("ruling"))
    ]
    # A block missing its FIX is dropped rather than half-used: the revision
    # prompt is built from the fix, and a claim without one asks the model to
    # invent the correction, which is the failure this module exists to catch.
    # A fix that only says the claim was right is dropped for the same reason.
    flagged = [c for c in flagged if c.claim and not _is_not_a_fix(c.fix)]
    if not flagged and "VERDICT" not in body.upper():
        return FactCheckResult(ran=True, raw=text, error="unparsed")
    return FactCheckResult(ran=True, flagged=flagged, raw=text)


def accept_revision(
    original: str,
    revised: str | None,
    flagged: list[FactCheckClaim],
    *,
    cta_options: list[str],
    min_chars: int,
    min_words: int,
    max_length_drift: float,
) -> tuple[str | None, str]:
    """Whether a revision is safe to ship, and why not when it is not.

    Pure, so every guard is testable without a network. Returns the accepted
    script or None with a reason; the caller ships the original on None.

    The containment guard is the important one. Measured precision is 14 in 16,
    so roughly one flag in eight is wrong, and without containment a bad
    correction could rewrite a script that was fine. With it, a false positive
    can damage the sentences it named and the one following each, and nothing
    beyond that.
    """
    from src.ai.script_generator import validate_script_completeness
    from src.utils.script_sanitizer import sanitize_script

    # First, because this judges the checker's answer rather than the
    # revision, so it must be the reason recorded when it is the cause. A
    # claim quoting more of the script than one sentence turns containment
    # off: every sentence it spans becomes rewritable, and a claim quoting the
    # whole script leaves nothing protected. The prompt asks for one sentence,
    # and `_covers` exists only to absorb a residual mismatch between that
    # sentence and how the text splits, so a span this wide means the answer
    # is unusable rather than that the whole script was flagged. Refusing
    # loses one repair; accepting hands a one-in-eight false positive the run
    # of the script.
    old_sentences = sentences(original)
    spans = [
        [i for i, s in enumerate(old_sentences) if _covers(c.claim, s)] for c in flagged
    ]
    for spanned in spans:
        if len(spanned) > _MAX_SPAN_PER_CLAIM:
            return (
                None,
                f"a flagged claim spans {len(spanned)} sentences of the script",
            )

    if revised is None or not revised.strip():
        return None, "the reviser returned nothing"

    # Sanitised first, because the sanitised form is what ships; validating the
    # raw form would check text nobody sees.
    candidate = sanitize_script(revised)
    if not candidate.strip():
        return None, "the revision was empty after sanitising"
    if _normalise(candidate) == _normalise(original):
        return None, "the revision changed nothing"

    ok, reason = validate_script_completeness(
        candidate, min_chars, min_words, cta_options
    )
    if not ok:
        return None, f"the revision failed validation: {reason}"

    original_words = len(original.split())
    if original_words:
        drift = abs(len(candidate.split()) - original_words) / original_words
        if drift > max_length_drift:
            return None, f"the revision changed length by {drift:.0%}"

    # Containment: a sentence the checker did not name, and which does not
    # immediately follow one it named, must come back unchanged. The successor
    # is allowed because a fix often has to carry into the step that referenced
    # the wrong thing.
    new_sentences = sentences(candidate)
    touchable: set[int] = set()
    for spanned in spans:
        touchable.update(spanned)
        # One successor for the claim, not one per sentence it spans.
        if spanned:
            touchable.add(spanned[-1] + 1)
    kept_old = [(i, s) for i, s in enumerate(old_sentences) if i not in touchable]
    new_norm = [_normalise(s) for s in new_sentences]

    # The per-claim bound is not an aggregate one: a few claims naming
    # alternate sentences, each within its own span, between them make every
    # sentence of a short script touchable -- and then the checks below have
    # nothing left to compare and a wholly fabricated rewrite is accepted.
    #
    # Measured on what came back, not on what the claims permitted. Counting
    # permission refused a correct repair of the first three sentences of
    # five, because the fourth was touchable merely for following a flagged
    # one and the fifth was the CTA, and the recorded reason then said the
    # claims covered the script when they covered three of five. That is also
    # why this sits here rather than beside the span check it otherwise
    # belongs with: survival cannot be read before the revision is split.
    #
    # A script with no body sentence left standing is the checker rejecting
    # it wholesale, and an ungrounded sentence-by-sentence patch is not the
    # instrument for that. Refusing ships the original with its wrong claims,
    # which is the worse-looking half of a real trade: the alternative is a
    # rewrite bounded by nothing but its length.
    survivors = [
        i
        for i, sentence in enumerate(old_sentences)
        if i != len(old_sentences) - 1 and _normalise(sentence) in new_norm
    ]
    if not survivors:
        return (
            None,
            "the revision left no sentence of the original but its closing line",
        )

    missing = [s for _, s in kept_old if _normalise(s) not in new_norm]
    if missing:
        return (
            None,
            f"the revision altered {len(missing)} sentence(s) it was not asked to",
        )

    # The survivors must also come back in the order they went out. Membership
    # alone let a pure reordering through: every sentence "unchanged", nothing
    # added, length identical, and the steps of a how-to swapped. The revise
    # prompt forbids re-ordering, but this module's own position is that a
    # prompt rule is not a guard -- which is why `RULING:` and `_is_not_a_fix`
    # exist.
    #
    # A forward cursor rather than `list.index`, which returns the *first*
    # occurrence: a script that says the same line twice, or twice after
    # normalising away case and punctuation, then reads as re-ordered when
    # nothing moved, and the repair is thrown away.
    cursor = -1
    for _, kept in kept_old:
        try:
            cursor = new_norm.index(_normalise(kept), cursor + 1)
        except ValueError:
            return None, "the revision re-ordered sentences it was not asked to"

    # Containment has a second half. Keeping every unflagged sentence still
    # allows the model to *insert* one, which is the shape a helpful rewrite
    # takes when it adds context nobody checked -- an unchecked new claim, in
    # a module whose entire purpose is that claims get checked. Length drift
    # does not catch it: one extra sentence in six is well inside 25%. A fix
    # may legitimately split a sentence in two, so the allowance is one per
    # claim rather than none.
    allowed = len(old_sentences) + len(flagged)
    if len(new_sentences) > allowed:
        return (
            None,
            f"the revision added {len(new_sentences) - len(old_sentences)} "
            "sentence(s) beyond the repair",
        )

    return candidate, "accepted"


async def check_script(
    script: str,
    subject: str,
    *,
    api_key: str,
    settings: ScriptFactCheckConfig,
) -> FactCheckResult:
    """Ask a grounded model which falsifiable claims in the script are wrong.

    Builds its own client, the way the stock judge does, because `call_llm`
    carries no tools. Never raises: every failure is `ran=False`, which the
    caller reads as "ship the original".
    """
    import asyncio
    from pathlib import Path

    unavailable = FactCheckResult(ran=False)
    try:
        from google import genai
        from google.genai import errors as genai_errors

        client = genai.Client(api_key=api_key)
        config = genai.types.GenerateContentConfig(
            tools=[genai.types.Tool(google_search=genai.types.GoogleSearch())],
            temperature=0.0,
        )
    except (ImportError, ValueError, OSError, RuntimeError) as e:
        # A SOCKS proxy without socksio raises from the constructor alone.
        logger.warning("Fact check unavailable: %s", e)
        return unavailable

    template = Path(__file__).parent / "prompts" / "script_fact_check.md"
    try:
        prompt = template.read_text(encoding="utf-8").format(
            SUBJECT=subject, SCRIPT=script
        )
    except (OSError, KeyError, IndexError) as e:
        logger.warning("Fact check prompt unusable: %s", e)
        return unavailable

    try:
        async with asyncio.timeout(settings.timeout_seconds):
            answer = await client.aio.models.generate_content(
                model=settings.model, contents=prompt, config=config
            )
    except (
        TimeoutError,
        OSError,
        ValueError,
        RuntimeError,
        genai_errors.APIError,
    ) as e:
        logger.warning("Fact check call failed: %s", e)
        return unavailable
    finally:
        aclose = getattr(client.aio, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except (OSError, RuntimeError) as e:  # closing is best effort
                logger.debug("Closing the fact-check client failed: %s", e)

    return parse_check_answer(answer.text)


async def revise_script(
    script: str,
    flagged: list[FactCheckClaim],
    product: Any,
    settings: Any,
    api_key: str,
    session: Any,
    api_settings: Any = None,
    *,
    secrets: dict[str, str] | None = None,
    narrator_profile: str = "",
    pillar: str | None = None,
    pillar_preambles: dict[str, str] | None = None,
    debug_mode: bool = False,
) -> str | None:
    """Rewrite only the flagged sentences. Not grounded, by design.

    The research already happened in `check_script`; rewriting a sentence
    around a fact you have been handed is not a search problem, and keeping
    this call ungrounded is what holds the bill at one grounded query per
    script. Routed through `generate_with_llm`, which applies the narrator
    profile so the correction does not drift in voice, and returns None rather
    than raising.
    """
    from pathlib import Path

    from src.ai.platform_metadata.utilities import generate_with_llm

    listing = "\n".join(
        f"- SENTENCE: {c.claim}\n  WRONG BECAUSE: {c.reason}\n  CORRECT FACT: {c.fix}"
        for c in flagged
    )
    return await generate_with_llm(
        Path(__file__).parent / "prompts" / "script_revise.md",
        product,
        settings,
        api_key,
        session,
        api_settings,
        debug_mode,
        # Without this the configured fallback provider is unreachable: it
        # looks its key up in `secrets`, which defaults to empty, so a
        # rate-limited primary loses the repair and ships the wrong claim
        # while a funded fallback sits unused.
        secrets=secrets,
        video_script=script,
        narrator_profile=narrator_profile,
        pillar=pillar,
        pillar_preambles=pillar_preambles,
        extra_placeholders={"FLAGGED_CLAIMS": listing},
    )


async def fact_check_and_revise(
    script: str,
    subject: str,
    product: Any,
    settings: Any,
    secrets: dict[str, str],
    session: Any,
    api_settings: Any = None,
    *,
    narrator_profile: str = "",
    pillar: str | None = None,
    pillar_preambles: dict[str, str] | None = None,
    debug_mode: bool = False,
) -> FactCheckOutcome:
    """Check the script, repair what is wrong, and return what should ship.

    Never raises, and never returns an empty script. Every path that fails
    returns the original with the reason recorded, because a fabricated
    sentence is a worse outcome than a lost render only if the render still
    happens -- and a check that can lose one is the worse trade.
    """
    cfg = settings.script_fact_check
    record: dict[str, Any] = {
        "ran": False,
        "model": cfg.model,
        "subject": subject,
        "flagged": [],
        "revision": {"attempted": False, "accepted": False, "reason": "not reached"},
    }
    if not cfg.enabled:
        record["revision"]["reason"] = "disabled"
        return FactCheckOutcome(script=script, record=record)

    api_key = secrets.get(settings.api_key_env_var) if secrets else None
    if not api_key:
        logger.debug("Fact check skipped: no API key (%s)", settings.api_key_env_var)
        record["revision"]["reason"] = "no api key"
        return FactCheckOutcome(script=script, record=record)

    try:
        result = await check_script(script, subject, api_key=api_key, settings=cfg)
    except Exception as e:  # noqa: BLE001 -- the render must survive anything
        logger.warning("Fact check raised unexpectedly: %s", e)
        record["revision"]["reason"] = f"checker raised: {e}"
        return FactCheckOutcome(script=script, record=record)

    record["ran"] = result.ran
    record["raw_answer"] = result.raw
    record["error"] = result.error
    record["flagged"] = [
        {"claim": c.claim, "reason": c.reason, "fix": c.fix} for c in result.flagged
    ]
    if not result.ran or not result.flagged:
        record["revision"]["reason"] = (
            "nothing flagged" if result.ran else "checker unavailable"
        )
        return FactCheckOutcome(script=script, record=record)

    flagged = result.flagged[: cfg.max_flags_to_revise]
    record["revision"]["attempted"] = True
    try:
        revised = await revise_script(
            script,
            flagged,
            product,
            settings,
            api_key,
            session,
            api_settings,
            secrets=secrets,
            narrator_profile=narrator_profile,
            pillar=pillar,
            pillar_preambles=pillar_preambles,
            debug_mode=debug_mode,
        )
    except Exception as e:  # noqa: BLE001 -- same rule
        logger.warning("Script revision raised unexpectedly: %s", e)
        record["revision"]["reason"] = f"reviser raised: {e}"
        return FactCheckOutcome(script=script, record=record)

    sv = settings.script_validation
    accepted, reason = accept_revision(
        script,
        revised,
        flagged,
        cta_options=settings.script_templates.cta_options_for(
            bool(getattr(product, "topic", None))
        ),
        min_chars=sv.min_chars,
        min_words=sv.min_words,
        max_length_drift=cfg.max_length_drift,
    )
    record["revision"]["reason"] = reason
    if accepted is None:
        logger.warning(
            "Fact check flagged %d claim(s); keeping the original because %s",
            len(flagged),
            reason,
        )
        return FactCheckOutcome(script=script, record=record)

    record["revision"]["accepted"] = True
    logger.info("Fact check revised %d claim(s) in the script", len(flagged))
    return FactCheckOutcome(script=accepted, record=record)
