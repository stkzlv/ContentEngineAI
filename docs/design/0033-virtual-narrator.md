# 0033. Virtual narrator

- **Status:** Implemented
- **Issue:** #805
- **Requirements:** REQ-CNT-173

## Context

A synthetic narrator can't buy, hold, try on or test anything, so a script that has it say so makes a false claim. Two things make that more than a style problem:

- **Models write such claims by default.** A review of 30 product scripts found 20 claiming the narrator owned, bought, received or used the product (#700). Design 0024 removed the prompt examples that modelled them and REQ-CNT-157 forbids them in the product prompt, but nothing checks the model's answer, and topic and commentary scripts carry no such rule.
- **The claim costs trust and can break the rules.** An endorsement that implies use requires that the endorser is a bona fide user (16 CFR 255.1(c)) [A]. Perceived authenticity predicts trust in an influencer more than whether they are human or virtual, and a disclosure label mostly doesn't move it (Barros and Fetscherin 2025, a 599-person 2x2 experiment, conference paper) [B]. Fans of an AI streamer treat persona consistency, not humanness, as its authenticity (arXiv 2509.10427, 334 survey responses and 550,000 chat messages) [B]. A narrator presented as synthetic that claims a physical act contradicts itself.

The script lint (REQ-CNT-053) already rejects machine-writing phrases and retries, and the copied-example check (REQ-CNT-155) already shows how to ship a script without one bad sentence.

## Goals

- A narrator identity can be marked virtual, per content kind.
- Its scripts are asked not to claim a physical act, and a script that claims one is retried, then shipped without the claiming sentence.
- A kind not marked virtual is unchanged, byte for byte.

## Non-goals

- Judging whether a claim is true. Any first-person physical act is refused, because the narrator can't have done it.
- Third-person claims ("people who bought it say"). The fact check and REQ-CNT-157's research voice cover those.
- Disclosure on the frame or in captions. The AI labels cover that (designs 0016, 0027).

## Design

- **The flag.** `NarratorIdentity.virtual: bool = False` in `src/video/config/identity_models.py`. It is not a pinned dimension: it fixes no drawn choice, so it doesn't enter the render-choices `pinned` list (REQ-CNT-170).
- **The prompt rule.** `config_for_record` (design 0031) sets `script_templates.virtual_narrator` on the kind's config copy, and `narrator_for` appends one rule to the text it assembles when the flag is set, so the rule reaches whichever profile the kind uses, an empty topic profile's product fallback included. The rule says the narrator is synthetic and never says it bought, held, tried, tested, unboxed, wore or went somewhere; first-person opinion and research are fine. It quotes no example sentence (REQ-CNT-159).
- **The check.** `script_validation.physical_claims` (a list of case-insensitive regular expressions, shipped with a default set). It is not part of the lint: a lint failure is a soft near miss that can ship (REQ-CNT-164), and the lint can be off. With `virtual_narrator` set, the script step's validation checks for a claim first, beside the copied-example check, whatever the lint setting, and fails naming the first sentence that matches. The default set covers a first-person subject ("I", "I've", "I just", "we") followed by a physical verb: bought, ordered, held, unboxed, wore, tested, tried it or them, picked up, plugged in, took it to or on, spent an amount. "Tried to" and verbs of reading, thinking and checking don't match.
- **When every attempt claims.** A claim is never kept as a soft near miss, because shipping it ships the false claim. The script step drops the claiming sentence the way it drops a borrowed example (`drop_borrowed`) and keeps the result as the same kind of near miss, so it ships if the rest passes validation; otherwise the run falls back as any failed script does.
- **Tests.** Positive and negative sentence lists against the default patterns; the prompt for a virtual kind carries the rule and a non-virtual kind's prompt equals the bundled one; a script whose every attempt claims ships without the claiming sentence.

## Alternatives considered

- **Prompt rule only.** That is REQ-CNT-157 today, and #700 shows the model ignores such rules often enough to need a check.
- **A banned phrase list or a lint rule.** `banned_phrases` is global, a lint failure ships as a near miss, and the lint can be off; the check must apply only to a virtual narrator's kind and must not ship a claim.
- **An LLM judge.** A second call per script for a pattern a regular expression finds.

## Rollout

The flag ships false and the bundled identities set nothing, so nothing changes until an installation sets `virtual: true` in its overlay. Measure before relying on it: render ten scripts per kind with the flag on and count matches and false positives in the log.

Remove the switch when: never; whether a narrator is synthetic is an installation's lasting choice.

## Open questions

None recorded.
