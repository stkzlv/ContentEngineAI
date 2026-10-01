# 0007. Script lint and search-phrase placement

- **Status:** Accepted
- **Issue:** #548
- **Requirements:** REQ-CNT-053, REQ-CNT-054, REQ-CNT-055

## Context

`validate_script_completeness` checks truncation, length floors and the CTA; the narrator profiles carry banned-phrase prose, which the model may ignore.

The technique comes from [creator-research.md](../creator-research.md).

## Goals

- Reject a script that uses common machine-writing phrases, exceeds a sentence-length cap or exceeds a word count derived from the target duration, and retry.
- Lead the hook headline and every platform caption with the search phrase.
- Report, per render, where the search phrase appears.

## Non-goals

- Losing a render to the lint. A script that fails only the lint ships with a warning.

## Design

- `script_validation.lint`: `enabled` (default false), `banned_phrases` (a list: "it's not X, it's Y" shapes as regexes, "game-changer", "say goodbye to", "elevate", "seamless", "delve", "whether you're"), `max_sentence_words` (default 16), `max_words_per_sec` (default 2.8, applied to the profile's target duration).
- Run inside the existing `_validate` closure, so a failing script re-enters the retry loop like a missing CTA, with its own reason string. The last-resort fallback today rescues only a script whose sole defect is the CTA; extend it so a script that fails only the lint also ships (with a warning) rather than losing the render.
- Prompt rules for the hook's concreteness and the "but/therefore" chain render into `{CTA_RULE}` after the existing rules, like naturalism, behind `script_templates.hook_rules.enabled` (default false).
- Search-phrase placement: a prompt rule for the hook headline and each platform caption prompt to lead with the search phrase (the product keyword or the topic keyword), behind the same switch as the hook rules. The first spoken sentence already carries it through the existing audio-keyword rule.
- Search-phrase report: the phrase checked against the first spoken sentence, the hook headline and the first 60 characters of each platform caption, counted per render in the [0006](0006-render-choices-and-variety-report.md) report.

**Tests.** Each banned shape rejects a fixture script; the length caps reject an over-long script; off leaves `_validate` unchanged; the report counts a fixture where the phrase is missing from the caption.

## Alternatives considered

None recorded.

## Rollout

- The search-phrase report is measurement only and can land before the reach-test readout (#540).
- The lint ships off: `script_validation.lint.enabled` defaults to false. Set it after the readout, once rejection rates on a batch stay low (the retry loop is paid per call) and the scripts read better on review.
- The hook and placement rules ship off: `script_templates.hook_rules.enabled` defaults to false. Set it after the readout; the placement rules wait for the report.

## Open questions

None recorded.
