# 0008. Remove engagement-bait lines from the CTA pools

- **Status:** Accepted
- **Issue:** #549
- **Requirements:** REQ-CNT-041

## Context

Both pools carry a share request: "Share with someone who needs this." in `cta_options` and "Share it with whoever needs it." in `cta_options_topic`. Meta lists share requests as engagement bait.

The finding comes from [creator-research.md](../creator-research.md).

## Goals

- No configured call to action or closing-line example asks viewers to share, tag, vote or reply with a specific word.

## Non-goals

None recorded.

## Design

- A bait-pattern list in a test (share, tag, vote, "comment <word>", emoji requests, follow-for-reward), checked against `cta_options`, `cta_options_topic` and the closing-line examples in every script template.
- Replace the flagged lines with genuine opinion or save prompts (for example "Save this for your next setup."). The first-comment extractor's `_CTA_MARKERS` has to gain the replacement openers, and its existing test asserts every configured CTA starts with one.

**Tests.** The bait test fails on a pool containing a share request; the marker test passes with the edited pool.

## Alternatives considered

None recorded.

## Rollout

Changing the pool changes the closing lines of both reach-test arms, so there is no switch: the pool edit itself waits for the reach-test readout (#540). The test lands first with the current offenders listed as known exceptions, which the pool edit then removes.

## Open questions

None recorded.
