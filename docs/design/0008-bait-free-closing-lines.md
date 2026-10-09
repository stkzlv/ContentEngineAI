# 0008. Remove engagement-bait lines from the CTA pools

- **Status:** Implemented
- **Issue:** #549
- **Requirements:** REQ-CNT-041

## Context

Both pools carry a share request: "Share with someone who needs this." in `cta_options` and "Share it with whoever needs it." in `cta_options_topic`. Meta lists share requests as engagement bait.

Evidence ([evidence grades](README.md#evidence-grades)):

- Meta demotes posts that ask for specific words, emojis, votes, shares or tags, demotes repeat offenders harder, and exempts genuine requests for advice or opinions. [A] [Meta engagement-bait guidelines](https://transparency.meta.com/features/approach-to-ranking/content-distribution-guidelines/engagement-bait/)
- TikTok's For You feed excludes engagement manipulation ("like-for-like" promises, false incentives for gifting or following), so a payoff promised only for a follow ("follow for part 2") is a risk. [A] [TikTok For You feed standards](https://www.tiktok.com/community-guidelines/en/fyf-standards)
- TikTok's Creator Rewards criteria exclude content "primarily designed to attract followers, likes, or advertisement clicks". [A] [TikTok Creator Rewards eligibility](https://www.tiktok.com/creator-academy/article/eligibility)
- In a preprint on 30 Estonian brand accounts, call-to-action type was the main comment predictor, led by engagement bait; no study isolates a genuine closing question. [B]

## Goals

- No configured call to action or closing-line example asks viewers to share, tag, vote or reply with a specific word.

## Non-goals

None recorded.

## Design

- A bait-pattern list in a test (share, tag, vote, "comment <word>", emoji requests, follow-for-reward), checked against `cta_options`, `cta_options_topic` and the closing-line examples in every script template.
- Replace the flagged lines with genuine opinion or save prompts (for example "Save this for your next setup."). The first-comment extractor's `_CTA_MARKERS` has to gain the replacement openers, and its existing test asserts every configured CTA starts with one.

**Tests.** The bait test fails on a pool containing a share request; the marker test passes with the edited pool.

## As built

The pool edit replaced the product share line with "Save this for when you need one." and dropped the topic one, leaving three topic lines; the profile line became "Check the link in bio if you want one.", so every configured call to action opens on an imperative (REQ-CNT-148). The first-comment extractor keeps the old openers among its markers, since scripts written before the edit still end on them.

## Alternatives considered

None recorded.

## Rollout

Changing the pool changes the closing lines of both reach-test arms, so there is no switch: the pool edit itself waits for the reach-test readout (#540). The test lands first with the current offenders listed as known exceptions, which the pool edit then removes.

## Open questions

None recorded.
