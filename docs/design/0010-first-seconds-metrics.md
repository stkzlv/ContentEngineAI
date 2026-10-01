# 0010. First-seconds metrics in the analytics sweep

- **Status:** Accepted
- **Issue:** #551
- **Requirements:** REQ-PUB-084

## Context

The sweep reads view timelines through the scheduling provider and stores day-2 and day-7 views and a durability ratio.

The other designs' rollout checks (swipe-away, completion, average percentage viewed) read the metrics this design stores.

## Goals

- Store each first-seconds and quality metric a platform exposes, per post.
- Segment each metric by `content_format` and by the render choices [0006](0006-render-choices-and-variety-report.md) records.

## Non-goals

- Storing an unavailable metric as zero. It is stored as unknown, the rule the day-N figures already follow.

## Design

- Inventory what the provider's analytics endpoint returns per platform (likes, comments, shares, saves, impressions, reach, watch time). Store every available field per post in the metrics store, beside the view figures.
- YouTube's "viewed vs swiped away" and engaged views are exposed by the YouTube Analytics API (`engagedViews`), not necessarily by the provider. If the provider lacks them, add an optional YouTube Analytics reader behind its own credentials, off by default.
- Store an unavailable metric as unknown, never zero.
- Extend the reports to segment each metric by `content_format` and by the [0006](0006-render-choices-and-variety-report.md) render choices.

**Tests.** The store round-trips the added fields and keeps unknowns unknown; a report segments a fixture by format and choice.

## Alternatives considered

None recorded.

## Rollout

Measurement only. It doesn't change rendered output, so it can land before the reach-test readout (#540). The optional YouTube Analytics reader ships off and needs its own credentials.

## Open questions

- Whether the provider exposes YouTube's viewed-vs-swiped-away and engaged views, which decides whether the YouTube Analytics reader is needed.
