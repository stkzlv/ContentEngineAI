# 0010. First-seconds metrics in the analytics sweep

- **Status:** Accepted
- **Issue:** #551
- **Requirements:** REQ-PUB-084

## Context

The sweep reads view timelines through the scheduling provider and stores day-2 and day-7 views and a durability ratio.

The other designs' rollout checks (swipe-away, completion, average percentage viewed) read the metrics this design stores.

Evidence ([evidence grades](README.md#evidence-grades)):

- YouTube ranks Shorts on whether the viewer chose to watch or swiped away, the share who viewed, average view duration and percentage viewed, likes and survey responses. [A] [YouTube Help](https://support.google.com/youtube/answer/11914225)
- YouTube Studio shows "viewed vs swiped away" per Short, the first gate. [A] [YouTube Help community](https://support.google.com/youtube/community-video/273390203/new-youtube-shorts-metric-viewed-vs-swiped-away)
- Since 31 March 2025 every Shorts play and replay counts as a view and "engaged views" (which monetisation uses) exclude them, so raw counts are inflated by loops. [A] [YouTube Help community thread](https://support.google.com/youtube/thread/333869549)
- "Beginning August 24, 2026, views are counted the moment a video starts to play across all formats"; engaged views stay the measure of viewers who "stayed to watch past the initial seconds". A raw view is now close to an impression. [A] [YouTube Help](https://support.google.com/youtube/answer/2991785), [YouTube Help](https://support.google.com/youtube/answer/12220281)
- Across 5,400 Shorts on 33 channels, Shorts below 60% viewed-vs-swiped-away rarely performed well, and likes, comments and shares had no strong relationship with performance; the study is from 2023, before the view-count change. [B] [Galloway thread](https://threadreaderapp.com/thread/1646898356419981315.html)
- TikTok ranks on user interactions, video information and device settings, and finishing a longer video carries more weight than weak signals (2020 statement). [A] [TikTok newsroom](https://newsroom.tiktok.com/en-us/how-tiktok-recommends-videos-for-you)
- Instagram's top signals are watch time, likes per reach and sends per reach, as quoted from Adam Mosseri. [B, secondary] [Hootsuite](https://blog.hootsuite.com/instagram-algorithm/)
- In TikTok's coding of its ads, the first 2 seconds matter most for ad recall and the first 2.5 seconds for awareness (2021, paid ads). [B] [TikTok Creative Center](https://ads.tiktok.com/business/creativecenter/quicktok/online/Power_Creative_Elements/pc/en)
- Not supported: fixed numeric weight tables for any platform, or TikTok's "batches of 300-500 viewers". No platform publishes its weights, so the reports segment raw metrics rather than scoring against invented weights.

## Goals

- Store each first-seconds and quality metric a platform exposes, per post.
- Segment each metric by `content_format`, by the render choices [0006](0006-render-choices-and-variety-report.md) records, and by duration band once each post records its duration ([0025](0025-video-length.md)).

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

Remove the switch when: never; it is a lasting option, because the YouTube Analytics reader needs credentials of its own that not every operator holds.

## As built

- The provider exposes no YouTube engaged views or viewed-vs-swiped-away, and no TikTok watch time or completion; the TikTok fields need the account connected through TikTok for Business. These stay unknown.
- The YouTube stand-in is the per-video retention curve the provider returns: once a video is `analytics.retention_curve_min_age_days` old (7), the sweep reads it once and stores `earlyWatchingPct`, the share of viewers still watching at `analytics.early_retention_sec` (3 s), and `earlyRelativeRetention`, YouTube's score there against videos of similar length (0 worst, 0.5 median, 1 best). The reading is the first curve point at or after that second; a curve with no duration, no points, or a gap of more than 5% of the video at that second is unknown and read again on the next sweep. Replays of the opening can put `earlyWatchingPct` above 100.
- The curve call spends the provider's shared YouTube Analytics quota, so a stored reading is never read again, and a rate-limit refusal ends the YouTube calls for the rest of the sweep, as for the daily stats.
- On a real sweep, settled videos with views got a reading; videos with very few views got the provider's note that YouTube returned no curve, and stayed unknown.
- The quality report segments `earlyWatchingPct`, `earlyRelativeRetention` and the daily stats' `avgViewPercentage` beside the exposed fields.
- No separate YouTube Analytics reader was built.

## Open questions

- Whether the retention curve's early reading tracks viewed-vs-swiped-away closely enough that the YouTube Analytics reader is not needed; the provider exposes neither viewed-vs-swiped-away nor engaged views.
