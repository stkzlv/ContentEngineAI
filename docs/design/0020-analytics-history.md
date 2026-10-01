# 0020. Analytics history

- **Status:** Accepted
- **Issue:** none
- **Requirements:** REQ-PUB-072 to REQ-PUB-084

## Context

A module owns per-post performance history: it captures metrics on a schedule, stores them locally, and reports by the dimensions the pipeline already varies (content format, pillar, template, voice profile, hook variant). Day-2 and day-7 views and a 30-day durability ratio per post have shipped, with a command that captures them and ranks by durability.

That work began from an assumption that turned out to be false, and the correction is the reusable part. The item originally read that the scheduler returns a cumulative per-post timeline, "so any day-N figure is a lookup rather than a scheduled job". Measured against the live API, the timeline stops reaching back after roughly five weeks, so day-2 and day-7 are available only while the post is young. They are a scheduled job or they are nothing, which is why this design is built around capture cadence rather than querying on demand.

The two figures answer different questions. A 7-day window captures the launch curve and can't tell content that accumulates search traffic from content that spiked and stopped. Anything claiming a format is evergreen needs the 30-day-plus ratio; see [the tutorial guide](../tutorial-video-best-practices.md).

## Goals

- The local store is the system of record, and the providers are sources feeding it.
- Capture follows each source's expiry, so no figure is lost while it was still reachable.
- Reports segment performance by content format, and by the other dimensions the pipeline varies.

## Non-goals

- Going direct to each platform's API as a first step (see Alternatives).
- Averaging retention across platforms that don't all report it.

## Design

**Own the history.** Every upstream expires. The scheduler's per-post timeline stops reaching back after roughly five weeks: past that, a post's rows begin at a recent date instead of at publication, and `from_date` doesn't widen it. The same shape appears elsewhere: one platform's post data freezes a year after publication, and its watch-time fields empty out after a week without engagement. A figure not captured while it was reachable can't be recovered.

**Capture cadence follows expiry, not convenience.** Most of a short-form post's views arrive within the first day or two, so the early curve needs frequent sampling and the tail very little. Two constraints shape the schedule: one platform's analytics rows take 48-72 hours to finalise, so a same-day pull records figures still settling; and one platform exposes only lifetime counters with no daily series, so its day-N figures exist only as differences between snapshots this module took. Readings are stored append-only and deltas are derived rather than overwritten.

**Start with the scheduler, because it is already authenticated.** It exposes more than the pipeline reads: per-video daily views for one platform, account insights and demographics for another, plus content decay, posting frequency and best time to post. The first useful version reads those through the existing client. Only the affiliate-program and link-in-bio halves need their own sources.

**Retention is the metric worth adding next, and it is platform-asymmetric.** One platform's API exposes a full retention curve per video, another reports average watch time and the share of viewers who leave within three seconds, and a third offers no watch-time signal on its generally available API. Any hook comparison is therefore two-platform evidence, and the module says so rather than averaging across the gap. First-seconds metrics are [design 0010](0010-first-seconds-metrics.md).

## Alternatives considered

- **Platform APIs directly.** They hold far longer history than the scheduler and are the only route to retention data, but each needs its own OAuth app, and one requires a business account plus an access application. Deferred: build the store first, then decide per platform whether the extra metrics justify an integration.
- **Paid aggregators.** A couple are usable by developers, but they don't remove the need for a local snapshot table, which is the part that solves history.

## Rollout

Measurement only, so it ships on. The capture runs as a daily timer.

## Open questions

- Which platforms justify a direct integration once the store is in place.
