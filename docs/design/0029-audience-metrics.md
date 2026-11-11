# 0029. Audience metrics

- **Status:** Implemented
- **Issue:** #781
- **Requirements:** REQ-PUB-162, REQ-PUB-163, REQ-PUB-164, REQ-PUB-165

## Context

The analytics sweep stores per-post views and interaction counts. It stores nothing about the audience itself: whether viewers follow, how much of a video they watch, or whether they look up the profile. Those are the signals for judging a recurring format over weeks. They are read per account and per day, so they need no post-by-post attribution.

A read-only check of the publishing provider's API found these available and unused [B]:

- Daily follower counts per connected account (`accounts/follower-stats`), a history by date for every platform.
- YouTube per-video daily stats, with `averageViewPercentage`, `averageViewDuration` and `subscribersGained` per day. These carry real values, unlike `averageViewDuration` in the per-post analytics, which comes back as 0 (REQ-PUB-084).
- Instagram account insights, with `profile_views` and `profile_links_taps` as totals over a range. The API refuses a daily series for these two (`only support metricType=total_value`).
- Daily metrics per platform. The provider groups them by the day a post was published, with each post's figures as of the call, not by the day the views happened.

## Goals

- Store daily follower counts per platform, so that net follows over any range are known.
- Store YouTube average view percentage, average view duration and subscribers gained per post.
- Store Instagram profile views and link taps per day.
- Report, for a date range: net follows per 1,000 views per platform and summed, shares plus saves and comments per 1,000 views, the YouTube median average view percentage, and Instagram profile views per 1,000 views.

## Non-goals

- TikTok watch time and completion, YouTube viewed-vs-swiped-away and returning viewers: the provider doesn't expose them (#551).
- Attributing follows to single posts.

## Design

- **Per post.** For each swept post with a YouTube leg, the sweep reads the video's daily stats from its publish date to today. It stores view-weighted `avgViewPercentage` and `avgViewDurationSec`, and the summed `subscribersGained`, in that post's `platform_metrics["youtube"]` beside the existing counts. The existing per-field merge keeps a stored value when a later reading is unknown. With no views to weigh by, each field is unknown. The stats come out of a YouTube Analytics quota the provider shares between its customers and resets at midnight Pacific time; once it refuses a call with a rate limit, every later call in that sweep would be refused too, so the sweep stops asking and the stored values stand.
- **Per account.** After the per-post figures are saved, the sweep stores `outputs/state/audience_metrics.json`, which holds three maps:
  - `followers`: platform to day to count, for the last 30 days. Two accounts on one platform add up; one outputs dir holds one API account's figures, as its other state files do.
  - `daily`: platform to publish day to views, post count and the interactions the platform exposes, for the last 30 days. The provider fills a field a platform doesn't expose with 0 (YouTube shares and saves, TikTok saves), so for the platforms `QUALITY_METRICS` describes, shares and saves are kept only where exposed; views and comments always stay.
  - `instagram_profile`: day to profile views and link taps, for the first connected Instagram account. These are read as one-day windows over the last 7 days, since Instagram reports up to 48 hours late.
  
  Each sweep overlays its readings on the file. A day the provider omits keeps its stored value, and an unreadable file is refused rather than overwritten. Each call is guarded, so one endpoint failing keeps the others' readings and never fails the sweep.
- **Report.** `python -m src.publisher.late audience --since D --until D` reads the stored files and makes no network call.
  - Net follows are the count at `--until` minus the count the day before `--since`.
  - Views are those of posts published in the range, as the provider groups them.
  - A rate whose inputs are missing prints `-`, and the summed row is unknown when any platform's input is.
  - Shares plus saves counts the fields a platform exposes, so it is unknown on YouTube.
  - Instagram profile views per 1,000 views needs profile figures for every day of the range: a partial sum over the whole range's views would read low.

## Alternatives considered

- **Views by the day they happened, from the per-post timelines.** The sweep doesn't store timelines, only day-N figures. Storing them would be a larger schema change for a small gain over a weeks-long window. The report names its basis instead.
- **Instagram `follows_and_unfollows`.** It returned 0 over a month in which the follower count moved, so the daily follower count is used for every platform.

## Rollout

No switch: the sweep only reads and stores figures, and changes no render or post. The daily analytics timer picks it up on the next run.

## Open questions

None recorded.
