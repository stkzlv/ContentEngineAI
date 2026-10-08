# Video length: limits, signals and evidence

This page explains how long the pipeline's videos are, what the platforms allow, what their ranking rewards, and what the evidence says about length for product videos and for tutorials. The spoken-length target lives in the narrator profiles in `config/ai_services.yaml` (`script_templates.narrator_profile`, `narrator_profile_topic`); a render's duration follows its voiceover (`REQ-VID-001`). Tutorial lengths by step count are in [the tutorials page](tutorials.md#length). Each finding carries a grade, defined in [the evidence grades](../design/README.md#evidence-grades).

The short answer: no source shows that length by itself makes a video perform. Longer videos collect more total views and watch time, shorter ones get better completion and more likes and shares per view, and studies disagree mostly because they measure different things. The first three seconds decide more than the last ten.

## Platform limits

| Platform | Upload limit | What matters for length |
|---|---|---|
| YouTube Shorts | 3 minutes, square or vertical, since October 15, 2024 | A Short over one minute with any active Content ID claim, manual claims included, is blocked worldwide. Most Shorts library tracks are capped at 90 s, some at 60 s or 30 s. [A] [YouTube Help](https://support.google.com/youtube/answer/15424877) |
| TikTok | 3 minutes for every creator, 5 or 10 minutes for some; the posting API accepts up to 10 minutes | The ads page sets no recommended length (10 minutes, no limit for Spark Ads). [A] [TikTok developers](https://developers.tiktok.com/doc/content-posting-api-media-transfer-guide), [TikTok Ads help](https://ads.tiktok.com/help/article/tiktok-auction-in-feed-ads) |
| Instagram Reels | 3 minutes in the app since January 2025; the API takes 3 s to 15 minutes | Instagram said it would recommend reels up to 3 minutes, and a 2025 creator-event slide advised against going past that. [A] for the API, [B] for the rest. [Instagram API](https://developers.facebook.com/docs/instagram-platform/instagram-graph-api/reference/ig-user/media), [Social Media Today](https://www.socialmediatoday.com/news/instagram-will-recommend-longer-reels/737913/) |

The pipeline's renders stay well under every limit; the limit that binds is the one-minute Content ID rule on YouTube, because a render's background music comes from providers whose tracks may be registered for Content ID, and published renders have run past 60 s. The producer warns, and records it in the run state, when a render with music runs past 60 s (`video_settings.music_claim_ceiling_sec`, `REQ-VID-160`).

## How views are counted

- **YouTube counts a view from the first frame.** "Beginning August 24, 2026, views are counted the moment a video starts to play across all formats", while Partner Program earnings stay on engaged views. Shorts had counted plays and replays since March 2025. [A] [YouTube Help](https://support.google.com/youtube/answer/2991785)
- **Engaged views and "Stayed to watch" are the YouTube length metrics.** YouTube defines engaged views as "how many times viewers stayed to watch past the initial seconds, not including any loops". A raw view count is close to an impression count, so it can't tell a short video that held from one that didn't. [A] [YouTube Help](https://support.google.com/youtube/answer/12220281)
- **Instagram views include replays.** The insights API counts every play, and separately reports average watch time and a skip rate (`reels_skip_rate`). The skip rate is described elsewhere as skips in the first 3 seconds; that definition was not confirmed on the rendered API page. [A] for the fields. [Instagram insights](https://developers.facebook.com/docs/instagram-platform/reference/instagram-media/insights/)
- **TikTok's view definition is not documented** on any official page reachable without a login.

Each render records its final length (`REQ-VID-159`), and the analytics report groups the stored metrics by duration band (`analytics.duration_bands_sec`, `REQ-PUB-084`), so length is judged on the channel's own figures rather than on cross-channel data. The provider doesn't expose engaged views, so YouTube bands read raw views ([design 0010](../design/0010-first-seconds-metrics.md)).

## What the platforms say they rank on

- **TikTok:** finishing a longer video weighs more than a weak signal (2020 statement). [A] [TikTok newsroom](https://newsroom.tiktok.com/en-us/how-tiktok-recommends-videos-for-you)
- **Instagram:** the likelihood to reshare, watch to the end, like and open the audio page (2023); watch time, likes per reach and sends per reach as quoted from Adam Mosseri in 2025. [A], then [B, secondary] [Instagram](https://about.instagram.com/blog/announcements/instagram-ranking-explained)
- **YouTube:** no recommended length. The Shorts product lead declined to name one and described a view as encoding "your intent of watching that thing" (2023). [B] [Search Engine Journal](https://www.searchenginejournal.com/youtube-explains-how-shorts-algorithm-works/494953/)
- **Recommenders correct for length.** Kuaishou's production model compares watch time only among videos of similar duration, because raw watch time favours long videos (20 billion samples). Competing on completion against videos of the same length matters more than length itself (inference). [A] [KDD 2022](https://arxiv.org/abs/2206.06003)

## What the data says

- **Per-view engagement falls with length.** Across 248 million Kuaishou videos, likes per view drop from 9.3% at 10 s to 3.3% at about 1,000 s, and shares from 0.67% to 0.39%, while absolute likes rise with length up to 500 s. Knowledge videos run twice as long as lifestyle ones: a 30 s median against 14-16 s. [B, preprint] [Chen et al. 2024](https://arxiv.org/abs/2410.16058)
- **Most drop-off is early.** In 9.2 million donated TikTok views, 45% were watched to the end and 24% were skipped before a fifth of the video. [A] [Zannettou et al., CHI 2024](https://dl.acm.org/doi/10.1145/3613904.3642433)
- **Shorts median views by length** (108,138 Shorts at least 14 days old, no split by category). [B] [Quso](https://quso.ai/research/youtube-shorts-length)

  | Length | Median views |
  |---|---|
  | Under 10 s | 514 |
  | 11-20 s | 901 |
  | 21-30 s | 542 |
  | 31-45 s | 347 |
  | 46-60 s | 188 |
  | 61-90 s | 186 |
  | Over 90 s | 51 |

- **TikTok reach rises with length, completion falls.** Across 1.1 million TikToks, videos over 60 s got 43.2% more reach and 63.8% more watch time, but the median watch on them was 11.3 s. [B] [Buffer](https://buffer.com/resources/longer-tiktoks-get-more-views-data/)
- **One ad finding.** TikTok's 2021 analysis of "thousands of ads" found 21-34 s ads had a 280% lift in conversion. Paid ads, no sample or method given. [B] [TikTok](https://ads.tiktok.com/business/en-US/blog/creative-that-drives-conversions)
- **One task per video.** A meta-analysis of 56 studies found small to medium gains in retention and transfer from splitting instruction into segments, and in edX data the 0-3 minute videos held three quarters of sessions past 75%. [A] [Rey et al. 2019](https://doi.org/10.1007/s10648-018-9456-4), [Guo et al. 2014](https://dl.acm.org/doi/10.1145/2556325.2566239)

Why the datasets disagree: views and total watch time rise with length, completion and engagement per view fall with it, and bigger, better-produced accounts post longer videos, which confounds every cross-channel comparison. The education research transfers only partly, since a learner chose the video and can rewind, while a feed viewer swipes for free.

## What top tech creators post

Median length of the latest Shorts on each channel, measured with yt-dlp in October 2026 (372 Shorts). [A, primary measurement]

| Channel | Kind | Shorts | Median | Over 60 s |
|---|---|---|---|---|
| Dave2D | Product | 7 | 37 s | 0% |
| ShortCircuit | Product | 30 | 37 s | 0% |
| Mrwhosetheboss | Product | 30 | 50 s | 7% |
| MKBHD | Product | 30 | 55 s | 43% |
| TechBurner | Product | 7 | 58 s | 29% |
| Zone of Tech | Product | 30 | 79 s | 70% |
| Unbox Therapy | Product | 30 | 108 s | 87% |
| JerryRigEverything | Product | 30 | 121 s | 93% |
| Kevin Stratvert | Tutorial | 30 | 33 s | 10% |
| iDB | Tutorial | 28 | 48 s | 0% |
| HowToMen | Tutorial | 30 | 82 s | 83% |
| Brandon Butch | Tutorial | 30 | 88 s | 100% |
| TechSpurt | Tutorial | 30 | 92 s | 83% |
| AppleInsider | Tutorial | 30 | 97 s | 100% |

Across all product channels the median is 59 s and 47% run past a minute; across tutorial channels, 76 s and 63%. Within 9 of the 12 channels with at least 28 Shorts, the most-viewed third of Shorts ran longer than the least-viewed third. All of these channels show a person and have large audiences who chose to follow them, so their length is evidence of what an established creator's viewers tolerate, not of what a faceless channel's feed viewers will watch.

## What the pipeline targets

| Video | Target | Why |
|---|---|---|
| Product | 30-40 s of speech, about 75-100 words (`REQ-CNT-142`) | Inside the range the evidence supports. Shorts data across all channels peaks shorter (11-20 s) and per-view engagement falls with length, so a 20-30 s product target is the first length to test. Keep under 60 s while the music can carry a claim. |
| Short profile | 15-30 s, about 50-60 words (`REQ-VID-095`) | A test canvas for hooks and the shorter target. |
| Tutorial, one setting | 15-30 s | One task per video; length follows the steps ([tutorials](tutorials.md#length)). |
| Tutorial, multi-step | 40-75 s | Tech-help creators run 33-97 s; past about 90 s, split into a series. |

Planned ([design 0025](../design/0025-video-length.md), `REQ-CNT-163`): the spoken-length target comes from config per content type and profile, with today's text as the default, so a shorter product target can be tested after the reach-test readout without editing the prompts.

## Gaps in the evidence

- **No public dataset separates product videos or tech help by length** on any short-video feed, and none covers faceless channels.
- **The bands are B-grade at best.** They come from cross-channel datasets confounded by account size, and differ by the metric they report.
- **Several sources predate 2025**, before YouTube changed how it counts views; the TikTok ranking statement is from 2020.
- **Your own channel decides.** Compare duration bands on the same account, interleaved by day, reading engaged views or "Stayed to watch" on YouTube and skip rate and average watch time on Instagram, not raw views.
