# 0025. Video length: recorded, bounded and configurable

- **Status:** Implemented
- **Issue:** #705
- **Requirements:** REQ-VID-159, REQ-VID-160, REQ-CNT-163

## Context

A render's length follows its voiceover (`REQ-VID-001`), and the voiceover follows a spoken-length target written into the narrator profiles: 30-40 seconds, about 75-100 words (`REQ-CNT-142`). Three things are missing:

- **Nothing records the final duration** where the analytics can read it. Length can only be judged against raw views, which since August 2026 YouTube counts from the first frame.
- **Nothing bounds a render with music.** Published topic renders have run to 77 s and product renders to 67 s, and step-list tutorials can reach about 75 s ([design 0017](0017-tutorial-step-lists.md), [tutorials](../explanation/tutorials.md#length)). Background music comes from Jamendo, Freesound or local files, and the pipeline doesn't know whether any track is registered for Content ID.
- **The target is fixed text.** The short profile can't size its script (`REQ-VID-095`), and testing a shorter product target means editing the profile text.

Evidence ([evidence grades](README.md#evidence-grades), detail in [video length](../explanation/video-length.md)):

- "Any Short that is over one minute in duration with an active Content ID claim of any type, including manual claims, will be blocked globally on YouTube." [A] [YouTube Help](https://support.google.com/youtube/answer/15424877)
- "Beginning August 24, 2026, views are counted the moment a video starts to play across all formats"; engaged views remain. [A] [YouTube Help](https://support.google.com/youtube/answer/2991785)
- Likes per view fall from 9.3% at 10 s to 3.3% at about 1,000 s across 248 million Kuaishou videos. [B, preprint] [Chen et al. 2024](https://arxiv.org/abs/2410.16058)
- Median Shorts views peak at 11-20 s and fall past 20 s across 108,138 Shorts. [B] [Quso](https://quso.ai/research/youtube-shorts-length)
- Recommenders compare watch time within a duration group. [A] [KDD 2022](https://arxiv.org/abs/2206.06003)

## Goals

- Record each video's final duration in the run state and the publish history.
- Warn when a render with music runs past a configured ceiling, 60 s by default.
- Segment the analytics report by duration band.
- Take the spoken-length target from config per content type and profile, rendering today's text at the default.

## Non-goals

- **Changing the default target.** A shorter product target is a test after the reach-test readout, not a default this design sets.
- **Cutting or trimming a render to fit.** The warning reports; the script decides length.
- **Clearing music for Content ID.** The warning names the risk; which tracks get claimed is an open question.

## Design

- **Duration recorded (`REQ-VID-159`).** After assembly the producer probes the final video (the probe `REQ-VID-002` already runs) and writes `video_duration_sec` into `pipeline_state.json`. The publisher copies it into the publish history beside the post ids, so the analytics sweep can join it to each post without reading the outputs tree.
- **Ceiling warning (`REQ-VID-160`).** `video_settings.music_claim_ceiling_sec` (default 60; 0 turns the check off). When a render carries background music and its duration exceeds the ceiling, the producer logs a warning naming the duration, the ceiling and the music source, and records `over_music_claim_ceiling: true` in the run state. It never fails the render.
- **Duration bands (`REQ-PUB-084`).** The analytics report groups stored metrics by `analytics.duration_bands_sec` (default `[20, 30, 45, 60]`: under 20, 20-30, 30-45, 45-60, over 60), per content format, with the post count and the median of each metric. A post with no recorded duration falls in an "unknown" band.
- **Configurable target (`REQ-CNT-163`).** `script_templates.target_length` holds `seconds` and `words` ranges per content type (`product`, `topic`), and a profile may override them (`target_length` in a video profile). The narrator profiles carry `{TARGET_SECONDS}` and `{TARGET_WORDS}` placeholders filled from it. The defaults (30-40 s, 75-100 words) render the profile text byte-identical to today. The script lint's `target_duration_sec` reads the upper bound when set.

**Tests.** The run state and publish history round-trip the duration; a render fixture past the ceiling with music logs and records the warning, one without music doesn't, and a ceiling of 0 skips the check; the report places fixtures in the right bands, unknown included; the default target renders the narrator profiles byte-identical to the shipped text, and a profile override reaches the prompt the script step sends.

## Alternatives considered

- **Cap renders at 60 s.** Cuts off the end of a script, its call to action included; the script should be written shorter instead.
- **Drop the music past 60 s.** Removes the claim risk, but changes output during the reach-test hold and costs every long tutorial its music. Kept as an option for later if claims appear.
- **Read duration from the platforms.** TikTok and Instagram report it unevenly through the provider; the render already knows it exactly.

## Rollout

The duration record, the warning and the report are measurement and ship on: they change no rendered output. The configurable target ships with today's values, so the prompt stays byte-identical ([decision 0002](../decisions/0002-output-changes-ship-off-by-default.md)). After the reach-test readout, a 20-30 s product target runs as its own arm against 30-40 s, interleaved by day, read on engaged views and "Stayed to watch" on YouTube and skip rate and average watch time on Instagram.

Remove the switch when: not applicable; the ceiling and the target are lasting operator settings.

## As built

- **Where the duration lands.** The producer writes `video_duration_sec` into the run state and into the render's row in the render-choices record (`state/render_choices.jsonl` under the outputs root), not into the publish history. The quality report already joins posts to render choices through that record, and it survives the product directory's cleanup after a publish, so the publisher needs no copy of its own.
- **Measured after assembly.** The length is probed when assembly finishes, before the caption burn replaces the file; on a checked render the two differed by 0.02 s, under a frame. A re-render clears the previous length and flag before recording its own.
- **Mean, not median.** The duration band is one more segment of the existing quality report, which shows the mean of each metric with its post count; a band uses the same figures as every other segment.
- **The lint reads an override only.** `script_validation.lint.target_duration_sec` stays the lint's word cap unless a video profile sets `target_length`, whose upper seconds then replace it; the content-type defaults leave the lint as configured.

## Open questions

- Whether Jamendo or Freesound tracks draw Content ID claims on the channel's Shorts. The channel's copyright tab answers it; if none appear, the ceiling can rise.
- Whether the provider exposes YouTube engaged views; without it the YouTube band report reads raw views ([design 0010](0010-first-seconds-metrics.md)).
