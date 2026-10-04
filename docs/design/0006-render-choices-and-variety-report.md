# 0006. Record render choices and report output variety

- **Status:** Implemented
- **Issue:** #547
- **Requirements:** REQ-PUB-083

## Context

`pipeline_state.json` records `script_template`, `cta`, the voice profile, `signoff` and some subtitle choices; the registry records `content_format`.

Every drawn choice in the other designs goes into `pipeline_state.json` (see [the rules every design follows](README.md#rules-that-apply-to-every-design)). This design turns those records into a variety report, and [0010](0010-first-seconds-metrics.md) segments metrics by them.

Evidence ([evidence grades](README.md#evidence-grades)):

- YouTube renamed "repetitious" to "inauthentic content" in July 2025; similar or repetitive content across uploads is its central category, and the policy names "generic or unoriginal templates giving the impression of mass production" while channels using AI stay eligible. [A] [YouTube inauthentic-content policy](https://support.google.com/youtube/answer/1311392)
- Journalists' reconstructions of YouTube's detection name upload frequency against production complexity, format similarity, and uniform titles and descriptions. [C] [Tech Times](https://www.techtimes.com/articles/320629/20260715/youtube-wiped-35m-subscribers-over-ai-slop-now-its-judging-your-taste.htm)
- TikTok's For You feed excludes reused content without creative edits and low-quality or minimally edited content. [A] [TikTok For You feed standards](https://www.tiktok.com/community-guidelines/en/fyf-standards)
- Automation used to send repetitive content is a TikTok spam rule. [A]
- TikTok's feed avoids consecutive videos from the same creator or with the same sound. [A] [TikTok transparency center](https://www.tiktok.com/transparency/en/recommendation-system)
- Since 30 April 2026, Instagram accounts that mainly post content they did not create or meaningfully edit are not recommended to non-followers; Meta asks for "fresh information, analysis, or substantial improvements". [A] [TechCrunch, 30 April 2026](https://techcrunch.com/2026/04/30/instagram-restricts-reach-of-content-aggregators-in-new-crackdown/)
- Mass producibility is one of three defining features of AI slop. [B] [arXiv 2601.06060](https://arxiv.org/abs/2601.06060)

## Goals

- Every choice that shapes a render is recorded.
- A report shows their distribution over recent renders, with an alert when one value dominates or two scripts are near-identical.

## Non-goals

- Blocking a script for similarity. The check warns, it does not block.

## Design

- Record every choice that shapes a render in the state: hook archetype, caption template, music track id, motion moves, transitions, effect variants, voice chain. Carry them into the published-products registry at publish time, as `content_format` is.
- A `variety` report (a report type beside the existing analytics reports): for the last N published renders (default 14), the distribution per dimension, and an alert when one value exceeds a share threshold (default 60%) where the pool has more than one option.
- A script similarity check: character 5-gram Jaccard similarity between the generated script and each of the last N scripts, with a warning above a threshold (default 0.5) logged at generation time and counted in the report. Warn, do not block.

**Tests.** A render records every listed choice; the report flags a dimension dominated by one value in a fixture; two near-identical scripts cross the similarity threshold and two unrelated ones do not.

## As built

Where the shipped feature differs from the design above, and why:

- **Where the choices live.** Each finished render appends a row to `state/render_choices.jsonl` under the outputs root rather than adding columns to the published-products registry. The report then covers what was rendered, published or not, and needs no change to the registry's format; a consumer such as [0010](0010-first-seconds-metrics.md) joins on the product id.
- **Window.** The last N rendered videos, not published ones, for the same reason (`--last`, default 14).
- **Dimensions.** Script template, pillar, CTA, hook headline, voice profile and voice, caption engine and pycaps template, music track, cold-open variant, assembly mode, pre-motion and transition duration. There is no hook archetype in the pipeline, so the hook headline stands for it; the FFmpeg engine's per-product ASS effect and a voice chain are not recorded, since the bundled engine is pycaps, whose effect is its template, and no voice chain exists yet.
- **The report** is `python -m src.video.render_choices`, not a report type of the analytics command, because it reads render records rather than platform figures.

The similarity check, its threshold, and the warning logged at generation time follow the design.

## Alternatives considered

None recorded.

## Rollout

Measurement only. It doesn't change rendered output, so it ships on and can land before the reach-test readout (#540).

## Open questions

None recorded.
