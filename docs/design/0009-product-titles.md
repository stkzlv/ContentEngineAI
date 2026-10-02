# 0009. YouTube titles for products

- **Status:** Accepted
- **Issue:** #550
- **Requirements:** REQ-PUB-008

## Context

Product videos go to YouTube with the store listing title, cut to fit (the metadata validation warns on `data.json` titles over 100 characters).

Evidence ([evidence grades](README.md#evidence-grades)):

- Across 10,000 trending Shorts, the median title was about 8 words, 20-40 characters. [B]
- YouTube's title and thumbnail testing excludes Shorts, and testing three cuts of a Short is announced for 2027, so a title change is judged on the pipeline's own analytics. [A] [YouTube Help](https://support.google.com/youtube/answer/16391400), [TechCrunch](https://techcrunch.com/2026/09/23/youtube-adds-new-creator-tools-like-video-a-b-testing-dynamic-thumbnails-and-live-dubbing/)
- Not supported: A/B testing Shorts titles. The 2027 tool tests cuts, not titles.

## Goals

- Product videos publish to YouTube with the generated title, within `title_length_max`, keyword first, not the store listing title.

## Non-goals

- The Instagram hashtag range. Issue #550 also aligned the three places that set it (`platform_metadata.instagram` and `platform_metadata_config` in `config/ai_services.yaml`, and `PLATFORM_LIMITS[Platform.INSTAGRAM]` in `src/publisher/models.py`); that half is superseded by the per-platform hashtag rework (#567) and is out of scope here.

## Design

- Find the path that sets the YouTube title for a product render (the metadata loader reads `title` from the product record) and have it use the generated YouTube title from the platform metadata step, bounded by `title_length_max`, keyword first.
- Fall back to a shortened listing title only when generation failed.

**Tests.** A product render's YouTube payload title is the generated one and within the maximum.

## Alternatives considered

None recorded.

## Rollout

Titles are held constant across both reach-test arms by the protocol. The title fix applies only to the product arm, so it waits for the reach-test readout (#540).

## Open questions

- The spec names no setting that holds the title fix off until the readout.
