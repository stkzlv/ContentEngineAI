# 0009. YouTube titles for products

- **Status:** Implemented
- **Issue:** #550
- **Requirements:** REQ-PUB-008, REQ-CNT-149

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

None.

## As built

Built and held off behind `description_settings.short_product_titles` (v0.184.0). The generated YouTube title in optimized mode already came from the model within `title_length_max`; the listing title reached YouTube through unified mode, the bundled default, which wrote `product.title` into `metadata.json`. With the switch on, unified mode writes `<Keyword>: <hook headline>`, the headline alone when it already names the keyword, or the listing title's first clause when there is no headline, whichever first fits `title_length_max`, cut on a word otherwise. Building it from the hook headline makes the title and the burned-in hook make the same promise (REQ-CNT-149) with no extra model call. The smartwatch render's title would read "Smartwatch that takes calls from your wrist" (43 characters) instead of its 120-character listing title.
