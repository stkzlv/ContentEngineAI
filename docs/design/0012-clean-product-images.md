# 0012. Prefer clean product images over seller infographics

- **Status:** Accepted
- **Issue:** #554
- **Requirements:** REQ-VID-015, REQ-VID-016

## Context

The producer uses the downloaded listing images in listing order (`step_gather_visuals` reads `downloaded_images`). Most are marketing composites with dense text, and captions and the hook headline are drawn over them.

Evidence ([evidence grades](README.md#evidence-grades)):

- When readers suspected AI, trust fell about 50% whether or not the content was AI-made, and adjacent ads lost 14% in purchase consideration: being perceived as automated is the penalty. [B] [PPC Land on Raptive](https://ppc.land/raptive-study-shows-ai-content-cuts-reader-trust-by-half/)
- In a February 2026 survey of 2,250 adults in the US, UK and Australia, 56% saw "AI slop" often, and half of Gen Z had blocked, muted or unfollowed a brand or creator whose content felt like slop. [B] [Sprout Social](https://sproutsocial.com/insights/press/social-media-is-now-the-top-source-for-breaking-news-new-sprout-social-research-finds/)
- Seller listing images are usually marketing composites (dense text, spec icons, stock models), and captions plus a hook headline over them give three layers of text on one frame; no study measures this, it follows from the reuse and minimal-edit rules. [A for the rules, C for the specifics]
- Meta lists borders, captions, speed changes and narration over existing material as not meaningful edits. [A] [Meta](https://about.fb.com/news/2026/03/rewarding-original-creators-on-facebook/)

## Goals

- A product render prefers clean product images over text-heavy seller infographics, using text-heavy ones only when too few clean images exist.
- When the listing has a product video and the profile accepts video, prefer it over stills.

## Non-goals

- Removing an image on a failed judgement. A failed judgement is unknown and never removes an image.

## Design

- `video_settings.image_curation`: `enabled` (default false), `max_text_share` (default 0.15), `min_clean_images` (default 3).
- Score each image once after download with the multimodal judge the stock-relevance step already uses (`src/video/stock_relevance.py`): one call per image asking for the share of the frame covered by overlaid text and whether it is a composite. Cache the score beside the image, so a re-render pays nothing. A failed judgement is unknown and sorts after known scores, the stock judge's rule.
- Order images clean first. Drop images above `max_text_share` while at least `min_clean_images` remain; otherwise keep the least text-heavy ones.
- When the listing has a product video and the profile accepts video, prefer it over stills.
- Record the per-image scores and the chosen order in the state.

**Tests.** A fixture set with known scores is reordered clean first and trimmed only above the minimum; a failed judgement never removes an image; off leaves today's order.

## As built

- The judge has its own `model` setting (default `gemini-2.5-flash`) and borrows the stock judge's concurrency, timeout and API key. On a ten-image smartwatch listing, `gemini-2.5-flash-lite` scored every image 0.22-0.45, counting the watch's own screen as text, under two prompts and a categorical question; `gemini-2.5-flash` with a prompt that excludes on-product text scored the four plain shots 0.0 and the six marketing images 0.2-0.3. The `composite` flag separated the same set on both models and is recorded, not used.
- Curation runs in `gather_visuals` after media validation, and never trims below the image count validation asked for, so it cannot fail a render validation passed.
- The cache sits beside each image as `<image>.text_score.json`, keyed to the file's size and modification time, because the scraper rewrites images under stable names.
- The assembly step shuffles the visuals, so the "clean first" order matters only for which images are kept; the order is recorded but not used.
- Preferring the product video (REQ-VID-016) is not built: profiles that take scraped video already build from the clips through their assembly mode.

## Alternatives considered

None recorded.

## Rollout

Ships off: `video_settings.image_curation.enabled` defaults to false. Set it after the reach-test readout (#540), once a side-by-side review of renders with and without it prefers the curated set, and swipe-away ([0010](0010-first-seconds-metrics.md)) is no worse.

Remove the switch when: `video_settings.image_curation.enabled` has been on in the bundled config for two weekly batches with swipe-away ([0010](0010-first-seconds-metrics.md)) no worse than without it; the key and the uncurated-order path then go in a minor release with a `**Breaking**:` CHANGELOG entry, and `max_text_share` and `min_clean_images` stay as the tuning.

## Open questions

None recorded.
