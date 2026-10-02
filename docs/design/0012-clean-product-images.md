# 0012. Prefer clean product images over seller infographics

- **Status:** Accepted
- **Issue:** #554
- **Requirements:** REQ-VID-015, REQ-VID-016

## Context

The producer uses the downloaded listing images in listing order (`step_gather_visuals` reads `downloaded_images`). Most are marketing composites with dense text, and captions and the hook headline are drawn over them.

The finding comes from [ai-slop-research.md](../ai-slop-research.md), which found that looking fully automated is itself the penalty.

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

## Alternatives considered

None recorded.

## Rollout

Ships off: `video_settings.image_curation.enabled` defaults to false. Set it after the reach-test readout (#540), once a side-by-side review of renders with and without it prefers the curated set, and swipe-away ([0010](0010-first-seconds-metrics.md)) is no worse.

## Open questions

None recorded.
