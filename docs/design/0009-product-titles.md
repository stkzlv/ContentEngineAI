# 0009. YouTube titles for products

- **Status:** Accepted
- **Issue:** #550
- **Requirements:** REQ-PUB-008

## Context

Product videos go to YouTube with the store listing title, cut to fit (the metadata validation warns on `data.json` titles over 100 characters).

The finding comes from [creator-research.md](../research/creator-research.md).

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
