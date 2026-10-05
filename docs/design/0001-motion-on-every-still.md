# 0001. Motion on every still

- **Status:** Held
- **Issue:** #542
- **Requirements:** REQ-VID-010

## Context

`_build_ken_burns_filter` in `src/video/assembler/visual_builder.py` applies a settle-zoom to the first image only. Later stills are static.

Evidence ([evidence grades](README.md#evidence-grades)):

- YouTube's inauthentic-content policy names "image slideshows, templated storylines, or scrolling text with minimal or no narrative". [A] [YouTube inauthentic-content policy](https://support.google.com/youtube/answer/1311392)
- TikTok's Creator Rewards criteria exclude "low-quality images, or slide videos". [A] [TikTok Creator Rewards eligibility](https://www.tiktok.com/creator-academy/article/eligibility)
- TikTok's For You feed excludes low-quality or minimally edited content, under guidelines effective 24 September 2026. [A] [TikTok For You feed standards](https://www.tiktok.com/community-guidelines/en/fyf-standards)
- Instagram counts "unique text, creative edits, and voiceover" as original since 30 April 2026, and watermarks and speed changes as not. [A] [TechCrunch, 30 April 2026](https://techcrunch.com/2026/04/30/instagram-restricts-reach-of-content-aggregators-in-new-crackdown/)
- How long a still can hold before viewers leave is unmeasured. [C]

## Goals

- Every still image carries slow, jitter-free motion.
- The direction varies per image and per product, so consecutive stills differ and two products differ.

## Non-goals

- Changing the first image's settle-zoom. It stays as it is; this design covers the rest.

## Design

- A per-profile `still_motion` block in `video_production.yaml`: `enabled` (default false), `moves` (the pool: `push_in`, `pull_out`, `pan_left`, `pan_right`, `pan_up`), `max_zoom` (default 1.15) and `min_zoom` (1.0).
- Per still, draw a move from the pool with the seed `<product_id>:motion:<index>`, so consecutive stills differ and two products differ.
- Implement with an upscale then an animated `crop` (sub-pixel) and a final `scale` to the frame size.
- Respect the image band and the caption safe zone the assembler already computes, so motion never pushes the product under the captions.

**Tests.** The filter string for a three-still render carries a motion clause per still, with at least two distinct moves; the off state produces today's filter graph byte for byte; a rendered test clip shows frame-to-frame change on every still with no single-pixel oscillation in the motion path.

## As built

- The block lives on `video_settings.still_motion`; a profile replaces it as a whole.
- Each still moves inside its own image box, not the whole frame: a per-frame `scale` grows the image by an even number of pixels and a `crop` cuts the box back out. The box, the band and the caption zone are the ones the assembler already computed.
- The crop's centring offset is computed from the zoom expression, because the crop keeps the input size it was set up with. With `exact=1` the offset never snaps to the chroma grid, so the centre holds still on odd box sizes.
- A draw that repeats the previous still's move takes the next move in the pool.
- The first image keeps its settle-zoom where `first_frame_pre_motion` is on, and moves like the rest where it is off.

## Alternatives considered

- **`zoompan`.** Rejected: it rounds positions to whole pixels and shudders on slow moves.

## Rollout

Ships off: `still_motion.enabled` defaults to false in every profile. A profile turns it on by setting `still_motion.enabled` after the reach-test readout (#540), once swipe-away and completion (#551) on a batch with motion are no worse than without.

Remove the switch when: `still_motion.enabled` has been on in the bundled profiles for two weekly batches with swipe-away and completion (#551) no worse than in the batches without motion; the key and the static-still path then go in a minor release with a `**Breaking**:` CHANGELOG entry.

## Open questions

None recorded.
