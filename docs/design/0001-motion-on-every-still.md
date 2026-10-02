# 0001. Motion on every still

- **Status:** Accepted
- **Issue:** #542
- **Requirements:** REQ-VID-010

## Context

`_build_ken_burns_filter` in `src/video/assembler/visual_builder.py` applies a settle-zoom to the first image only. Later stills are static.

The technique comes from [creator-research.md](../creator-research.md).

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

## Alternatives considered

- **`zoompan`.** Rejected: it rounds positions to whole pixels and shudders on slow moves.

## Rollout

Ships off: `still_motion.enabled` defaults to false in every profile. A profile turns it on by setting `still_motion.enabled` after the reach-test readout (#540), once swipe-away and completion (#551) on a batch with motion are no worse than without.

## Open questions

None recorded.
