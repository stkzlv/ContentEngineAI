# 0019. Explanatory graphics

- **Status:** Accepted
- **Issue:** #561
- **Requirements:** REQ-VID-124

## Context

The detailed design is in the issue body, from sections 2 and 8 to 10 of [tutorial-video-best-practices.md](../research/tutorial-video-best-practices.md). This doc records how it fits the [rules every design follows](README.md#rules-that-apply-to-every-design).

## Goals

- A tutorial carries templated graphics, each tied to a script fact.

## Non-goals

- Failing a render on a graphic. A failed graphic is skipped.
- More than one graphic on screen at a time.

## Design

- HTML and CSS templates rendered to transparent images in the existing Chromium and composited with FFmpeg overlays, from a validated JSON spec.
- One graphic at a time, every graphic tied to a script fact.
- A DOM overflow and safe-zone check, and a failed graphic skipped.
- Behind `video_settings.graphics.enabled` (default false).

## Alternatives considered

None recorded.

## Rollout

Ships off: `video_settings.graphics.enabled` defaults to false. Set it after the reach-test readout (#540). The switch gains a check in `tests/test_reach_test_holdout.py` when it lands.

## Open questions

None recorded.
