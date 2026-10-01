# 0018. A visual per step

- **Status:** Accepted
- **Issue:** #560
- **Requirements:** REQ-VID-123

## Context

Visual planning is one keyword search over the script. The detailed design is in the issue body, from sections 2 and 8 to 10 of [tutorial-video-best-practices.md](../tutorial-video-best-practices.md). This doc records how it fits the [rules every design follows](README.md#rules-that-apply-to-every-design).

## Goals

- Each tutorial step is shown as it is spoken.

## Non-goals

- Stock footage for the steps themselves. Stock covers only the opening symptom.

## Design

- Visual planning moves from one keyword search over the script to one plan per step, timed from the Whisper word timings.
- Source order: a real capture, a UI mockup with exact labels, a diagram, and stock only for the opening symptom.
- Behind `video_settings.step_visuals.enabled` (default false).

## Alternatives considered

None recorded.

## Rollout

Ships off: `video_settings.step_visuals.enabled` defaults to false. Set it after the reach-test readout (#540). The switch gains a check in `tests/test_reach_test_holdout.py` when it lands.

## Open questions

None recorded.
