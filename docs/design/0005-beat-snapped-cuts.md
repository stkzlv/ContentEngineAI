# 0005. Snap visual cuts to music beats

- **Status:** Accepted
- **Issue:** #546
- **Requirements:** REQ-VID-013

## Context

Segment boundaries come from the assembly strategy and the voiceover length; the music is chosen independently in the `download_music` step.

Evidence ([evidence grades](README.md#evidence-grades)):

- Cuts on accented downbeats, even unnoticed ones, increase perceptual pleasure in a lab study. [A] [Neuroscience Letters](https://www.sciencedirect.com/science/article/pii/S030439402200180X)

## Goals

- Visual cuts move to the nearest music beat within a small window.

## Non-goals

- Changing caption timing. Boundaries move, the voiceover does not, so caption timing is untouched.
- Making librosa a required dependency. It is optional; without it the option warns and does nothing.

## Design

- `video_settings.beat_snap`: `enabled` (default false), `window_ms` (150), `min_segment_sec` (the profile's existing minimum).
- Detect beats once per track after the music is downloaded, with `librosa.beat.beat_track`, and cache them beside the track (`<track>.beats.json`).
- Before building the `concat` or `xfade` offsets, move each boundary to the nearest beat within the window, skipping any move that would break the minimum segment length or push a boundary past the voiceover end.

**Tests.** With a synthetic click track at a known tempo, most moved boundaries land within 30 ms of a beat; no segment drops below the minimum; a missing librosa leaves boundaries unchanged with a warning; off produces today's offsets.

## Alternatives considered

None recorded.

## Rollout

Ships off: `video_settings.beat_snap.enabled` defaults to false. Set it once a blind listening comparison prefers it, or completion improves. The evidence is a lab result, so treat it as low priority.

## Open questions

None recorded.
