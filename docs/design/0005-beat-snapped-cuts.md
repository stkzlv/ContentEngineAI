# 0005. Snap visual cuts to music beats

- **Status:** Held
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

## As built

- The setting is `video_settings.beat_snap` with `enabled` and `window_ms`; the minimum segment is the existing `min_visual_segment_duration_sec`, not a key of its own.
- A cut is the middle of a crossfade. Moving it lengthens one neighbour and shortens the other by the same amount, so the timeline's length holds; a move that would lengthen a video clip is skipped, since a clip may have no frames to spare.
- Beats are detected in the assembler, not after `download_music`, so a resumed `assemble_video` run gets them too; the cache beside the track makes the second render free.
- On a bundled lofi track librosa found 253 beats, and five of seven cuts on a 3.1 s timeline moved onto a beat; the other two had none within 150 ms and stayed. On a click track at 120 BPM, librosa's beats sat within 35 ms of the clicks, bounded by its 23 ms frame hop.
- Each render records how many cuts beat snapping moved in `state/render_choices.jsonl` (`beat_snap_moved`), empty when snapping did not run (#659).

## Alternatives considered

None recorded.

## Rollout

Ships off: `video_settings.beat_snap.enabled` defaults to false. Set it once a blind listening comparison prefers it, or completion improves. The evidence is a lab result, so treat it as low priority.

Remove the switch when: `video_settings.beat_snap.enabled` has been on in the bundled config for two weekly batches with completion no worse than without it; the key and the unsnapped path then go in a minor release with a `**Breaking**:` CHANGELOG entry, and `window_ms` stays as the tuning.

## Open questions

None recorded.
