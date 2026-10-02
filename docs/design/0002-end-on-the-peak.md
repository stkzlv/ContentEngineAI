# 0002. End on the peak, with an optional seamless loop

- **Status:** Accepted
- **Issue:** #543
- **Requirements:** REQ-VID-011

## Context

`outro_duration_sec: 1.0` (`config/core.yaml`) is added after the voiceover, partly to avoid AAC truncating the last word, and the music fades out over `music_fade_out_duration` (3.0 s).

The work relates to roadmap item 1.8 in [the roadmap](../roadmap.md). Evidence ([evidence grades](README.md#evidence-grades)):

- One creator found a one-second retention cliff at the end of a Short, and cutting it took retention from 83% to 88%; nobody has measured it at scale. [C] [Creator Science podcast](https://podcast.creatorscience.com/jenny-hoyos/)
- Since 31 March 2025 every Shorts play and replay counts as a view, while engaged views exclude loops, so a seamless loop raises the replay signal. [A] [YouTube Help community thread](https://support.google.com/youtube/thread/333869549)

## Goals

- With `peak`, a render ends on its last spoken word with no silent or fading tail.
- With `loop`, its last frame also matches its first, so a replay reads as continuous.

## Non-goals

- Moving the CTA. The CTA is spoken, so it stays the last sentence. `peak` removes only the silence after it.

## Design

- A per-profile `ending` setting: `outro` (default, today's behaviour), `peak` or `loop`.
- `peak`: the video ends `peak_margin_sec` after the last spoken word (default 0.25 s, enough for the AAC frame padding the outro was protecting; verify on a render). The music is cut at the same point with a fade no longer than the margin, not faded to silence over three seconds.
- `loop`: `peak`, plus the last visual segment ends on the frame 0 composition (the first image at its starting scale, without the hook overlay), so a replay reads as continuous. Implement by reusing the first image as the last segment's source and matching its crop.

**Tests.** A `peak` render's duration equals the voiceover duration plus the margin within one frame; the last spoken word is intact in a Whisper transcript of the output; a `loop` render's last and first frames differ by less than a set mean pixel difference; `outro` produces today's command.

## Alternatives considered

None recorded.

## Rollout

Ships off: `ending` defaults to `outro`, today's behaviour. A profile turns it on by setting `ending` to `peak` or `loop` after the reach-test readout (#540), once average percentage viewed rises and the last-word transcript check still passes.

## Open questions

- Whether the 0.25 s default margin covers the AAC frame padding; verify on a render.
