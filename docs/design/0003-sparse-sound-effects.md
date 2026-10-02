# 0003. Sparse event sound effects

- **Status:** Accepted
- **Issue:** #544
- **Requirements:** REQ-VID-012

## Context

The mix is voiceover, music and (optionally) the signature sting, built by `AudioFilterBuilder.build_mix`.

The technique comes from [creator-research.md](../creator-research.md).

## Goals

- Sparse effects mark a few beats (a transition, the reveal, the call to action), drawn per product from a pool, capped per 10 seconds and mastered with the rest of the mix.

## Non-goals

- Bundled effect files. The feature ships without them; the config points at a local directory.

## Design

- `audio_settings.sound_effects`: `enabled` (default false), `level_db` relative to the voice (default -15), `max_per_10_sec` (default 2), and one pool per event type: `transition`, `reveal`, `cta`. Each pool is a list of local files, at least five per type to avoid a template sound.
- Events come from data the pipeline already has: transition times from the visual chain, the reveal as the start of the sentence after the hook (from the Whisper word timings), the CTA as the start of the last sentence.
- Per event, draw a file with the seed `<product_id>:sfx:<event>`. Add each as an input with `volume` and `adelay` into the same `amix` as the music and sting, so `loudnorm` masters it with the rest.
- Enforce the per-10-second cap by dropping the lowest-priority events (transitions first). Never place an effect under a spoken word's first 100 ms.
- A missing file warns and is skipped (the sting's rule).

**Tests.** With three events configured, the filter graph carries three delayed inputs at the configured level; the cap drops transition effects first; two products draw different variants; off adds nothing to the command.

## Alternatives considered

None recorded.

## Rollout

Ships off: `audio_settings.sound_effects.enabled` defaults to false. Set it after the reach-test readout (#540), once an A/B over a batch shows completion no worse with effects. Stop at the first sign of lower engagement (the inverted-U finding).

## Open questions

None recorded.
