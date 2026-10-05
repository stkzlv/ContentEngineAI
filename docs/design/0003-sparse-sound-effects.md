# 0003. Sparse event sound effects

- **Status:** Held
- **Issue:** #544
- **Requirements:** REQ-VID-012

## Context

The mix is voiceover, music and (optionally) the signature sting, built by `AudioFilterBuilder.build_mix`.

Evidence ([evidence grades](README.md#evidence-grades)):

- Message sensation value (cuts, motion, sound and text combined) raises engagement up to a point, then heavy effects reduce likes, shares and comments, in a 2026 preprint of 1,200 rated short videos validated on 14,492 more. [B] [arXiv 2604.19995](https://arxiv.org/abs/2604.19995)
- No retention study measures sound effects; editors agree an effect on every cut is worse than none, so effects are reserved for the hook, the reveal and the CTA. [C]
- No public controlled study measures the organic retention effect of an individual edit effect (zooms, effects, transitions) on Shorts or Reels, so the pipeline's own A/B data decides.
- Not supported: any unsourced "+N% retention" figure for an effect.

## Goals

- Sparse effects mark three beats (the hook, the reveal, the call to action), drawn per product from a pool, capped per 10 seconds and mastered with the rest of the mix.

## Non-goals

- Bundled effect files. The feature ships without them; the config points at a local directory.

## Design

- `audio_settings.sound_effects`: `enabled` (default false), `level_db` relative to the voice (default -15), `max_per_10_sec` (default 2), and one pool per event type: `hook`, `reveal`, `cta`. Each pool is a list of local files, at least five per type to avoid a template sound.
- Events come from data the pipeline already has: the hook on the first frame, under the hook headline, the reveal as the start of the sentence after the hook (from the Whisper word timings), the CTA as the start of the last sentence.
- Per event, draw a file with the seed `<product_id>:sfx:<event>`. Add each as an input with `volume` and `adelay` into the same `amix` as the music and sting, so `loudnorm` masters it with the rest.
- Enforce the per-10-second cap by dropping the lowest-priority events (the hook first, since the voice's opening line carries the hook too). Never place an effect under a spoken word's first 100 ms.
- A missing file warns and is skipped (the sting's rule).

**Tests.** With three events configured, the filter graph carries three delayed inputs at the configured level; the cap drops the hook effect first; two products draw different variants; off adds nothing to the command.

## As built

- The hook effect sits on the first frame, moved past the first word's onset. The reveal and the call to action come from the raw Whisper transcript, which the pycaps engine writes; with the FFmpeg caption engine there is none, so only the hook plays.
- A first build put an effect on every crossfade instead of the hook. That contradicted the evidence above (an effect on every cut is worse than none), so the events were set to the hook, the reveal and the call to action, and the `transition` pool became `hook`.
- An effect that would start in a word's first 100 ms moves to 100 ms after that word's start; the reveal and the call to action, placed at a sentence's first word, always do, and so does the hook when the voice starts on the first frame.
- Only event kinds with a file on disk are planned, so an empty pool cannot take the cap's places from one that can sound. The cap keeps the call to action, then the reveal, then the hook, and drops any event that would put more than `max_per_10_sec` in a 10-second window.
- Each file is drawn with the seed `<product_id>:sfx:<event>:<n>`, the `n`th event of its kind.
- The effect level is the voice's `voiceover_volume_db` plus `level_db`.

## Alternatives considered

None recorded.

## Rollout

Ships off: `audio_settings.sound_effects.enabled` defaults to false. Set it after the reach-test readout (#540), once an A/B over a batch shows completion no worse with effects. Stop at the first sign of lower engagement (the inverted-U finding).

Remove the switch when: `audio_settings.sound_effects.enabled` has been on in the bundled config for 30 days with completion and engagement no worse than in the batches without effects; the key and the effects-off path then go in a minor release with a `**Breaking**:` CHANGELOG entry, and `level_db` and `max_per_10_sec` stay as the tuning.

## Open questions

None recorded.
