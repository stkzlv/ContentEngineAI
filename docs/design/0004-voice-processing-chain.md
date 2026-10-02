# 0004. Optional voice processing chain

- **Status:** Accepted
- **Issue:** #545
- **Requirements:** REQ-CNT-073, REQ-CNT-074

## Context

The voiceover enters the mix with only a volume adjustment.

The technique comes from [creator-research.md](../creator-research.md).

## Goals

- Treat the voiceover with filtering, gentle compression, de-essing and limiting before the mix, without changing its loudness target or its transcript.
- Record per render whether the chain was on, beside the voice name.

## Non-goals

- Changing the captions. Captions are transcribed from the TTS output file, so the processed audio never reaches Whisper; the transcript is unchanged by construction.

## Design

- `audio_settings.voice_chain`: `enabled` (default false) and the parameters: `highpass_hz` (80), `harsh_cut_hz` (3000), `harsh_cut_db` (-2), `compressor` (threshold -18 dB, ratio 3, attack 5 ms, release 80 ms), `deess` (on), `air_shelf_db` (+1.5 above 9 kHz), `limiter` (on).
- Applied in `build_mix` to the voice chain before the `volume` stage: `highpass,equalizer,acompressor,deesser,highshelf,alimiter`.
- Record `voice_chain` (on or off) and the voice name per render, so #551 can compare voices and chains.

**Tests.** The filter clause matches the configured parameters; integrated loudness of a mixed test clip with the chain on stays within 0.5 LU of the same clip with the chain off (the mix already lands about 1 LU under the target, so compare the two, not either with the target); off produces today's command.

## Alternatives considered

None recorded.

## Rollout

Ships off: `audio_settings.voice_chain.enabled` defaults to false. Set it after the reach-test readout (#540), once a voice-by-chain comparison over at least 20 posts per cell shows no loss.

## Open questions

- The research suggests also trying a lower-pitched voice; [0015](0015-narrator-voice-evaluation.md) covers the voice comparison.
