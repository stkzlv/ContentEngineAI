# 0004. Optional voice processing chain

- **Status:** Accepted
- **Issue:** #545
- **Requirements:** REQ-CNT-073, REQ-CNT-074

## Context

The voiceover enters the mix with only a volume adjustment.

Evidence ([evidence grades](README.md#evidence-grades)):

- The usual polish chain for TTS is a high-pass filter, a small cut at a harsh 2-4 kHz peak, gentle compression, de-essing, a slight air shelf and a limiter. [C]
- AI voiceovers drew lower engagement than human voices on real TikTok ads, and a lower-pitched AI voice narrowed the gap. [A] [International Journal of Information Management](https://www.sciencedirect.com/science/article/abs/pii/S0268401225000945)

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

## As built

- The parameters are flat keys on `audio_settings.voice_chain` (`compressor_threshold_db` and the rest) rather than a nested `compressor` block.
- `deesser` does nothing at its own default intensity of 0, so de-essing is `deess_intensity` (default 0.4; 0 turns it off). `air_shelf_db` 0 and `limiter: false` drop their stages too.
- The limiter runs at -1 dBFS with `level=0`: `alimiter` otherwise raises the output back up to its ceiling.
- The harsh cut is a one-octave peaking `equalizer` at `harsh_cut_hz`.
- Loudness: on a bundled voiceover with music, the mastered mix measured -14.2 LUFS with the chain and -14.8 without, against the -14 target. The test compares each with the target rather than with each other: the chain may move the mix towards the target, not past it.

## Alternatives considered

None recorded.

## Rollout

Ships off: `audio_settings.voice_chain.enabled` defaults to false. Set it after the reach-test readout (#540), once a voice-by-chain comparison over at least 20 posts per cell shows no loss.

Remove the switch when: `audio_settings.voice_chain.enabled` has been on in the bundled config for 30 days and the voice-by-chain comparison, at 20 or more posts per cell, still shows no loss for the chain; the key and the unprocessed-voice path then go in a minor release with a `**Breaking**:` CHANGELOG entry, and the chain parameters stay.

## Open questions

- The research suggests also trying a lower-pitched voice; [0015](0015-narrator-voice-evaluation.md) covers the voice comparison.
