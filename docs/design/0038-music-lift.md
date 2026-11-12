# 0038. Music that rises between voice lines

- **Status:** Held
- **Issue:** #592
- **Requirements:** REQ-CNT-178

## Context

The audio research puts music at -6 to -10 dB in voice-free beats and 18-24 dB below the voice under narration ([audio](../explanation/audio.md)). The mix holds the music at `music_volume_db` (-24 dB, about 27 dB below the voice) for the whole video, and the optional voice-keyed duck only attenuates: in a gap the music returns to that same level, never above it.

## Goals

- Music at a gap level between voice lines and lowered under narration, from one switch.

## Non-goals

- Changing the fixed-level mix or the existing duck when the switch is off.
- A duck shape other than `sidechaincompress` keyed by the voice.

## Design

- `audio_settings.music_lift`: `enabled` (default false), `gap_volume_db` (-10), `duck_threshold` (0.05), `duck_ratio` (6).
- On, the music leg plays at `gap_volume_db` and the voice-keyed duck runs whether or not `music_ducking_enabled` is set, with the lift's threshold and ratio and the duck's attack and release.

## As built

- The audio builder picks the music level, whether to duck and the duck depth from the lift when it is on, and from `music_volume_db` and the `music_ducking_*` keys when it is off; nothing else in the chain changes, so the sting, the sound effects and loudness normalisation follow as before.
- The duck defaults are the 0.05 / 6 row of the measured duck table in `config/video_production.yaml`: about 10.7 dB of attenuation under real narration, so music sits near -21 dB under speech, about 24 dB below the voice, and at -10 dB in the gaps.
- On a synthetic voice with a gap, the music leg in the gap rises by the 14 dB between `music_volume_db` and `gap_volume_db`, and drops under the voice.

## Alternatives considered

- **Raise `music_volume_db` and turn the duck on by hand.** The same mix, but three keys that each change other behaviour; one switch keeps the fixed mix intact and the pairing explicit.

## Rollout

Ships off. A listening A/B of one render with and without it kept the fixed level, so the bundled config leaves the lift off; the switch stays for a later retest.

Remove the switch when: the lift has been on in the bundled config for two weekly batches with completion no worse than without it; `music_volume_db` and the `music_ducking_*` keys then go in a minor release with a `**Breaking**:` CHANGELOG entry, and the lift's values stay as the tuning.

## Open questions

None recorded.
