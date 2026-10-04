# 0014. Normalise numbers, units and model names before TTS

- **Status:** Accepted
- **Issue:** #556
- **Requirements:** REQ-CNT-075, REQ-CNT-076

## Context

The sanitised script goes to the voice unchanged.

Evidence ([evidence grades](README.md#evidence-grades)):

- Misread names, numbers and units ("five thousand M A H") are widely cited as a giveaway of synthetic narration, with no study behind it; product scripts are full of them. [C]

## Goals

- Numbers, units and model names the voice misreads are rewritten to speakable words in the text sent to TTS.
- The captions show what the voice said.

## Non-goals

- Rewriting the script file or the state. They keep the written form.
- Rewriting strings the voice reads correctly. Only entries the probe shows are misread go in the table.

## Design

- A probe (a sibling of `tools/tts_tag_probe.py`) voices a fixed list of strings (`5000mAh`, `65W`, `2.4 GHz`, `1.83-inch`, `USB-C`, `IP68`, a SKU) and records the transcript, so misreadings are measured before anything is rewritten.
- A `tts_normalisation` block in config: `enabled` (default false) and a table: unit spellings applied only after a number (`mAh` to "milliamp hours", `W` to "watts", `GHz` to "gigahertz"), decimal and range handling, and a small lexicon for brand and model terms.
- Applied in `TTSManager.generate_speech` to the text sent to the provider only; the script file and state keep the written form. Captions come from Whisper on the audio, so they show the spoken form ("5000 milliamp hours").
- Only entries the probe shows are misread go in the table.

**Tests.** Each table entry rewrites its fixture and leaves unit letters inside ordinary words alone ("Watch" stays "Watch"); the script file is unchanged; off sends today's text.

## As built

- The block is `tts_config.tts_normalisation` (`enabled`, `units`, `lexicon`) in `config/subtitles.yaml`, applied in `TTSManager.generate_speech` before the pause plan or markup rules.
- Captions built from the script, the fallback when speech-to-text returns no timings, get the same rewritten copy (`spoken_script`), so they show what the voice said.
- A unit matches only straight after a digit, with an optional space or hyphen, and not before a letter, digit or hyphen; a lexicon term matches only as a whole term.
- The probe (`tools/tts_normalisation_probe.py`) voices each case as written and spelled out in the same sentence and compares the transcripts, since Whisper writes units back in either form. On `charon` (Gemini 2.5 Flash TTS) all twelve cases read as intended: the transcripts that differed did so because Whisper abbreviated a spoken unit or spelled out a written one, and the pair durations agreed within about 5% for every unit. No entry qualified, so both tables ship empty and the switch stays off.

## Alternatives considered

None recorded.

## Rollout

Ships off: `tts_normalisation.enabled` defaults to false. Set it after the reach-test readout (#540), once the probe has measured which strings the voice misreads.

Remove the switch when: `tts_normalisation.enabled` has been on in the bundled config for 30 days and a probe run transcribes every table entry in its spoken form; the key and the raw-text path then go in a minor release with a `**Breaking**:` CHANGELOG entry, and the table stays as the place for further entries.

## Open questions

None recorded.
