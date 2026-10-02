# 0014. Normalise numbers, units and model names before TTS

- **Status:** Accepted
- **Issue:** #556
- **Requirements:** REQ-CNT-075, REQ-CNT-076

## Context

The sanitised script goes to the voice unchanged.

The finding comes from [ai-slop-research.md](../ai-slop-research.md).

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

## Alternatives considered

None recorded.

## Rollout

Ships off: `tts_normalisation.enabled` defaults to false. Set it after the reach-test readout (#540), once the probe has measured which strings the voice misreads.

## Open questions

None recorded.
