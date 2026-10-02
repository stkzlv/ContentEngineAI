# 0015. Evaluate a distinctive or owned narrator voice

- **Status:** Draft
- **Issue:** #557
- **Requirements:** none

## Context

An evaluation, not a feature.

Evidence ([evidence grades](README.md#evidence-grades)):

- AI voiceovers drew lower engagement than human voices on real TikTok ads, and a lower-pitched AI voice narrowed the gap. [A] [International Journal of Information Management](https://www.sciencedirect.com/science/article/abs/pii/S0268401225000945)
- WPP Media found listeners identified generic AI voices less than half the time; AI voices matched human ones on attention and purchase intent, voices believed human scored higher on relatability, and prosody aligned with the information structure raised human-likeness. [B] [WPP Media](https://www.wppmedia.com/news/ai-voices-audio-ads)
- Listeners spot synthetic speech by intonation, rhythm, fluency, pauses, speed and breathing, though detection was only 59% accurate in one listening experiment. [B] [arXiv 2512.09221](https://arxiv.org/abs/2512.09221)
- Missing breath is a known synthetic cue. [B] [arXiv 2404.15143](https://arxiv.org/pdf/2404.15143)
- Humanlike voices are rated less eerie and more likable. [A] [Frontiers in Neurorobotics](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2020.593732/full)
- Some library voices are sold as "one of the most recognizable voices on the internet" in faceless genres, and a prebuilt TTS voice shared by many channels carries the same risk. [C]
- YouTube needs no disclosure for AI-written scripts, captions or a clone of your own voice. [A] [YouTube Help](https://support.google.com/youtube/answer/14328491)

## Goals

- A recorded decision on the narrator voice.

## Non-goals

- Changing the voice. A voice change waits for the reach-test readout (#540).

## Design

- Compare the available voices, including lower-pitched ones, with the #439 pauses and the [0004](0004-voice-processing-chain.md) chain, in a blind listening test on three scripts.
- Separately, record whether a clone of the operator's own voice is possible through the TTS providers in use, its cost, and each platform's rule for it (YouTube exempts an owned-voice clone from disclosure).

The output is a recorded decision.

## Alternatives considered

None recorded.

## Rollout

Nothing ships from the evaluation itself. A voice change it recommends waits for the reach-test readout (#540).

## Open questions

- Which voice the blind listening test prefers.
- Whether an owned-voice clone is possible through the TTS providers in use, at what cost, and under which platform rules.
