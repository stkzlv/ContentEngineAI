# 0015. Evaluate a distinctive or owned narrator voice

- **Status:** Draft
- **Issue:** #557
- **Requirements:** none

## Context

An evaluation, not a feature. The question comes from [ai-slop-research.md](../research/ai-slop-research.md).

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
