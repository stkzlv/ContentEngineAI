# 0016. Revisit the TikTok AI label

- **Status:** Accepted
- **Issue:** #558
- **Requirements:** REQ-CMP-017, REQ-CMP-018, REQ-CMP-019

## Context

`tiktok_settings.video_made_with_ai` is on for every post, and `docs/compliance.md` gave AI voiceover as the reason. TikTok's 2026-H2 guidelines exempt generic TTS narration.

The finding comes from [ai-slop-research.md](../research/ai-slop-research.md).

## Goals

- AI disclosure on each platform follows that platform's rule, recorded with its source.
- A label beyond what a rule requires is a documented, voluntary choice.
- An optional statement says what AI did and what a person did.

## Non-goals

- A statement that overstates the person's role. Here an LLM writes the whole script, a person curates the topic pool and keywords, and no person edits each video, so a line like "researched and edited by a person" would be false.

## Design

- Correct the compliance row in [compliance.md](../compliance.md) (the row carries a pending-correction note) and record the policy decision: keep the label on voluntarily, or turn it off, with the reason.
- Optionally add a bounded AI-role statement to the profile bio or the caption template, behind a config key. It must describe the real process. The study behind the idea found the label penalty disappears when AI's role is limited (polishing, a first draft) and persists when AI writes the whole piece, so a truthful statement helps only to the extent a person really reviews each video.
- Inspect a rendered file with `exiftool` or a C2PA reader for SynthID or C2PA metadata carried through from the TTS audio, and record whether it survives the mux.

## Alternatives considered

- **Keep the label on voluntarily.** One of the two outcomes the policy decision chooses between.
- **Turn the label off.** The other outcome, since the 2026-H2 guidelines exempt generic TTS narration.

## Rollout

The label changes reach for both reach-test arms, so the config change to `tiktok_settings.video_made_with_ai` waits for the reach-test readout (#540). The optional AI-role statement ships off behind its config key.

## Open questions

- Keep the label on voluntarily, or turn it off.
- Whether SynthID or C2PA metadata from the TTS audio survives the mux.
- The spec names no config key for the AI-role statement.
