# 0018. A visual per step

- **Status:** Implemented
- **Issue:** #560
- **Requirements:** REQ-VID-123

## Context

Visual planning is one keyword search over the script. The detailed design is in the issue body, from "Length", "What makes a short tutorial useful", "Visuals that show the spoken step" and "Explanatory graphics" in [the tutorials explanation](../explanation/tutorials.md). This doc records how it fits the [rules every design follows](README.md#rules-that-apply-to-every-design).

Evidence ([evidence grades](README.md#evidence-grades)):

- Meta lists borders, captions, speed changes and narration over existing footage as not meaningful edits, so narrating stock footage is not original on its own. [A] [Meta](https://about.fb.com/news/2026/03/rewarding-original-creators-on-facebook/)

## Goals

- Each tutorial step is shown as it is spoken.

## Non-goals

- Stock footage for the steps themselves. Stock covers only the opening symptom.

## Design

- Visual planning moves from one keyword search over the script to one plan per step, timed from the Whisper word timings.
- Source order: a real capture, a UI mockup with exact labels, a diagram, and stock only for the opening symptom.
- Behind `video_settings.step_visuals.enabled`.

## As built

- The source is the UI mockup: each step's screen is drawn with Pillow as a full frame, a panel headed by the screen the step happens on, the step's target row marked with a tap ring, and the rows the step list can't name drawn as blank bars, so no label is invented. Pillow rather than the caption engine's Chromium: no browser start per render, and the image is the same on every machine that has the font. Screen capture and diagrams are not built.
- Timing starts from the step cards' matcher, which finds each step by its path's last segment, then moves back to the start of the sentence that first names any of the path's segments: a step spoken as "Now, find Auto-Lock. Open the list. Select 30." starts at "Now", not at "30". The step cards keep their own timing. If any step's narration can't be found the render keeps its stock visuals, since a screen at the wrong time teaches the wrong step.
- The header names the screen only where the step list does: the path's parent segment, or the start screen for the first step. Otherwise it is a blank bar, because the previous step's target ("30") is often not the screen the next step happens on.
- Stock plays for at most `opening_max_sec` (3 s); from there to the first step the start screen from the step list shows. The video ends on a recap of every step's path and the last step's expected result, from the second sentence after the last step's (its result, then the script's closing recap), at least `recap_sec` (2.5 s), leaving the last step at least a second.
- On a real four-step tutorial re-assembled with the switch on, each screen started at the sentence that opens its step and the recap at the spoken recap; before the change, the same render ran stock footage under every step.
- Each screen is an image input overlaid on the footage in its window, under the captions and every text overlay. A screen carries the step counter, so the step cards aren't drawn on a render that draws screens (one graphic at a time, design 0019).
- The render's choices record how many screens were drawn (`step_visuals`), which the removal condition below compares against.

## Alternatives considered

None recorded.

## Rollout

On in the bundled config, since a look at a sample tutorial render with it on. The model default stays `false`, so a config with no `step_visuals` section draws no screens.

Remove the switch when: `video_settings.step_visuals.enabled` has been on in the bundled config for two weekly batches with topic-render swipe-away and completion (#551) no worse than without it; the key, its holdout check and the single-search path then go in a minor release with a `**Breaking**:` CHANGELOG entry.

## Open questions

None recorded.
