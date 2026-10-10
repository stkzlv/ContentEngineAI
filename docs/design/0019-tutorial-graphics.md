# 0019. Explanatory graphics

- **Status:** Held
- **Issue:** #561
- **Requirements:** REQ-VID-124

## Context

The detailed design is in the issue body, from "Length", "What makes a short tutorial useful", "Visuals that show the spoken step" and "Explanatory graphics" in [the tutorials explanation](../explanation/tutorials.md). This doc records how it fits the [rules every design follows](README.md#rules-that-apply-to-every-design).

## Goals

- A tutorial carries templated graphics, each tied to a script fact.

## Non-goals

- Failing a render on a graphic. A failed graphic is skipped.
- More than one graphic on screen at a time.

## Design

- HTML and CSS templates rendered to transparent images in the existing Chromium and composited with FFmpeg overlays, from a validated JSON spec.
- One graphic at a time, every graphic tied to a script fact.
- A DOM overflow and safe-zone check, and a failed graphic skipped.
- Behind `video_settings.graphics.enabled` (default false).

## Alternatives considered

None recorded.

## Rollout

Ships off: `video_settings.graphics.enabled` defaults to false. Set it after the reach-test readout (#540). The switch gains a check in `tests/test_reach_test_holdout.py` when it lands.

Remove the switch when: `video_settings.graphics.enabled` has been on in the bundled config for two weekly batches with topic-render completion (#551) no worse than without it and a skipped graphic on fewer than one render in ten; the key, its holdout check and the graphics-off path then go in a minor release with a `**Breaking**:` CHANGELOG entry.

## Open questions

None recorded.

## As built

- The first graphic type is the step card: "Step 2 of 4" over the step's menu path, drawn with FFmpeg `drawtext` in the assembler's overlay chain, like the hook overlay, rather than HTML rendered in Chromium. A text card needs no browser, and the Chromium route stays open for the graphics that draw shapes.
- The spec is the step list the script step records (`step_list.json`): each step's `ui_path` gives the path line, so no new model call writes graphic text.
- Timing comes from Whisper's word timings: a card starts at the first word of its step's last path segment, spoken after the first sentence, preferring a match right after an action verb ("Tap General" over "The General menu opens"). Words are compared with spaces and punctuation dropped and "&" read as "and", so "BackTap" matches "Back Tap". A step is looked for within 40 words of the previous one, so a step that doesn't match its own narration isn't found in the closing recap, and a path segment that is only a placeholder ("[App Name]") is matched by the action's first word instead. Timings come from the pycaps engine's transcript, so a render on the FFmpeg engine draws no cards. A card ends when the next starts, after at most `max_sec`; a step not found, or on screen less than `min_sec`, gets no card.
- Cards sit where the hook overlay sits (28% from the top) and start after it ends, so one graphic is on screen at a time. A path longer than `max_path_words` keeps its last segments after "...".
- On a real topic render (five steps, 49 s) all five cards were drawn, each starting within a word of its step's narration.
