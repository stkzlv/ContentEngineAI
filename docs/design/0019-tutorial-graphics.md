# 0019. Explanatory graphics

- **Status:** Accepted
- **Issue:** #561
- **Requirements:** REQ-VID-124, REQ-VID-153, REQ-VID-164

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
- Timing comes from Whisper's word timings: a card starts at the first word of its step's last path segment, spoken after the first sentence, preferring a match within three words after an action verb ("Tap General" over "The General menu opens"). Words are compared with spaces and punctuation dropped and "&" read as "and", so "BackTap" matches "Back Tap". A step is looked for within 60 words of the last step found, 60 more for each step missed since (real scripts put up to 42 words between two steps), so a step that doesn't match its own narration isn't found in the closing recap. Scripts often name an earlier label or describe the control instead of saying the last segment ("go back to Display" for "Display > Night Shift", "tap the small i" for "Details"), so a step whose last segment isn't spoken in its window is looked for by its earlier segments and by the action's first word that is neither a stopword nor an action verb ("Click the trash icon" is found at "trash", since every step says "click"; a possessive is dropped, so "app's" is "app"), taking the first of those spoken within three words after an action verb, else the first spoken (a label with no verb before it is often the last step's result, "The Display screen opens"). When none of those is spoken, up to five content words after the action's first action verb are tried the same way (so a leading "To finish the capture," is skipped) ("any area outside the toolbar" spoken as "anywhere outside the toolbar" is found at "outside"); they come last because a later word is more often in the previous step's result sentence too. A segment or action word the next step's path also has is left out, since the next step's narration names it too, and a segment that is only a placeholder ("[App Name]") is never looked for. Matched over the kept step lists and their scripts, this found every step in 14 tutorials of 15, against 7 by the last segment alone. A skipped step's log line names the words searched. Timings come from the pycaps engine's transcript, so a render on the FFmpeg engine draws no cards. A card ends when the next starts, after at most `max_sec`; a step not found, or on screen less than `min_sec`, gets no card.
- Cards sit where the hook overlay sits (28% from the top) and start after it ends, so one graphic is on screen at a time. A path longer than `max_path_words` keeps its last segments after "...".
- On a real topic render (five steps, 49 s) all five cards were drawn, each starting within a word of its step's narration.
- The step card has three layouts (REQ-VID-153): `centred` (the counter over the path in the middle), `left` (both lines from the left edge of `max_width_fraction`, pulled left when a line would cross the right edge, since the width estimate under-reads real glyphs, but never further than centred) and `inline` ("Step 2 of 4: Settings > General" as one line at the path's size). Each video draws one from an md5 hash of its product id, so a re-render keeps it. An `inline` card whose line still overflows at the smallest size is stacked instead, so a long path stays in the frame. The drawn layout goes into the render's assembly choices and the render-choices row (`step_card_layout`, empty when no card is drawn). The bundled `graphics.layouts` is `[centred, inline]`; `left` is built but not listed, and a one-entry list keeps every card the same.
- The checklist (REQ-VID-164) is the second type, behind `graphics.checklist` (off). Its text is the step list's common mistake, asked for as a short "don't" and "do" of at most eight words each in the same grounded call, and only when the checklist is on, so the prompt is unchanged otherwise. A pair with either line missing or longer is dropped rather than cut. It takes the second half of the mistake step's card time ("Don't: ..." over "Do: ...", each line in its own colour, stacked in every layout), or the whole step when half is under `min_sec`. Eight words can still be wider than the frame and drawtext doesn't wrap, so a checklist whose line doesn't fit at the smallest size shows the step's own card instead. A render that draws step screens draws no cards, so no checklist either. Like the card it follows, it ends where the next step's key word is spoken, so on a real re-render it stayed up through the first words of the next step's sentence.
