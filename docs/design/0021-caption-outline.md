# 0021. A black outline round pycaps captions

- **Status:** Held
- **Issue:** #591
- **Requirements:** REQ-VID-158

## Context

The caption research asks for a white fill with a black stroke at 8-10% of the font size ([captions](../explanation/captions.md), "Colour and contrast"); its own starter recipe uses 7 px at 144 px. Neither bundled pycaps template draws a stroke: `word-focus` rings each word with a 2 px blurred text shadow, and `explosive` with an orange glow. A side-by-side render of one product with a 3 px and a 6 px outline was judged, and the 6 px one read better.

## Goals

- An opaque black outline of a configured width round each caption word on the pycaps engine, in frame pixels.

## Non-goals

- The FFmpeg caption presets, which keep their 2-4 px outline (`REQ-VID-057`).
- The pictex renderer, which renders glows and shadows differently; the outline is drawn on `css` only.

## Design

- `subtitle_settings.pycaps.outline_px` (default 0, which renders the template as shipped).
- Above 0, the renderer appends `.word { text-shadow: none; -webkit-text-stroke: Wpx #000; paint-order: stroke fill; }` after the template's CSS. The stroke is painted under the fill, so half of it shows outside the glyph, and pycaps draws template CSS at `2 x height / 1280` frame pixels; `W = 2 x outline_px / scale`, 4 CSS px for 6 frame px on a 1920-pixel frame. The frame height is probed from the input video, falling back to 1920.
- The template's text shadow is dropped so the outline is the only edge, which also removes `explosive`'s glow.

**Tests.** The width in CSS for a frame height; the rule appended only when the width is above 0, after the sentence-case rule; pictex skipped with a warning; the probe on a real clip and its fallback; the bundled config holding it at 0.

## Alternatives considered

- **3 px.** Rendered side by side with 6 px; 6 px read better.
- **Fork the templates.** The appended rule keeps the templates as shipped, the way sentence case already works.

## Rollout

Ships off: `outline_px` is 0 in the bundled config. Set it to 6 after the reach-test readout (#540), since it restyles every caption.

Remove the switch when: an outline is the default on every profile; the 0 path then goes in a minor release with a `**Breaking**:` CHANGELOG entry, and `outline_px` stays as the tuning.

## Open questions

None recorded.

## As built

- Built as designed. The comparison renders used the same appended rule.
