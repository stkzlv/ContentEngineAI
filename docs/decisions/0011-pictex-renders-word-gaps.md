# 0011. pictex is a usable renderer, CSS stays the default

- **Status:** Accepted
- **Date:** 2026-10-05

## Context and problem

[Decision 0004](0004-caption-engine.md) chose pycaps with the CSS renderer and kept pictex for previews only, because pictex rendered caption words with no gaps between them. The cause was in pycaps, not pictex: the builder kept a template's CSS on the renderer that `with_custom_subtitle_renderer` then replaced, so pictex rendered with no template style. pycaps 0.3.0 keeps the CSS across the swap (#565).

## Options considered

- **Make pictex the default.** No browser to install or launch. Its glows and soft shadows differ from the CSS renderer's, so `explosive` would change look on every render, which the reach test holds ([decision 0002](0002-output-changes-ship-off-by-default.md)).
- **Keep CSS as the default and allow pictex.** Renders stay as they are; pictex is available where a browser is not.

## Decision

The CSS renderer stays the bundled default. pictex is no longer limited to previews: on a fixed clip it matches the CSS renderer on `word-focus` in word gaps, font and colour, and differs on glows and soft shadows.

## Consequences

A run without a browser can use pictex after checking a frame. Making pictex the default is a later choice, after the readout, with `explosive`'s glow compared.
