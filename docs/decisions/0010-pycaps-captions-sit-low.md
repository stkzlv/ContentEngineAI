# 0010. pycaps captions sit low, below the research's caption band

- **Status:** Accepted
- **Date:** 2026-10-05

## Context and problem

The caption research puts the caption block around 52% from the top, with its lowest pixel above 65% of the frame, to clear every platform's bottom interface ([captions](../explanation/captions.md), [platform safe zones](../explanation/platform-safe-zones.md)). The pycaps engine places the block as a lower third instead, its bottom at 75% (`REQ-VID-053`). #99 tried to raise it and was closed; the reason lived in that issue and in the requirement's "Why" line.

## Options considered

- **Raise the block to the research's band** (`vertical_align_offset` near -0.40). It clears the platforms' bottom interface, and it puts the captions over the product image or the centred product video on every bundled profile, which ends near 66% of the frame.
- **Move the imagery up to make room.** Top-anchoring the slideshow imagery still left the caption inside the image area, since the assembler sizes the imagery around a caption band reserved at the bottom of the frame.
- **Keep the lower third.** The product stays clear, and the lowest caption lines can sit under the platforms' bottom interface.

## Decision

Keep the lower third for pycaps. Clearing the product matters more than the bottom interface on these profiles, and moving the caption band means reworking how the assembler reserves caption space, not a config change.

## Consequences

The FFmpeg caption engine still clamps to the safe zone (`REQ-VID-051`); pycaps does not (`REQ-VID-052`). A layout that reserves the caption band higher in the frame would reopen this decision.
