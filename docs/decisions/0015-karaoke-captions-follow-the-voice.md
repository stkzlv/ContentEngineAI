# 0015. Karaoke captions follow the voice, not a reading-speed cap

- **Status:** Accepted
- **Date:** 2026-10-10

## Context and problem

The caption research recommends 15-17 characters per second, and REQ-VID-156 asked for a cap that merges a faster segment into its neighbour (#591). That figure comes from subtitle practice, where the viewer reads a block of text. The pipeline's captions are word-by-word karaoke timed from the voiceover, so a segment is on screen for exactly as long as the voice takes to say it, and its rate is the voice's rate. Across 54 segments of three saved renders the median was 14.2 characters per second and the 90th percentile 17.2; 6 segments were over 17 and one over 20, "Your connection should be" at 21.2.

## Options considered

- **Merge a segment over the cap into a neighbour.** Lowers that segment's rate, but the merged caption holds more words than the 3-5 the same research sets as the most on screen, and the karaoke highlight still moves at the voice's pace.
- **Slow the voice.** Changes every render's length and pacing to fix the fastest tenth of segments.
- **Keep segmenting by characters and let the rate follow the voice.** The fastest segments stay a little over 17.

## Decision

Keep segmenting by the templates' character limits and let the reading rate follow the voice. The cap protects a reader who has to finish a block before it leaves; a karaoke caption leaves when its last word is spoken, and the highlight tells the viewer where the voice is.

## Consequences

- REQ-VID-156 is deprecated.
- If a voice change or a faster TTS setting pushes the median past 17, measure again: the case for slowing the voice changes before the case for merging does.
