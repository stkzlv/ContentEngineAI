# Captions: why the defaults are what they are

This page explains how the pipeline styles, places and times the burned-in captions on a 9:16 render, and the evidence behind each default. The settings live in `subtitle_settings` in `config/subtitles.yaml` (engine, the `pycaps` block, `timing_smoothing`, segmentation limits, `safe_zone`, `font_pool`, `color_pool` and `style_presets`), in the per-profile `subtitle_settings` blocks in `config/video_production.yaml`, and in the Pydantic models in `src/video/config/subtitle_models.py`. The requirements are in [the video requirements](../requirements/video.md). The pycaps engine itself is described in [pycaps subtitles](pycaps-subtitles.md), the overlay numbers in [platform safe zones](platform-safe-zones.md), and the defects behind the caption code in [the subtitle module notes](../notes/subtitles.md). The `[A]`, `[B]` and `[C]` marks are defined in [the evidence grades](../design/README.md#evidence-grades).

Related pages: [audio](audio.md) for the sound-on layer, [promotional videos](promotional-videos.md) for the hook, CTA and disclosure, and [tutorials](tutorials.md) for the how-to format.

## Why captions are burned in

Every render carries burned-in captions while `subtitle_settings.enabled` is on (the default). The pycaps engine is the bundled default (`subtitle_engine: "pycaps"`, `REQ-VID-031`); it burns word-by-word karaoke captions after assembly (`REQ-VID-035`). The FFmpeg engine burns SRT or ASS captions during assembly and is the fallback when pycaps isn't installed (`pycaps.fallback_policy: fallback_ffmpeg`, `REQ-VID-038`, `REQ-VID-041`). The choice between the two is recorded in [decision 0004](../decisions/0004-caption-engine.md).

Why: many viewers watch muted, and for them the captions are the content, though most TikTok users spend time with sound on (see [audio](audio.md)). Captions are reported to raise retention and engagement [C] ([ContentFries](https://www.contentfries.com/blog/the-science-of-video-captions-how-they-impact-audience-retention), [3Play Media](https://www.3playmedia.com/blog/studies-find-captions-improve-engagement/), [NCAM captions research](https://www.accessible-social.com/audio-and-video/captions)).

Not supported: platform auto-captions as the only captions. Viewers can switch on TikTok's auto-captions, which can overlap the burned-in ones, and burned-in captions can't be switched off; where TikTok draws its auto-captions isn't documented here, so check one test post with them on.

## Font and size

Captions use a bold sans-serif of weight 700 or more (`REQ-VID-055`). On the FFmpeg engine the style presets name Montserrat or Gabarito with `bold: true`, and when `randomize_fonts` is on the font is drawn per product from `font_pool` (Montserrat, Poppins, Gabarito and Rubik, all Bold) (`REQ-VID-073`). The size is `font_size_percent: 0.075`, 7.5% of the frame height, about 144 px on 1920 (`REQ-VID-056`). Pycaps templates ship their own `@font-face` and size in their CSS, so `font_pool` applies to the FFmpeg engine only.

Why:

- Bold fonts read better on mobile than thin weights, and Montserrat is the most-used caption font in a 2M-video corpus [C] ([Submagic](https://www.submagic.co/blog/best-font-for-subtitle), [Blitzcut](https://blitzcutai.com/blog/best-caption-fonts-tiktok), [Kapwing](https://www.kapwing.com/resources/font-for-subtitles/)).
- Serifs, thin weights and decorative fonts break down at mobile size on busy backgrounds [C] (same sources).
- 7-9% of the frame height (about 135-172 px on 1920) is the readable band for vertical video; 7.5% sits inside it [C] ([md-subs](https://www.md-subs.com/blog/saa-subtitle-font-size), [Nimdzi](https://www.nimdzi.com/subtitling-vertical-videos-guidelines-where-art-thou/)).

## Casing

Pycaps captions keep the transcript's casing: `pycaps.force_sentence_case: true` appends `.word { text-transform: none; }` after the template's CSS, overriding the `text-transform: uppercase` that `word-focus` and `line-focus` ship (`REQ-VID-044`). The model default is `false`, so a programmatic build renders a template as shipped.

Why: mixed case reads faster than ALL CAPS, because ascenders and descenders carry the word's shape, while ALL CAPS makes the reader parse letter by letter. ALL CAPS suits deliberate shout-style captions, which a product video doesn't need.

## Colour and contrast

Captions have a white fill and an opaque black outline of 2-4 px depending on the style preset, a drop shadow on every preset except `minimal`, and no background box (`REQ-VID-057`). `modern` is the default FFmpeg preset. When `randomize_colors` is on (off globally, on in several slideshow profiles), the fill is drawn per product from `color_pool`: white, yellow (`#FFFF00`), neon green (`#00FF4C`) or saturated yellow (`#FFEB00`), each on a black outline.

Why:

- White fill against a black stroke is 21:1, far above the WCAG AA minimum of 4.5:1 and the AAA target of 7:1, and it holds on any background [A] ([WCAG 1.4.3](https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html), [WCAG 1.4.6](https://www.w3.org/WAI/WCAG21/Understanding/contrast-enhanced.html)). The 21:1 is the fill against the stroke: against a bright backdrop the stroke does the separating, which is why the blurred backdrop behind the captions is darkened (see [aspect modes](../reference/video-producer.md#aspect-modes)).
- Yellow and green on black are the most common highlight colours in short-form captions, and every pool entry keeps a black outline because coloured outlines lower readability [C] ([Submagic, Hormozi captions](https://www.submagic.co/blog/how-to-make-alex-hormozi-captions)).
- A background box adds visual weight over a clean product shot, so the presets ship without one; a box helps only on photographically noisy footage.

Not supported: "yellow beats brand colours in A/B tests" and "highest-converting preset" claims trace back to vendor marketing, not published tests.

## Highlighting and animation

The pycaps templates highlight the word being narrated (`REQ-VID-035`); the bundled `template_pool` is `["explosive", "word-focus"]`, chosen per product, and each template owns its own animation timing in its CSS and template file. The FFmpeg effects are karaoke, fade and typewriter (`REQ-VID-071`), one per render (`REQ-VID-072`). The `animated` preset uses karaoke rather than per-word movement, and the `random` preset's effect pool was narrowed to those three after rotation, glow and movement effects were judged fatiguing.

Why:

- Word-by-word highlighting keeps the viewer's eye where the audio is instead of reading ahead, and is reported to have the highest retention of the caption styles for longer clips [C] ([Opus Clip](https://www.opus.pro/blog/best-caption-presets-styles-boost-retention), [Blitzcut](https://blitzcutai.com/blog/best-caption-style-tiktok)).
- Pulsing or moving every word adds fatigue and a motion-sickness risk. If you add an effect, keep any colour flash under three per second, the WCAG photosensitivity threshold [A] ([WCAG 2.3.1](https://w3c.github.io/wcag21/understanding/three-flashes-or-below-threshold.html)).

## AI-driven highlighting

When `pycaps.enable_ai_tagging` is on (the bundled default) and the Gemini key is set, templates with an AI tagging rule (`explosive` in the bundled pool) ask Gemini which words to emphasise. `pycaps.ai_tag_prompt_override` replaces each template's own instruction with a recipe: tag prices, numbers, product nouns, outcome verbs and factual superlatives, never articles, prepositions, auxiliaries or absolute praise words, and around 15% of the words. [Pycaps subtitles](pycaps-subtitles.md) describes how the override reaches the tagger.

Why:

- The only peer-reviewed study (Weingartner et al., MUM 2024, n=66) found that keyword highlights improve recall, but that time-synchronised highlights were too distracting to replace standard captions in everyday viewing [A] ([arXiv 2307.05870](https://arxiv.org/abs/2307.05870)). Highlighting every word is overkill; highlighting a few is the defensible middle.
- Tagging around 15-20% of the words, in the order prices and quantities, the product noun, outcome verbs and factual superlatives, is where caption vendors converge, and at most three highlight colours in one video [C] ([Submagic, highlighting colours](https://care.submagic.co/en/article/how-to-apply-highlighting-colors-to-your-words-16ttppq/), [Submagic, emphasis captions](https://care.submagic.co/en/article/how-to-do-emphasis-captions-1p5w99b/), [a Submagic example](https://www.aiedgeforrealtors.com/blog/submagic-ai-video-captions-for-realtors)).
- The templates' stock instructions ("the most important phrase") made Gemini tag filler such as `also`, `can`, `all` and `from`, the bucket that adds no information.

Not supported: vendor figures such as "+X% retention from dynamic highlighting" come from marketing copy, not controlled trials. The defensible claim is directional: selective highlighting beats both none and everything.

## Sound effects

Caption templates add no sound while `pycaps.mute_template_sound_effects` is on, the bundled default (`REQ-VID-045`). `explosive` plays a `ding` on every highlighted word, and the 15% in the tagging recipe would make that 10-17 dings in a 30-second render. The 15% is a figure about visual highlighting; nothing chose it as a rate for sound, so muting the effect keeps the highlighting and drops the noise. Don't lower the tagging coverage to quiet a render.

Why: no retention study supports sound effects on captions, and editors agree an effect on every cut is worse than none [C]. Engagement follows an inverted U with total stimulation, and caption motion counts toward it [B] ([arXiv 2604.19995](https://arxiv.org/abs/2604.19995), a 2026 preprint).

Built and held off: sparse sound effects at each crossfade, the reveal and the call to action ([design 0003](../design/0003-sparse-sound-effects.md), `REQ-VID-012`; see [Audio](audio.md)).

## Layout and positioning

The FFmpeg engine anchors captions below the content (`anchor: "below_content"`, `content_aware: true`) with a 4% margin (`REQ-VID-046`, `REQ-VID-047`, `REQ-VID-049`) and clamps them to the platform safe zone (`REQ-VID-051`). The safe zone is the union of the TikTok, YouTube Shorts and Instagram Reels overlays, 270 px top, 670 px bottom and 180 px right on 1080x1920, set in `subtitle_settings.safe_zone` and overridable per profile (`REQ-VID-050`). Lines hold at most 3 words (`max_words_per_line: 3`), a caption at most 2 lines, and a line at most 80% of the frame width (`max_subtitle_width_fraction: 0.80`) (`REQ-VID-058`).

The pycaps engine places the block as a lower third: `vertical_align: "bottom"` with `vertical_align_offset: -0.20` puts its bottom edge at 75% of the frame height, below the 65% safe-zone floor (`REQ-VID-053`). Its width (`max_width_ratio: 0.80`) is clamped to the safe zone at render time, but it doesn't enforce the vertical boundaries (`REQ-VID-052`). The assembler fits a product image above the caption block wherever it sits (`REQ-VID-021`); with no explicit offset, the template places the block and the assembler assumes it is centred around 52% of the frame (`src/video/assembler/visual_band.py`).

Why:

- The bottom of the frame is interactive UI: 35% on Reels since Meta's March 2026 change, and about 25% on TikTok. A block centred around 52% of the frame, its lowest pixel above y=1250 (65%), clears all three platforms in one render [C] ([Kreatli](https://kreatli.com/guides/tiktok-safe-zone), [Zeely](https://zeely.ai/blog/tiktok-safe-zones/), [Postplanify](https://postplanify.com/blog/social-media-safe-zones-2026-complete-guide)). [Platform safe zones](platform-safe-zones.md) holds the canonical numbers; defer to it when the two pages diverge.
- The pycaps block stays lower than that recommendation to keep clear of centred product video on the `product_video_*` profiles, which ends near 66% of the frame. [Decision 0010](../decisions/0010-pycaps-captions-sit-low.md) records why raising the block was dropped.
- Two lines of 3-5 words is the readable maximum on a phone; three lines becomes a wall of text, and 80% of the width leaves a margin inside every platform's side overlay [C] ([Opus Clip, TikTok](https://www.opus.pro/blog/tiktok-caption-subtitle-best-practices), [Opus Clip, Shorts](https://www.opus.pro/blog/youtube-shorts-caption-subtitle-best-practices), [Nimdzi](https://www.nimdzi.com/subtitling-vertical-videos-guidelines-where-art-thou/)).

## Timing and reading

A caption segment lasts between 0.6 s and 2.5 s (`min_duration`, `max_duration`, `REQ-VID-059`). The FFmpeg engine breaks a segment on the word count, the line length (`max_line_length`), the duration cap, or a sentence end once the segment holds three words.

Captions are timed from the voiceover by vanilla `openai-whisper`. Before either engine sees them, `subtitle_settings.timing_smoothing` (on by default, `REQ-VID-063`) applies four rules in `src/video/subtitle_timing_smoother.py`:

1. Minimum word duration `min_word_sec: 0.12`, so short words don't flash past.
2. Gaps shorter than `gap_merge_sec: 0.08` merge into the preceding word, removing micro-flicker.
3. The last word of a segment is held `hold_last_sec: 0.20` past the audio, so the viewer finishes reading.
4. Every word leads its audio by `lead_sec: 0.04` (`REQ-VID-060`).

The first `hook_lead_word_count: 3` words get an extra `hook_lead_sec: 0.20` on top of the base lead (`REQ-VID-061`), so a muted viewer reads the opening before any audio cue, in step with the hook overlay. Set `hook_lead_sec` to 0 to turn it off.

Why:

- Under 0.6 s even a one-word caption doesn't register; past 2.5 s the viewer has read it and scanned ahead, and the caption feels stale.
- Showing a word slightly before it is spoken matches perception: reading takes longer than hearing.
- Vanilla Whisper's word timestamps drift by tens to hundreds of milliseconds, enough to make a karaoke highlight visibly early or late [C] ([OpenAI Whisper discussion](https://github.com/openai/whisper/discussions/435)). Forced alignment (WhisperX) or attention-based alignment ([whisper-timestamped](https://github.com/linto-ai/whisper-timestamped)) fixes it at the cost of a heavy dependency chain. Issue #90 closed that upgrade as not planned until a render shows timing drift the smoother can't absorb.

## Numbers and punctuation

Captions never split or alter a number: thousands separators, decimals and hyphenated tokens stay whole and keep their punctuation (`REQ-VID-062`). The pycaps renderer drops a template's punctuation-removing effect, because `word-focus` deletes every period in a word and turns `2.4GHz` into `24GHz`.

Not supported: stripping terminal punctuation from captions. Karaoke segment breaks do act as visual punctuation, but no template effect can tell a full stop from a decimal point, and a caption stating a different number costs more than one that keeps a full stop.

Planned: TTS normalisation of numbers, units and model names, with captions showing what the voice said ([design 0014](../design/0014-tts-text-normalisation.md), `REQ-CNT-076`).
