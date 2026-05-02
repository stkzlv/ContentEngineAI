# Subtitle Best Practices

Practical, opinionated rules for designing captions that hold attention on
short-form vertical video. Applies to both subtitle engines in this project
(the bundled-default pycaps engine and the FFmpeg + SRT/ASS fallback path).
The recommendations are distilled from industry research across Submagic,
Opus Clip, Captions.ai, TikTok creator tooling, WCAG, and subtitling
research — see the sources at the bottom.

**Audience**: 30–60 second 9:16 vertical e-commerce product videos watched
on mobile, often with sound off. Captions aren't a nice-to-have — for a
large fraction of viewers they *are* the content.

**Related docs**:
- [promotional-video-best-practices.md](promotional-video-best-practices.md)
  — promotional / e-commerce video strategy (hook patterns, sound-off
  audience, CTA staging, FTC compliance, trust signals). Apply both docs
  together for promo content.
- [pycaps-subtitles.md](pycaps-subtitles.md) — pycaps engine reference
  including AI word tagging via Gemini.
- [platform-safe-zones.md](platform-safe-zones.md) — TikTok / Shorts /
  Reels UI overlay zones.

Open follow-up work is tracked as GitHub Issues with the `subtitles` and
`pycaps` labels.

---

## The 15 rules that matter (cheat-sheet)

1. **Bold sans-serif only, weight 700–900**. Montserrat Black, Inter Black, Poppins Black, Proxima Nova, Anton, Bebas Neue, TikTok Sans. Never a serif, never a thin weight.
2. **Font size 7–9% of frame height** — roughly 135–172 px on a 1080×1920 canvas (55–75 pt).
3. **White fill, black stroke at 8–10% of font size, subtle drop shadow**. Universal readable base, ~21:1 contrast.
4. **Word-by-word karaoke highlight is the dominant style**. Active word gets a color swap (yellow/green) and a small scale pop.
5. **Max 3–5 words on screen, max 2 lines**. Break on phrase boundaries, never mid-phrase.
6. **Sentence case, not ALL CAPS** (mixed case reads faster — ascenders/descenders carry word shape). ALL CAPS only for deliberate shout-style.
7. **15–17 CPS (~170 WPM)** reading speed. Minimum 500–600 ms per segment.
8. **Vertical center (45–60% from top)**, not lower-third. TikTok's UI eats the bottom 480 px.
9. **Design to TikTok's safe zone**: top 220 / bottom 480 / right 180. Works on Shorts and Reels automatically.
10. **Entrance 100–250 ms, no exit animation**. Use ease-out-quint (`cubic-bezier(0.22, 1, 0.36, 1)`).
11. **Strip terminal punctuation** — karaoke segment breaks ARE the punctuation.
12. **Smooth Whisper timings**: merge gaps <80 ms, min word duration 120 ms, hold last word +200 ms, lead audio by 40 ms.
13. **Contrast ≥4.5:1** (WCAG AA). White-on-black-stroke is 21:1, yellow-on-black is ~19:1, neon green ~17:1. All safe.
14. **One emoji per 8–10 word segment max**. Zero is fine too. Spam looks amateur.
15. **No flashing, scale cap 1.15×, pulse frequency <3 Hz**. Motion sickness and WCAG photosensitivity safety.

For promotional / e-commerce video, the
[promotional-video-best-practices.md](promotional-video-best-practices.md)
companion adds 4 more rules covering hook content, CTA staging, FTC
disclosure, and trust signals. Apply both lists together.

---

## 1. Typography

**Rule: Bold sans-serif, weight 700+**. Bold fonts score ~31% better mobile
readability than thin weights. Montserrat is the most-used caption font
across a 2M-video corpus (Submagic data). Specific picks that ship with
solid Latin + extended coverage:

- **Montserrat Black (900)** — default choice, wide ecosystem support
- **Inter Black (900)** — modern, tight tracking, excellent on mobile
- **Poppins Black (900)** — rounder, friendlier than Montserrat
- **Anton Regular** — ultra-condensed display weight, good for stacked layouts
- **Bebas Neue** — upper-case-only, narrow, works for shout-style
- **Proxima Nova Black** — if licensed, excellent readability
- **TikTok Sans** — TikTok's open-source caption font (2025+)

**Avoid**: serifs at mobile size (unreadable), thin/light weights,
decorative scripts, low x-height display fonts, all-caps condensed fonts
with ambiguous I/l/1.

**Size**: `7–9% of frame height`. For 1080×1920: ~135–172 px (55–75 pt).
Test at arm's length on a real phone — if you squint, size up.

**Casing**: Sentence case by default. Mixed case reads faster because word
shapes carry meaning; ALL CAPS loses ascenders/descenders so readers parse
letter-by-letter. Reserve ALL CAPS for deliberate shout-style (Hormozi,
MrBeast) or short 1–2 word emphasis hits.

**Letter spacing / line height**:
- `letter-spacing: -0.01em` to `-0.02em` on black weights (tight tracking)
- `line-height: 1.1` to `1.15` (generous leading so stroked text doesn't collide)

**Max width**: 80% of frame width. On 1080×1920 that's 864 px.

**Font loading in headless Chromium (pycaps)**:
- Embed with `@font-face` pointing at a local `.woff2` or `.ttf` in the template's `resources/` dir
- Set `font-display: block` (you want FOIT, not FOUT — swapping fonts mid-render produces broken frames)
- Pre-warm by injecting a hidden `<span>` with the font before rendering the first frame
- Ship Noto Emoji or Twemoji alongside your display font to avoid black-square fallbacks on Unicode

## 2. Color and contrast

**Base style: white fill, black stroke, subtle drop shadow**. This is the
universal default for a reason — it survives any background. WCAG AA needs
4.5:1; white/black is 21:1.

**Stroke thickness**: 8–10% of font size. With 70 pt text that's ~6–8 px.
Thin outlines only work on clean backgrounds; thick outlines provide
maximum readability on any background.

In CSS, layer `-webkit-text-stroke` with a text-shadow fan for true thick
outlines:

```css
-webkit-text-stroke: 7px #000000;
text-shadow:
  2px  2px 0 #000,
 -2px  2px 0 #000,
  2px -2px 0 #000,
 -2px -2px 0 #000,
  0 3px 6px rgba(0,0,0,0.55);
```

In ASS format, use `\bord7` for outline thickness and
`PrimaryColour=&H00FFFFFF&` / `OutlineColour=&H00000000&` in the style block.

**Drop shadow**: yes. Subtle: `0 4px 8px rgba(0,0,0,0.5)`. Separates text
from busy product shots.

**Background box**: skip by default. Background boxes kill momentum on
clean product shots and add visual weight. Use them only on photographically
noisy backgrounds (unboxing, street, UGC). If used: semi-transparent black
`rgba(0,0,0,0.7)`, 12–16 px padding, 8 px border radius.

**Active-word highlight color** (for karaoke-style):
- **Yellow `#FFEB00`** — Hormozi / Submagic default, highest perceived contrast swap from white
- **Neon green `#00FF4C`** — second most-used (also Hormozi palette)
- **Brand color** — works for established brands but yellow tends to outperform in A/B tests
- **Red** — reserved for "don't" / "avoid" words (reads as warning)

**Gradient fills**: look dated. Skip. The only exception is a solid
brand-colored fill on a single emphasized word.

**Accessibility**: WCAG 2.1 minimum 4.5:1 (AA), target 7:1 (AAA). The base
white-on-black-stroke style achieves ~21:1 — AAA comfortably.

## 3. Animation and effects

**Word-by-word reveal is the dominant trend** because it creates a
micro-anticipation loop. Viewers focus exactly where the audio is instead
of scan-reading ahead. Opus Clip reports this style has the highest
retention for educational/storytelling/longer clips.

**Current-word highlight technique**:
- Color swap from white → yellow (or green) on the active word
- Subtle scale pop: 1.0 → 1.10 → 1.0 over ~120 ms
- Avoid full bounce/overshoot unless the video's tone is playful
- Underline wipes look dated
- Background-box swap on the active word is the MrBeast move — heavy but effective

**Entrance animation** (per-word reveal):
- 120–200 ms fade + 4 px upward slide, or 150 ms scale from 0.85 → 1.0
- Keep it **under 250 ms** or the word is gone before the viewer reads it
- No stagger — each word appears exactly on its start timestamp

**Exit animation**: near-zero. Either hard cut on segment change, or 80 ms
opacity fade. Long exits create overlap confusion.

**Easing**: `cubic-bezier(0.22, 1, 0.36, 1)` (ease-out-quint) for
entrances. Ease-out always feels more "professional" than ease-in or
linear because it mimics real-world deceleration.

**Emoji insertion**: Submagic and Opus Clip insert an emoji roughly
once per 8–10 word segment, keyed off named entities or sentiment.
For e-commerce, this can highlight a product benefit (star for rating,
fire for deal, check for feature). **Don't place emoji mid-sentence** —
it fragments reading. Put them at the end of the emphasized phrase or
as a brief overlay beside the key word.

**Punctuation stripping**: drop all terminal punctuation (`.`, `,`, `!`,
`?`) because karaoke captions already have segment breaks as visual
punctuation. Keep apostrophes (`don't`, `it's`) and hyphens in compound
words.

**Sound effects on captions**: pop/whoosh on segment entrance is
standard on TikTok. Keep them subtle (-18 dB relative to voiceover) and
only trigger on emphasized words — every-word SFX becomes fatiguing fast.

**Safety limits**:
- Cap scale pulses at 1.15×
- Cap color-flash frequency at 3 Hz
- No full-screen flashes
- WCAG 2.1 photosensitivity guideline: max 3 flashes per second

## 4. Layout and positioning

**Vertical position: 45–60% from top** (roughly center, slightly above
center). This survives TikTok's bottom UI overlay, Reels' caption sticker,
and Shorts' progress bar simultaneously. Center-ish outperforms strict
lower-third because TikTok's UI controls occupy the bottom 480 px of a
1920-tall frame.

**Safe zones** (union of all three platforms):

| Platform | Top | Bottom | Right |
|---|---|---|---|
| TikTok | 160 px | 480 px (ads: ~600 px) | 180 px |
| YouTube Shorts | 120 px | 300 px (expanded: ~400 px) | 84 px |
| Instagram Reels | 220 px | 450 px | 84 px |

**Design to the union**: top 220 px, bottom 480 px, left/right 90 px on a
1080×1920 canvas. Caption center Y between **860–1100 px**. One design
covers all three platforms.

**Line break strategy**: 3–5 words per line, max 2 lines on screen. Break
on natural phrase boundaries (after verbs, before prepositions) — never
split noun phrases or compound names.

**Segment duration / reading speed**: target 15–17 CPS (~170 WPM). BBC
uses 160–180 WPM for general audiences. Minimum segment duration: 600 ms
(even "yes" needs time to register). Maximum: 2.5 s before the segment
feels stale.

**Width**: cap at 80% of frame width (~864 px on 1080), wrap at phrase
boundaries not word boundaries.

## 5. Timing and reading

**Vanilla Whisper rounds word timestamps to whole seconds** — the
timestamps are unreliable for karaoke-style captions. Two viable fixes:

- **WhisperX**: wav2vec2 forced alignment on top of Whisper output. Most
  accurate, drop-in replacement.
- **whisper-timestamped**: DTW on cross-attention weights. Works with
  existing Whisper models, slightly less accurate than WhisperX.

ContentEngineAI currently uses vanilla `openai-whisper`. Upgrading is
tracked as a GitHub Issue with the `subtitles` label.

**Post-processing rules** (apply after STT, before handing to either
engine):
- Clamp minimum word duration to **120 ms** (otherwise individual words flash imperceptibly)
- Merge inter-word gaps under **80 ms** into the preceding word
- Hold the last word of each segment **+200 ms** after audio end so viewers finish reading
- Lead the audio by **~40 ms** (show the word a hair before it's spoken) — eye→brain→recognition takes longer than ear→brain

**Reading speed budget**: 15–17 CPS (~170 WPM) for general audiences.
If a segment exceeds this, merge shorter words into the previous group
instead of cramming.

## 6. Platform-specific nuances

**TikTok**: the most aggressive UI. Bottom 480 px eaten by
like/comment/share/caption/sound. Right 180 px by icon column. TikTok's
auto-captions are on by default in 2026 — your custom captions must look
clearly better or viewers toggle yours off. Style preference: bold,
colorful, karaoke.

**YouTube Shorts**: progress bar at bottom, subscribe button bottom-center.
Keep bottom 300 px clear (expand to 400 px when description is open).
Shorts' AI captions are plainer — your custom design differentiates.

**Instagram Reels**: tightest top margin (220 px eaten by username/music).
IG's native caption sticker has a specific look that viewers associate
with "lazy creator" — custom captions should look clearly more intentional.

**Cross-platform rule**: design to TikTok's safe zone (tightest bottom).
Center-Y around 960–1050 px. One template, three platforms.

## 7. What the pros ship

- **Submagic "Hormozi 1" / "Beast"**: Montserrat Black, ALL CAPS, white + yellow + green highlight, thick black stroke, 4–6 words per 2 lines, pop animation on active word. Highest-converting preset for business/coaching content.
- **Opus Clip defaults**: cleaner than Hormozi. White text, thin scale pop on current word, occasional emoji, sentence case. Works for explainer/tutorial.
- **Captions.ai "Ali"**: minimal, white, thin shadow. Calmer — good for productivity content.
- **MrBeast style**: Komika Axis font, heavy black stroke, keyword highlights. High energy, NOT suited for e-commerce product shots where you want the product to be the hero.

**E-commerce recommendation**: lean toward the **Opus Clip / Submagic "Devin"** look — sentence case, white base, single accent color (brand or yellow) on benefit/price words, minimal emoji. The product imagery does the heavy lifting; captions should emphasize, not compete.

## 8. Anti-patterns (avoid)

- Serifs, thin weights, decorative fonts, cursive
- Full-sentence captions held on screen for 4+ seconds
- Tiny text (under 5% of frame height)
- Low contrast pairs (gray-on-white, yellow-on-white, color-on-color)
- Bottom-edge placement that TikTok UI clips
- Raw Whisper timings without smoothing (flicker artifacts)
- 5+ emoji per segment
- Over-animated pulsing on every word (fatigue + motion sickness risk)
- Caption speed mismatched to voiceover tempo
- Mixing 3+ highlight colors in one video
- Platform auto-captions as final output (viewers read these as "creator didn't try")

## 9. AI-driven highlighting — what to tag

The only peer-reviewed study to date (Weingärtner et al., MUM '24, n=66)
tested standard captions vs. keyword highlights vs. time-synchronized
keyword highlights. Both highlight conditions improved *recall*, but
time-synced highlights were rated "too distracting to replace standard
captions in everyday viewing." Translation: aggressive karaoke + color on
every spoken word is overkill.

**The "tag 15–20% of words" heuristic** is vendor-converged across
Submagic, Captions.ai, and OpusClip. Targets in priority order:

1. Prices and quantities (`$15`, `30 seconds`, `4.6 stars`)
2. The product noun (`backpack`, `ring light`, `bottle`)
3. Outcome verbs (`fixed`, `saves`, `lasts`, `replaces`)
4. Factual superlatives only (`loudest`, `lightest` — not `best`, `amazing`)

**What to skip**: articles, prepositions, auxiliaries, filler verbs,
absolute superlatives. The frames we sampled in the explosive-template
test had Gemini tagging `also`, `can`, and `all` — exactly the auxiliary /
filler bucket that adds no information density.

**Color cap: 3 highlight classes maximum.** Submagic enforces this in their
UI. More colors = noise.

**Concrete prompt template for the AI tagger** (override the built-in
template prompt when needed):

> Tag the most concrete, information-dense words: prices, numbers, product
> nouns, outcome verbs, and factual superlatives. Never tag articles
> (`a`, `the`), prepositions (`from`, `to`, `with`), auxiliaries
> (`can`, `also`, `just`), or absolute praise words (`amazing`,
> `incredible`). Aim for ~15% of total words.

**Honest caveat**: vendor numbers like "+X% retention from dynamic
highlighting" trace back to marketing copy, not controlled trials. The
defensible directional claim is "selective highlighting outperforms
either extreme (none / everything)."

## 10. Starter recipe — short-form vertical video

A concrete set of choices that produces a high-retention caption style
directly implementable in either engine.

**Font**: Montserrat Black (weight 900), loaded via `@font-face` or
embedded as a font file in the project's `static/fonts/` dir.

**Size**: `72 px` on a 1080×1920 canvas (≈7.5% of frame height).

**Case**: sentence case. Terminal punctuation stripped in timing
post-processing.

**Position**: container centered, `top ≈ 52vh` (slightly above center to
survive TikTok's bottom UI). Max width `864 px` (80% of 1080).

**Line structure**: 3–4 words per line, max 2 lines, break on phrase
boundaries.

**Base styling** (CSS — pycaps engine):

```css
.word {
  font-family: 'Montserrat Black', sans-serif;
  font-weight: 900;
  font-size: 72px;
  color: #FFFFFF;
  -webkit-text-stroke: 7px #000000;
  text-shadow: 0 4px 10px rgba(0,0,0,0.55);
  letter-spacing: -0.01em;
  line-height: 1.12;
  text-align: center;
}

.word-being-narrated {
  color: #FFEB00;
  transform: scale(1.10);
  transition: all 120ms cubic-bezier(0.22, 1, 0.36, 1);
}
```

**ASS equivalent** (FFmpeg + libass engine):

```
Style: Default,Montserrat,72,&H00FFFFFF,&H0000FFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,7,0,2,90,90,480,1
```

Fields: `Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding`.
Note the 7-px outline, white primary, yellow secondary (karaoke fill color), 480-px vertical margin.

**Entrance animation** (per word): `opacity 0 → 1` + `translateY(6px → 0)`
over **160 ms**, ease-out-quint, no stagger. Word appears on its start
timestamp.

**Exit**: hard cut on segment change. No exit animation.

**Emoji**: one contextual emoji per high-salience segment (rating → star,
price → money-bag, feature → checkmark). Max one emoji per 8 words,
placed at end of phrase.

**Timing post-processing**:
- Minimum word duration: 120 ms
- Merge inter-word gaps < 80 ms
- Lead audio by 40 ms
- Hold last word of segment +200 ms
- Target 15 CPS; merge short words forward if exceeded

**Segment rules**: 3–5 words per segment, 600–1800 ms duration, break on
sentence/phrase boundaries not word count alone.

**Safe zone**: designed to TikTok (tightest union) — top 220 / bottom 480 /
right 180. Single render covers all three platforms.

**Accessibility**: ~21:1 contrast, WCAG AAA. No flashes, scale pulses
capped at 1.10.

---

## Sources

- Blitzcut — [TikTok Caption Font 2026](https://blitzcutai.com/blog/best-caption-fonts-tiktok) / [Best Caption Style for TikTok 2026](https://blitzcutai.com/blog/best-caption-style-tiktok)
- Submagic — [15 Best Fonts for Subtitles](https://www.submagic.co/blog/best-font-for-subtitle) / [MrBeast Captions](https://www.submagic.co/blog/how-to-make-captions-like-mrbeast) / [Hormozi Captions](https://www.submagic.co/blog/how-to-make-alex-hormozi-captions) / [E-commerce product videos](https://www.submagic.co/business/e-commerce)
- Opus Clip — [Best Caption Presets](https://www.opus.pro/blog/best-caption-presets-styles-boost-retention) / [TikTok Caption Best Practices 2026](https://www.opus.pro/blog/tiktok-caption-subtitle-best-practices) / [YouTube Shorts Caption Best Practices](https://www.opus.pro/blog/youtube-shorts-caption-subtitle-best-practices) / [Best Text Animation Packs 2026](https://www.opus.pro/blog/best-text-animation-packs-captions-titles)
- Captions.ai — [Styles documentation](https://captions.ai/help/docs/captions/styles)
- Platform safe zones — [Kreatli TikTok Safe Zone Guide 2026](https://kreatli.com/guides/tiktok-safe-zone) / [Zeely TikTok Safe Zones 2026](https://zeely.ai/blog/tiktok-safe-zones/) / [Postplanify Social Safe Zones](https://postplanify.com/blog/social-media-safe-zones-2026-complete-guide)
- Accessibility — [W3C WCAG 1.4.3 Contrast Minimum](https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html) / [W3C WCAG 1.4.6 Contrast Enhanced](https://www.w3.org/WAI/WCAG21/Understanding/contrast-enhanced.html) / [NCAM captions research](https://www.accessible-social.com/audio-and-video/captions)
- Retention research — [ContentFries captions and retention study](https://www.contentfries.com/blog/the-science-of-video-captions-how-they-impact-audience-retention) / [3PlayMedia captions improve engagement](https://www.3playmedia.com/blog/studies-find-captions-improve-engagement/)
- Subtitling conventions — [md-subs font size research](https://www.md-subs.com/blog/saa-subtitle-font-size) / [Nimdzi vertical video subtitling](https://www.nimdzi.com/subtitling-vertical-videos-guidelines-where-art-thou/) / [Kapwing best subtitle fonts](https://www.kapwing.com/resources/font-for-subtitles/)
- Whisper timing accuracy — [WhisperX / whisper-timestamped](https://github.com/linto-ai/whisper-timestamped) / [OpenAI Whisper timestamp discussion](https://github.com/openai/whisper/discussions/435)
- AI-driven highlighting (academic) — [Useful but Distracting: Keyword Highlights in Captions for Language Learning (MUM '24)](https://arxiv.org/abs/2307.05870)
- AI-driven highlighting (vendor) — [Submagic: How To Apply Highlighting Colors](https://care.submagic.co/en/article/how-to-apply-highlighting-colors-to-your-words-16ttppq/) / [Submagic: How To Do Emphasis Captions](https://care.submagic.co/en/article/how-to-do-emphasis-captions-1p5w99b/) / [Submagic vertical example for realtors](https://www.aiedgeforrealtors.com/blog/submagic-ai-video-captions-for-realtors)
