# Promotional Video Best Practices

Strategy and content rules for short-form vertical promotional / e-commerce
video (30-60 s, 9:16, watched on autoplay social feeds). Independent of the
caption engine — these recommendations apply whether you use the FFmpeg
path, the pycaps engine, or render captions in some other tool entirely.

**Audience**: 30-60 second 9:16 vertical product reviews, demos, and ads
served via TikTok, Instagram Reels, and YouTube Shorts. Watched mostly
with sound off.

**Related docs**:
- [subtitle-best-practices.md](subtitle-best-practices.md) — universal
  subtitle/caption design rules and the project's pycaps + FFmpeg
  starter recipe. Apply both docs together.
- [pycaps-subtitles.md](pycaps-subtitles.md) — pycaps engine reference
  including AI word tagging via Gemini.
- [platform-safe-zones.md](platform-safe-zones.md) — TikTok / Shorts /
  Reels UI overlay zones.

---

## The 4 promo-video rules that matter (cheat-sheet)

1. **First-3-second hook is a static title card, not karaoke.** 5-12 words, on screen for the full 1.5-3 s, larger than narration captions. The decision window is ~1.7 s; if the hook is still revealing word-by-word at 1.5 s, the viewer has already swiped.
2. **CTA gets its own staging**, distinct from narration. Pair an early **soft CTA** (3-5 s, neutral) with a **hard CTA** at the end (full-frame, accent color, larger, static, ≥1.5 s on screen). Red/orange beats green in independent A/B tests.
3. **`#ad` disclosure is a persistent on-frame overlay AND first-line caption text.** FTC requires both. Same font family as captions, ~50-60% size, fixed corner, full-clip duration. Penalties up to $53,088 per violation (2025).
4. **State at least one trade-off per video.** Trust converts; absolute superlatives ("life-changing", "obsessed") now actively reduce trust in 2025-2026 data. A dedicated downside beat is the strongest trust signal in the de-influencing era — disclosed sponsorships do not depress engagement.

---

## 1. Hook patterns (first 1-3 seconds)

The decision window on a vertical autoplay feed is ~1.7 s (WARC eye-tracking,
TikTok Creative Center). TikTok's first ranking signal triggers at 1.5 s; a
failed hook produces a cold start the video rarely escapes. **84.3% of 2025
viral TikToks** used a recognized psychological hook trigger in the first 3
seconds (TTS Vibes scrape).

**Word budget**: read aloud in ≤3 s. ~5-12 words, ideally 5-8.

**Patterns that work for product video specifically:**

| Pattern | Example | Why it works |
|---|---|---|
| Price-first reveal | "The $15 [thing] that…" | Specific number = credibility + curiosity gap |
| Regret / contrarian | "I regret buying this", "Don't buy X until you see this" | Negative framing > positive in cold audiences |
| POV: | "POV: you finally found a [thing] that doesn't [pain]" | Reader becomes protagonist; instant context |
| Outcome-first | "This fixed my [pain] in 30 seconds" | Result hook; works on warm audiences |
| Numbered teardown | "3 reasons I'm returning this" | Lists promise structure → lower cognitive cost |
| Comparison | "$15 vs $200 — same thing?" | Pattern interrupt + value framing |

**Caption-rendering implication**: render the hook as a static title card,
not karaoke. Karaoke is still revealing word-by-word at 1.5 s — the viewer
has already swiped. Pin the full hook caption on screen for the entire
1.5-3 s, 1.2-1.5× the size of narration captions, single fade-in (no
scale-pop, no per-word reveal). See subtitle-best-practices for caption
design rules; the deviation here is "no per-word reveal on the hook."

Hook caption sizing and word budget converge across Captions.ai, OpusClip,
and Submagic 2025-2026 guidance.

## 2. Sound-off as the primary audience

**85% of social video views are sound-off** (Manchester Digital,
Clicks.video, Zebracat 2025 — number converges across sources). Design
the entire video assuming sound-off is the default; treat sound-on as the
accessibility layer for the 15%, not the source of truth.

**What this changes:**

- **Captions must be self-sufficient narration**, not ornament. The viewer
  should be able to follow the entire script with audio off.
- **Embed the CTA inside the visual track**, not only in the voiceover.
  "Link in caption" + arrow on a frame, not just spoken `tap the link`.
  The on-frame text IS the call to action for the 85% on autoplay.
- **Visual hierarchy**: captions should be the second-largest element
  after the product itself.
- **Beat punctuation matters more sound-off.** Voice intonation is the
  natural pacing cue; without it, punctuation (period, em-dash, ellipsis)
  is the only signal of "this is a beat, not a continuation." Strip
  mid-script periods for karaoke flow but keep beat punctuation at
  intentional pause points (hook → narration handoff, before CTA).
- **Mute resistance**: 39% of viewers mute video ads by default; only 22%
  skip if captions are present (vs. higher skip rates without). Captions
  don't directly boost CTR — they prevent the skip that kills CTR.

## 3. Trust signals & FTC `#ad` disclosure

**FTC compliance (US, 2023-updated Endorsement Guides) is the floor**:

- Disclosure must be **clear and conspicuous**. For Reels/TikToks, the
  agency wants a **two-punch**: on-screen overlay (`#ad`, `Sponsored`,
  `Paid partnership`) PLUS the same disclosure at the **top of the
  caption text**, before any other text or hashtags.
- Disclosure must be **on screen long enough to read**, in a font/color
  that contrasts the background, **persistent for the entire clip** (not
  a flash card).
- Penalties: **up to $53,088 per violation** as of 2025.
- Practical rendering: same font family as the project's caption
  configuration, ~50-60% caption size, fixed top-left or top-right
  corner, full-clip duration.

**Trust signals beyond the legal floor:**

- **71% of consumers** name "transparency about brand relationships" as a
  top trust factor; **79%** want "authentic reviews, even if negative"
  (Stack Influence, Skeepers 2025).
- **Disclosed sponsorships do not depress engagement.** TikTok 2023-2025
  data shows no performance gap between disclosed and undisclosed
  branded content.
- **2026 vertical data**: unscripted/unedited formats outperformed
  polished branded content by **+62% engagement, +38% conversion** across
  beauty/food/fitness niches (Amra & Elma 2026).

**Script copywriting patterns that build trust:**

- **Hedge phrases** ("for $15, you get…", "the catch is…", "won't replace
  your X but…") read as honest. Absolute superlatives ("life-changing",
  "obsessed") read as ad copy and now actively *reduce* trust.
- **Trade-off mention** as its own beat (one segment dedicated to a
  downside) is the strongest trust signal in 2025-2026 data — the
  de-influencing aesthetic without the negativity.
- **Source citations on factual claims** ("4.6★, 12k reviews") render
  well as small caption supers below the main caption line.

This aligns with the project's trade-off-honesty rule baked into the
script template prompts (CHANGELOG `0.43.1`).

## 4. CTA staging — soft + hard, two-stage

**Wistia State of Video 2025** analyzed 36,000+ video CTAs:

- Post-roll CTAs convert at **~16% on average**, well-placed CTAs reach **~40%**.
- For videos under 60 s, **CTAs in the first quarter** convert nearly
  40% of viewers.
- Surprising data point: a **soft early CTA + hard late CTA double-tap**
  outperforms a single end-card.

**Two-stage pattern (vendor consensus):**

| Stage | When | Style | Purpose |
|---|---|---|---|
| Soft CTA | 3-5 s | Small caption line, neutral color, e.g. "link in caption" + arrow | Plant intent without interrupting the hook |
| Hard CTA | Last 2-4 s | Full-frame text, accent color, 1.3-1.5× narration size, static or single pulse, ≥1.5 s on screen | Convert the warm viewer |

**Hard-CTA caption rendering:**

- Different color from the karaoke highlight (e.g., brand accent vs. the
  per-word highlight color). Red/orange beat green in independent A/B
  tests (CapCut/Nemo case studies).
- Larger font (1.3-1.5× narration).
- **Static**, not karaoke. Karaoke on the CTA reads as "more text
  coming"; static reads as "stop, act."
- On screen ≥1.5 s minimum. Faster than that, the CTA blinks past in
  autoplay.

**Verb choice**: imperative + specific outcome. "Get the $15 fix" beats
"Shop now". "Link to the bag in bio" beats "Click here".

## 5. Honest gaps in the evidence

The vendor literature on captions and promo video is louder than the
empirical record. Things to flag rather than assert:

- **No clean caption-isolated A/B benchmark exists for short-form
  commerce.** The "+34% conversion" numbers floating around all confound
  caption design with thumbnail / script / audio / product changes.
- **Empirical data on which specific words to highlight is thin**: the
  Weingärtner study (MUM '24) is the only academic piece, and it's a
  language-learning context, not commerce. The "highlight nouns and
  numbers" heuristic is vendor-converged but not study-backed.
- **Quantitative engagement gap between disclosed and undisclosed
  sponsorship** is asserted by TikTok's own data; no independent study
  confirms.
- **"Viewers without dynamic captions are ignored by 65%"** is
  folkloric. Treat any viral statistic without primary methodology with
  skepticism.

The defensible posture: captions are a **precondition for sound-off
completion**, completion is the **precondition for the CTA frame to be
seen**, and selective highlighting + trust signals + a well-placed CTA
maximize the conversion *given that the viewer stayed*. Frame project
decisions around enabling that funnel rather than chasing direct
caption-to-CTR multipliers.

---

## Sources

- Hook patterns — [Captions.ai Hook Writing Guide](https://captions.ai/help/guides/marketing/hook-writing) / [OpusClip TikTok Hook Formulas](https://www.opus.pro/blog/tiktok-hook-formulas) / [OpusClip Best TikTok Hooks 2026](https://www.opus.pro/blog/tiktok-hooks-that-go-viral-2026) / [TTS Vibes — TikTok First 3 Seconds Retention Stats](https://insights.ttsvibes.com/tiktok-first-3-seconds-hook-retention-rate/)
- Sound-off audience — [Clicks.video — Silent-First Editing](https://www.clicks.video/blog/silent-first-editing-captions-text-overlays-and-visual-hooks-for-sound-off-viewing) / [Manchester Digital — Mute Is the New Norm 2025](https://www.manchesterdigital.com/post/title-productions/mute-is-the-new-norm-why-captions-win-in-2025-video) / [Zebracat — 150+ Video Marketing Statistics 2025](https://www.zebracat.ai/post/video-marketing-statistics)
- E-commerce conversion — [Firework — Video Content & E-commerce Conversion](https://firework.com/blog/how-video-content-boosts-conversion-rates) / [Outfy — TikTok Affiliate Marketing 2025](https://www.outfy.com/blog/tiktok-affiliate-marketing/) / [Influencers Time — TikTok Shop Add-to-Cart Rates](https://www.influencers-time.com/tiktok-shop-product-links-that-boost-add-to-cart-rates/)
- CTA placement — [Wistia — Using Video CTAs (State of Video 2025)](https://wistia.com/learn/marketing/using-video-ctas) / [Nemo Video — CTA Examples 2026](https://www.nemovideo.com/blog/video-cta-examples) / [CapCut — CTA Video Examples](https://www.capcut.com/resource/best-cta-videos)
- FTC `#ad` disclosure — [FTC Endorsement Guides FAQ](https://www.ftc.gov/business-guidance/resources/ftcs-endorsement-guides-what-people-are-asking) / [Influencer Marketing Hub — FTC Disclosure Checklist 2025](https://influencermarketinghub.com/ftc-disclosure-checklist-by-platform/) / [Luthor — 2025 Reels Compliance Checklist](https://www.luthor.ai/guides/ultimate-2025-instagram-reels-compliance-checklist-ftc-influencer-disclosure-rules)
- Trust signals & de-influencing — [Stack Influence — Authenticity & Transparency 2025](https://stackinfluence.com/authenticity-transparency-influencer-2025-guide/) / [Amra & Elma — Consumer Trust in Influencers 2026](https://www.amraandelma.com/consumer-trust-in-influencers-statistics/) / [HBR — Influencer Marketing That Customers Actually Trust (Dec 2025)](https://hbr.org/2025/12/how-to-do-influencer-marketing-that-customers-actually-trust) / [BBB Programs — 2025 Influencer Trust Index](https://bbbprograms.org/media/insights/blog/influencer-trust-index) / [Frontiers — De-influencing Wave](https://www.frontiersin.org/journals/communication/articles/10.3389/fcomm.2025.1600657/full) / [Skeepers — TikTok Branded Content Policy](https://community.skeepers.io/blog/tiktok-new-policy/)
- AI-driven highlighting (academic) — [Useful but Distracting: Keyword Highlights in Captions for Language Learning (MUM '24)](https://arxiv.org/abs/2307.05870)
