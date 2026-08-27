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
  Reels UI overlay zones (canonical safe-zone numbers).
- [audio-best-practices.md](audio-best-practices.md) — the sound-on layer:
  trending audio, voiceover/music mix levels, ducking.

---

## The 6 promo-video rules that matter (cheat-sheet)

1. **First-3-second hook is a large, legible statement, distinct from the running captions.** 5-8 words, on screen for the full 1.5-3 s, visibly larger than narration captions. The swipe decision lands in the first ~1-1.5 s. Whether the hook is a static card or animated word-by-word is not settled (caption tools argue a moving fixation point holds better through the retention cliff); what matters is that it's big, readable sound-off, and lands fast. **Do not render the hook as the same words as the bottom captions at the same time** — that reads as redundant clutter. Make the hook a distinct authored headline (the pattern Submagic/OpusClip use), or suppress the captions while the card is up. The hook should also front-load the search keyword in spoken audio within the first 5 s; spoken audio is one TikTok search signal alongside caption / on-screen text / hashtags, not the dominant one.
2. **CTA gets its own staging**, distinct from narration. Pair an early **soft CTA** (3-5 s, neutral) with a **hard CTA** at the end (full-frame, accent color, larger, static, ≥1.5 s on screen). Red/orange beats green in independent A/B tests.
3. **`#ad` disclosure is an on-frame overlay AND first-line caption text.** FTC wants the disclosure in the video itself and at the top of the caption before other text. Same font family as captions, sized for legibility (not a fixed ratio), fixed corner; full-clip persistence is a safe short-form default rather than a codified duration. Max statutory penalty is $53,088 per violation (2025 figure, still current 2026) — a cap reached via a Notice of Penalty Offense or consent-order violation, not an automatic per-post fine.
4. **State at least one trade-off per video.** Trust converts; absolute superlatives ("life-changing", "obsessed") now actively reduce trust in 2025-2026 data. A dedicated downside beat is the strongest trust signal in the de-influencing era — disclosed sponsorships do not depress engagement.
5. **End with an engagement-bait closing line right before the hard CTA.** Personal and storytelling content closes with a two-option opinion question (comment-fork); analytical and comparison content closes with a debatable but defensible spec claim. The closing line drives comments and saves, both of which feed the algorithm. It is additive to the CTA, not a replacement; generic "Comment YES if..." asks are spam-filtered.
6. **Keep something moving every 1.5-3 seconds.** Each slide change, punch-in, or text pop resets the attention clock. Hold no single static frame past 4-5 s without a visual change. Younger audiences need the tighter end of the band (cut every 2-4 s); a high-density profile pushes to 1.5-3 s with a transition between slides.

---

## 1. Hook patterns (first 1-3 seconds)

The swipe decision on a vertical autoplay feed lands in the first ~1-1.5 s, and
a video that fails the first ~3 s of retention gets little algorithmic push.
(Precise figures that circulate in vendor blogs — a "1.7 s window," a "1.5 s
ranking trigger," "84.3% of viral TikToks used a hook trigger" — trace to
single un-sourced posts with no disclosed methodology. Treat them as narrative,
not data; measure your own 2 s / 3 s view-through by hook style instead.)

**Word budget**: read aloud in ≤3 s. 5-8 words.

**Patterns that work for product video specifically:**

| Pattern | Example | Why it works |
|---|---|---|
| Price-first reveal | "The $15 [thing] that…" | Specific number = credibility + curiosity gap |
| Regret / contrarian | "I regret buying this", "Don't buy X until you see this" | Negative framing > positive in cold audiences |
| POV: | "POV: you finally found a [thing] that doesn't [pain]" | Reader becomes protagonist; instant context |
| Outcome-first | "This fixed my [pain] in 30 seconds" | Result hook; works on warm audiences |
| Numbered teardown | "3 reasons I'm returning this" | Lists promise structure → lower cognitive cost |
| Comparison | "$15 vs $200 — same thing?" | Pattern interrupt + value framing |

**Caption-rendering implication**: the hook needs its own on-screen treatment,
**distinct from the running captions**. Two shapes both work: a static title
card (single fade-in, no per-word reveal) or an animated word-by-word hook. The
evidence doesn't settle which holds better through the retention cliff, so A/B
test it. What is NOT defensible is rendering the hook as the same words as the
bottom captions at the same time: for ~1.5 s the viewer reads the first sentence
twice, which is clutter. The standard (what Submagic and OpusClip do) is a
distinct authored hook headline — different, punchier copy than the first spoken
line — layered above the running captions; the alternative is to suppress the
captions while the card is up. Size the hook visibly larger than narration
captions. See subtitle-best-practices for caption design rules.

**Title card vs. text-over-frame — two shapes, different timing**:

- **Static title card** (default on `slideshow_short_20s`):
  1.0-1.5 s, **hard cut to motion** (no fade between card and the first
  slideshow segment), 3-5 words capped at 7, ALL CAPS-leaning or bold
  weight, 10-15% of frame height. The card is the first thing on screen
  and gives way to motion immediately.
- **Text-over-mid-action-frame** (longer profiles): 1.5-3.0 s, can fade in,
  text sits over a frame that already carries motion (Ken Burns settle-zoom
  or a video clip mid-action). The hook reads alongside visible movement
  rather than as a separate beat.

The pipeline's `hook_overlay` setting in `config/video_production.yaml`
controls the duration. Set `duration_sec: 1.5` for title-card behaviour,
`duration_sec: 2.5` for text-over-frame. Both share the same drawtext
implementation; the duration choice carries the design intent.

**Pre-motion on static product photos**: the first slide should already be
mid-motion when frame 1 lands. The sourced practice is continuous slow Ken
Burns motion on any still (2-5 s per image is the cited range); the specific
**0.3-0.5 s settle-zoom** micro-timing here is a project heuristic, not an
industry-sourced number, so tune it by measured retention rather than treating
it as a spec. 0.2 s is defensible only when burned-in text is on frame 1 (the
text alone gives the eye a focal point). The
pipeline's `first_frame_pre_motion: true` + `pre_motion_peak_zoom: 1.10`
defaults sit at the upper edge of the band; tune `pre_motion_peak_zoom`
between 1.05 and 1.15 to taste.

**Opener fatigue**: identical pattern-interrupt structure for ~3 weeks
shows measurable diminishing returns. Rotate at least 2-3 cold-open
variants per pillar so the channel doesn't read as a template factory
at the aggregate level. The pipeline's `cold_open_variant_pool`
deterministically picks one variant per product render and persists the
chosen variant in `pipeline_state.json` for downstream analytics.

Hook caption sizing and word budget converge across Captions.ai, OpusClip,
and Submagic 2025-2026 guidance.

**Audio is one search signal.** TikTok indexes spoken-audio transcripts
alongside captions, on-screen text, and hashtags. Front-loading the hook
keyword (product category, price band, audience cue, pain point) in the first
5 seconds of spoken audio is a robust, low-risk move regardless of the exact
ranking weight. (Vendor claims that ASR is "the primary" signal, or that a
triple-mention ranks "2-3x" better, are un-sourced precision — keep the tactic,
drop the numbers.) All six patterns above embed the keyword naturally
without the opener reading as a search-bar query. **Anti-pattern**: literal
Google-query syntax like "Best [category] under $[N] for [audience]." Reads
as keyword stuffing in voiceover; sacrifices the conversational register
that holds the viewer through the first few beats. The project's script
templates encode the six patterns as the rule for line one and call out
the Google-query shape as an explicit anti-pattern.

## 2. Cut cadence and motion density

Vertical feeds reward visual energy. A frame that hasn't changed in a few
seconds reads as "nothing happening" and the viewer swipes. The fix is a steady
beat of visual change, tuned to the audience.

**Shot-length bands** (vendor-converged across 2026 editing tooling):

| Audience / profile | Cut every | Notes |
|---|---|---|
| General short-form | 1.5-3 s | The default high-retention band |
| Younger / Gen Z feeds | 2-4 s, pushing to 1-2 s on high-energy edits | The algorithm reclassifies into faster feeds; match it |
| First shot (hook) | change within 1-1.5 s | Signals pace immediately, see section 1 |
| Hard ceiling | never hold a static frame past 4-5 s | Add a punch-in, cut, B-roll, or text pop |

**Motion within a shot counts as a cut.** A Ken Burns settle-zoom, a punch-in,
or a text pop resets the attention clock without an actual edit. On static
product photos the pipeline's pre-motion (section 1) supplies this; on a
slideshow, the slide change itself is the beat.

**Transition vocabulary**, lightest to heaviest:

- **Hard cut** -- the default. Keeps momentum without drawing attention to
  itself. Most cuts should be hard cuts.
- **Whip pan** -- energetic scene-to-scene transition; use sparingly so it
  stays a pattern interrupt, not a tic.
- **Zoom punch** -- quick scale-in to emphasize a price, spec, or benefit beat.
- **J-cut / L-cut** -- audio leads or trails the video edit; smooths a
  narration handoff so the cut doesn't feel abrupt.

**One transition style per video.** Mirrors the subtitle one-effect rule:
mixing whip pans, zoom punches, and slides in one clip reads as amateur. Pick
the cadence and the transition from the profile, not per-slide.

**Pipeline mapping**: the high-density cut profile (`cut_density: high`,
roadmap 1.4) drops the minimum slide duration to 1.5-3 s and inserts one
transition between every slide. Keep the slower-cut profile available for
audiences and platforms where a calmer pace fits. Match cut speed to content
energy; a calm productivity review and a Gen Z gadget teardown should not share
a cadence.

## 3. Sound-off as the primary audience

**A large majority of feed video is watched sound-off.** (The widely-cited
"85%" traces to a 2016 Facebook, publisher-reported figure, so treat the exact
number as dated; the direction is solid and repeated across 2025-2026 sources.)
Design the entire video assuming sound-off is the default; treat sound-on as
the accessibility layer, not the source of truth.

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
- **Mute resistance**: the defensible stat behind the folklore is a skip
  comparison, not a mute rate — roughly 22% skip captioned videos vs ~39%
  skip uncaptioned ones. Captions don't directly boost CTR; they prevent the
  skip that kills CTR.

## 4. Trust signals & FTC `#ad` disclosure

**FTC compliance (US, 2023-updated Endorsement Guides) is the floor**:

- Disclosure must be **clear and conspicuous**. For Reels/TikToks, the
  agency wants a **two-punch**: on-screen overlay (`#ad`, `Sponsored`,
  `Paid partnership`) PLUS the same disclosure at the **top of the
  caption text**, before any other text or hashtags.
- Disclosure must be **on screen long enough to read**, in a font/color
  that contrasts the background. Full-clip persistence is a safe short-form
  default (a viewer can enter mid-scroll), not a codified FTC duration; the
  binding test is "clear and conspicuous / unavoidable."
- Penalties: statutory **maximum of $53,088 per violation** (2025 figure,
  still current in 2026). It's a cap reached via a Notice of Penalty Offense
  or consent-order violation, not an automatic fine on a single mislabeled post.
- Practical rendering: same font family as the project's caption
  configuration, fixed top-left or top-right corner, full-clip duration. The
  FTC judges legibility and unavoidability (large enough to read on a phone,
  high contrast, stable placement clear of platform UI), not a pixel ratio —
  "~50-60% of caption size" is a design heuristic, not an FTC rule.
- **Corner conflict**: when the burned-in hook overlay sits centre-upper
  (the pipeline's default), the `#ad` disclosure goes top-left (or
  top-right). Don't stack both at top-centre — they compete for the
  same attention zone in the 1.5-second decision window.

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

## 5. Closing-line beat — comment-fork or spec-correction

One short engagement-bait line, right before the hard CTA, not replacing it.
TikTok's algorithm rewards comments and saves as high-intent engagement signals;
a closing line that invites a reply outperforms a video that ends on the
sponsor pitch. Generic engagement bait ("Comment YES if you agree") is widely
reported to be down-ranked by the moderation layer while specific, opinion-driven
prompts pass. This is industry consensus and aligns with TikTok's stated stance
against engagement-bait, though the exact "spam-filtered" mechanism isn't
documented by the platform — keep the tactic, hold the mechanism loosely.

**Two flavours, picked by template framing:**

| Flavour | When to use | Example | Why it works |
|---|---|---|---|
| **Comment-fork** | Personal, storytelling, discovery framing (POV / regret / outcome-first hooks) | "USB-C or Lightning — which still annoys you more?" / "Team magnetic or team plug-in?" | Two-option opinion questions invite a one-tap pick; viewers reply because the answer is fast. |
| **Spec-correction** | Analytical, comparison, numbered-teardown content (price-first / comparison / numbered hooks) | "Most people only need two ports, but three is usually better." / "65W is the sweet spot for laptops." | Debatable but defensible claims invite "well, actually" replies. The audience that knows the spec types because they want to correct you. |

**Word budget**: ≤15 words. The line is one beat, not a paragraph. Read it
out loud before locking — if it reads as a rhetorical question with no real
options ("Isn't that crazy?"), rewrite it as a specific opinion fork or
spec claim.

**Render staging**: the closing line lives in the script (TTS reads it), in
the on-screen subtitle (subtitles ARE the script in this project), and
optionally in the platform caption. It comes BEFORE the hard CTA frame, not
on the same frame.

**Anti-patterns**:
- "Comment YES if you agree" — generic, spam-filtered
- "What do you think?" — too open, no reply hook
- A question with no opinion attached ("Have you seen this before?")
- Replacing the CTA with the closing line — both beats are needed, not one

The project's script templates encode comment-fork for personal/storytelling
templates and spec-correction for analytical/comparison templates. See
`src/ai/prompts/scripts/` for the per-template Rules block.

## 6. CTA staging — soft + hard, two-stage

**Wistia State of Video 2025** analyzed 36,000+ video CTAs:

- Video CTAs convert at **~16% on average** across the Wistia platform (verified).
- For videos under 60 s, place the **CTA in the first quarter** (verified Wistia guidance).
- The "well-placed CTAs reach ~40%" figure and the "soft-early + hard-late
  double-tap outperforms a single end-card" framing are NOT in Wistia's report:
  the 40% is an unattributed secondary citation, and the double-tap is a
  playbook tactic (a sound one worth testing), not a Wistia finding.

**Two-stage pattern (vendor consensus):**

| Stage | When | Style | Purpose |
|---|---|---|---|
| Soft CTA | 3-5 s | Small caption line, neutral color, e.g. "link in caption" + arrow | Plant intent without interrupting the hook |
| Hard CTA | Last 2-4 s | Full-frame text, accent color, 1.3-1.5× narration size, static or single pulse, ≥1.5 s on screen | Convert the warm viewer |

**Hard-CTA caption rendering:**

- High-contrast accent color, different from the karaoke highlight. "Red/orange
  beats green" replicates in A/B data, but the tests measure **contrast, not
  color** — a CTA wins by standing out against its surrounding palette, and copy
  and placement beat color as levers. Use whatever contrasts your palette.
- Larger font (~1.3-1.5× narration), **static** not karaoke, on screen ≥1.5 s.
  These are sound UX defaults (static reads as "stop, act"; sub-1.5 s blinks
  past in autoplay), not sourced findings — treat as craft heuristics to A/B.

**Verb choice**: imperative + specific outcome. "Get the $15 fix" beats
"Shop now". "Link to the bag in bio" beats "Click here".

**Where the CTA can actually point, per surface.** The wording only matters if
the destination is reachable, and that varies by platform in ways that decide
the whole CTA design:

| Surface | Clickable destination? |
|---|---|
| YouTube Shorts description | No. URLs render as plain text |
| YouTube Shorts comments (incl. pinned) | No. Same anti-spam rule |
| YouTube channel profile / About | Yes, up to 14 links |
| Instagram Reels caption | No. Bio link only |
| Instagram Stories link sticker | Yes, open to all accounts |
| TikTok caption | No. Bio link only |

YouTube's restriction is the one that surprises people, and it is decisive for
short vertical video: a video classified as a Short cannot carry a clickable
link on any surface the uploader controls per-video. Classification is
automatic from aspect ratio and duration, so opting out is not available to a
9:16 promo clip. The practical consequences: point the CTA at the profile
("link in bio", "links on my channel") rather than at a description or comment
URL; treat the pinned comment as a device for earning a profile visit, not for
delivering a destination; and do not spend engineering effort on injecting
destination URLs into YouTube Shorts metadata, because the platform will render
them inert.

Sources: [Sharing links with your audiences — YouTube Help](https://support.google.com/youtube/answer/13748639?hl=en).

## 7. AI-content disclosure (platform policy, not FTC)

YouTube and TikTok treat AI-generated content separately from sponsored
content. The disclosure surfaces are different and both apply on top of
the FTC `#ad` overlay above.

- **YouTube**: the July 2025 policy update (renamed "repetitious" to
  "inauthentic content") targets mass-produced, templated, low-human-input
  content for reduced reach and demonetization — the trigger is *unoriginality*,
  not AI generation itself. For an automated render pipeline the real exposure
  is shipping near-identical videos at scale, so vary hooks, scripts, templates,
  voices, and cadence across renders (see the originality note below).
  Separately, YouTube's "altered or synthetic content" disclosure applies only
  to *realistic* synthetic media; a product slideshow with AI voiceover over
  obvious product imagery doesn't require it, and YouTube's own examples list
  AI-written scripts and synthetic voiceover among the cases that don't. So
  `containsSyntheticMedia` is opt-in and off by default
  (`config/publisher.yaml::synthetic_media_disclosure`). Turn it on for output
  that does meet the bar — AI-generated music, or AI-generated footage of a
  real place.
- **TikTok**: as of 2026, TikTok auto-flags AI content via C2PA detection.
  Auto-flagging suppresses distribution BEFORE removal; explicit
  disclosure via TikTok's AI-content label keeps reach intact. The
  publisher sets this flag on every TikTok payload, on by default,
  configurable at `config/publisher.yaml::tiktok_settings.video_made_with_ai`.
  No manual per-render step is needed. The opt-out does not reach a global
  batch run, which uses the dataclass defaults and so keeps the label on
  (issue #255); see `docs/compliance.md`.
- **Instagram / Reels**: enforced now, not draft. As of 2026 Meta runs
  automatic AI detection-and-enforcement, strongest on the paid/ads surface
  (auto-applies a disclosure label, rejects undisclosed AI creative, can
  retroactively flag running campaigns); organic-Reels detection is more
  metadata- and self-declaration-driven. Self-declare AI provenance (embed
  C2PA / IPTC at generation) and set the platform AI toggle — self-labeling
  preserves reach; getting auto-flagged undisclosed is what costs distribution.

The pipeline already prepends `#ad` to caption text and burns the corner
disclosure. AI-content disclosure is additive, not a replacement.

**Originality is the real 2026 AI-reach risk.** No platform penalizes *labeled*
AI content on its own, but all three deprioritize unoriginal, templated,
mass-produced output, and that is exactly the failure mode of an automated
pipeline that renders the same shapes repeatedly. Treat variety as a
reach-preservation requirement: rotate hook patterns, script templates, voice,
cut cadence, and cold-open variant per render so the aggregate output doesn't
read as a template factory. The pipeline already has the variant frameworks;
the discipline is using them.

**2026 platform shifts a 2025 playbook would miss.** TikTok's US operation was
divested to a US joint venture in January 2026 and its recommendation algorithm
is being retrained on US-only data, so US-reach assumptions are provisional
right now. Instagram removed the longer-Reels penalty (recommends up to ~3 min
to non-followers), weights DM-sends ~3-5x above likes for reaching new
audiences, and capped hashtags at 5 (December 2025) — the old "use 30 tags"
guidance now hurts reach.

## 8. Honest gaps in the evidence

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
