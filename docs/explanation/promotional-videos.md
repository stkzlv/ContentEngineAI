# Promotional videos: why the pipeline builds them this way

This page explains the hook, the cut cadence, the closing line, the call to action, disclosure and the trust signals in a product render: what the pipeline does and the evidence behind it. The script rules live in the prompt templates in `src/ai/prompts/scripts/` and in `script_templates` in `config/ai_services.yaml` (narrator profile, `cta_options`). The on-frame elements live in `video_settings` in `config/video_production.yaml` (`hook_overlay`, `disclosure_overlay`, `cold_open_variant_pool`) and in the video profiles. Disclosure detail is in [the compliance guide](compliance.md). Caption design is in [the captions page](captions.md), and the UI overlay zones are in [the platform safe zones](platform-safe-zones.md).

Each finding carries a grade, `[A]`, `[B]` or `[C]`, defined in [the evidence grades](../design/README.md#evidence-grades). Where a vendor figure disagrees with a graded finding, the graded finding wins.

## Hook patterns

Every product template tells the model to open with a conversational hook that carries the search keyword (product category, price band, audience cue or pain point) within the first five seconds of speech (`REQ-CNT-016`). The rule lists six shapes and names the literal search-query shape ("Best [X] under $[N] for [Y]") as an anti-pattern (`REQ-CNT-017`). Line one states a concrete fact, result or observation, and setup framings such as "Today I'll show you" or "In this video" are named as anti-patterns (`REQ-CNT-018`).

| Pattern | Example | Why it works |
|---|---|---|
| Price-first reveal | "The $15 [thing] that..." | A specific number gives credibility and a curiosity gap |
| Regret or contrarian | "I regret buying this", "Don't buy X until you see this" | Negative framing beats positive in cold audiences |
| POV | "POV: you finally found a [thing] that doesn't [pain]" | The viewer becomes the protagonist, with instant context |
| Outcome-first | "This fixed my [pain] in 30 seconds" | A result hook; works on warm audiences |
| Numbered teardown | "3 reasons I'm returning this" | A list promises structure, at a lower cognitive cost |
| Comparison | "$15 vs $200, same thing?" | A pattern interrupt with value framing |

Why:

- The swipe decision on a vertical autoplay feed lands in the first 1 to 1.5 seconds, and a video that fails the first three seconds of retention gets little algorithmic push. The hook shapes and the word budgets converge across Captions.ai, OpusClip and Submagic guidance from 2025 and 2026 (see [Sources](#sources)).
- Each negative word raised click-through by about 2.3% across 105,000 headline variants, which supports honest "mistake" and "don't buy" hooks. [A] [Nature Human Behaviour](https://www.nature.com/articles/s41562-023-01538-4)
- Concrete beats mysterious: name the product, number or situation and withhold only the answer; "you won't believe this" is past the optimum ([design 0007](../design/0007-script-lint.md) has the source). [A]
- Across 14,424 videos from 355 accounts, spoken question hooks averaged 10.08 times account baseline views against 7.04 for statements, but only specific, audience-targeted questions won, and topic confounds the result. [B] [The Content Labs](https://thecontentlabs.app/blog/question-hooks-data-study)
- Across 4,148 hooks from million-view videos, the median Shorts hook was 11 words, a third used "you" or "your", and no single feature tracked with higher view tiers. [B] [Overseer](https://www.overseeros.com/blog/best-youtube-hooks)
- In TikTok's coding of its ads, the first 2 seconds matter most for ad recall and the first 2.5 seconds for awareness (2021, paid ads, not organic reach). [B] [TikTok Creative Center](https://ads.tiktok.com/business/creativecenter/quicktok/online/Power_Creative_Elements/pc/en)
- Treat the opening like a thumbnail: in a 2023 study of 5,400 Shorts, those below 60% viewed-vs-swiped-away rarely performed well. [B] [Galloway thread](https://threadreaderapp.com/thread/1646898356419981315.html)

Not supported: the precise figures that circulate in vendor blogs (a "1.7 s window", a "1.5 s ranking trigger", "84.3% of viral TikToks used a hook trigger") trace to single posts with no method. Measure your own 2 s and 3 s view-through by hook style instead.

Built and held off ([design 0007](../design/0007-script-lint.md)): a script lint behind `script_validation.lint.enabled` (`REQ-CNT-053`), and stricter hook rules (concreteness, a "but/therefore" chain) with the search phrase leading the headline and captions, behind `script_templates.hook_rules.enabled` (`REQ-CNT-054`).

## Spoken search phrase

The hook rule puts the keyword in the first five seconds of speech because TikTok indexes spoken-audio transcripts alongside captions, on-screen text and hashtags. Front-loading the keyword is a low-risk move whatever its exact ranking weight, and all six hook shapes carry it without reading as a search-bar query. The literal query shape is an anti-pattern because it reads as keyword stuffing in a voiceover and loses the conversational register that holds the viewer.

Not supported: vendor claims that speech recognition is "the primary" signal, or that saying the phrase three times ranks "2-3x" better, have no published method. Keep the tactic, drop the numbers.

Planned: where the hook rules are on, the hook headline and every platform caption lead with the search phrase, and a report shows per render whether the phrase appears in the first spoken sentence, the headline and each caption ([design 0007](../design/0007-script-lint.md), `REQ-CNT-054`, `REQ-CNT-055`).

## Hook overlay

The render draws a hook headline as static centre-upper text over the first `hook_overlay.duration_sec` seconds (default 1.5), with no per-word reveal (`REQ-VID-077`). The producer writes the headline for the screen, separately from the spoken script, so it doesn't repeat the first caption (`REQ-VID-083`); a product headline names the category and never a model designation (`REQ-VID-084`). The headline is capped at `max_words` (default 7, `REQ-VID-086`), wraps and shrinks to fit (`REQ-VID-080`), and falls back to the first spoken sentence when no headline is available (`REQ-VID-088`). The font is `size_factor` 1.1 times the subtitle base font. The `#ad` disclosure is drawn above the overlay in its own corner (`REQ-VID-078`).

Why:

- The first frame has to work on mute; one top Shorts creator keeps hooks at a fifth-grade reading level and foreshadows the payoff within three seconds. [C] [Creator Science podcast](https://podcast.creatorscience.com/jenny-hoyos/)
- Text overlay works best when it stands out from the frame while fitting its style. [A] [Journal of Marketing](https://journals.sagepub.com/doi/10.1177/00222429251322773)
- The same words in the headline and in the bottom captions at the same moment make the viewer read the first sentence twice. The pattern caption tools such as Submagic and OpusClip use is a distinct authored headline above the running captions, which is what the producer writes.
- Whether a static headline or an animated word-by-word hook holds better is not settled; caption tools argue a moving fixation point holds better through the retention cliff. A/B test it rather than assume it.

The shipped `size_factor` of 1.1 gives about 5.5% of frame height, below the 10-15% band vendors quote and possibly smaller than the rendered captions. Measure one render before relying on it. Set a longer `duration_sec` (for example 2.5) when the headline should read alongside visible motion rather than as its own beat.

Planned: a cover image carrying the hero visual and the hook headline ([design 0011](../design/0011-cover-frames.md), `REQ-VID-014`).

## Cold open

Where `first_frame_pre_motion` is on, the first image starts at a slight zoom and settles to 1.0 over its segment, so frame 0 is already in motion (`REQ-VID-074`). It is off by default and on in `slideshow_short_20s` (`REQ-VID-075`); `pre_motion_peak_zoom` defaults to 1.10 (`REQ-VID-076`). Where `still_motion` is on, every still after it moves too, with a push, pull or pan drawn per product and per image (`REQ-VID-010`, held off until the reach-test readout). Each render also picks a named variant from `cold_open_variant_pool` per product and records it for analytics, but the variants render identically, so the pool adds no visual variety (`REQ-VID-090`, partial).

Why: identical pattern-interrupt openings over weeks are reported to lose effect [C], and templated sameness is what the platforms' originality rules target (see [Originality and variety](#originality-and-variety)).

Planned: motion on every still, not only the first ([design 0001](../design/0001-motion-on-every-still.md), `REQ-VID-010`).

## After the hook

The narrator profile in `script_templates.narrator_profile` targets 30-40 seconds of speech (roughly 75-100 words). The `slideshow_short_20s` profile is a shorter canvas (about 50-60 words) for testing hooks against a fixed body.

Why:

- Hook, retain, reward: the hook earns the watch, the middle pays small curiosity loops, the end fulfils the hook's promise. The testing method suits automation: one fixed body with several hooks, then iterate on the winners. [C] [Dickie Bush on Alex Hormozi](https://dickiebush.substack.com/p/i-invested-45000-in-alex-hormozis)
- "But" and "then" beats and a visible progression ("three steps") keep viewers to the end. [C] [Creator Science podcast](https://podcast.creatorscience.com/jenny-hoyos/)
- Storytelling outperformed other post types on views in a four-month field experiment across 202 TikTok posts (working paper, small study), which is one reason the template pool includes `story_driven`. [B] [MPRA](https://mpra.ub.uni-muenchen.de/123280/1/MPRA_paper_123280.pdf)
- Length: one top creator targets 34 seconds, and in 2023, when Shorts were capped at 60 seconds, Shorts with an average view duration above 50 seconds averaged 4.1 million views. 30-45 seconds is defensible; lengthen only when the retention curve holds. [B] [Galloway thread](https://threadreaderapp.com/thread/1646898356419981315.html)

Planned: the "but/therefore" chain as a prompt rule ([design 0007](../design/0007-script-lint.md)).

## Cut cadence and motion

A slideshow divides the voiceover duration evenly across its images (`REQ-VID-003`), so the pace follows script length and image count, not a fixed setting. A 35-second script over five images holds each slide for 7 seconds, past the 4-5 second ceiling below. Consecutive media elements join with a crossfade (`REQ-VID-005`, `video_transition_duration` 0.5 s on the profiles that set it). The render applies no colour filter or grade, and draws no progress bar.

Why:

- Stimulation follows an inverted U: moderate pace gets the most engagement. [B] [arXiv 2604.19995](https://arxiv.org/abs/2604.19995) (a 2026 preprint)
- A visual change every 3-5 seconds, list items about every 3 seconds, measured frame by frame on four of one creator's Shorts. [C] [WritePanda](https://www.writepanda.ai/blog/how-to-edit-shorts-like-ali-abdaal/) Hold no static frame past 4-5 seconds; motion within a shot (a settle-zoom, a punch-in, a text pop) resets the clock without an edit.
- In TikTok's ad coding, seamless transitions gave 14% more view time and surprising transitions 53% more brand recall (paid ads). [B] [TikTok Creative Center](https://ads.tiktok.com/business/creativecenter/quicktok/online/Power_Creative_Elements/pc/en)
- In 9,654 brand TikToks, having no visual filter predicted better performance, and editing pace had only modest predictive value (one preprint). [B] [arXiv 2606.16053](https://arxiv.org/html/2606.16053)

Not supported: a cut every 2 seconds has no measured support. Progress bars show gains only in the reports of vendors that sell them, and a bar on every video is a template tell. [C]

Planned: motion on every still ([design 0001](../design/0001-motion-on-every-still.md), `REQ-VID-010`), cuts snapped to music beats ([design 0005](../design/0005-beat-snapped-cuts.md), `REQ-VID-013`), and cut density as a recorded variant dimension ([design 0006](../design/0006-render-choices-and-variety-report.md)).

## Sound off and sound on

Captions are burned into every render and carry the whole script, so a muted viewer can follow it, and the call to action appears on screen as well as in the voiceover. The voice, music and mix carry the render for viewers who listen.

Why: 93% of US TikTok users spend time with sound on (TikTok Marketing Science, 2020). [B] [TikTok Creative Center](https://ads.tiktok.com/business/creativecenter/quicktok/online/Power_Creative_Elements/pc/en) Muted viewing is common on other feeds, so every beat has to work both ways.

Not supported: the widely cited "85% watch muted" traces to a 2016 publisher-reported Facebook figure and does not describe TikTok.

## Closing line

Every product template ends the script with one short closing question or claim right before the call to action, not in place of it (`REQ-CNT-019`). Personal and storytelling templates close on a two-option opinion question; analytical and comparison templates close on a debatable but defensible spec claim (`REQ-CNT-020`), grounded in a measurement the description states, or in a material, shape or use claim when it states none (`REQ-CNT-021`, `REQ-CNT-022`). The caption generator places the closing line in the caption body (`REQ-CNT-039`), and the bundled config posts it as the YouTube first comment (`REQ-PUB-037`).

| Flavour | Templates | Example |
|---|---|---|
| Comment-fork | Personal, storytelling, discovery framing | "USB-C or Lightning, which still annoys you more?" |
| Spec-correction | Analytical, comparison, numbered teardown | "65W is the sweet spot for laptops." |

Why:

- A two-option question invites a one-tap pick; a debatable but defensible claim invites "well, actually" replies from viewers who know the spec. An open "What do you think?" or a question with no opinion attached gives no reply hook.
- Comments are a weak ranking signal on Shorts (2023) [B] ([Galloway thread](https://threadreaderapp.com/thread/1646898356419981315.html)), so the line is for conversation and profile visits, not rank.
- Meta documents the demotion of comment, share, tag and vote baiting [A] ([Meta](https://transparency.meta.com/features/approach-to-ranking/content-distribution-guidelines/engagement-bait/)), and TikTok's For You feed standards exclude engagement manipulation, including false incentives for following [A] ([TikTok](https://www.tiktok.com/community-guidelines/en/fyf-standards)). Genuine requests for opinions or experiences are exempt, which is the line this beat stays on. [Design 0008](../design/0008-bait-free-closing-lines.md) carries the rest of the evidence.

Planned: the CTA pools carry share requests ("Share with someone who needs this."), which Meta lists as bait; [design 0008](../design/0008-bait-free-closing-lines.md) replaces them, so that no configured call to action or closing example asks viewers to share, tag, vote or reply with a specific word (`REQ-CNT-041`).

## Calls to action

Every script ends on exactly one configured call to action, verbatim, as its final sentence: `script_templates.cta_options` for products, `cta_options_topic` for topics (`REQ-CNT-033`). One line is chosen per record by a salted hash, and only that line is rendered into the template rules (`REQ-CNT-034`); validation rejects a script whose last sentence is not a configured line (`REQ-CNT-035`), and `--cta` or `fixed_cta` forces one (`REQ-CNT-037`). The call to action is spoken and captioned; the render has no separate end card and no early "soft" call to action ([decision 0009](../decisions/0009-no-staged-cta-until-a-click-path-works.md)).

Why:

- Choosing one line per product rather than letting the model pick spreads the closing lines across a batch; quoting all four let the model pick the first every time, which is the templated sameness the platforms throttle.
- A standalone end card after the last spoken line can become the "goodbye second" where viewers leave. One creator measured it; nobody has at scale. Shorts have no end screens; the substitute is the "related video" link. [A] [YouTube Help](https://support.google.com/youtube/answer/14075157)

Not supported: Wistia's State of Video 2025 report does not contain the "well-placed CTAs reach about 40%" figure or the claim that a soft-early plus hard-late call to action beats a single end card. The 40% is an unattributed secondary citation, and the two-stage pattern is a playbook tactic, not a finding. [Wistia](https://wistia.com/learn/marketing/using-video-ctas)

Built and held off until the reach-test readout: ending on the last spoken word, with no silent tail after the call to action, and optionally closing on the opening frame so a replay reads as continuous (`ending: peak` or `loop`, [design 0002](../design/0002-end-on-the-peak.md), `REQ-VID-011`). The call to action stays the last sentence.

## Where a call to action can point

One product CTA points at the profile ("Link in bio if you want one."); the others ask for a follow, a comment or a share until the pool is reworked (#549). The publisher adds each product's affiliate link to a link-in-bio page after publishing (`REQ-PUB-062`). The Instagram first comment carries a link-in-bio pointer (`REQ-PUB-037`).

| Surface | Clickable destination? |
|---|---|
| YouTube Shorts description | No. URLs render as plain text |
| YouTube Shorts comments, pinned included | No. Same anti-spam rule |
| YouTube channel profile or About | Yes, up to 14 links |
| Instagram Reels caption | No. Bio link only |
| Instagram Stories link sticker | Yes, open to all accounts |
| TikTok caption | No. Bio link only |

Why: a video classified as a Short cannot carry a clickable link on any surface the uploader controls per video, and classification follows aspect ratio and duration, so a 9:16 promo clip cannot opt out. [A] [YouTube Help: sharing links with your audiences](https://support.google.com/youtube/answer/13748639?hl=en) Injecting destination URLs into Shorts metadata is wasted effort; the pinned comment earns a profile visit rather than delivering a destination.

## Disclosure

A render with a material connection carries a persistent `#ad` overlay in a fixed corner for the full clip (`video_settings.disclosure_overlay`, top-right at 0.45 times the subtitle font by default, `REQ-CMP-001`, `REQ-CMP-002`; the FTC asks for clear and conspicuous, not a size ratio), and its caption leads with the disclosure on its own line (`REQ-CMP-006`). The overlay and the caption follow the same recorded decision (`REQ-CMP-007`). The hook overlay sits centre-upper, so the two don't compete for the same zone in the first seconds.

AI-content labels are a separate platform-policy layer on top of `#ad`. The TikTok AI label is on by default (`REQ-CMP-016`) and under review ([design 0016](../design/0016-tiktok-ai-label.md), `REQ-CMP-017`); the YouTube synthetic-media flag is opt-in (`REQ-CMP-015`, held).

The regulators, the rules, the platform flags, the manual steps and the penalty surface are in [the compliance guide](compliance.md).

## Trust signals

Every product template asks for one trade-off or limitation of the product, one sentence at most (`REQ-CNT-023`). The narrator profile bans superlatives such as "game changer", "ultimate", "must-have" and "revolutionary", asks for one concrete detail per script, and keeps the delivery calm rather than hyped (`REQ-CNT-008`).

Why:

- Two-sided messages raise credibility, best when the negative is small, real and tied to a positive. [A] [International Journal of Research in Marketing](https://www.sciencedirect.com/science/article/abs/pii/S0167811606000267)
- A stated downside is the trust signal of the de-influencing era, and absolute superlatives ("life-changing", "obsessed") read as ad copy ([HBR](https://hbr.org/2025/12/how-to-do-influencer-marketing-that-customers-actually-trust), [BBB Programs](https://bbbprograms.org/media/insights/blog/influencer-trust-index), [Frontiers](https://www.frontiersin.org/journals/communication/articles/10.3389/fcomm.2025.1600657/full))
- Disclosing a sponsorship does not cost reach: TikTok's guidance says correctly disclosed branded content performs as well as or better than undisclosed. No independent study confirms it ([Skeepers](https://community.skeepers.io/blog/tiktok-new-policy/)).

## Originality and variety

The pipeline draws each render's script template, call to action, voice and cold-open variant per product from a hash of the product id, so a product renders the same way every time and a batch varies (`REQ-CNT-009`, `REQ-CNT-034`, `REQ-CNT-064`, `REQ-VID-090`). Where randomisation is on, the caption font and colour are drawn the same way (`REQ-VID-073`).

Why: all three platforms deprioritize unoriginal, templated, mass-produced output ([design 0006](../design/0006-render-choices-and-variety-report.md) has the sources). YouTube's inauthentic-content policy (July 2025, with channel terminations in January 2026, [The Next Web](https://thenextweb.com/news/youtube-ai-slop-crackdown-faceless-creators-collateral-damage)) targets unoriginality, not AI generation itself. Instagram's originality rules (30 April 2026) count "unique text, creative edits, and voiceover", and not watermarks or speed changes. TikTok tightened its For You feed standards on 24 September 2026. An automated pipeline that renders the same shapes repeatedly is exactly the failure mode, so variety is a reach requirement.

Planned: record every drawn choice and report output variety, with an alert when one value dominates or two scripts are near-identical ([design 0006](../design/0006-render-choices-and-variety-report.md), `REQ-PUB-083`).

## Platform changes since 2025

- TikTok's US operation was divested to a US joint venture in January 2026, and its recommendation algorithm is retrained on US-only data, so US-reach assumptions are provisional.
- Instagram removed the longer-Reels penalty (it recommends up to about 3 minutes to non-followers) and names watch time, likes per reach and sends per reach as its top signals. [B] [Hootsuite](https://blog.hootsuite.com/instagram-algorithm/)
- Instagram has capped hashtags at 5 since December 2025. [A] [Social Media Today](https://www.socialmediatoday.com/news/instagram-implements-new-limits-on-hashtag-use/808309/) The shipped Instagram caption settings in `config/ai_services.yaml` ask for 15-30 hashtags; keeping each platform within its own limit is planned (#567, `REQ-PUB-108`).

## Claims without support

The vendor literature on captions and promo video is louder than the empirical record. Don't act on these without your own measurement:

- No clean caption-isolated A/B benchmark exists for short-form commerce. The "+34% conversion" figures confound caption design with thumbnail, script, audio and product changes.
- The "highlight nouns and numbers" heuristic is vendor-converged but not study-backed; the only academic study is in a language-learning context, not commerce ([MUM '24](https://arxiv.org/abs/2307.05870)).
- "Viewers ignore videos without dynamic captions 65% of the time" is folklore with no primary method.
- The engagement gap between disclosed and undisclosed sponsorship rests on TikTok's own data.

The defensible chain: captions are a precondition for sound-off completion, completion is the precondition for the call to action being seen, and the trade-off, the closing line and a well-placed call to action raise conversion given that the viewer stayed.

## Sources

Sources for the hook shapes and word budgets: [Captions.ai Hook Writing Guide](https://captions.ai/help/guides/marketing/hook-writing), [OpusClip TikTok Hook Formulas](https://www.opus.pro/blog/tiktok-hook-formulas), [OpusClip Best TikTok Hooks 2026](https://www.opus.pro/blog/tiktok-hooks-that-go-viral-2026), [TTS Vibes: TikTok first 3 seconds retention stats](https://insights.ttsvibes.com/tiktok-first-3-seconds-hook-retention-rate/).
