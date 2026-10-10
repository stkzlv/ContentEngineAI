# Tutorials: why topic videos are built this way

This page explains why the pipeline renders a topic video (a how-to or fix-it Short with no product behind it) the way it does: answer first, at its length, over stock visuals, written for search, and judged on a 30-day clock. A topic enters through `--topic` or `--topics-file` on the producer and the batch, or through the `topics:` block and `topics_file` in `config/pipeline.yaml`. The scripts come from the topic templates in `config/ai_services.yaml` (`script_templates.topic_templates`, `narrator_profile_topic`, `cta_options_topic`), whose prompts live in `src/ai/prompts/scripts/topic_*.md`; the captions and titles come from the `*_topic.md` prompts in `src/ai/prompts/`. The visuals come from the `slideshow_stock` profile in `config/video_production.yaml`. Each finding carries a grade, defined in [the evidence grades](../design/README.md#evidence-grades). Captions, hooks and disclosure work as they do for product videos ([promotional videos](promotional-videos.md), [captions](captions.md)) and are not repeated here.

## Answer first, method second

A topic script runs **problem -> answer -> method -> result**, the inverse of a product video's build to a reveal. The three bundled templates (`topic_answer_first`, `topic_symptom_cause`, `topic_mistake_fix`) each instruct the model to:

- state the fix within the first three seconds (REQ-CNT-026),
- speak the search phrase within the first five seconds (REQ-CNT-027),
- give one instruction per sentence (REQ-CNT-028),
- close on the result rather than on a claim (REQ-CNT-031), right before the one configured call to action (REQ-CNT-033).

A topic render draws only from `script_templates.topic_templates` and uses the topic narrator profile (REQ-CNT-013, REQ-CNT-014); the topic narrator talks the viewer through the fix "over the phone", with "you" often and "I" sparingly. On a plain task ("How to back up your iPhone") the symptom-first and mistake-first templates invent a problem the task doesn't have; `topic_routing` sends task titles to `topic_answer_first` only (REQ-CNT-161).

Why: a viewer who arrives from search already has the problem, so restating it spends the retention window on something they know, and a tutorial that withholds the answer loses the viewer who came for it. A viewer from the feed needs the problem framed, in one line. One instruction per visual change is a comprehension rule rather than a pacing one: a step narrated over unrelated footage does not land ([Search Engine Journal](https://www.searchenginejournal.com/from-article-to-short-form-video-that-holds-attention/565238/), [Swarmify](https://swarmify.com/blog/how-to-make-a-how-to-video/), [Socialync](https://www.socialync.io/blog/short-form-video-structure-guide-2026)).

A workable split for a 40-second tutorial; if the answer first appears past 8 seconds, the structure is promotional, not instructional:

| Beat | Budget | Content |
|---|---|---|
| Problem | 0-3 s | One line. The symptom, in the viewer's words. |
| Answer | 3-8 s | The fix, stated plainly, before any explanation. |
| Method | 8-32 s | The steps. One instruction per visual change. |
| Result | 32-40 s | What changed, plus the closing line and call to action. |

## Length

The topic narrator profile targets 30-40 seconds, roughly 75-100 words, for every topic. That budget is a prompt instruction that nothing enforces: published topic renders have run from 21 to 77 seconds, depending on how much the topic description gives the script. Platform limits, view counting and the cross-platform evidence are in [video length](video-length.md).

Length follows the number of steps, from a sourced step list (REQ-VID-121, [design 0017](../design/0017-tutorial-step-lists.md)). A fixed word count pads a one-step shortcut and cuts a six-step fix, which is part of why fixed-length renders feel generic. The bands the design works from are an inference [C]:

| Tutorial type | Example | Steps | Duration | Words |
|---|---|---|---|---|
| Single setting or shortcut | "Screenshot on a Pixel: Power + Volume Down" | 1-2 | 15-30 s | 40-80 |
| Multi-step fix | "Stop an iPhone overheating while it charges" | 3-6 | 40-75 s | 110-200 |
| Concept explainer | "Why wifi drops at night" | a cause, then 1-2 checks | 35-60 s, ending in one testable action | 100-160 |
| More than 6 steps, or forks by device | "Fix wifi drops" in general | - | a numbered series, one per device or cause | - |

With step lists on, a topic titled "Why ..." is written to the explainer row's shape and band (REQ-VID-161); the step-list topic filter accepts a "why" question that names one device family or app and one symptom, its steps being the checks for the usual causes.

Why:

- **Short holds attention [A, correlational].** Across 6.9 million edX sessions, median engagement time was at most about 6 minutes whatever the video's length, and in the shortest videos three quarters of sessions watched more than 75% ([edX study](https://dl.acm.org/doi/10.1145/2556325.2566239)).
- **Tutorial viewers take what they need and leave [A].** In the same study, viewers watched 2-3 minutes of a tutorial regardless of its length and re-watched tutorials more than lectures. Length matters less than being able to find the step. These were motivated learners, not feed scrollers, so treat the numbers as an upper bound on patience.
- **Fast speech is fine [A].** Engagement rose with speaking rate; viewers followed even 254 words per minute. Pauses belong between steps, not within them.
- **Knowledge runs longer than entertainment, but not long [B, preprint].** On Kuaishou, knowledge videos have a 30 s median against 14-16 s for lifestyle videos, and likes per view fall as any video gets longer ([Chen et al. 2024](https://arxiv.org/abs/2410.16058)).
- **Tech-help creators run 33-97 s [A, primary measurement].** Six tutorial channels' latest Shorts have a 76 s median, and 63% run past a minute ([video length](video-length.md#what-top-tech-creators-post)). They show a person and have audiences who chose them, so treat their lengths as a ceiling for a faceless channel.
- **Platform limits are not targets.** YouTube Shorts accept up to 3 minutes since October 2024, but a Short over one minute with any Content ID claim is blocked worldwide, which matters for a multi-step render with music [A]; Reels and TikTok allow longer.

A Short loops, which helps re-watching one step, but a viewer cannot jump to step 5, so a topic that forks by device is set aside for a series rather than rendered (REQ-VID-121).

## Visual source when there is no product

A topic renders only with a profile whose visuals come entirely from stock (REQ-VID-119); the bundled one is `slideshow_stock` (`use_scraped_images: false`, `use_stock_images: true`, `stock_image_count: 8`). It declares `stock_media_keywords: []` rather than inheriting the product-oriented global terms, and a topic's own `--topic-keywords` replace the profile and global lists rather than joining them (REQ-VID-100): the provider joins every term into one query, so a mixed list searches for neither.

Because no visual comes from a product, this profile writes the script first and then searches for footage on phrases derived from the narration (REQ-VID-101, the `visual_search_terms.md` prompt). `visual_search_terms.max_phrases` (default 3) is how many different shots a render draws on, since each phrase is a separate search (REQ-VID-102); the library answers a long combined query with results skewed toward whichever phrase dominates. Duplicate results are dropped (REQ-VID-103), and if deriving the phrases fails the render keeps the existing terms (REQ-VID-104). The stock relevance judge (`stock_relevance`, on by default, `min_score: 2`, `max_candidates: 80`) scores each candidate's thumbnail 0-3 against the script and uses the best, falling back to a random sample when it returns no scores (REQ-VID-108, REQ-VID-109).

The footage tracks the script as a whole, not the sentence playing over it, so "one instruction per visual change" is only partly met. That gap is the main reason these renders read as generic.

Why: there are three sources, in increasing order of cost ([Vidyard](https://www.vidyard.com/blog/different-styles-of-videos/), [Swarmify](https://swarmify.com/blog/how-to-make-a-how-to-video/)):

- **Stock footage**, keyed to the problem rather than the product. Cheapest and fully automatable, and the weakest at teaching: stock shows a person near a laptop, not the setting you are telling them to change.
- **Generated visuals**, which can depict a specific state but need a generation step and, when realistic, carry platform AI-disclosure obligations. On TikTok a label narrows reach slightly ([design 0016](../design/0016-tiktok-ai-label.md)); generic TTS narration alone needs none. The pipeline does not generate visuals.
- **Screen recordings**, which are what tutorials want, because the instruction and the visual are the same artifact, and which conflict hardest with full automation.

Published guidance agrees that stock alone underperforms for instructional content and works best with UI, typography and captions carrying the specific information: stock as the bed, on-screen text as the teaching layer.

Planned:

- A stock clip or image used in a recent render loses to a fresh one that is relevant enough (REQ-VID-110, [design 0013](../design/0013-stock-clip-reuse-guard.md)).
- Each step gets its own visual, timed to its narration ([Visuals that show the spoken step](#visuals-that-show-the-spoken-step)).

## Search discovery

A topic video is a search product, not a feed product, and the prompts are written for that ([Marketing Agent](https://marketingagent.blog/2026/02/16/building-a-search-first-youtube-content-strategy-seo-tips-for-2026/), [Miraflow](https://miraflow.ai/blog/youtube-shorts-seo-2026-how-to-rank-in-search)):

- **The search phrase in three places.** The script speaks it in the first five seconds (REQ-CNT-027), the on-screen hook headline names the symptom or the fix (REQ-VID-085, `hook_headline_topic.md`), and the TikTok, Instagram and YouTube caption prompts front-load the symptom in the words a person would type (`*_caption_topic.md`, `youtube_metadata_topic.md`). Platforms transcribe audio and read on-screen text, so the spoken line is a search asset, not only narration.
- **Title.** The YouTube topic title front-loads the symptom in its first 5-7 words, 50-60 characters (`platform_metadata.youtube.title_length_max: 60`). A Shorts title made only of hashtags wastes the primary text ranking signal; the publisher refuses a YouTube post with no title (REQ-PUB-110). Across 10,000 trending Shorts the median title was about 8 words, 20-40 characters [B], shorter than the bundled maximum; settling the length is part of #550.
- **Description.** The YouTube description names the symptom in its first sentence and keeps the first 150 characters for the search phrase.
- **No link and no `#ad` on a topic.** The topic prompts forbid a link, an advertising disclosure and any product the script does not cover: there is no material connection to disclose.
- **Hashtags.** In optimized mode the bundled config asks for 3-5 on YouTube (with `#Shorts` first), 2-4 on TikTok (#ad included on a product) and 3-4 on Instagram (`platform_metadata.*.hashtag_count_min` and `hashtag_count_max`), and the publisher keeps every composed caption within Instagram's five, `#ad` and the product id included (REQ-PUB-108). Hashtags categorise; they do not add reach, so "hashtags boost Instagram reach" is not supported.
  - TikTok's own advice is two or three, "less is more" [A, [TikTok Creator Academy](https://www.tiktok.com/creator-academy/en/article/elements-of-tiktok-video?lang=en)].
  - Instagram caps a post at five since December 2025 [A, [Social Media Today](https://www.socialmediatoday.com/news/instagram-implements-new-limits-on-hashtag-use/808309/)]; a caption over it does not post.
  - YouTube ignores every hashtag on a video that carries more than 60 [A].
  - A topic caption carries no id tag: its slug would be cut at the first hyphen.
- **Caption length.** The Instagram SEO caption is capped at 240 characters (the `caption_length_seo` default). Instagram posts under 30 words had higher engagement across 9.1 million posts [B, [Socialinsider](https://www.socialinsider.io/blog/instagram-caption-length/)].

YouTube restructured its search filters on 2026-01-08, adding a Type filter that selects Shorts only, long-form only, or a mix ([Tubefilter](https://www.tubefilter.com/2026/01/09/youtube-search-filters-shorts-vs-long-form/)). Most coverage framed it as letting users exclude Shorts. It cuts both ways: Shorts became an explicitly selectable result type and an explicitly excludable one, and no data on the split is published. The safe reading is that Shorts search behaviour changed recently enough to invalidate older guidance, not that it improved.

## What viewers ask about

A topic earns its search traffic only if people type it. The evidence below was gathered in October 2026 to pick topics by demand rather than by what seems interesting.

**How much people ask.** "How do I..." searches reached an all-time high in 2025, up 25% on the year [A, [Google](https://blog.google/products-and-platforms/products/search/year-in-search-2025/)]. Google Trends gives relative interest, not volumes. Over the 12 months to October 2026 in the US, against "iphone storage full" as 1.0 [A, Google Trends through the unofficial `pytrends` client, compared within shared requests]:

| Search | US | UK |
|---|---|---|
| clear cache | 22.3 | 27.8 |
| screenshot on windows | 8.3 | 6.0 |
| scan qr code | 6.8 | n/a |
| share location iphone | 5.5 | n/a |
| factory reset iphone | 5.1 | 5.9 |
| wifi not working | 3.3 | 6.3 |
| block spam calls | 2.7 | 1.7 |
| bluetooth not connecting | 2.2 | 2.9 |
| transfer data to new iphone | 1.0 | 1.2 |
| whatsapp backup | 0.9 | 3.8 |
| iphone battery drain | 0.7 | 0.9 |
| forgot iphone passcode | 0.6 | 0.7 |

In India, "whatsapp backup" runs about 20 times "iphone storage full".

**What people type.** Google's autocomplete shows the phrasing, not the volume [A]. "How to turn off" completes first with "ai on google" in the US and UK. "Why is my phone" completes with not charging, getting hot and slow. "Samsung how to" completes with screenshot, block spam calls, check battery health and transfer to a new phone. The fastest-rising US searches over the year included sharing location on iPhone, scanning a QR code on Android and uploading contacts to Gmail.

**What support sites put first.** Apple's iPhone support page leads with a forgotten Apple Account password, a disabled iPhone, backup and updates [A, curated, [Apple](https://support.apple.com/iphone)]. Microsoft's Windows support page leads with installation and updates [A, curated, [Microsoft](https://support.microsoft.com/en-us/windows)]. Neither says how it ranks them.

**When.** Searches for moving data to a new iPhone peak in the September launch weeks and at Christmas (five years of Trends) [A]. Phone storage searches peak in April-June.

**What this means for the pool.** Demand clusters around everyday chores: storage and cache, moving to a new phone, spam calls and texts, screenshots and screen recording, resets, Wi-Fi and Bluetooth, lockouts and battery. A topic phrased the way people search, one device or app and one outcome, passes the topic filter (REQ-VID-151): of a pool of claim-style titles ("One repeater makes your wifi slower"), the filter kept 1 in 17, and of searches phrased as tasks, 16 in 26. The filter drops some of the highest-demand tasks as default behaviour (a QR code scan, a factory reset) and splits others whose steps fork by device (screen recording on iPhone) into series.

## Durability metrics

The `analytics` command stores, for each published post, its cumulative views at day 2 and day 7 and a durability ratio: views after the first 30 days divided by views within them (REQ-PUB-072, REQ-PUB-073). A figure a post has not reached is unknown rather than the running total, and a post with no views in its first 30 days has an unknown ratio rather than 0.0 (REQ-PUB-074, REQ-PUB-075). Reports rank posts by durability (REQ-PUB-079), and each registry row records whether the video came from a topic or a scraped product (REQ-PUB-093). The capture cadence and why it is a scheduled job are in [design 0020](../design/0020-analytics-history.md).

Why: short-form views arrive fast and stop, so on a short window every video looks like a spike and a tutorial looks identical to a trend post. The metric that separates them is whether the video earns views after the launch. The industry shorthand is an *evergreen score*, the same ratio, where 1.0 or higher means the video gathered more attention later than at launch ([Miraflow](https://miraflow.ai/blog/what-happens-youtube-shorts-after-30-days-old-content-views)). Two consequences:

- **A 7-day window cannot tell evergreen content from a spike.** It captures the launch curve for both. If the reason for making tutorials is durable search traffic, only 30-plus days tests it.
- **Day-2 and day-7 measure the launch; day-30-plus measures durability.** One does not substitute for the other.

No figure for the decay curve is given here: pull your own, because it varies by channel, niche and posting cadence. Do not assume durability because the format is educational.

On YouTube, every play and replay has counted as a Shorts view since 31 March 2025, while "engaged views" exclude replays [A, [YouTube Help community](https://support.google.com/youtube/thread/333869549)]. Loops inflate raw counts.

Planned: the sweep stores engaged views and "viewed vs swiped away" where a platform exposes them (REQ-PUB-084, [design 0010](../design/0010-first-seconds-metrics.md)).

## Trust and fact checks

A promo video that oversells a product costs credibility; a tutorial that gets a fact wrong costs the reason the viewer came. The pipeline guards that in the script and after it:

- A topic script names a menu path or a URL only when it can state it exactly for a platform it names; otherwise it names an observable on the device or says that it differs by device (REQ-CNT-030).
- It carries one honest limit, the case where the fix does not work, placed among the steps (REQ-CNT-032). A tutorial with no failure condition reads as untested.
- It never invents a product to recommend (REQ-CNT-029).
- After generation, `script_fact_check` (on by default) checks the script's falsifiable claims with one grounded web search, revises at most `max_flags_to_revise` sentences (default 3), refuses a revision that drifts more than `max_length_drift` (default 25%) in length, and ships the original if anything fails (REQ-CNT-049 to REQ-CNT-051). A claim the check says to remove is deleted rather than rewritten when nothing after it depends on it, and a rewrite may not restate it (REQ-CNT-151). A fabricated setting name or menu path is immediately checkable and immediately disqualifying.

Every step cites a source before the video renders, and a topic whose steps cannot be sourced is dropped (REQ-VID-122, [design 0017](../design/0017-tutorial-step-lists.md)).

## What makes a short tutorial useful

A review of the pipeline's own topic renders found that the picture had nothing to do with what the voice said, and the advice was what most viewers already know. One render ("Why your router needs a reboot") ran 24 s over stock photos of a raised hand, legs among cables and a stressed man at a laptop, while the voice said "unplug the power for thirty seconds". The learning-science evidence ranks exactly that as the most damaging design error. This section is the evidence behind [design 0017](../design/0017-tutorial-step-lists.md), [design 0018](../design/0018-tutorial-step-visuals.md) and [design 0019](../design/0019-tutorial-graphics.md).

Effect sizes from Mayer's 2017 review [A, [Mayer](https://doi.org/10.1111/jcal.12197)], applied to a 30-90 s video:

| Principle | d | What it means here |
|---|---|---|
| Temporal contiguity | 1.30 | The visual for step N is on screen while step N is spoken |
| Redundancy | 0.87 | Full captions plus narration plus busy footage overloads; during the steps, show the menu path, not only a transcript. Captions still help muted and second-language viewers, so this is a balance |
| Spatial contiguity | 0.79 | Put the label next to the thing it names |
| Personalization | 0.79 | Conversational "you"; the topic narrator profile already does this |
| Voice | 0.74 | Human voices beat machine voices in the studies reviewed, which predate modern TTS |
| Coherence | 0.70 | Remove footage that does not teach; decorative material lowers learning |
| Segmenting | 0.70 | One step per visual segment, with a counter ("2/4") |
| Signaling | 0.46 | Highlight the control being tapped |
| Pre-training | 0.46 | Name the starting place first ("everything is in Settings, Battery") |

Beyond the table:

- **Show the task done, then recap it [A, small sample].** Demonstration tutorials built procedural skill, and a short recap beat demonstration alone ([van der Meij 2016](https://doi.org/10.1007/s11251-016-9394-9)). In a Short the recap is the last 2-3 s: the whole path on one card. Planned in REQ-VID-123.
- **First-person view, not a presenter [A].** Showing an instructor did not improve learning; showing the task as the viewer sees it did ([Fiorella and Mayer 2018](https://doi.org/10.1016/j.chb.2018.07.015)). Faceless is not the problem.
- **Be specific [C].** An exact menu path plus the device and OS version, said once and shown on screen, separates a tutorial from advice and makes it checkable. Shipped as REQ-CNT-030 for the script.
- **Show the result, and name the common mistake [C].** "Don't hold Power too long, or you get the power menu" adds information stock footage cannot carry. The scripts close on the result (REQ-CNT-031); `topic_symptom_cause` names one thing the viewer might wrongly blame, and `topic_mistake_fix` is built around a common mistake.

**Topic filter [C].** Applied with the step list (REQ-VID-151, [design 0017](../design/0017-tutorial-step-lists.md)): accept a topic only if it is specific (one device family or app, one outcome), searchable (phrased the way people type it), demonstrable (a visible path or result) and surprising or non-default (a hidden setting, a shortcut, a counter-intuitive cause). "Screenshot anything on any device" fails "specific" and becomes a series. "Why wifi drops at night" has many causes and cannot be shown with stock footage; pick one cause and one check. Keep health, finance and legal advice out of the topic pool: YouTube's July 2026 clarification of its inauthentic-content policy added AI personas giving that advice [A, [Tubefilter](https://www.tubefilter.com/2026/07/13/youtube-inauthentic-content-monetization-policy-update/)].

## Visuals that show the spoken step

Planned ([design 0018](../design/0018-tutorial-step-visuals.md), REQ-VID-123, behind `video_settings.step_visuals.enabled`, default false): the visual for each step comes from the step, timed to its narration from the Whisper word timings, not from a keyword search over the whole script. In order of preference:

1. **A real screen capture** where it can be automated. On Android, an emulator driven over adb records with `screenrecord` (MP4, 180 s maximum) and draws each touch with Developer options > Show taps. Stock Android differs from Samsung and Pixel skins, and routers, Windows and iOS need another source.
2. **A rendered UI mockup** with the exact labels from the source page, rendered in the headless Chromium the caption engine already runs, generic in its phone chrome and exact in its labels.
3. **A diagram** for a concept ([Explanatory graphics](#explanatory-graphics)).
4. **Stock footage only for the opening symptom** (a spinner, a hot phone), at most about 3 s, and never under a step.

The video ends on the result, then a 2-3 s recap card with the full path.

Why: temporal contiguity (d = 1.30) and coherence (d = 0.70) in [What makes a short tutorial useful](#what-makes-a-short-tutorial-useful). Narrating over stock footage is not an original edit on its own either; design 0018 carries that evidence.

## Explanatory graphics

Built in part, behind `video_settings.graphics.enabled` (default false, on in the bundled config; [design 0019](../design/0019-tutorial-graphics.md), REQ-VID-124): the step card, a counter over the step's menu path, timed to the step's narration. The other types below are planned. Graphics when they explain, never when they decorate. HTML and CSS templates are rendered to transparent images in the existing Chromium and animated with FFmpeg overlays, driven by a validated JSON spec from the script step. One graphic on screen at a time, about eight words at most, at least 1.5 s for each text segment that appears, and a failed graphic is skipped rather than losing the render. No AI image generation for anything the viewer must read: it garbles text and invents UI.

Why:

- **Signaling works [A].** Arrows, highlights and labels: d = 0.46 in Mayer's 2017 review, and positive across a meta-analysis of 29 studies ([meta-analysis](https://link.springer.com/article/10.1007/s11423-020-09748-7)).
- **Animation beats static pictures, most for procedures [A].** d = 0.37 overall and d = 1.06 for procedural-motor knowledge, larger when the animation shows the content itself rather than decorating it. "Tap here, then here" is procedural ([meta-analysis](https://www.sciencedirect.com/science/article/abs/pii/S0959475207001077)).
- **Keep each graphic short and single [A].** Animation's advantage shrinks on long sections because transient information overloads working memory ([Wong et al. 2012](https://eric.ed.gov/?id=EJ978021)): one idea per graphic, on screen long enough to read.
- **Build-on drawing beats static slides [B].** In the edX data, continuously drawn tutorials were more engaging than slides or screencasts.
- **Decorative graphics hurt [A].** Interesting but irrelevant material lowers learning. Every graphic encodes a fact from the script.
- **Graphics count as transformation [A].** YouTube credits substantive edits, and Instagram counts unique text and creative edits, as original ([TechCrunch](https://techcrunch.com/2026/04/30/instagram-restricts-reach-of-content-aggregators-in-new-crackdown/)). No retention data exists for arrows or circles. The same five cards with swapped words in every video would read as a template, so the layout varies and the geometry follows the content.

The graphic types, in build order:

| Type | Why | Data | Note |
|---|---|---|---|
| Menu-path breadcrumb | Signaling plus segmenting | The path segments | Revealed segment by segment with the voice |
| Step card with counter | Segmenting | Step number, total, a short action | Doubles as progress |
| Callout on a real screenshot | Signaling on real content | A box on a known capture | Never guessed on stock footage |
| Spec or comparison card | Signaling on numbers | Values from the scraped data | For product videos; never LLM-invented specs |
| Before/after split | A concrete comparison | Two real states | Speed test, cluttered and clean menu |
| Checklist ("do this, not that") | Segmenting, coherence | Items with a state | Low risk |
| Simulated phone tap path | Procedural animation | Screens and taps | Later; generic chrome, exact labels |
| Parametric concept diagram | Build-on animation | Numbers only | Later; the template draws, the LLM supplies numbers |

## Gaps in the evidence

Most published short-form guidance is vendor marketing for editing tools, and this page's sourcing is weaker than it looks.

- **The length bands are an inference.** "25-40 s for tutorials" appears across several vendor blogs with no disclosed method. The bands above rest on the segmenting research, the Kuaishou knowledge medians and what tech-help creators post, none of which measures feed-served tutorials by length. Measure engaged views and skip rate by duration band on your own content (`REQ-PUB-084`).
- **No peer-reviewed study covers feed-served short tutorials by length**; the length guidance transfers from MOOC and marketing data.
- **The evergreen threshold of 1.0 is a convention**, not a validated cutoff. The ratio is useful; the specific line is arbitrary.
- **This page cites no decay percentages.** The figures in circulation are either platform-wide aggregates that hide content-type differences, or single-channel measurements whose content mix goes unstated. A channel publishing mostly one format cannot tell you how the other decays, and even your own curve cannot settle whether tutorials behave differently unless both formats are measured on the same account at the same time.
- **Large-scale academic work on short-form dynamics does not answer this.** The Kuaishou study covering 248 million videos characterises creator and attention distribution, not per-video decay by content type ([arXiv](https://arxiv.org/abs/2410.16058)).
- **Retention percentages in editing-tool articles are not supported.** Figures like "45-55% up to 70-85%" appear without method, sample or platform, from companies selling editing tools ([Shortzly](https://shortzly.com/blog/short-form-video-retention-strategies)).
- **No published comparison of stock footage against screen capture exists for Shorts**; the pipeline has to measure it.
- **The human-over-machine voice effect (d = 0.74) comes from studies that predate modern TTS.**
- **Topic demand is relative.** Google Trends compares searches and gives no volumes, and autocomplete shows phrasing, not frequency. Published keyword-volume lists for these searches state no method. No platform publishes which tech tips perform on TikTok or Shorts.
- **Nothing here substitutes for a test on your own channel.** Two arms, interleaved by day, same voice and cadence, differing only in format; sequential comparison confounds the format change with whatever else moved. The reach test (#540) is that comparison, which is why every planned change above ships off until it reads out.
