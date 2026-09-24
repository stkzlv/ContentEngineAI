# Tutorial Video Best Practices

Strategy and content rules for short-form vertical **tutorial / how-to**
video (25-60 s, 9:16, watched on autoplay social feeds). Companion to
[promotional-video-best-practices.md](promotional-video-best-practices.md),
which covers product and promo video. Both describe the same pipeline output;
the difference is what the video is for, and that changes the structure, the
length band, the visual source, and which metric tells you it worked.

**Audience**: problem-first tech-help, fix-it, and explainer content served
via TikTok, Instagram Reels, and YouTube Shorts. Often watched muted, though most TikTok users spend time with sound on.

**Related docs**:
- [promotional-video-best-practices.md](promotional-video-best-practices.md) —
  hook patterns, cut cadence, CTA staging, and `#ad` disclosure. Everything
  there about hooks, legibility without sound, and disclosure applies here;
  only the sections below differ.
- [subtitle-best-practices.md](subtitle-best-practices.md) — caption design.
- [creator-research.md](creator-research.md): graded 2026 evidence on ranking signals,
  search placement, the Shorts view-count change and what to measure.
- [platform-safe-zones.md](platform-safe-zones.md) — UI overlay zones.

---

## What actually differs from promo video (cheat-sheet)

1. **The payoff goes first, then the method.** Promo video builds to a
   reveal. Tutorial video that withholds the answer until the end loses the
   viewer who came for the answer. State the fix in the first 3 seconds, then
   show how.
2. **Longer band: 25-40 s, up to 60.** Promo lands at 21-34 s. A tutorial
   needs room for problem, method, and result, and cutting the method to fit
   an entertainment-length band produces a video that names a fix without
   teaching it.
3. **One instruction per visual change.** Not a pacing rule, a comprehension
   rule. Every step needs a matching visual; a step narrated over unrelated
   footage does not land.
4. **The title is a ranking signal, not decoration.** Tutorial content is
   found by search, and a Shorts title made only of hashtags throws away the
   strongest signal available. Write the query the viewer typed.
5. **Put the search phrase in three places**: the first spoken line, the
   on-screen text and the start of the caption. Platforms transcribe audio and
   read on-screen text, so on tutorial content the spoken line is a search
   asset, not just narration.
6. **Judge it on a longer clock than promo.** Promo is scored on day-1 reach.
   Tutorial content is scored on whether it still earns views after 30 days,
   and reading it on a 48-hour window will call every tutorial a failure.

---

## 1. Structure: answer first, method second

The reliable shape is **problem -> answer -> method -> result**, which
inverts the promo build-to-reveal.

A viewer arriving from search already has the problem. Restating it for ten
seconds spends the retention window on something they know. A viewer arriving
from the feed does not have the problem and needs it framed, but in one line,
not a paragraph.

A workable split for a 40-second tutorial:

| Beat | Budget | Content |
|---|---|---|
| Problem | 0-3 s | One line. The symptom, in the viewer's words. |
| Answer | 3-8 s | The fix, stated plainly, before any explanation. |
| Method | 8-32 s | The steps. One instruction per visual change. |
| Result | 32-40 s | What changed, plus the closing line and CTA. |

The common failure is a tutorial that spends 15 seconds on the problem and
5 on the fix. Read the script aloud and mark where the answer first appears;
past 8 seconds, the structure is promotional, not instructional.

**Pace by difficulty, not by clock.** Slow down on the step that is easy to
get wrong, speed through the obvious ones. A uniform cut cadence, which is
right for promo, flattens exactly the moment that needed room.

## 2. Length

**Length follows the number of steps, not a fixed word count.** A fixed
~100-word script is about 37 s whatever the task, which is itself part of the
generic feel: a one-step shortcut gets padded and a six-step fix gets cut.

| Tutorial type | Example | Steps | Duration | Words |
|---|---|---|---|---|
| Single setting or shortcut | "Screenshot on a Pixel: Power + Volume Down" | 1-2 | 15-30 s | 40-80 |
| Multi-step fix | "Stop an iPhone overheating while it charges" | 3-6 | 40-75 s | 110-200 |
| Concept explainer | "Why wifi drops at night" | a cause, then 1-2 checks | 35-60 s, ending in one testable action | 100-160 |
| More than 6 steps, or forks by device | "Fix wifi drops" in general | - | a numbered series, one per device or cause, or long-form with chapters | - |

The evidence behind the bands (the bands themselves are an inference, grade C):

- **Short holds attention (A, correlational).** Across 6.9 million edX
  sessions, median engagement time was at most about 6 minutes whatever the
  video's length, and in the shortest videos three quarters of sessions
  watched more than 75%.
  https://dl.acm.org/doi/10.1145/2556325.2566239
- **Tutorial viewers take what they need and leave (A).** In the same study,
  viewers watched 2-3 minutes of a tutorial regardless of its length,
  and re-watched tutorials more than lectures. Length matters
  less than being able to find the step. These were motivated learners, not
  feed scrollers, so treat the numbers as an upper bound on patience.
- **Fast speech is fine (A).** Engagement rose with speaking rate; viewers
  followed even 254 words per minute. Add pauses between steps, not within
  them.
- **Platform limits are not targets.** YouTube Shorts accept up to 3 minutes
  since October 2024 (A); Reels and TikTok allow longer.

A Short loops, which helps re-watching one step, but a viewer cannot jump to
step 5, so a many-step fix belongs in a series or a chaptered long-form video.

## 3. Visual source when there is no product to show

Tutorial content has no product photography, which is the practical problem
when a pipeline was built around product imagery. Three sources, in
increasing order of cost:

**Stock footage**, keyed to the problem rather than the product ("router",
"desk setup", "phone charging"). Cheapest, fully automatable, and the weakest
at actually teaching: stock shows a person near a laptop, not the setting you
are telling them to change.

**Generated visuals**, which can depict a specific state but need a
generation step and, when they look realistic, carry platform
AI-disclosure obligations. On TikTok a label narrows reach slightly (see
creator-research.md section 1); generic TTS narration alone needs none.

**Screen recordings**, which are what tutorials actually want, because the
instruction and the visual are the same artifact. They also conflict hardest
with full automation, since each one is bespoke to the fix being shown.

Published guidance is consistent that stock alone underperforms for
instructional content and works best combined with UI, typography, and
captions carrying the specific information. For an automated pipeline the
practical reading is: stock as the bed, on-screen text as the teaching layer.
The text is doing the work the footage cannot.

**In this project**, the `slideshow_stock` profile renders entirely from stock,
paired with `--topic` for content that has no product to scrape. Because no
visual comes from a product, this profile writes the script first and then
searches for footage using phrases taken from the narration.

Each phrase is a separate search, so the phrase count is how many different
shots the render draws on. Passing them together would not work: the provider
concatenates a keyword list into one query, and the library answers a long
query with results skewed toward whichever phrase dominates. Some beats go
unrepresented entirely. The same reason is why the profile
declares an empty keyword list rather than inheriting the product-oriented
global defaults.

The footage tracks the script as a whole, not the sentence playing over it, so
"one instruction per visual change" is only partly met. Matching each shot to
the instruction it illustrates needs a search per segment.

That gap is the main reason these renders read as generic. Section 9 sets the
order to move to: a real capture or a UI mockup per step, a diagram for a
concept, and stock only for the opening symptom (#560).

## 4. Discovery: search-first, not feed-first

Promo video is a feed product. Tutorial video is a search product, and the
distinction changes what to optimise.

- **Title.** Write the query, not a label. Shorts titles composed only of
  hashtags waste the primary text ranking signal.
- **Search phrase in three places.** The first spoken line, the on-screen text
  and the start of the caption. Platforms transcribe audio and read on-screen text.
- **Description.** Two or three sentences carrying the target phrase.
- **Hashtags.** Two or three on TikTok (its own advice), at most five on Instagram (a hard cap since December 2025). They categorise; they do not add reach.
- **Topical clustering.** Consecutive videos answering adjacent questions in
  one theme build topical authority; scattered one-off topics do not.

YouTube restructured its search filters on 2026-01-08, adding a Type filter
that selects Shorts only, long-form only, or a mix. Read it carefully before
treating it as good news: most coverage framed it as users finally being able
to **exclude** Shorts from search results. It cuts both ways. Shorts became an
explicitly selectable result type, and also an explicitly excludable one, and
which effect dominates depends on what searchers choose. Nobody has published
data on that split yet. The safe reading is that Shorts search behaviour
changed recently enough to invalidate older guidance, not that it improved.

## 5. Measure it on the right clock

This is where tutorial content is most often misjudged.

Short-form views arrive fast and stop. The bulk of a post's lifetime views
land within the first day or two, and the curve is steep enough that on a
short window every video looks like a spike and a tutorial looks identical to
a trend post.

Do not take a figure for this from any document, including this one. Pull
your own curve: it is the baseline every comparison below is measured
against, and it varies by channel, niche, and posting cadence. Scheduling and
analytics APIs commonly expose a per-post timeline or a content-decay
endpoint, which makes this a query rather than a project.

On YouTube, every play and replay has counted as a Shorts view since
31 March 2025, while "engaged views" exclude replays. Loops inflate raw
counts, so compare engaged views and "viewed vs swiped away" where you can,
and note which one a comparison used.

The metric that separates them is **whether the video earns views after the
initial spike**. The industry shorthand is an *evergreen score*: views after
the first 30 days divided by views during the first 30 days, where 1.0 or
higher means the video accumulated more attention later than at launch.

Two consequences:

- **A 7-day window cannot tell evergreen content from a spike.** It captures
  the launch curve for both. If the reason for making tutorials is durable
  search traffic, 7 days does not test it; 30-plus does.
- **Report both.** Day-2 and day-7 for launch performance, day-30-plus for
  durability. They answer different questions and one does not substitute for
  the other.

Do not assume durability because the format is educational. Measure the ratio
per video and let it decide.

## 6. What the strongest channels do differently

Published analysis of high-performing short-form channels converges on a few
points, and the striking part is how unglamorous they are.

**Retention across the whole clip, not just the hook.** The opening 3 seconds
get most of the attention in creator guidance, but platforms weight sustained
retention through the body. A video with a strong hook and a slack middle
loses to one that holds evenly. For tutorial content that means the method
section is not filler between the hook and the CTA; it is where the video is
won or lost.

**Cut the dead air.** Top-creator editing advice is consistent about removing
every pause longer than roughly 0.3 s. For a pipeline using synthetic speech
there are no filler words to cut, but the same principle applies to silence:
gaps between sentences accumulate into dead frames that cost retention with
no content benefit. This project already trims trailing silence during audio
processing; the parameter that controls it is worth checking against this
guidance rather than left at whatever value happened to work.

**Discipline over tricks.** The recurring conclusion across 2026 analyses is
that the edge comes from better hooks, clearer structure, and clean captions
rather than from algorithm exploits. That is convenient for an automated
pipeline, which is good at consistency and bad at chasing trends.

Treat the specific retention percentages in this genre of article
(“45-55% up to 70-85%”) as marketing. They appear without methodology,
sample, or platform, and they are published by companies selling editing
tools.

## 7. Trust rules carry over, and matter more

Everything in the promo doc about trade-off honesty applies here with a
sharper edge: a promo video that oversells a product costs credibility, while
a tutorial that gets a fact wrong costs the reason the viewer came.

- State the case where the fix does not work. A tutorial with no failure
  condition reads as untested.
- Do not invent specifics. A fabricated setting name or menu path is
  immediately checkable and immediately disqualifying.
- Disclosure obligations are unchanged. A tutorial that recommends a paid
  tool under an affiliate relationship is an endorsement for compensation and
  needs the same on-frame plus first-line caption disclosure as a product
  video.

## 8. What makes a short tutorial useful

Diagnosis of this pipeline's own topic renders: the picture has nothing to do
with what the voice says, and the advice is what most viewers already know. A
reviewed render ("Why your router needs a reboot") ran 24 s over stock photos
of a raised hand, legs among cables and a stressed man at a laptop, while the
voice said "unplug the power for thirty seconds". The learning-science
evidence ranks exactly that as the most damaging design error.

Effect sizes from Mayer's 2017 review (A, https://doi.org/10.1111/jcal.12197),
applied to a 30-90 s video:

| Principle | d | What it means here |
|---|---|---|
| Temporal contiguity | 1.30 | The visual for step N is on screen while step N is spoken |
| Redundancy | 0.87 | Full captions plus narration plus busy footage overloads; during the steps, show the menu path, not only a transcript. Captions still help muted and second-language viewers, so this is a balance |
| Spatial contiguity | 0.79 | Put the label next to the thing it names |
| Personalization | 0.79 | Conversational "you"; the templates already do this |
| Voice | 0.74 | Human voices beat machine voices in the studies reviewed, which predate modern TTS |
| Coherence | 0.70 | Remove footage that does not teach; decorative material lowers learning |
| Segmenting | 0.70 | One step per visual segment, with a counter ("2/4") |
| Signaling | 0.46 | Highlight the control being tapped |
| Pre-training | 0.46 | Name the starting place first ("everything is in Settings, Battery") |

Beyond the table:

- **Show the task done, then recap it (A, small sample).** Demonstration
  tutorials built procedural skill, and a short recap beat demonstration alone
  (van der Meij 2016, https://doi.org/10.1007/s11251-016-9394-9).
  In a Short the recap is the last 2-3 s: the whole path on one card.
- **First-person view, not a presenter (A).** Showing an instructor did not
  improve learning; showing the task as the viewer sees it did (Fiorella and
  Mayer 2018, https://doi.org/10.1016/j.chb.2018.07.015). Faceless is not
  the problem.
- **Be specific (C).** An exact menu path plus the device and OS version, said
  once and shown on screen, is what separates a tutorial from advice and makes
  it checkable.
- **Show the result, and name the common mistake (C).** "Don't hold Power too
  long, or you get the power menu" adds information stock footage cannot carry.

**Topic filter (C).** Accept a topic only if it is specific (one device family
or app, one outcome), searchable (phrased the way people type it),
demonstrable (a visible path or result) and surprising or non-default (a
hidden setting, a shortcut, a counter-intuitive cause). "Screenshot anything
on any device" fails "specific" and should be a series. "Why wifi drops at
night" has many causes and cannot be shown with stock footage; pick one cause
and one check. Every step should cite a vendor support page before the video
renders, and a topic whose steps cannot be sourced is dropped. Tracked in
#559.

## 9. Visuals must show the step being spoken

The visual for each step comes from the step, timed to its narration, not
from a keyword search over the whole script. In order of preference:

1. **A real screen capture** where it can be automated. On Android, an
   emulator driven over adb records with `screenrecord` (MP4, 180 s maximum)
   and draws each touch with Developer options > Show taps. Stock Android
   differs from Samsung and Pixel skins, and routers, Windows and iOS need
   another source.
2. **A rendered UI mockup** with the exact labels from the source page,
   rendered in the headless Chromium the caption engine already runs. It must
   stay generic in its phone chrome and exact in its labels.
3. **A diagram** for a concept (section 10).
4. **Stock footage only for the opening symptom** (a spinner, a hot phone), at
   most about 3 s, and never under a step.

End on the result, then a 2-3 s recap card with the full path. Tracked in
#560.

## 10. Infographics and simple animation

Yes, when the graphic explains; no when it decorates.

- **Signaling works (A).** Arrows, highlights and labels: d = 0.46 in Mayer's
  2017 review, and positive across a meta-analysis of 29 studies.
  https://link.springer.com/article/10.1007/s11423-020-09748-7
- **Animation beats static pictures, most for procedures (A).** d = 0.37
  overall and d = 1.06 for procedural-motor knowledge, larger when the
  animation shows the content itself rather than decorating it. "Tap here,
  then here" is procedural. https://www.sciencedirect.com/science/article/abs/pii/S0959475207001077
- **Keep each graphic short and single (A).** Animation's advantage shrinks
  on long sections because transient information overloads working memory
  (Wong et al. 2012, https://eric.ed.gov/?id=EJ978021): one idea per graphic,
  on screen long enough to read.
- **Build-on drawing beats static slides (B).** In the edX data, continuously
  drawn tutorials were more engaging than slides or screencasts.
- **Decorative graphics hurt (A).** Interesting but irrelevant material lowers
  learning. Every graphic must encode a fact from the script.
- **Graphics count as transformation (A).** YouTube credits substantive
  edits, and Instagram counts unique text and creative edits, as original. The same five cards
  with swapped words in every video would read as a template, so vary the
  layout and tie the geometry to the content.

The graphic types worth building first, for tech help:

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

Build them as HTML and CSS templates rendered to transparent images in the
existing Chromium and animated with FFmpeg overlays, driven by a validated
JSON spec from the script step. One graphic on screen at a time, about eight
words at most, at least 1.5 s per new text segment, and a failed graphic is
skipped rather than losing the render. No AI image generation for anything the
viewer must read: it garbles text and invents UI. Tracked in #561.

## 11. Honest gaps in the evidence

Most published short-form guidance is vendor marketing for editing tools, and
this doc's sourcing is weaker than it looks.

- **The length bands are conventional, not measured.** "25-40 s for
  tutorials" appears across several vendor blogs with no disclosed
  methodology and no primary data. Treat it as a starting point and measure
  view-through by length on your own content.
- **The evergreen-score threshold of 1.0 is a convention**, not a validated
  cutoff. The ratio is useful; the specific line is arbitrary.
- **This doc deliberately cites no decay percentages.** The figures in
  circulation are either platform-wide aggregates that hide content-type
  differences, or single-channel measurements whose content mix goes
  unstated, which matters because a channel publishing mostly one format
  cannot tell you how the other decays. Your own curve is the only one that
  describes your channel, and even that cannot settle whether tutorial
  content behaves differently unless both formats are measured concurrently
  on the same account.
- **Large-scale academic work on short-form dynamics exists but does not
  answer this.** The Kuaishou study covering 248 million videos characterises
  creator and attention distribution, not per-video decay by content type.
- **Nothing here is a substitute for an A/B on your own channel.** Two arms,
  interleaved by day, same voice and cadence, differing only in format.
  Sequential comparison confounds the format change with whatever else moved.
- No peer-reviewed study covers feed-served short tutorials by length; the
  length guidance transfers from MOOC and marketing data.
- No published comparison of stock footage against screen capture exists for
  Shorts; the pipeline would have to measure it.
- The human-over-machine voice effect (d = 0.74) comes from studies that
  predate modern TTS.

## Sources

- [Shorter Is Different: Characterizing the Dynamics of Short-Form Video Platforms](https://arxiv.org/abs/2410.16058) — 248M-video Kuaishou analysis; creator and attention distribution.
- [YouTube's new search filters make clearer distinctions between long-form videos and Shorts](https://www.tubefilter.com/2026/01/09/youtube-search-filters-shorts-vs-long-form/) — the 2026-01-08 filter change, and the framing that it lets users exclude Shorts.
- [Short-form video pacing: the editing rhythm guide](https://shortzly.com/blog/short-form-video-pacing-editing-guide) — cut cadence, trimming pauses over 0.3s.
- [Short-form video retention](https://shortzly.com/blog/short-form-video-retention-strategies) — sustained retention over hook-only optimisation. Retention percentages in it are unsourced.
- [YouTube Shorts SEO in 2026](https://miraflow.ai/blog/youtube-shorts-seo-2026-how-to-rank-in-search) — titles as ranking signal, spoken-keyword indexing, hashtag counts.
- [Building a search-first YouTube content strategy](https://marketingagent.blog/2026/02/16/building-a-search-first-youtube-content-strategy-seo-tips-for-2026/) — search-first vs feed-first framing, topical authority.
- [What happens to your YouTube Shorts after 30 days](https://miraflow.ai/blog/what-happens-youtube-shorts-after-30-days-old-content-views) — evergreen score definition.
- [Short-form video structure: hook, body, payoff](https://www.socialync.io/blog/short-form-video-structure-guide-2026) — structure and tutorial length band.
- [From article to short-form video that holds attention](https://www.searchenginejournal.com/from-article-to-short-form-video-that-holds-attention/565238/) — pacing by difficulty, one instruction per visual.
- [How to make a how-to video](https://swarmify.com/blog/how-to-make-a-how-to-video/) — step sequencing, matching visual per instruction.
- [Video styles explained](https://www.vidyard.com/blog/different-styles-of-videos/) — screen recording vs stock for instructional content.
