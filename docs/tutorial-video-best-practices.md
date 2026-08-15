# Tutorial Video Best Practices

Strategy and content rules for short-form vertical **tutorial / how-to**
video (25-60 s, 9:16, watched on autoplay social feeds). Companion to
[promotional-video-best-practices.md](promotional-video-best-practices.md),
which covers product and promo video. Both describe the same pipeline output;
the difference is what the video is for, and that changes the structure, the
length band, the visual source, and which metric tells you it worked.

**Audience**: problem-first tech-help, fix-it, and explainer content served
via TikTok, Instagram Reels, and YouTube Shorts. Watched mostly with sound off.

**Related docs**:
- [promotional-video-best-practices.md](promotional-video-best-practices.md) —
  hook patterns, cut cadence, CTA staging, and `#ad` disclosure. Everything
  there about hooks, sound-off legibility, and disclosure applies here
  unchanged; only the sections below differ.
- [subtitle-best-practices.md](subtitle-best-practices.md) — caption design.
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
5. **Say the keyword out loud in the first 5 seconds.** Platforms transcribe
   audio and index it. On tutorial content the spoken line is a search asset,
   not just narration.
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

Published guidance clusters at **25-40 s for tutorial Shorts**, against
21-34 s for entertainment, with educational formats stretching to 60 s and
occasionally 90 s when the topic needs it. YouTube Shorts accepts up to
3 minutes, which is not a reason to use it.

Treat the band as a constraint on scope, not a target to fill. A fix that
genuinely needs 90 seconds is usually two videos: one that answers the common
case, one that handles the exception.

## 3. Visual source when there is no product to show

Tutorial content has no product photography, which is the practical problem
when a pipeline was built around product imagery. Three sources, in
increasing order of cost:

**Stock footage**, keyed to the problem rather than the product ("router",
"desk setup", "phone charging"). Cheapest, fully automatable, and the weakest
at actually teaching: stock shows a person near a laptop, not the setting you
are telling them to change.

**Generated visuals**, which can depict a specific state but need a
generation step and carry platform AI-disclosure obligations.

**Screen recordings**, which are what tutorials actually want, because the
instruction and the visual are the same artifact. They also conflict hardest
with full automation, since each one is bespoke to the fix being shown.

Published guidance is consistent that stock alone underperforms for
instructional content and works best combined with UI, typography, and
captions carrying the specific information. For an automated pipeline the
practical reading is: stock as the bed, on-screen text as the teaching layer.
The text is doing the work the footage cannot.

**In this project**, a stock-only visual profile is a config change, not new
code: set `use_scraped_images: false` on the profile and populate
`media_settings.stock_media_keywords`. Stock media is already merged into the
same visual pool as scraped media.

## 4. Discovery: search-first, not feed-first

Promo video is a feed product. Tutorial video is a search product, and the
distinction changes what to optimise.

- **Title.** Write the query, not a label. Shorts titles composed only of
  hashtags waste the primary text ranking signal.
- **Spoken keyword in the first 5 s.** Platforms transcribe and index audio.
- **Description.** Two or three sentences carrying the target phrase.
- **Hashtags.** 3-5 relevant, not 30.
- **Topical clustering.** Consecutive videos answering adjacent questions in
  one theme build topical authority; scattered one-off topics do not.

YouTube added a dedicated Shorts type to its search filters in January 2026,
which makes Shorts a first-class search result rather than only a feed
surface. That is recent enough that any strategy written before it
underweights search as a Shorts discovery path.

## 5. Measure it on the right clock

This is where tutorial content is most often misjudged.

Short-form views arrive fast and stop. Measured on one real channel, roughly
79% of a post's final views landed within 6 hours and 92% within 24. Judged
on that curve, every video looks like a spike, and a tutorial looks identical
to a trend post.

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

## 6. Trust rules carry over, and matter more

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

## 7. Honest gaps in the evidence

Most published short-form guidance is vendor marketing for editing tools, and
this doc's sourcing is weaker than it looks.

- **The length bands are conventional, not measured.** "25-40 s for
  tutorials" appears across several vendor blogs with no disclosed
  methodology and no primary data. Treat it as a starting point and measure
  view-through by length on your own content.
- **The evergreen-score threshold of 1.0 is a convention**, not a validated
  cutoff. The ratio is useful; the specific line is arbitrary.
- **The decay figures above come from one channel's own analytics**, and that
  channel's content was predominantly product rather than tutorial. They
  describe the population that was measured, and cannot settle whether
  tutorial content decays differently. That comparison needs both arms
  measured concurrently on the same account.
- **Large-scale academic work on short-form dynamics exists but does not
  answer this.** The Kuaishou study covering 248 million videos characterises
  creator and attention distribution, not per-video decay by content type.
- **Nothing here is a substitute for an A/B on your own channel.** Two arms,
  interleaved by day, same voice and cadence, differing only in format.
  Sequential comparison confounds the format change with whatever else moved.

## Sources

- [Shorter Is Different: Characterizing the Dynamics of Short-Form Video Platforms](https://arxiv.org/abs/2410.16058) — 248M-video Kuaishou analysis; creator and attention distribution.
- [YouTube Shorts SEO in 2026](https://miraflow.ai/blog/youtube-shorts-seo-2026-how-to-rank-in-search) — titles as ranking signal, spoken-keyword indexing, hashtag counts.
- [Building a search-first YouTube content strategy](https://marketingagent.blog/2026/02/16/building-a-search-first-youtube-content-strategy-seo-tips-for-2026/) — search-first vs feed-first framing, topical authority.
- [What happens to your YouTube Shorts after 30 days](https://miraflow.ai/blog/what-happens-youtube-shorts-after-30-days-old-content-views) — evergreen score definition.
- [Short-form video structure: hook, body, payoff](https://www.socialync.io/blog/short-form-video-structure-guide-2026) — structure and tutorial length band.
- [From article to short-form video that holds attention](https://www.searchenginejournal.com/from-article-to-short-form-video-that-holds-attention/565238/) — pacing by difficulty, one instruction per visual.
- [How to make a how-to video](https://swarmify.com/blog/how-to-make-a-how-to-video/) — step sequencing, matching visual per instruction.
- [Video styles explained](https://www.vidyard.com/blog/different-styles-of-videos/) — screen recording vs stock for instructional content.
