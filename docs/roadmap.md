# Roadmap

Where ContentEngineAI is going, grouped into phases by horizon: **Now**, **Next** and **Later**. Items are aspirational, not commitments, and the order within a phase is rough priority. Each item states the outcome and links to where the detail lives: its issue, its [design doc](design/README.md) and its [requirements](requirements/README.md). What has shipped is in the [CHANGELOG](../CHANGELOG.md).

Issues and pull requests are welcome on any item. To pick one up, open an issue first so the scope can be agreed.

Output-changing items ship off by default and are turned on only on measured results ([decision 0002](decisions/0002-output-changes-ship-off-by-default.md)).

## Phase 0: Disclosure compliance baseline (Now, gates 1.0.0)

Affiliate creators carry disclosure obligations under the FTC Endorsement Guides, the Amazon Associates Operating Agreement and each platform's policy, so compliance is the default render output rather than a per-video checklist. The on-frame overlay, caption disclosure, platform tags, the affiliate phrase and the language-matched disclosure have shipped; see [docs/explanation/compliance.md](explanation/compliance.md).

## Phase 1: Hook and retention (Now)

Hold the viewer past the opening seconds and keep output varied enough that platforms don't treat it as mass-produced.

### 1.4 High-density cut profile

An optional profile with 1.5-3 s slides and a transition between each, built as an option to test rather than a default. The creator research finds stimulation follows an inverted U, so faster is not assumed better.

**Done when:** a high-density profile renders without subtitle desync and is selectable per platform. No issue yet.

### 1.7 Hook-variant A/B measurement

The registry records each video's hook variant and a report segments retention by it. It needs the first-seconds metrics of 5.6. Instagram Trial Reels, shown to non-followers first, are the one native A/B lever; check whether the publishing provider can post them.

**Done when:** the registry carries the hook variant per video and a report segments retention by it. No issue yet.

### 1.8 Loop-friendly ending

End on the peak, with an optional seamless loop to the opening frame. [Design 0002](design/0002-end-on-the-peak.md), #543.

**Done when:** a profile with the loop flag renders a last frame that matches its first within a tolerance.

### 1.10 Output-variety guard

Selection spreads renders across the variant dimensions (hook, template, voice, cut density, cold open), and a report shows recent-render variety. [Design 0006](design/0006-render-choices-and-variety-report.md), #547.

**Done when:** a batch spreads renders across the variant dimensions and a report shows their distribution.

### 1.11 Humanization layer

Conversational naturalism, context-varied pauses and an author signature are built and held off. Enabling them is #540; naturalism is re-measured first (#541).

**Done when:** each piece is enabled with a measured setting or left off with the reason recorded.

### 1.12 Motion on every still

Slow, varied, jitter-free motion on every still image. [Design 0001](design/0001-motion-on-every-still.md), #542.

**Done when:** a still-image profile renders motion on every still, varied between products, with the default unchanged.

### 1.13 Sound design

Sparse event sound effects (#544), a voice processing chain (#545) and beat-snapped cuts (#546), each optional. Designs [0003](design/0003-sparse-sound-effects.md), [0004](design/0004-voice-processing-chain.md) and [0005](design/0005-beat-snapped-cuts.md).

**Done when:** each option renders as designed and a batch comparison decides which to enable.

### 1.14 Remove AI-slop signals

Clean product images (#554), no stock-clip reuse (#555), TTS text normalisation (#556), a narrator-voice evaluation (#557) and the TikTok AI-label decision (#558). Designs [0012](design/0012-clean-product-images.md) to [0016](design/0016-tiktok-ai-label.md) carry the evidence.

**Done when:** each item is enabled on measured results or left off with the reason recorded, and the label decision is written down.

## Phase 2: Non-affiliate and tutorial content (Now/Next)

Topic renders and per-profile stock keywords have shipped. What remains is making tutorials useful and letting a pillar opt out of affiliate links.

### 2.2 Non-affiliate pillar mode

A pillar flag that skips the affiliate URL and the link-in-bio registration, so an educational track runs beside the affiliate one.

**Done when:** a video under a non-affiliate pillar publishes with no affiliate URL and no bio link, while still naming products in the script. No issue yet.

### 2.5 Useful tutorials

Scripts built from a sourced step list and sized by step count (#559), a visual per step timed to its narration (#560), and templated explanatory graphics (#561). Designs [0017](design/0017-tutorial-step-lists.md), [0018](design/0018-tutorial-step-visuals.md) and [0019](design/0019-tutorial-graphics.md).

**Done when:** a topic render shows each step as it is spoken, with sourced steps, a length set by their count, and graphics that encode script facts, enabled on measured results.

## Phase 3: Conversion infrastructure (Next)

Make attribution and calls to action measurable.

### 3.1 UTM tagging at publish

Platform- and pillar-keyed UTM parameters on affiliate URLs, configurable per platform, skipped for the link-in-bio destination.

**Done when:** every post ships with platform-tagged links that the analytics layer can attribute. No issue yet.

### 3.2 Price-anchored calls to action

Call-to-action wording chosen by price band, with an end-of-video arrow pointing at the bio link.

**Done when:** every video ends with a band-appropriate call to action in voice, on-screen text and the arrow overlay. No issue yet.

### 3.3 A/B caption variants

A reproducible per-product caption variant, recorded in the registry for per-variant conversion.

**Done when:** the registry records the variant per post and one A/B test is running. No issue yet.

### 3.4 Hashtag pools per pillar

Pillar- and platform-keyed hashtag pools, within each platform's own cap. Builds on the hashtag rework in #567.

**Done when:** every post carries a pillar-appropriate, platform-capped hashtag set without manual curation.

### 3.5 Cross-platform watermark check

A test that every platform publishes the source render, never a re-downloaded copy carrying another platform's watermark.

**Done when:** the test exists and each platform path provably uses the source render. No issue yet.

### 3.6 Instagram Reels delivery audit

The Instagram path posts as a Reel, guarded by a payload test.

**Done when:** Instagram posts publish as Reels, confirmed end to end, with a test against drift. No issue yet.

### 3.7 Pre-production conversion gate

The scraper drops weak candidates (low rating, thin reviews, out of stock, price outside a band) before rendering, with the reason logged.

**Done when:** a scrape rejects below-threshold products before the producer stage. No issue yet.

### 3.8 Bait-free closing lines

No configured call to action or closing line matches an engagement-bait pattern, guarded by a test. [Design 0008](design/0008-bait-free-closing-lines.md), #549.

**Done when:** no configured call to action or closing-line example matches a bait pattern.

### 3.9 Script lint and search-phrase placement

A lint for machine-writing tells and pace, and a report of where the search phrase lands. [Design 0007](design/0007-script-lint.md), #548.

**Done when:** the lint rejects the listed shapes when enabled and the report counts placement per render.

## Phase 4: Per-platform optimisations (Next)

### 4.1 Instagram Stories re-share with link sticker

Each Reel is re-shared as a Story with a link sticker, opt-in per pillar.

**Done when:** every Reel triggers a Story re-share with a link sticker. No issue yet.

### 4.3 Comment-reply video mode

A producer mode that renders a short reply clip from a comment, a product and a parent video, into a drafts folder for manual review.

**Done when:** the producer renders a reply clip to a drafts folder. No issue yet.

### 4.4 Amazon Influencer Storefront target

Upload each render to a configured Amazon Influencer Storefront beside the social platforms.

**Done when:** the publisher uploads each render to the storefront. No issue yet.

### 4.5 Amazon OneLink localisation

Wrap affiliate URLs with OneLink so non-US viewers reach their local store with the tracking id kept.

**Done when:** non-US clicks reach the local store with the tracking id preserved. No issue yet.

### 4.6 Link-in-bio funnel hygiene

A new product rotates the featured slot rather than appending, so the bio carries two or three links.

**Done when:** the bio shows at most three links, with the newest product featured. No issue yet.

### 4.7 Cover frames

A cover image per render, set as the poster where the platform accepts one. [Design 0011](design/0011-cover-frames.md), #552.

**Done when:** every render produces a cover and the publish payload sets it on the platforms that accept one.

### 4.8 Episodic series framing

A per-pillar episode counter in titles and captions, with each episode standing alone.

**Done when:** titles and captions carry a per-pillar episode number. No issue yet.

### 4.9 Product titles per platform

A short written YouTube title for product videos, keyword first. [Design 0009](design/0009-product-titles.md), #550. Hashtag caps per platform are #567.

**Done when:** product videos carry a written YouTube title within the configured maximum.

## Phase 5: Analytics and learning (Next)

### 5.1 Analytics module

A local store owns per-post performance history, captured on a schedule that follows each source's expiry. [Design 0020](design/0020-analytics-history.md).

**Done when:** a scheduled capture writes readings without losing figures already taken, and one command reports performance by content format.

### 5.2 Listing drift diagnostic

Flag published products whose listing drifted (price up, out of stock, rating down), with a hook to drop them from the bio.

**Done when:** a report flags drifted listings and offers a remove-from-bio action. No issue yet.

### 5.3 Segmented performance reports

Reports per pillar, template, voice, caption variant and hook style over rolling four-week windows.

**Done when:** one command answers which pillar converts best on a platform over the last four weeks. No issue yet.

### 5.4 Reach by content-format arm

Join performance, keyed by post, with the format arm, keyed by product, through the publish history.

**Done when:** one command reports day-N and durability figures by format arm and states how many posts it could not place. No issue yet.

### 5.6 First-seconds metrics

Store every first-seconds and quality metric the platforms expose, segmented by format arm and render choice. [Design 0010](design/0010-first-seconds-metrics.md), #551.

**Done when:** the store carries each available metric per post, unavailable ones as unknown, and the reports segment them.

## Phase 6: Threshold-gated unlocks (Later)

Blocked on platform features, eligibility, or earlier items.

- **6.1 Long-form profile (60-120 s)**, for YouTube long-form and TikTok's 60-second Creator Rewards threshold.
- **6.2 TikTok Shop product tags**, once the account is Shop-approved.
- **6.3 Instagram native affiliate tags**, once the account and market qualify.
- **6.4 YouTube end-screen subscribe overlay**, which needs 6.1, since Shorts have no end screens.
- **6.5 Zernio SDK migration**, from `late-sdk` to `zernio-sdk`, with the publisher's next substantive change.
- **6.6 Pycaps follow-ups**, tracked as issues with the `pycaps` label.

## Toward 1.0.0

Most of the feature surface is built. What stands between the pre-production line and 1.0.0 is consolidation: API stability, test coverage, distribution, and proof that the pipeline runs reliably at volume.

**API stability**
- Config schema frozen for one full minor cycle. New fields are additive with sensible defaults.
- CLI flags stable across producer, scraper, publisher and global batch. Removals go through a one-release deprecation with a `DeprecationWarning`.
- Public Python entry points (`src/pipeline.global_batch`, `src/video.producer.cli`, `src/scraper.amazon.scraper`, `src/publisher.late.cli`) treated as a stable surface; signature changes need a major bump.
- Every flag pair shared between a standalone module CLI and the global batch stays in sync, enforced in CI by a parity test.

**Test coverage on the critical paths**
- Critical-path coverage is the gate, not a global percentage: scraper, producer and per-platform publisher each covered by unit and integration tests, at 80% or more on those modules.
- Overall line coverage holds the `--cov-fail-under=50` floor and trends up. The 90% target in `docs/testing.md` is post-1.0.0.
- One real-API smoke test in CI that runs scrape, produce and publish on a fixture product with sandbox credentials, optional so forks without secrets stay green.

**Documentation completeness**
- The installation guide is tested from a clean Linux box and a clean macOS box by someone new to the project.
- The configuration reference covers every YAML field with type, default and an example.
- The troubleshooting guide covers the top issues from the tracker.
- A quickstart takes a fresh clone to a published video on a sandbox account in under 5 minutes of human time.

**Operational maturity**
- A documented performance baseline (seconds per product, peak RSS, API cost per video), tracked in CI with an alert at a 20% slowdown over 10 runs.
- Structured logging with consistent field names across modules.
- Every external integration has a circuit breaker and retry policy, with defaults documented and overrides in YAML.

**Distribution**
- A PyPI package installable in a clean venv and runnable with documented system dependencies (FFmpeg, Playwright Chromium), with a decision on extras-gating pycaps.
- A Docker image with all system dependencies, ideally CPU-only and CUDA variants.
- `docs/versioning.md` updated for the 1.0.0 promise.

**Security and dependencies**
- Bandit and Safety stay clean.
- Secret masking covers every log path, with a test that no env-var value appears in a log file.
- No HIGH or CRITICAL CVE in pinned dependencies for more than 7 days.

**Roadmap items in scope**
- Every Phase 0 item.
- Every Phase 1, 2 and 3 item, except the experiments held for the reach-test readout (1.4, 1.8, 1.11, 1.12, 1.13, 1.14, 2.5, 3.8 and 3.9), which ship off and are enabled only on measured results.
- At least half of Phase 4 shipped or in review.
- Phase 6 is out of scope.

**Real-world proof**
- 100 or more videos produced and published end to end across all three platforms.
- Subtitled videos render correctly under manual QA on each platform's safe-zone overlays.
- Affiliate links land in the link-in-bio destination with no manual cleanup.

## Update rules

- An item is an outcome with a horizon, a "done when" and a link to its issue or design doc. Detail goes in the design doc or the issue, not here.
- A shipped item leaves the roadmap; the CHANGELOG records it. A dropped item leaves with a one-line reason on its issue.
- An item that sits in Phase 6 for two quarters without movement is pruned or rewritten to name its real blocker.
