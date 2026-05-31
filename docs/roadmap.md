# Roadmap

Last updated: 2026-05-31

Forward-looking work on ContentEngineAI, grouped into phases by horizon. Items are aspirational, not commitments. Order within each phase is rough priority.

The phases are sequenced so each one builds on the previous. Phase 0 is the disclosure-compliance baseline that every promotional render needs to satisfy and has to land before anything else ships. Phase 1 fixes the retention foundation (hook quality and 3-second hold). Phase 2 organises content by theme. Phase 3 makes attribution and CTAs measurable. Phase 4 layers per-platform optimisations. Phase 5 closes the analytics loop. Phase 6 captures unlocks blocked on platform thresholds or features.

Issues and PRs are welcome on any item. If you want to pick something up, open an issue first so we can talk through scope.

## Phase 0 — Disclosure compliance baseline (partial, blocking)

Targeted to gate the 1.0.0 release. Creators using the pipeline for affiliate marketing have legal disclosure obligations under the FTC Endorsement Guides (any US-facing content), the Amazon Associates Operating Agreement (any Amazon affiliate program participant), and per-platform policy (TikTok / Instagram / YouTube). Compliance is not a polish item: FTC penalties reach $53,088 per violation per post in 2025 and Amazon Associates enforces with account termination, no warning. The pipeline should make compliance the default render output, not a per-video checklist.

The persistent on-frame overlay, first-line caption disclosure, platform-tag audit, regression suite, and `docs/compliance.md` shipped together. What remains gates 1.0.0:

### 0.3 Affiliate program literal-phrase rendering

Add a configurable literal-phrase block that ships the affiliate program's required identification text. For Amazon Associates: "As an Amazon Associate I earn from qualifying purchases" (or a substantially similar pre-approved statement). Render in at least one of: profile bio, the closing frame of the video, or the caption body. Drive from a new `affiliate_disclosure` config keyed by program name so non-Amazon programs (ShareASale, Impact, eBay Partner Network) can plug in their own phrases.

**Done when:** every published video includes the configured affiliate program literal phrase in at least one of bio / on-frame / caption.

### 0.4 Localized disclosure variants

When the script language is not English, the disclosure must match. Add per-language disclosure variants keyed off the existing TTS / script language: `en` → `#ad`, `es` → `#publi` or `#publicidad`, etc. Required by FTC's same-language rule and by Spain's Royal Decree 444/2024 for creators in the EU. A config-load-time validator should warn (not silently default) when the configured disclosure language doesn't match the script language for a given run.

**Done when:** a Spanish-language render emits a Spanish-language disclosure overlay and caption first line; an English render emits English; mismatches raise a config-load warning.

## Phase 1 — Hook and retention surgery (Now)

The foundational items shipped across 0.48.0-0.49.0 (audio-keyword opener, engagement-bait closing line, caption mirror) and on the current branch (punchline-first opener with visual interrupt, short profile). One Gen Z cut-density profile remains.

### 1.4 High-density cut profile

Add a `cut_density: high` profile setting in `config/video_production.yaml` that drops the minimum slide duration to 1.5-3s and adds a transition (whip pan, hard cut, zoom punch) between every slide. Useful for younger audiences on platforms whose feeds reward visual energy density. Keep the existing slow-cut profile available for use cases where it fits better. Strategy and shot-length bands are in `docs/promotional-video-best-practices.md` section 2.

**Done when:** a high-density profile renders without subtitle desync and is selectable per platform.

### 1.6 Cross-platform safe-zone refresh

The safe-zone docs were aligned to 2026 platform specs (`docs/platform-safe-zones.md` is now the canonical source; the subtitle and promo docs cite it). The runtime constants in `src/video/config/constants.py` still carry the older union (top 200 / bottom 1440 / left 50 / right 840). Update them to the 2026 union (top 270 / bottom 1250 / left 60 / right 900 on 1080x1920), driven by Meta's March 2026 Reels unification (14% top, 35% bottom). The current `max_y` of 0.75 lets captions land inside Reels' bottom interactive zone. A single cross-platform render should clamp to the union, not `platform=tiktok` only. Tracked as GitHub issues.

**Done when:** the runtime safe-zone defaults match the canonical doc, and a render's lowest caption pixel stays above y=1250.

### 1.7 Hook-variant A/B measurement

The cold-open variant framework already selects one of several hook variants per product and writes it to `pipeline_state.json`. Persist that variant into the published-products registry (a `hook_variant` column) and surface it in the analytics reports so per-variant retention is measurable. Hook hold is the primary retention lever; without per-variant data there's no way to learn which opener holds past the 3-second mark. Pairs with the high-density cut profile (1.4) as the two retention experiments. Note this is the measurement layer; 1.2 and 1.4 are the production layers.

**Done when:** the registry carries the hook variant per video and a report segments retention by hook variant.

### 1.8 Loop-friendly ending

Optionally match the final frame to the opening frame so the clip loops seamlessly on autoplay. Replay rate is a ranking signal on short-form feeds. Config flag per profile, off by default.

**Done when:** a profile with the loop flag renders a video whose last frame matches its first within a tolerance, selectable per profile.

## Phase 2 — Non-affiliate pillar mode (Now/Next)

Targeted for weeks 3-4. The pillar system itself shipped in 0.43.0 (default pillars: `value`, `novelty`, `utility`; keyword pool grouped by pillar in `config/scraper.yaml`; templates mapped to pillars in `config/ai_services.yaml::script_templates.pillars`; `--pillar` flag on both `src/video/producer/cli.py` and `src/pipeline/global_batch.py`; per-pillar preambles and audiences). What's still missing: an opt-out for affiliate URL injection so the same pipeline can carry an educational track.

### 2.2 Non-affiliate pillar mode (educational / how-to track)

Add a `non_affiliate: true` flag at the pillar level. When set, the publisher skips appending the affiliate URL and skips the link-in-bio registration. Lets creators run an educational or how-to track alongside an affiliate track from the same pipeline. Educational content earns trust and search SEO; the audio script can still mention specific products by name (audio-keyword crossover indexes the video for both the help query and the product query) without an explicit affiliate push.

**Done when:** a video produced under a `non_affiliate: true` pillar publishes with platform-appropriate captions, no affiliate URL, and no bio-link registration, while still naming products in the spoken script.

## Phase 3 — Conversion infrastructure (Now/Next)

Targeted for weeks 5-6. Once retention is up and pillars exist, attribution and CTAs become measurable.

### 3.1 UTM tagging at publish

Append `utm_source`, `utm_medium`, `utm_campaign` query parameters to affiliate URLs at publish time, with values keyed off the target platform and the video's pillar. Configurable on/off per platform in `config/publisher.yaml`. Skip when the destination is the link-in-bio service itself.

**Done when:** every cross-platform post ships with platform-tagged links, and click attribution becomes possible at the analytics layer.

### 3.2 Price-anchored CTA template system

Add a CTA template system to the publisher with platform-specific defaults. Sub-$50 products use action-band wording ("Shop Now", "$X here →"); $50+ products use trust-band wording ("Learn More", "Link in bio"). Wire the price band into the publisher metadata generation (the scraper already pulls price; thread it through). Add an emoji-arrow overlay sticker in the final 3 seconds of every video pointing at the bio-link area on each platform's UI. CTA placement defaults to bottom-center, slightly above the native button area (the bottom 12% of frame is clipped by username/caption on TikTok).

**Done when:** every video ends with a price-anchored CTA in TTS, on-screen text, and emoji-arrow overlay, with band-appropriate wording.

### 3.3 A/B caption variants

Add an `ab_variant` field to the publish step that selects A or B per product via deterministic salted MD5. Two variant pools per platform (`cta_variants_a`, `cta_variants_b`) in `config/publisher.yaml`. Persist the chosen variant in the registry so downstream analytics can compute conversion per variant.

**Done when:** the registry has an `ab_variant` column, the publisher logs which variant was used, and one A/B test is in flight.

### 3.4 Hashtag pools per pillar

Add `hashtag_pools` to `config/publisher.yaml`, keyed by pillar and platform. Per-platform caps reflect 2026 platform rules: Instagram has a hard 5-tag limit on Posts and Reels (Meta change, December 2025), TikTok favours 4-8 with volume bias, YouTube hashtags barely matter beyond the top 3 in description. Each post draws a baseline pillar-relevant set plus 2-3 product-specific tags from the LLM-generated metadata. Removes the need to curate hashtags per video.

**Done when:** every published video carries a pillar-appropriate, platform-capped hashtag set, no manual curation.

### 3.5 Cross-platform watermark check

Add a smoke test in `src/publisher/` that asserts the platform-bound video file path matches the original render output, not a TikTok-source-derived file. Logs the file checksum for cross-platform comparison. Meta's 2026 originality rules de-rank Instagram content with TikTok watermarks, so the publisher must provably use the source render, not a re-downloaded copy.

**Done when:** the test exists and passes, and each platform publisher provably uses the source render.

### 3.6 Instagram Reels delivery audit

Verify the Instagram path in the publisher posts as a Reel, not as a Feed Post. Zernio accepts separate `instagramSettings` for each format. Add a regression test that asserts the payload carries the Reels flag.

**Done when:** the next batch publishes to IG as Reels, confirmed end-to-end, with a test in place to prevent drift.

### 3.7 Pre-production conversion gate in the scraper

Score and filter product candidates before rendering, using the data the scraper already pulls (price, rating, review count, stock, Prime/shipping). Drop weak-converting candidates up front: rating below a floor, thin review count, out of stock, price outside a configurable impulse band. Renders are expensive; spending them on products that won't convert is the largest avoidable waste in the funnel. This sits upstream of the listing-drift diagnostic (5.2), which only monitors drift after publish. Thresholds live in `config/scraper.yaml`.

**Done when:** a scrape run rejects below-threshold products before the producer stage, with the rejection reason logged.

## Phase 4 — Per-platform optimisations (Next)

Targeted for the following quarter. Builds on Phases 1-3.

### 4.1 Instagram Stories auto-publish with link sticker

After publishing a Reel, the publisher automatically schedules a Story re-share with a link sticker pointing at the same destination URL. Optional poll or quiz sticker for engagement signal. The link sticker is open to all Instagram accounts regardless of follower count (changed in 2021), so this is a near-zero-threshold reach and CTR unlock. Make this opt-in per pillar so non-affiliate pillars don't carry an affiliate link sticker.

**Done when:** every Reel published triggers an automatic Story re-share with a link sticker, confirmed live on the IG account.

### 4.2 YouTube engagement-bait pinned comment

Add a "pinned comment" field to the YouTube publish path. Generate the pinned comment from the script's closing fork (Phase 1.5) — the comment is the spec-correction or two-option fork phrased as a direct question. Don't pin a "subscribe" CTA; pinned subscribe asks underperform, and the closing 5s of the video already carries a script-level subscribe CTA. End screens are not available on Shorts (long-form only), so the pinned comment is the equivalent placement.

**Done when:** every YouTube Short publishes with an engagement-bait pinned comment derived from the script.

### 4.3 Comment-reply video mode

Add a `--mode reply` option to the producer that takes a parent video ID, a comment text, and a product ID, and produces a 10-15s response clip with the comment text overlaid and the product image/video. Doesn't auto-publish; surfaces the rendered clip in `outputs/<asin>/reply_drafts/` for manual review and submission. Future hook: an analytics step that scans recent comments for product questions and pre-draws reply videos.

**Done when:** producer renders a reply-mode video from a comment text + product ID + parent video ID, output to a drafts folder.

### 4.4 Amazon Influencer Storefront publisher target

Add the Amazon Influencer Storefront as a publisher target alongside YouTube, TikTok, and Instagram. The storefront accepts MP4 uploads; videos appear on relevant product detail pages and earn commission from on-Amazon traffic the creator didn't drive. The same render output that goes to YouTube can go to the storefront in v1; trim to under 60s if the storefront video length cap requires it (verify at integration time). Maps products to storefront slots (one or two products per uploaded video). Eligibility for the Amazon Influencer Program is separate from Amazon Associates and uses "demonstrated social influence" rather than a formal follower minimum.

**Done when:** the publisher uploads each rendered video to a configured Amazon Influencer Storefront alongside the other platforms.

### 4.5 Amazon OneLink localisation wrapper

Wrap every Amazon affiliate URL with the OneLink redirect at publish time so non-US viewers get routed to their local Amazon storefront automatically. Verify the wrapped URL preserves the affiliate tracking ID across locales. Free Amazon feature; recovers the share of traffic that lands on the wrong-region store.

**Done when:** non-US clicks route to the correct local Amazon storefront with tracking ID preserved.

### 4.6 Link-in-bio funnel hygiene

Update the link-in-bio integration in `src/publisher/link_in_bio/` so adding a new product rotates the featured slot rather than appending. Industry baseline: link-in-bio hubs see 20-40% click-through to the top destination URL when only 2-3 links are present; the first 3 positions get ~130% higher CTR than positions 4-10. Cap the bio at the most recent featured product plus an "all products" fallback.

**Done when:** the bio shows at most 2-3 links at any time, with the most recent product as the featured slot.

### 4.7 Cover / poster-frame generation

Generate a cover frame for each video (hero product image plus a bold three-word title) and set it as the poster frame. The Shorts/Reels grid and the profile page drive browse-tab click-through and the follow decision; right now nothing controls the thumbnail. Reuses the hook text and the product image the producer already has.

**Done when:** every render produces a cover image and the publish payload sets it as the poster where the platform supports it.

### 4.8 Episodic series framing per pillar

Add a per-pillar counter to the registry and thread it into title/caption templates (for example "Pillar pick #12"). Series framing is a documented return-viewership driver and targets the follower/subscriber conversion gap. Builds on the existing pillar system.

**Done when:** published titles/captions carry a per-pillar episode number that increments across the back-catalogue.

## Phase 5 — Analytics and continuous learning (Next)

Targeted for the following quarter. Closes the loop between pipeline output and platform signals.

### 5.1 Companion analytics module

Optional analytics tooling that pulls platform metrics (TikTok, Instagram, YouTube, Amazon Associates, link-in-bio) into a local SQLite store and produces weekly reports segmented by pillar, template, voice profile, and A/B variant. Either a sister module in this repo or a separately released companion tool, to be decided when the work starts. Avoid building against Amazon's PA-API — it deprecates May 15, 2026; use the new Creators API or scraping instead.

### 5.2 Listing-side drift diagnostic

Add a diagnostic step to the analytics tool that pulls the current state of every recently-published product's source listing (price, stock, availability flags, rating, review count) and flags products whose state has drifted from as-published thresholds (price up >15%, out of stock, rating dropped below 4.0, return-rate badge added). Per-product alert in the weekly report. Optional hook: a publisher endpoint to remove a flagged product from the link-in-bio rotation.

**Done when:** the dashboard flags drifted listings and exposes a "remove from bio" hook for manual action.

### 5.3 Per-pillar / per-CTA / per-hook performance reports

Add report types to the analytics tool for: per-pillar conversion, per-template engagement, per-voice watch-through (historical baseline; mostly constant after voice pinning), per-A/B-variant CTA click rate, per-hook style retention. 4-week rolling windows so seasonality doesn't pollute the comparison.

**Done when:** the user can answer "which pillar converts best on platform X over the last 4 weeks" with a single command.

## Phase 6 — Threshold-gated unlocks (Later)

Blocked on platform features, eligibility, or earlier items landing.

### 6.1 Long-form profile (60-120s)

Add a long-form profile (`slideshow_long_60s` or `video_long_90s`) to `config/video_production.yaml` with longer slide durations and a longer script template variant. Useful for YouTube long-form, TikTok Creator Rewards eligibility (which requires 60s+ videos), and platforms that reward depth over brevity. Word-count budget assumes ~150-180 wpm TTS pacing. Pairs with the Shorts-to-long-form bridge: the algorithmic Shorts→long-form recommendation spillover ended in late 2025, so the bridge is now user-driven via explicit CTAs and channel-page landing experience.

### 6.2 TikTok Shop product tagging in publisher

Add support for TikTok Shop product tags in the publisher payload. Requires the user's TikTok Shop affiliate approval and Shop-eligible product listings. Once enabled, posts in supported regions can carry tappable product cards, replacing or complementing the bio-link CTA.

### 6.3 Instagram native affiliate product tagging

Wire up Instagram's native affiliate product tagging through the publisher when the user's account meets the threshold (1K followers + Pro account) and Instagram has rolled the affiliate program out to the user's market. As of mid-2026 the affiliate program is live in only a small number of markets; check current availability before building. Solves the bio-click drop-off because the tag is in-feed.

### 6.4 YouTube end-screen subscribe overlay

Add an end-screen overlay step in `src/video/producer/` that bakes a subscribe CTA on the last few seconds. End screens are long-form-only — they don't appear on Shorts under 60s — so this depends on the long-form profile (6.1) shipping first. Pinned-comment subscribe asks can ride on the existing publisher path but underperform engagement-bait pinned comments (Phase 4.2), so end-screens are the better long-term lever.

### 6.5 Zernio SDK migration

Migrate from the legacy `late-sdk` package to `zernio-sdk`, rename `LATE_API_KEY` to `ZERNIO_API_KEY`, and update imports under `src/publisher/late/`. The old package keeps working during the SDK grace period; do this when the publisher gets its next substantive change so the migration rides on existing testing.

### 6.6 Pycaps follow-ups

Open items tracked as GitHub Issues with the `pycaps` label: AI word tagging via the Gemini key (in flight), two-part subtitles plus pycaps hybrid, CSS renderer integration test in CI, custom project template, WhisperX upgrade.

## Toward 1.0.0

The current line is `0.42.x`, status pre-production. Most of the feature surface is built. What stands between today and 1.0.0 is consolidation: API stability, test coverage at target, distribution, and proof that the pipeline runs reliably at volume.

Concrete gates for the 1.0.0 release:

**API stability**
- Config schema frozen for one full minor cycle. No breaking field renames or removals; new fields are additive with sensible defaults.
- CLI flags stable across producer, scraper, publisher, and global batch. Removals go through a one-release deprecation with a `DeprecationWarning`.
- Public Python entry points (`src/pipeline.global_batch`, `src/video.producer.cli`, `src/scraper.amazon.scraper`, `src/publisher.late.cli`) treated as a stable surface; signature changes need a major bump.
- Module/Batch Alignment Rule covers every flag pair (CLAUDE.md), enforced in CI by a parity test.

**Test coverage at the documented targets**
- Unit tests at the >=90% line CONTRIBUTING.md already promises (currently around 45%).
- Integration tests at >=80% on the critical paths: scraper end-to-end, producer end-to-end, publisher per-platform.
- One real-API smoke test in CI that exercises scrape → produce → publish on a fixture ASIN with sandbox credentials. Marked optional so forks without secrets stay green.

**Documentation completeness**
- Installation guide tested from a clean Linux box and a clean macOS box, top to bottom, by someone who hasn't seen the project before.
- Configuration reference covers every YAML field with type, default, and at least one example.
- Troubleshooting guide includes the top issues from the issue tracker.
- A working quickstart that takes a fresh clone to a published video on a sandbox account in under 5 minutes of human time (modulo download speeds and API latency).

**Operational maturity**
- A documented performance baseline: seconds per product at default profile, peak RSS, approximate Google API cost per video. Tracked in CI with a regression alert at >=20% slowdown over a 10-run window (the report types in `tools/performance_report.py` already exist).
- Structured logging with consistent field names across modules. Module summaries (already standardized in 0.35.0) extended to cover every long-running step.
- Every external integration has a configured circuit breaker and retry policy; defaults documented, overrides exposed in YAML.

**Distribution**
- PyPI package buildable, installable in a clean venv, and runnable with documented system deps (FFmpeg, Playwright Chromium). Decide whether to pin or extras-gate the heavy optional pieces (pycaps, coqui-tts).
- Docker image published with all system deps baked in, ideally one for CPU-only and one for CUDA.
- Versioning policy in `docs/versioning.md` updated to reflect the 1.0.0 promise.

**Security and dependencies**
- Bandit and Safety stay clean (already enforced).
- Secret masking covers every log path; one test asserts no env-var values appear in produced log files.
- No HIGH or CRITICAL CVEs in pinned dependencies for more than 7 days; Dependabot batched into patch releases per existing workflow.

**Roadmap items in scope for 1.0.0**
- All Phase 0 items shipped (compliance baseline; gating).
- All Phase 1, 2, and 3 items shipped.
- At least four of six Phase 4 items shipped or in review.
- Phase 6 items are explicitly out of scope; they're 1.x material once their gates clear.

**Real-world proof**
- The pipeline has produced and successfully published a meaningful volume of videos (target: 100+) end-to-end across all three target platforms.
- Subtitled videos render correctly on TikTok, YouTube Shorts, and Instagram Reels under manual QA on each platform's safe-zone overlays.
- Affiliate links land in the link-in-bio destination automatically with no manual cleanup.

When all gates are green, the next release is `1.0.0`, not `0.43.x`. Anything not on this list is a 1.x or 2.x conversation.

## Update rules

- Each shipped item moves to the Shipped section at the bottom (append-only) with a one-line summary and the release version.
- Items dropped from the roadmap also get a one-line note in the same section, with the reason.
- Don't let Phase 6 accumulate without bound. If something sits there for two quarters with no movement, either prune it or rewrite the description to reflect the actual blocker.

## Shipped

Backfilled from the changelog (0.1.0 through 0.42.x). Grouped by theme rather than version, with the release range that delivered each capability.

**Core pipeline (0.1.0 - 0.2.x)**
- Initial open-source release: Amazon scraper, multi-provider AI services, FFmpeg video assembly, audio-synced subtitles, background music, batch processing.
- AI-generated platform-aware video descriptions with `#ad` disclosure baked in.

**Subtitle configuration and styling (0.3.x - 0.8.x)**
- Deterministic per-product font and color randomization with style presets (`minimal`, `modern`, `bold`, `random`); `--preset` CLI override.
- Unified subtitle configuration with anchor-based positioning (`top`, `center`, `bottom`, `above_content`, `below_content`) and content-aware mode.
- One-effect-per-video rule enforced across all presets.
- Two-part subtitle system: upper line for affiliate URLs or product titles, lower line for voiceover, independently styled.

**Configuration architecture (0.4.x - 0.5.x, 0.13.x - 0.14.x)**
- Modular YAML split (core, video_production, ai_services, subtitles, performance, scraper, publisher, url_shortener, pipeline).
- Three-tier precedence: CLI > environment > YAML, validated by Pydantic at startup.
- Producer and scraper modules split into focused submodules; type-safe configuration models throughout.

**Scraper (0.7.x, 0.12.x, 0.21.x, 0.26.x, 0.28.x, 0.34.x)**
- URL shortening via PicSee.io with provider-agnostic registry.
- M3U8/HLS video extraction with strict product filtering.
- Platform detection registry pattern with Amazon ASIN validation.
- Two-tier product limits (`max_products` + `products_per_keyword`).
- Full-URL and shortened-URL input support (tr.ee, amzn.to, etc.); `--input-file` and `--batch-size` CLI options.
- Global batch page retry when products fail media validation.

**Video producer (0.10.x - 0.12.x, 0.16.x, 0.27.x, 0.30.x)**
- Multiple video assembly modes (`product_video_sequential`, `slideshow_images1`, `slideshow_images2`).
- CTA-based timing for upper subtitles with configurable keyword detection.
- Assembler split from a 3,311-line monolith into 7 focused modules.
- Script template system: 15 prompt variants with deterministic per-product selection via salted MD5.
- TTS voice profiles (`soft_intimate`, `calm_confident`, `gentle_storyteller`, etc.) with Gemini TTS provider added alongside Google Cloud TTS, automatic fallback.
- Inline markup preprocessing for sentence-boundary pauses.
- LLM provider fallback chain: Gemini primary, OpenRouter fallback with free-model discovery.

**Subtitle engines and timing (0.35.x - 0.42.x)**
- Platform-aware safe zones for TikTok, YouTube Shorts, and Instagram Reels.
- Optional pycaps animated subtitle engine alongside the FFmpeg path; CSS and pictex renderers.
- Whisper timing post-processing (min duration, gap merge, segment-end hold, audio lead).
- Style presets as the single source of truth for subtitle styling.
- Font and color pools moved from Python enums to YAML (data, not code).
- Subtitle config consolidated into a single strict `SubtitleSettings` Pydantic model with deep-merge profile overrides.

**Audio (0.34.x, 0.42.x)**
- Audio provider platform with `BaseAudioProvider` ABC and `AudioManager` chain.
- Jamendo Music provider (CC-licensed, `fuzzytags` search); Freesound provider as fallback; local files as last resort.
- TTS final-word truncation fix via accurate `silenceremove` semantics.

**Platform metadata (0.17.x, 0.23.x)**
- Platform-specific metadata generation: YouTube (5000-char descriptions), TikTok (2200-char captions), Instagram (2200-char captions).
- Metadata cache with TTL, A/B testing scaffolding, batch generation, multi-format export, trend-aware hashtag merging.

**Publisher (0.18.x - 0.19.x, 0.22.x, 0.24.x, 0.29.x, 0.33.x, 0.38.x)**
- Late.dev integration (rebranded as Zernio) with multi-platform support: YouTube, TikTok, Instagram, Facebook, Twitter, LinkedIn.
- Auto-scheduling with occupied-slot detection across an 8-week lookahead.
- Post-publication cleanup with safety checks.
- Multi-account support, conflict resolution with alternatives, retry queue, webhook handler with HMAC verification.
- Platform-specific publishing mode (separate posts per platform with optimized metadata).
- Link-in-bio integration via lnk.bio, with affiliate URL fallback and image fallback.
- Published-products registry (JSON + CSV) with rebuild support.
- First-comment support for YouTube and Instagram (affiliate links posted as the first comment).
- TikTok branded-content disclosure handled automatically.

**Pipeline orchestration (0.15.x, 0.19.x, 0.25.x)**
- Global batch pipeline: scrape → produce → publish in a single command.
- Pipeline resume from last successful phase via `--resume`.
- Dry-run mode (`--dry-run`) and JSON output format for machine consumption.
- Webhook notifications on phase completion and pipeline events.

**Resilience and operations (0.20.x, 0.31.x)**
- Network retry decorator with exponential backoff for HTTP requests.
- Circuit breaker pattern with pre-configured breakers for Freesound, Pexels, OpenRouter, Google STT, Scraper.
- Secret masking filter applied to all log handlers.
- Performance monitoring with summary, trends, detailed, comparison, and regression report types via `tools/performance_report.py`.

**Quality (0.9.x, 0.22.x, 0.31.x)**
- Compliance test suite (114 tests) validating all documented requirements.
- Comprehensive test coverage for video producer, scraper, publisher, audio, AI metadata modules.
- Performance regression detection with configurable window and threshold.

**Content pillars and prompt hygiene (0.43.x)**
- Content pillars system. Default pillars `value`, `novelty`, `utility`. Keyword pool grouped by pillar in `config/scraper.yaml`; script templates mapped to pillars in `config/ai_services.yaml::script_templates.pillars`; `--pillar` on both producer CLI and global batch.
- Per-pillar runtime preamble (`script_templates.pillar_preambles`) and per-pillar audience override (`script_templates.pillar_audiences`).
- Channel-wide narrator profile (`script_templates.narrator_profile`) prepended to every script prompt, with anti-AI-tells list, banned phrases, single-CTA rule, persona anchor, voice example.
- `{SHORT_PRODUCT_NAME}` placeholder resolved by a brand-plus-model heuristic in `format_prompt`.
- NFKC normalization of product titles and descriptions before prompt injection (kills Amazon's mathematical-alphabet bold tricks); em/en dash replacement in descriptions.
- Honest-tradeoff clause: per-template `## Rules` block requires one short trade-off or limitation per script.
- Phase 2.1 state-side: chosen pillar persists to `pipeline_state.json` for every produced video.

**Pycaps subtitle engine maturity (0.44.x)**
- AI word tagging via Gemini in pycaps. Reuses existing Gemini key. Built-in `neo-minimal` and `explosive` templates ship `type: ai` rules. Per-call errors governed by `pycaps.ai_tagging_on_error` (default `skip`).
- Default subtitle engine flipped from FFmpeg to pycaps; bundled `pycaps.fallback_policy` is `fallback_ffmpeg` so forks without the optional pycaps group degrade silently.
- Default pycaps template pool tightened to `["explosive", "word-focus"]` (50/50 AI-tagged / untagged); default template `explosive`.
- `--pycaps-template NAME` now actually forces the named template (was silently no-op against multi-entry pools).
- `make produce-lowpri` cgroup hardening with `MemorySwapMax=0` so producer memory pressure doesn't trigger systemd-oomd kills on unrelated session apps.

**TTS voice pinning (0.45.x)**
- Phase 2.3 default voice profile. `tts_config.default_voice_profile` pins one voice for unattended runs without a CLI flag. Voice selection precedence: CLI override > non-empty pool (random for A/B) > pinned default > random across all profiles. Bundled `default_voice_profile: charon`.

**Phase 0 disclosure compliance baseline (0.46.x)**
- Phase 0.1 persistent on-frame disclosure overlay (`#ad` by default, configurable text). Burned in a fixed corner of every produced video, full-clip duration, sized smaller than narration captions. Configurable per render so language-aware variants can ship without code changes.
- Phase 0.2 first-line caption disclosure on every platform. Disclosure leads each caption on its own line ahead of the description and hashtag block. `#ad` deduped from hashtags so it never appears twice.
- Phase 0.5 platform-tag audit completed: YouTube `containsSyntheticMedia: true` set on every publish payload alongside the TikTok branded-content flags already wired.
- Phase 0.6 cross-cutting disclosure regression suite at `tests/test_disclosure_stack.py` covers all four disclosure surfaces with consistency invariants.
- Phase 0.7 `docs/compliance.md` describes the disclosure stack, regulator coverage, and the per-video manual workarounds for SDK gaps.
- Phase 2.1 registry side: `pillar` column on the published-products registry; `--rebuild` retroactively tags rows from the producer state file. Backward-compatible loader.

**Script template hook and closing rules (0.48.x)**
- Phase 1.1 long-tail audio keyword required in line 1 of every script template. Six proven hook patterns listed in each template's Rules block; literal Google-query shape called out as anti-pattern.
- Phase 1.5 engagement-bait closing line required in every script template. Personal/storytelling templates use a two-option comment-fork; analytical/comparison templates use a debatable spec claim that invites a correction.

**Caption-side mirror of the engagement-bait closing line (0.49.x)**
- Per-platform caption generators receive the rendered spoken script and mirror its closing engagement-bait line into each caption body. Same line in spoken audio + on-screen subtitle + caption text (Rule of 3s). Empty or missing script produces a normal caption with no closing line.

**Hook retention surgery and short profile (Unreleased)**
- Phase 1.2 punchline-first opener. Anti-setup clause across all 15 script templates so line 1 states a concrete fact instead of a setup framing. Pre-motion (Ken Burns settle-zoom) on the first image segment (`first_frame_pre_motion` + `pre_motion_peak_zoom`). Burned-in hook overlay renders the first sentence of the spoken script as centre-upper static text on the first 1.5 s, drawn after subtitles and before the disclosure rewrite. Cold-open variant rotation framework: three named variants selected per product via salted MD5 and persisted to `pipeline_state.json` for analytics. Hook-line lead in the subtitle timing smoother (first 3 words led by an extra 200 ms on top of the base lead).
- Phase 1.3 short profile and per-platform routing. New `slideshow_short_20s` (15-30 s canvas at ~50-60 word script budget). `profiles: <platform>: <profile>` mapping in `config/publisher.yaml` routes each platform to a named profile; the publisher prefers `video_<asin>_<profile>.mp4` and falls back to the first matching render when unset.
- Phase 1.5 closing-line rule pivot. The 8 analytical templates now branch the spec-correction close on whether the description carries a contestable performance number; passive products close with a material-or-use claim instead. Fixes a fabrication case where the LLM invented numeric specs on products that didn't have them.

**Pillar infrastructure and caption voice (Unreleased)**
- Keyword-to-pillar attachment in the scraper config. Keywords in `config/scraper.yaml` are a dict keyed by pillar; each scraped product carries the pillar through to the producer and registry without `--pillar`.
- Narrator profile and pillar preamble shared with platform caption generators (YouTube, TikTok, Instagram) so captions match the video's conversational voice.

**Best-practices docs (Unreleased)**
- Safe-zone docs aligned to 2026 platform specs. `docs/platform-safe-zones.md` is the canonical source; subtitle and promo docs cite it. Refreshed for Meta's March 2026 Reels unification (14% top, 35% bottom) and TikTok's Jan 2026 playlist button. Runtime constant refresh tracked separately (roadmap 1.6).
- New `docs/audio-best-practices.md`: the sound-on layer (trending vs original audio, voiceover/music mix levels, ducking, audio hook, platform loudness).
- New cut-cadence section in `docs/promotional-video-best-practices.md` (shot-length bands, transition vocabulary) backing the high-density cut profile (1.4).
