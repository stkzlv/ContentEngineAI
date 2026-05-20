# Roadmap

Last updated: 2026-05-07

Forward-looking work on ContentEngineAI, grouped into phases by horizon. Items are aspirational, not commitments. Order within each phase is rough priority.

The phases are sequenced so each one builds on the previous. Phase 0 is the disclosure-compliance baseline that every promotional render needs to satisfy and has to land before anything else ships. Phase 1 fixes the retention foundation (hook quality and 3-second hold). Phase 2 organises content by theme. Phase 3 makes attribution and CTAs measurable. Phase 4 layers per-platform optimisations. Phase 5 closes the analytics loop. Phase 6 captures unlocks blocked on platform thresholds or features.

Issues and PRs are welcome on any item. If you want to pick something up, open an issue first so we can talk through scope.

## Phase 0 — Disclosure compliance baseline (Now, blocking)

Targeted to ship before any Phase 1 retention work, and gating the 1.0.0 release. Creators using the pipeline for affiliate marketing have legal disclosure obligations under the FTC Endorsement Guides (any US-facing content), the Amazon Associates Operating Agreement (any Amazon affiliate program participant), and per-platform policy (TikTok / Instagram / YouTube). The pipeline already wires TikTok branded-content disclosure but leaves the rest of the stack manual. Compliance is not a polish item: FTC penalties reach $53,088 per violation per post in 2025 and Amazon Associates enforces with account termination, no warning. The pipeline should make compliance the default render output, not a per-video checklist.

This phase ships as a single compliance bundle, not staged. Shipping pieces of it leaves the floor incomplete.

### 0.1 Persistent on-frame disclosure overlay

Burn a configurable disclosure overlay (`#ad`, `Sponsored`, `Paid partnership`, or localized equivalents like `#publi` / `#publicidad`) into every produced video. Full-clip duration, fixed corner, ~50-60% size of narration captions, contrasting font weight against the background. Configurable via a new `disclosure` block in `config/publisher.yaml` (or `config/video_production.yaml`, depending on which side of the assembler/publisher boundary the overlay lives). Required by FTC's two-punch guidance (overlay AND caption text); platform tags alone don't satisfy it.

**Done when:** every render emits a persistent on-frame disclosure overlay in the language of the script, visible across TikTok / YouTube Shorts / Instagram Reels safe zones (`docs/platform-safe-zones.md`).

### 0.2 First-line caption disclosure

Update the per-platform metadata generators in the publisher so the disclosure leads the caption text, before any product description, hook, hashtags, or affiliate link. Platform-specific quirks: Instagram and TikTok captions clip at the `…more` cut, so the disclosure must fit ahead of that fold; YouTube descriptions tolerate longer lead-ins. The disclosure language must match the script language, not the platform default.

**Done when:** every published caption leads with the disclosure on every platform, language-matched to the script.

### 0.3 Affiliate program literal-phrase rendering

Add a configurable literal-phrase block that ships the affiliate program's required identification text. For Amazon Associates: "As an Amazon Associate I earn from qualifying purchases" (or a substantially similar pre-approved statement). Render in at least one of: profile bio, the closing frame of the video, or the caption body. Drive from a new `affiliate_disclosure` config keyed by program name so non-Amazon programs (ShareASale, Impact, eBay Partner Network) can plug in their own phrases.

**Done when:** every published video includes the configured affiliate program literal phrase in at least one of bio / on-frame / caption.

### 0.4 Localized disclosure variants

When the script language is not English, the disclosure must match. Add per-language disclosure variants keyed off the existing TTS / script language: `en` → `#ad`, `es` → `#publi` or `#publicidad`, etc. Required by FTC's same-language rule and by Spain's Royal Decree 444/2024 for creators in the EU. A config-load-time validator should warn (not silently default) when the configured disclosure language doesn't match the script language for a given run.

**Done when:** a Spanish-language render emits a Spanish-language disclosure overlay and caption first line; an English render emits English; mismatches raise a config-load warning.

### 0.5 Platform-tag audit (TikTok / YouTube / Instagram)

Verify the platform-policy disclosure flags are set on every publish:

- TikTok: `commercial_content_type: "brand_organic"` and `is_brand_organic_post: true` (already wired in `src/publisher/late/client.py`).
- YouTube: `paid_promotion: true` (or the Zernio / late-sdk equivalent field) in the publish payload.
- Instagram: paid partnership label on Reels and Posts.

These satisfy each platform's policy floor and are additive to the FTC overlay/caption disclosure, not a substitute. Add a regression test that asserts each flag is present in the per-platform payload.

**Done when:** every per-platform publish payload carries the platform's paid-content flag, with a regression test in CI.

### 0.6 Disclosure-test suite

One smoke test per requirement above, living next to the existing publisher tests and running in CI on every PR. Asserts: overlay is rendered, caption first line is the disclosure, affiliate phrase is present in at least one rendered location, language matches script, platform flags are set in the payload. Fails the build on any drift.

**Done when:** the suite catches every documented disclosure regression.

### 0.7 Documentation

Add a `docs/compliance.md` that describes the disclosure stack the pipeline produces, what each layer satisfies (FTC, Amazon Operating Agreement, platform policy), what configuration knobs exist, and what creators are still expected to verify manually (e.g., bio identification on each social profile). Cross-link from `docs/publisher.md` and from the README's feature list.

**Done when:** `docs/compliance.md` exists and the publisher README points to it.

## Phase 1 — Hook and retention surgery (Now)

Targeted for the next 4-6 weeks. Highest-leverage block; downstream phases depend on retention being fixed first.

Industry benchmark for short-form: 70-80% stayed-to-watch on YouTube Shorts, 60%+ 3-second hold on Reels, sub-1.5s distribution decisions on TikTok. Faceless slideshow content with a generic intro typically lands well below those benchmarks. The items below address that gap at the script-prompt and assembler levels.

### 1.1 Front-loaded long-tail keyword in script prompts

Edit the script templates in `src/ai/prompts/scripts/` to require a long-tail audio hook in line 1, with structure `[audience or context] + [problem or benefit] + [price band]`. Mirror the hook in the on-screen text overlay and the platform caption (Rule of 3s — same keyword in caption, on-screen text, and spoken audio in the first 3 seconds). TikTok ranks spoken audio as a primary signal alongside captions, so the first 5 seconds of TTS determine search inclusion.

**Done when:** spot-check 10 produced videos shows the long-tail keyword appears in TTS within the first 5 seconds and again in the on-screen text and caption.

### 1.2 Punchline-first opener with visual interrupt

Modify the script templates so line 1 is the payoff, not the setup (Phase 1.2a, shipped: anti-setup clause added across all 15 templates). At the assembler level, ship two distinct hook shapes that share one drawtext implementation:

- **Static title card** (default on `slideshow_short_20s`): 1.0-1.5 s, hard cut to motion, 3-5 words capped at 7, ALL CAPS, 10-15% frame height.
- **Text-over-mid-action-frame** (default on the existing 30-45s profiles): 1.5-3.0 s, can fade in, sits over a frame that already carries motion.

If the first slide is a static product photo, inject **0.3-0.5 s of Ken Burns settle-zoom** so frame 1 is mid-motion. The pipeline's `first_frame_pre_motion: true` + `pre_motion_peak_zoom: 1.10` defaults sit at the upper edge of that band. Ship with at least 2-3 cold-open variants per pillar (rotated deterministically per product via `cold_open_variant_pool`) so the channel doesn't read as a template factory at the aggregate level. Burned-in big text on the first frame is mandatory because 85-92% of mobile views are sound-off.

**Done when:** the assembler emits a non-fade first frame with burned-in hook text, the static title card and text-over-frame shapes are both selectable per profile, the cold-open variant rotation is tracked in `pipeline_state.json`, and a sample of 30 produced videos averages above 50% retention at the 3-second mark on YouTube Shorts. Closes Issue #102.

### 1.3 Short profile (15-30s) for hook iteration

Add a `slideshow_short_20s` profile to `config/video_production.yaml` with shorter slide durations and a script word-budget around 50-60 words (tuned for ~150-180 wpm TTS pacing). 15-30s is the hook-iteration zone for getting fast retention deltas; 60-90s is needed for TikTok Creator Rewards eligibility, which Phase 6.1 (long-form profile) covers when that threshold comes into reach.

Make the profile selectable per-platform in the publisher via a `profiles: <platform>: <profile_name>` mapping in `config/publisher.yaml`, with a single `profile` field as the back-compat fallback. The publisher prefers `video_<asin>_<profile>.mp4` per platform and falls back to the first `video_<asin>_*.mp4` when no matching render exists.

**Done when:** the short profile renders a clean 15-30s output and the publisher accepts a per-platform profile-routing field.

### 1.4 High-density cut profile

Add a `cut_density: high` profile setting in `config/video_production.yaml` that drops the minimum slide duration to 1.5-3s and adds a transition (whip pan, hard cut, zoom punch) between every slide. Useful for younger audiences on platforms whose feeds reward visual energy density. Keep the existing slow-cut profile available for use cases where it fits better.

**Done when:** a high-density profile renders without subtitle desync and is selectable per platform.

### 1.5 Closing comment-fork or spec-correction line

Add a closing-line block to every script template that injects a comment-fork (two-option opinion question) or spec-correction line (a deliberately debatable claim that invites correction). The closing line stays in both the spoken script and the platform caption. Doesn't replace the affiliate CTA — adds an extra closing beat right before the CTA. Generic engagement bait ("Comment YES if...") is spam-filtered; specific opinion forks and spec-correction bait still drive comments.

**Done when:** sample of 10 produced scripts shows 9+ include a comment-fork or spec-correction in the closing 5 seconds.

## Phase 2 — Pillar persistence, non-affiliate mode, voice pinning (Now/Next)

Targeted for weeks 3-4. The pillar system itself shipped in 0.43.0 (default pillars: `value`, `novelty`, `utility`; keyword pool grouped by pillar in `config/scraper.yaml`; templates mapped to pillars in `config/ai_services.yaml::script_templates.pillars`; `--pillar` flag on both `src/video/producer/cli.py` and `src/pipeline/global_batch.py`; per-pillar preambles and audiences). What's still missing: pillar persistence into runtime state, an opt-out for affiliate URL injection so the same pipeline can carry an educational track, and voice pinning.

### 2.1 Persist active pillar in state and registry

Pillar is currently a runtime selector that filters the template pool and prepends a preamble, but the chosen pillar is not written into `pipeline_state.json` or the published-products registry. Without persistence, downstream analytics can't segment by pillar. Add `pillar` to the state file in `src/video/producer/state.py` and to the registry record in `src/publisher/product_registry.py`. Backfill the registry rebuild path so existing rows can be re-tagged from `pipeline_state.json` where available.

**Done when:** every produced video records its pillar in `pipeline_state.json` and the registry row, and `registry --rebuild` preserves pillars on existing rows.

### 2.2 Non-affiliate pillar mode (educational / how-to track)

Add a `non_affiliate: true` flag at the pillar level. When set, the publisher skips appending the affiliate URL and skips the link-in-bio registration. Lets creators run an educational or how-to track alongside an affiliate track from the same pipeline. Educational content earns trust and search SEO; the audio script can still mention specific products by name (audio-keyword crossover indexes the video for both the help query and the product query) without an explicit affiliate push.

**Done when:** a video produced under a `non_affiliate: true` pillar publishes with platform-appropriate captions, no affiliate URL, and no bio-link registration, while still naming products in the spoken script.

### 2.3 Default voice profile (voice pinning)

Add a `default_voice_profile` field to `tts_config` in `config/subtitles.yaml`. When set, unattended runs use that voice unless `--voice-profile` overrides. The random-voice path stays available for testing. Complements the channel-wide `narrator_profile` (text direction, shipped in 0.43.0) by pinning the synthesized voice itself.

**Done when:** an unattended batch picks the configured default voice every time without a CLI flag.

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

**Pycaps subtitle engine maturity (0.44.x)**
- AI word tagging via Gemini in pycaps. Reuses existing Gemini key. Built-in `neo-minimal` and `explosive` templates ship `type: ai` rules. Per-call errors governed by `pycaps.ai_tagging_on_error` (default `skip`).
- Default subtitle engine flipped from FFmpeg to pycaps; bundled `pycaps.fallback_policy` is `fallback_ffmpeg` so forks without the optional pycaps group degrade silently.
- Default pycaps template pool tightened to `["explosive", "word-focus"]` (50/50 AI-tagged / untagged); default template `explosive`.
- `--pycaps-template NAME` now actually forces the named template (was silently no-op against multi-entry pools).
- `make produce-lowpri` cgroup hardening with `MemorySwapMax=0` so producer memory pressure doesn't trigger systemd-oomd kills on unrelated session apps.
