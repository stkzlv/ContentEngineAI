# Roadmap

Last updated: 2026-04-24

Forward-looking work on ContentEngineAI, grouped by horizon. Items are aspirational, not commitments. Order within each horizon is rough priority.

Issues and PRs are welcome on any item. If you want to pick something up, open an issue first so we can talk through scope.

## Now

Targeted for the next 4-6 weeks.

### Pillar tagging across the pipeline

Add a `pillar` field to the keyword config so users can group products by content theme (e.g., gaming, home office, travel gear). Tag each script template in `src/ai/prompts/scripts/` with one or more pillar labels, and route the deterministic MD5 selection so a video produced under pillar X picks from templates tagged with X. Persist the chosen pillar in `pipeline_state.json` and the published-products registry. Add `--pillar <name>` to both `src/video/producer/cli.py` and `src/pipeline/global_batch.py` (Module/Batch Alignment Rule).

**Done when:** every produced video declares a pillar in its state file and registry row, and a single batch can run scoped to one pillar.

### Default voice profile (voice pinning)

Add a `default_voice_profile` field to `tts_config` in `config/subtitles.yaml`. When set, unattended runs use that voice unless `--voice-profile` overrides. The random-voice path stays available for testing. Helps creators who want a single recognizable voice across their channel.

**Done when:** an unattended batch picks the configured default voice every time without a CLI flag.

### UTM tagging at publish

Append `utm_source`, `utm_medium`, `utm_campaign` query parameters to affiliate URLs at publish time, with values keyed off the target platform and the video's pillar. Configurable on/off per platform in `config/publisher.yaml`. Skip when the destination is the link-in-bio service itself.

**Done when:** every cross-platform post ships with platform-tagged links, and click attribution becomes possible at the analytics layer.

### Honest-tradeoff clause in script prompts

Update the script templates in `src/ai/prompts/scripts/` to require one realistic downside or limitation per product review. Add an optional validation step in `src/ai/llm_client.py` that warns when a generated script lacks a tradeoff marker. Builds trust with viewers and reduces the "AI shilling" feel.

**Done when:** sampling 10 generated scripts shows at least 8 include a tradeoff marker.

### Instagram Reels delivery audit

Verify the Instagram path in the publisher posts as a Reel, not as a Feed Post. Zernio accepts separate `instagramSettings` for each format. Add a regression test that asserts the payload carries the Reels flag.

**Done when:** the next batch publishes to IG as Reels, confirmed end-to-end, with a test in place to prevent drift.

## Next

Targeted for this quarter. Most depend on Now items landing.

### Long-form profile (60-120s)

Add a long-form profile (`slideshow_long_60s` or `video_long_90s`) to `config/video_production.yaml` with longer slide durations and a longer script template variant. Useful for YouTube long-form Shorts, TikTok Creator Rewards eligibility (which requires 60s+ videos), and platforms that reward depth over brevity. Word-count budget assumes Google Chirp 3 HD pacing of 150-180 wpm.

### A/B caption variants

Add an `ab_variant` field to the publish step that selects A or B per product via deterministic salted MD5. Two variant pools per platform (`cta_variants_a`, `cta_variants_b`) in `config/publisher.yaml`. Persist the chosen variant in the registry so downstream analytics can compute conversion per variant.

### Hashtag pools per pillar

Add `hashtag_pools` to `config/publisher.yaml`, keyed by pillar and platform. Each post draws a baseline pillar-relevant set plus 2-3 product-specific tags from the LLM-generated metadata. Removes the need to curate hashtags per video.

### Non-affiliate pillar support

When a pillar is marked non-affiliate (e.g., tutorial or how-to content), the publisher skips appending the affiliate URL and skips the link-in-bio registration. Lets creators run a tutorial track alongside an affiliate track from the same pipeline.

### Companion analytics module

Optional analytics tooling that pulls platform metrics (TikTok, Instagram, YouTube, Amazon Associates, lnk.bio) into a local SQLite store and produces weekly reports segmented by pillar, template, voice profile, and A/B variant. Either a sister module in this repo or a separately released companion tool, to be decided when the work starts.

## Later

Blocked on platform features, eligibility, or earlier items landing.

### TikTok Shop product tagging in publisher

Add support for TikTok Shop product tags in the publisher payload. Requires the user's TikTok Shop affiliate approval. Once enabled, posts in supported regions can carry tappable product cards, replacing or complementing the bio-link CTA.

### Instagram native affiliate product tagging

Wire up Instagram's native affiliate product tagging through the publisher when the user's account meets the threshold. Solves the bio-click drop-off because the tag is in-feed.

### YouTube end-screen subscribe overlay

Add an end-screen overlay step in `src/video/producer/` that bakes a subscribe CTA on the last few seconds. Pinned-comment subscribe asks can ride on the existing publisher path. Most useful in combination with the long-form profile.

### Zernio SDK migration

Migrate from the legacy `late-sdk` package to `zernio-sdk`, rename `LATE_API_KEY` to `ZERNIO_API_KEY`, and update imports under `src/publisher/late/`. The old package keeps working during the SDK grace period; do this when the publisher gets its next substantive change so the migration rides on existing testing.

### Pycaps follow-ups

Open items tracked in `docs/pycaps-followups.md`: AI word tagging via the Gemini key, mypy pin cleanup, two-part subtitles plus pycaps hybrid, CSS renderer integration test in CI.

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
- All Now items shipped: pillar tagging, default voice profile, UTM tagging, honest-tradeoff clause, Instagram Reels delivery audit.
- At least three of the five Next items shipped or in review.
- TikTok Shop and Instagram native affiliate tagging (Later) are explicitly out of scope; they're 1.x material once the platforms unlock for real users.

**Real-world proof**
- The pipeline has produced and successfully published a meaningful volume of videos (target: 100+) end-to-end across all three target platforms.
- Subtitled videos render correctly on TikTok, YouTube Shorts, and Instagram Reels under manual QA on each platform's safe-zone overlays.
- Affiliate links land in the link-in-bio destination automatically with no manual cleanup.

When all gates are green, the next release is `1.0.0`, not `0.43.x`. Anything not on this list is a 1.x or 2.x conversation.

## Update rules

- Each shipped item moves to the Shipped section at the bottom (append-only) with a one-line summary and the release version.
- Items dropped from the roadmap also get a one-line note in the same section, with the reason.
- Don't let the Later section accumulate without bound. If something sits in Later for two quarters with no movement, either prune it or rewrite the description to reflect the actual blocker.

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
