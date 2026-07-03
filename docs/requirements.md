# Project Requirements

High-level requirements for ContentEngineAI.

---

## Global Requirements

### Configuration System
- **Three-tier precedence**: CLI arguments > Profile overrides > YAML defaults
- CLI arguments override only when **explicitly provided** by the user (use `default=None` in argparse, not hardcoded defaults)
- Environment variables (`.env`) store secrets only, never committed
- YAML files (`config/`) store application settings, safe to commit
- Nested dotted CLI overrides for sub-models (e.g., `subtitle_settings.pycaps.template_name`)
- Validate all configuration at startup with clear error messages
- Both producer CLI and global batch pipeline must expose identical override flags (Module/Batch Alignment Rule)

### Security
- Store API keys and secrets in `.env` file only
- Provide `.env.example` template for required variables
- Never commit `.env` or put secrets in YAML files

### Error Handling & Resilience
- Continue processing on individual item failures (graceful degradation)
- Retry transient network failures with exponential backoff (timeouts, rate limits)
- Circuit breaker pattern to prevent cascading failures from unavailable services
- Provide clear error messages for missing configuration
- Support fail-fast mode via CLI flag when strict behavior needed

### Logging & Monitoring
- Global debug mode across all components
- Progress tracking with `[N/total]` format for batch operations
- **Unified module summaries**: each module (scraper, producer, publisher, audio) logs a summary at the end of its work with consistent format, key counts, product IDs, and duration. No emojis in logs.

### Documentation Standards
- Required root files: README.md, CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md, CHANGELOG.md, LICENSE
- Use GitHub-Flavored Markdown (GFM)
- Update documentation in same PR as code changes
- Use relative paths for internal links
- Provide working code examples with context and expected output

### Documentation Structure
- Extended documentation organized in `docs/` directory
- README.md contains: project title, description, key features, quick start, links to detailed docs
- CHANGELOG.md follows Keep a Changelog format (Added, Changed, Deprecated, Removed, Fixed, Security)

### Outputs Directory
- Centralized `outputs/` directory for all pipeline artifacts
- Per-product directories: `outputs/<product_id>/` with `data.json`, `images/`, `videos/`
- Global directories: `cache/`, `logs/`, `reports/` for shared resources
- Customizable via `--outputs-dir` CLI flag
- Automatic cleanup of published products (configurable)

### Async Concurrency Control
- Semaphore-based limits for resource-intensive operations (ffmpeg, I/O, network)
- Configurable concurrency limits per operation type
- Prevent system overload during batch processing

### Resource Cleanup
- Use context managers for connections, file handles, temporary resources
- Automatic cleanup of partial files on failure
- Proper connection pool management for HTTP sessions

### Validation Framework
- Pydantic models for configuration and data structures
- Custom validators for domain-specific rules (product IDs, URLs, file formats)
- Early validation with clear error messages

### Performance Monitoring

#### Step-Level Measurement
- Track each pipeline step: wall-clock duration, memory (start/peak/end), CPU percent, disk I/O
- Continuous peak memory sampling during step execution (configurable interval, default 100ms)
- Capture errors per step for correlation with resource spikes
- Timing decorators for any async or sync function

#### Pipeline Run Tracking
- Each pipeline run recorded with: product ID, profile name, success/failure, total duration, aggregated resource usage
- Automatic history persistence after pipeline completion
- Retention limit with periodic cleanup (default: 100 runs, cleanup every 10 saves)
- Corrupt history entries skipped gracefully on load

#### Threshold Warnings
- Configurable per-step timing threshold (default 5s) and memory ceiling (default 1000MB)
- Warnings logged after pipeline completion for any step exceeding thresholds

#### Reporting
- **Summary**: success rate, duration stats with percentiles (p50/p95/p99), memory/CPU averages, product/profile distribution, step-level breakdown
- **Trends**: daily aggregates over configurable window (default 30 days), filterable by product, includes per-step daily averages
- **Detailed**: individual run records with per-step breakdown, exportable as CSV
- **Comparison**: side-by-side profile performance (run count, success rate, duration percentiles, memory)
- **Regression detection**: compare last N runs vs previous N, flag steps exceeding a slowdown factor (default 2x)

#### Configuration
- **History retention**: max runs to keep (default 100)
- **Cleanup interval**: how often to check retention limit (default every 10 saves)
- **Memory sampling interval**: peak memory polling frequency (default 0.1s)
- **Timing threshold**: warn if a step exceeds this duration (default 5s)
- **Memory threshold**: warn if peak memory exceeds this value (default 1000MB)
- **Report defaults**: summary limit (50), detailed limit (20), trends window (30 days), recent window (10 runs)

---

## Scraper Module

### Multi-Platform Architecture
- Modular design with platform-specific implementations (Amazon first)
- Separate core scraping logic from platform adapters
- Support both direct product ID lookups and keyword searches

### Product Discovery & Media Extraction
- Extract product data: title, price, description, ID, ratings, reviews
- Scraped price is recorded as a plain decimal number, parsed from both US (comma grouping, dot decimal) and European (dot grouping, comma decimal) price formats
- Product rating falls back to the search-results rating when the product page yields none, so a rating is recorded whenever the listing shows one
- Download high-resolution images and videos
- Filter out low-quality or invalid media
- Store media in dedicated directories per product ID

### Search & Filtering
- Keyword-based searches with filters (price, rating, shipping, brands)
- Sorting options and regional redirect handling
- Product ID validation against platform formats
- Skip products lacking essential data

### Stealth & Human Simulation
- Implement detection evasion techniques
- Simulate human-like browsing patterns
- Browser window size varies for detection evasion but stays desktop-width, so the site serves the desktop layout the product-card extraction depends on; a narrow window draws a mobile layout with no extractable cards
- Handle failures gracefully without halting batch

### Batch Mode
- Process lists of product IDs and/or keywords
- Sequential processing with rate limiting
- Deduplication across multiple sources
- Summary reporting of success/failure counts

### Product Limits
- **`max_products`**: Total products to collect across all keywords (global cap)
- **`products_per_keyword`**: Maximum products to scrape per individual keyword
- Both limits must be consistent across all configurations (pipeline, scraper, CLI)
- Processing continues through keyword list until `max_products` is reached
- Early termination when global limit is hit, even if keywords remain

### Affiliate URL handling
- Every scraped product's affiliate URL is canonicalised to `https://www.amazon.com/dp/<ASIN>?tag=<associate_tag>` before it lands in `data.json`
- The associate tag is read from the `AMAZON_ASSOCIATE_TAG` environment variable, with the YAML `scrapers.amazon.associate_tag` field as a fallback
- The standalone scraper CLI loads `.env` at startup; a tag set only in `.env` (not exported in the shell) is visible to the canonicaliser
- When no associate tag resolves, the canonicaliser returns the input URL unchanged and emits a WARNING-level log line indicating affiliate attribution will be lost

### URL shortener
- The shortener layer is provider-pluggable through a typed registry
- Bundled providers: `bare` (no-op, returns input unchanged) and `picsee` (opt-in, requires API key)
- The bundled default is the bare provider; fresh installs do not require any shortener API key
- The bare provider requires no API key and makes no network calls; the canonical affiliate URL passes through untouched
- A new provider is registered by implementing the shared shortener interface and adding it to the enum and registry

---

## Video Producer Module

### Video Assembly
- Adjust video duration to match voiceover length
- Display images with configurable duration (2-3 seconds)
- Smooth transitions (crossfade) between media elements
- Reuse images if needed to fill remaining time

### Image Positioning
- Width as percentage of frame (default 100%)
- **Horizontal alignment**: always centered in frame
- **Vertical alignment**: center (default) or top with configurable offset
- Preserve aspect ratio by default

### Video Positioning
- **Vertical alignment**: center (default) or top with offset (default 10%)
- **Content height**: configurable portion of frame (default 75%)
- **Aspect modes**: letterbox, crop-to-fit, smart-scale (auto-select when aspect ratios within 10%)
- **Assembly modes**: sequential, single-best, mixed-media, video-first-fallback
- **Audio handling**: remove original audio or mix at configurable volume
- Match final duration to voiceover (±1 second tolerance)

### Subtitles

**Two rendering engines** selectable per-profile or per-run. Bundled
`config/subtitles.yaml` selects the pycaps engine by default with
`fallback_policy: fallback_ffmpeg`, so forks without the optional pycaps
group degrade to FFmpeg without manual intervention.
- **Pycaps engine** (bundled default): CSS-styled animated captions burned post-assembly via the pycaps library. Word-by-word karaoke, per-word CSS animations, template-driven styling. Optional AI word tagging via Gemini. Single-line only (two-part not supported). Requires optional Poetry group install.
- **FFmpeg engine**: SRT (drawtext) or ASS (libass) burned during assembly. Supports two-part mode, karaoke, and all positioning anchors. The fallback path when pycaps is unavailable; also selectable explicitly via `--subtitle-engine ffmpeg`.

**Positioning:**
- **Anchors**: top, center, bottom, above-content, below-content
- **Margin** from anchor edge (default 4%)
- **Horizontal alignment**: left, center (default), right
- Content-aware positioning relative to actual media bounds
- **Platform safe zone**: boundaries avoid TikTok, YouTube Shorts, and Instagram Reels UI overlays. See `platform-safe-zones.md` for the canonical cross-platform union measurements. Configurable globally in YAML and per-profile via the nested `subtitle_settings.safe_zone` block (only the boundaries that differ need to be set).
- Both engines enforce the safe zone. FFmpeg clamps subtitle position to the zone, including the vertical floor (it accounts for line height so a centered caption's lowest pixel stays above the bottom boundary), and honors per-profile safe-zone overrides. Pycaps dynamically clamps `max_width_ratio` so centered text never extends past the right-side boundary (TikTok buttons). The clamping is automatic — no manual tuning needed per template.
- Pycaps default position: a deliberate lower-third (~75% of frame), which sits below the 2026 safe-zone bottom (65%) to avoid colliding with centered product imagery. Template's own alignment preserved unless explicitly overridden.

**Text formatting (best-practice aligned):**
- **Font**: bold sans-serif weight 700+ (Montserrat Black default)
- **Font size**: 7.5% of frame height (~144px on 1920)
- **Color**: white fill, opaque black outline (3-4px), no background box
- **Max 3 words per line**, max 2 lines, max 80% frame width
- **Segment duration**: 2.5s max, 0.6s min (~170 WPM reading speed)
- Terminal punctuation stripped in karaoke mode
- **Audio-lead timing**: each narration word appears slightly before its audio onset (read-ahead lead). The opening hook gets an extra lead applied to the first few words on top of the base, so sound-off viewers parse the hook before any audio cue. Lead duration and number of leading words configurable per render.

**Two-part mode** (FFmpeg engine only):
- Upper line: static URL/link, positioned above content
- Lower line: voiceover-synced transcription, positioned below content
- Dynamic repositioning per visual segment using assembler geometry
- CTA detection for styling emphasis on call-to-action phrases

**Style presets**: minimal, modern (default), bold, animated, random
- **Research-backed effects**: karaoke (word-by-word highlight), fade, typewriter
- One effect per video, deterministic by product ID
- Font and color randomization from curated pools

**Pycaps engine specifics:**
- 10+ built-in templates: word-focus, hype, minimalist, vibrant, explosive, etc.
- Deterministic template selection per product from a configurable pool
- CSS renderer (Playwright + Chromium, default) or Pictex renderer (browserless Skia)
- Fallback policy (3 options): a burn failure aborts under both `raise` and `fallback_ffmpeg`; only `warn_and_skip` keeps a caption-less video. `fallback_ffmpeg` degrades to the FFmpeg engine when pycaps is unavailable (caught before assembly)
- Render speed: ~0.7x realtime on CSS path, ~420 MB peak RSS

### Cold-Open Style

- **Pre-motion on first image segment**: when enabled, the first image starts at a slight zoom and settles to 1.0 over the segment duration, so frame 0 is mid-motion instead of a static still. Pairs with the burned-in hook overlay to defeat the static-still-then-fade pattern that burns the 1.5-second decision window. Off by default on existing profiles, on by default on the short profile. Peak zoom factor configurable globally and per profile.
- **Burned-in hook overlay**: the first sentence of the spoken script is rendered as centre-upper static text on the first 1.5 s (configurable). Sized 1.2-1.5x narration captions, capped at 7 words, no per-word reveal. Drawn after the subtitle pass and before the FTC `#ad` corner disclosure so the disclosure stays on top of the z-order. Source text is the rendered spoken script; an empty or missing script makes the overlay a silent no-op.
- **Cold-open variant rotation**: each render picks one named variant from a configurable pool (defaults: title-card with pre-motion, static title-card, pre-motion only). Selection is deterministic per product so re-renders match. The chosen variant name is persisted in the pipeline state for downstream analytics; v1 ships the framework only, with all variants rendering identically until variant-specific visual differentiation lands.

### Profile System
- **Precedence**: CLI > Profile > Global defaults
- All visual, subtitle, and video settings configurable per profile
- Subtitle overrides use a single nested `subtitle_settings` block on each profile; only fields that differ from the global value need to be set, and nested sub-blocks (`pycaps`, `two_part_subtitles`, `safe_zone`) deep-merge per-field rather than being replaced wholesale
- Cold-open knobs (pre-motion toggle, peak zoom) are per-profile overrides as well; the short profile enables pre-motion by default while the existing 30-45s profiles inherit the off default
- Strict validation: unknown keys in subtitle YAML or profile overrides fail at config load with a typed error, instead of being silently dropped at render time
- Legacy flat per-profile keys (`subtitle_anchor`, `pycaps_template`, `two_part_subtitles`, ...) still load with a deprecation warning during one-release migration window
- Typed Pydantic models for merged settings
- Deterministic random profile selection per product
- **Short profile (15-30 s)** sized for hook-iteration renders. Script word budget around 50-60 words at natural TTS pacing. 60-90 s long-form for platform-revenue-program eligibility lives in a separate planned profile, not in the short profile.

### Media Validation
- Scraper validates media against producer profile requirements
- If profile ignores videos, scraper counts images only
- Insufficient media = skipped (not failed)

### AI Service Integration
- Provider fallback chain: primary provider (Gemini) with automatic fallback to secondary (OpenRouter)
- OpenRouter free model discovery with blocklist filtering and context length minimum
- Configurable retry, validation thresholds, and model blocklist via `LLMSettings`
- Google Cloud TTS and Gemini TTS with voice prioritization

### Script Templates
- Multiple prompt templates with different styles (curiosity hook, problem-solution, storytelling, comparison, etc.)
- Templates target calm, conversational delivery. No high-energy, hype, or clickbait phrasing.
- Deterministic template selection per product for batch consistency
- Configurable template pool to restrict which styles get used
- CLI override to force a specific template
- Template metadata recorded in pipeline output for traceability
- Every template instructs the LLM to open with a natural conversational hook that carries the long-tail audio keyword (product category, price band, audience cue, pain point) embedded inside speech a person would actually say out loud. The template rule lists six proven hook patterns aligned with `docs/promotional-video-best-practices.md` §1 — price-first reveal, regret/contrarian, POV, outcome-first, numbered teardown, comparison — and names the literal Google-query shape as an anti-pattern. The keyword must land in the first 5 seconds of TTS because TikTok transcribes audio via ASR and indexes the transcript as a primary search signal.
- Line 1 must state a concrete fact, result, or observation about the product. Setup framings ("Today I'll show you", "Let me tell you", "In this video", "I want to share") are anti-patterns — they burn the 1.5-second decision window before the payoff lands.
- Every template instructs the LLM to end the spoken script with one short engagement-bait closing line, right before the CTA, not replacing it. Personal and storytelling templates use a two-option opinion question (comment-fork); analytical and comparison templates use a debatable but defensible spec claim that invites a correction. The closing line drives comments, which feeds the platform algorithm; generic "Comment YES if..." asks are spam-filtered and excluded.
- Analytical templates branch the closing claim on whether the product description carries a contestable performance number (units like W, mAh, Hz, GHz, MP, GB, ports, hours of battery, dB, Mbps, lumens, refresh rate). Spec-rich products close with a spec claim; passive products (mounts, hooks, organizers, brackets, kitchen tools, decor, manual gadgets) close with a material-or-use claim instead. The LLM is explicitly anchored against fabricating specs the product doesn't have (e.g. battery life for a phone holder).
- The per-platform caption generator receives the rendered spoken script and mirrors the script's closing engagement-bait line into the caption body before the hashtag block. Same line in spoken audio + on-screen subtitle + caption text (Rule of 3s for engagement bait). When no script is available, the caption falls back to the platform's standard search-optimised content with no closing line.

### TTS Voice Profiles
- Named voice presets with style direction, voice preferences, and text markup
- Multiple TTS providers with automatic fallback
- Style-directed speech (tone, energy, pacing). Profiles favor calm, confident delivery over high-energy pitch.
- Speaking rate and pitch tunable per profile. Rates near 1.0 (natural pace), avoid stacking slow rate + low pitch + "slow" style prompt.
- Inline markup rules for pause insertion at sentence boundaries (periods, exclamations, question marks)
- Deterministic profile selection per product for reproducibility
- Voice selection precedence (highest first): CLI override, random across configured pool, pinned default profile, random across all profiles
- Pinned default profile keeps unattended runs on a single voice for channel-wide consistency
- Configurable profile pool restricts selection to a named subset for A/B testing
- CLI override forces a specific profile for one-off runs
- Profile metadata (profile name and selected voice) recorded in pipeline output for traceability

### Stock Background Music
- Pluggable audio provider platform: `BaseAudioProvider` ABC with registry and factory pattern
- Provider chain tries each configured source in order, falls back to local files
- Default chain: Jamendo (primary) -> Freesound (fallback) -> local stock files
- Dynamic duration matching to voiceover length
- Per-provider circuit breakers for resilience
- Proper attribution and license tracking (source, author, license URL, track ID)

#### Jamendo Provider
- Jamendo Music API v3.0 with `client_id` authentication (no OAuth2 needed)
- `fuzzytags` search mode for genre/mood matching (OR relevance), configurable to `tags` (AND) or `search` (free text)
- Configurable search query pool with random selection per product for music variety
- Prefers `audiodownload` URL, falls back to stream URL if download not allowed

#### Freesound Provider
- Wraps existing FreesoundClient behind `BaseAudioProvider` interface
- OAuth2 for full quality downloads, API key for preview fallback
- Fully automated OAuth2 token setup: headless browser (Playwright) handles login + authorize + code capture
- Duration filter search with general filter fallback

### Batch Mode
- Automatic product discovery from outputs directory
- Filter to specific product IDs with `--product-ids`
- Random or fixed profile selection
- Inter-product delays for rate limiting
- Comprehensive summary reporting

---

## Content Pillars

Group products and scripts into a small set of named pillars (default 3). Each keyword, script template, and produced video declares one pillar so randomness operates within a focused theme instead of the whole catalog. The channel gets a through-line; variety stays inside each pillar.

### Pillar Definition
- Pillars listed in `config/scraper.yaml` as named buckets. Default labels: `value` (under-$30 staples, mass-appeal hooks), `novelty` (lesser-known or unique, "haven't seen this" hooks), `utility` (practical daily-use, problem/solution framing).
- Names are user-defined. Users can rename, add, or remove pillars without code changes.

### Tagging
- Keywords declare their pillar via a dict keyed by pillar name in `config/scraper.yaml` (`batch.keywords: {value: [...], novelty: [...], utility: [...]}`). Each scraped product carries the source keyword's pillar through to the producer and registry. A flat list is accepted for backward compatibility (no pillar attached). A keyword fitting more than one pillar can appear under each.
- Script templates are mapped to pillars via a central `pillars` dict under `script_templates` in `config/ai_services.yaml`. A template can appear under multiple pillars when its style works in more than one (e.g., `classic_promo` could land under both value and novelty).
- Deterministic per-product MD5 selection picks within the chosen pillar's templates instead of the full pool.

### Pipeline Behavior
- `--pillar <name>` filters a run to one pillar; without the flag, batch runs balance across all pillars.
- The flag is present on both `src/video/producer/cli.py` and `src/pipeline/global_batch.py` (Module/Batch Alignment Rule).
- Each script prompt is built by stacking three layers, in order: (1) a channel-wide narrator profile (`script_templates.narrator_profile`) that anchors voice, persona, and the anti-AI-tells rules; (2) a per-pillar preamble (`script_templates.pillar_preambles`) when a pillar is set, nudging the LLM toward that pillar's framing angle; (3) the chosen template's hook structure plus product data. Templates themselves stay pillar-agnostic and channel-agnostic so the same template can serve multiple pillars and personas.
- Platform caption generators (YouTube, TikTok, Instagram) receive the same narrator profile and pillar preamble so captions adopt the video's conversational voice rather than defaulting to SEO copy.
- When a pillar is set, the `{AUDIENCE}` placeholder substitutes the per-pillar audience hint (`script_templates.pillar_audiences[pillar]`) instead of the global `target_audience`. Falls back to the global value when the entry is missing or empty.
- If `--pillar` is given a name that isn't configured in any of `pillars`, `pillar_preambles`, or `pillar_audiences`, the run logs an info-level hint listing the configured pillars. The run continues; the template filter, preamble injection, and audience override each gracefully no-op.
- The fully-rendered script prompt (narrator profile + pillar preamble + template + product data) is written to `outputs/<asin>/temp/script_prompt.txt` on every run, useful for inspecting what the LLM actually saw.
- The chosen pillar is recorded in `pipeline_state.json` and the published-products registry so downstream analytics can segment by it.

### Prompt Hygiene
- Product titles and descriptions are Unicode-normalized before injection, folding mathematical-alphabet bold codepoints (e.g. Amazon's pseudo-bold section headers) to plain ASCII.
- Em dashes and en dashes in the description are replaced (em dash to a comma, en dash to a hyphen) so the LLM doesn't mimic them in output. Em dashes especially are a strong AI-writing tell.
- Templates receive both the full product title (`{FULL_PRODUCT_NAME}`) and a short alias (`{SHORT_PRODUCT_NAME}`) extracted heuristically from the listing title (a brand-plus-model handle, capped to a few words). Scripts refer to the product by the short alias rather than parsing the SEO-bloated title themselves.
- When the description contains a banned phrase or marketing fluff, the narrator profile instructs the LLM to paraphrase the underlying feature in its own words rather than quoting. Catches Amazon-SEO copy that contains banned-list words ("ultimate", "must-have", "revolutionary", etc.) before they leak into the script.

### Editorial Focus
- Pillar choice drives hook framing and product selection. Subtitle styling and TTS voice stay global so brand voice carries across pillars.
- A `value` video pitches the deal, a `novelty` video pitches discovery, a `utility` video frames the problem and the solution.
- The per-pillar audience hint reinforces the framing: `value` targets budget-conscious shoppers, `novelty` targets curious early discoverers, `utility` targets practical problem-solvers.
- Every generated script includes one short trade-off or limitation about the product, one sentence max, to keep the brand voice trustworthy rather than purely promotional. The rule lives in each template's `## Rules` block in `src/ai/prompts/scripts/` so the LLM sees it adjacent to the active task and applies it more reliably than when it sat in the channel-wide narrator profile.

---

## Batch Processing Module

### Global Pipeline
- **End-to-end execution**: scraping → video production → publishing
- Single command for complete workflow
- Phase isolation: failures don't cascade between phases
- Unified configuration with CLI overrides

### Execution Phases
1. **Scraping**: Collect product data, images, videos
2. **Handoff**: Filter products by media availability
3. **Production**: Generate videos with selected profile
4. **Publishing**: Upload and schedule to platforms

### Profile Randomization
- Random profile selection per product from configured pool
- When no pool is configured, the fallback is all profiles except `base` (the inheritance template, not a render target); `base` is still usable via an explicit profile choice
- Deterministic seeding for reproducibility
- Profile compatibility checking with skip on mismatch

### Summary Reporting
- Per-phase counts (success, failure, skipped)
- Overall pipeline statistics
- Profile usage distribution

---

## Publisher Module

### Service Integration
- Publish via third-party scheduling services (e.g., Late.dev)
- Multi-platform support: YouTube, TikTok, Instagram
- API-based upload and status tracking
- Reading a post's status or listing posts tolerates a published leg that reports no platform URL (some platforms return none), so status checks, first-comment verification, slot-occupancy detection, and media cleanup keep working instead of failing on such a post.
- Per-platform delivery is verifiable after posts go live: a sweep over recent posts flags any whose delivery is incomplete (top status `partial`, or a platform leg that failed) and reports the failing platform with its error. This catches a silently-dropped leg that the scheduling service reports without surfacing.

### Per-Platform Profile Routing
- Optional mapping from platform to video profile so the publisher uploads a platform-tailored render per platform (e.g., the short hook-iteration cut for YouTube, the longer cut for TikTok and Instagram).
- The publisher prefers the routed profile's render when present and falls back to the first available render in the product directory when the mapping is unset or the routed profile hasn't been produced for that product.
- Routing leaves the unified upload model intact: when one file is shared across platforms, the file picked is the render for the first platform in the post's target list. True per-platform uploads (different files to different platforms) live behind the platform-specific publishing mode and remain a planned extension.

### Scheduling
- Immediate or future publish times with timezone support
- Recurring schedule slots (e.g., "Monday 9am, Wednesday 2pm")
- Auto-scheduling to first available slot
- Batch scheduling of multiple videos via `schedule` command (with `--immediate` for direct publishing)
- Dry-run mode to preview schedule assignments without publishing

### First Comment
- Post affiliate links as the first comment instead of embedding them in captions
- Avoids algorithm penalties on platforms that deprioritize posts with outbound links in descriptions
- Supported on YouTube and Instagram; TikTok always skipped (not supported by scheduling API)
- Per-platform templates with placeholders: affiliate link, product title, hashtags
- Prefers shortened affiliate link, falls back to full link
- Optional hashtag migration: move Instagram hashtags from caption to first comment
- Non-blocking: missing data or affiliate link silently skipped (warning logged)
- Works in both unified and platform-specific publishing modes (each platform gets its own comment via per-platform data)
- Disabled by default, toggled via config
- Delivery is verifiable after a post goes live: the platform inbox is queried for the account owner's own comment, since the scheduling API reports the post published without confirming the comment posted. A sweep over recent published posts warns on any YouTube or Instagram post missing its first comment.

### Duplicate Publish Protection
- Tracks previously published products per platform
- Warns and skips duplicates by default
- `--force` flag to override and republish

### Post-Publication Cleanup
- Remove published product directories after confirmation
- Configurable per-platform enable/disable
- Safety checks verify publication success before deletion
- Require all platforms to succeed before cleanup
- Local publish history and the registry entry are written before a product directory is removed, on every publish path (single and scheduled), so a scheduled post that gets cleaned up still leaves a local record and stays visible to the duplicate-publish guard
- Trim the Vercel Blob upload store after each publish run under a config-driven retention policy: delete blobs older than a configurable age, then trim oldest-first to a configurable total size
- Blobs referenced by posts that aren't fully published yet are always kept, regardless of the retention policy
- Retention is non-blocking (failures log a warning, never affect publishing) and skips silently when disabled or no Blob token is configured

### Link-in-Bio Integration
- After publishing a video, add the product's Amazon affiliate link to a link-in-bio page
- Provider-agnostic: swappable between Lnk.Bio, Linktree, Beacons, etc.
- Configurable max links with automatic oldest-link rotation
- Non-blocking: failures never affect video publishing
- Disabled by default, toggled via config

### Published Products Registry
- Maintain a registry of all published products in the outputs directory
- Fields: product ID (ASIN), product title, canonical URL, affiliate URL, content pillar
- Dual format: JSON (machine-readable) and CSV (spreadsheet-friendly)
- Append new entries after each successful publish (no duplicates)
- Republish refreshes the existing row so registry fields reflect the latest publish, not the original. Identical-data calls don't trigger a save.
- Pillar is read from the producer's pipeline state at registration time so it captures what was actually rendered, not what was scraped. Empty when no pillar was set for the run.
- Backward-compatible loader: legacy rows without a pillar field load with the field empty.
- Support bulk import from existing scraped data directories
- CLI command to rebuild registry from existing data, retroactively populating the pillar field for any product whose state file carries one
- Rebuild merges scanned entries into the existing registry; rows whose product directories were cleaned up after publishing stay in the registry
- Each write of the registry renames the existing JSON/CSV file to `<name>.bak` first so a write that drops or corrupts entries can be recovered

---

## Content Metadata Module

### Platform-Specific Optimization
- **Unified mode**: Single metadata set for all platforms
- **Optimized mode**: Platform-tailored titles, descriptions, hashtags
- Character limit validation per platform
- Title and description that exceed the platform's hard cap are trimmed on a word boundary with an ellipsis before reaching the publisher. Hashtag-count violations are logged as warnings; the publisher does not invent or drop tags.

### Compliance
- Persistent on-frame disclosure overlay burned into every render. Fixed corner placement, full-clip duration, sized smaller than narration captions. Configurable text, position, size, color, outline, and background per render. Disabling the overlay is opt-in for non-affiliate renders (educational pillar mode).
- First-line caption disclosure on every platform. Disclosure leads the caption text on its own line, ahead of the description and hashtag block, satisfying the regulatory requirement for clear and conspicuous placement.
- Disclosure dedup: when a platform metadata generator emits the disclosure as a hashtag, the published caption renders it once (leading line) rather than twice.
- Platform-policy disclosure tags propagated automatically: TikTok branded-content disclosure, YouTube AI-content disclosure. Set on every publish payload that targets the relevant platform.
- Configurable disclosure text per render so language-matched variants are possible without code changes.
- Manual workarounds documented for platform-policy layers the publishing SDK does not expose (YouTube paid-promotion checkbox, Instagram paid-partnership label).
