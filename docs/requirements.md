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
- Topic renders use `outputs/topic-<slug>-<digest>/`, carrying no `images/` or `videos/` subdirectory since a topic has no scraped media. The identifier is derived from the title and is stable, so a re-run resumes its own directory; the digest is what keeps two titles from sharing one, which would otherwise let the second run inherit the first's completed state and return the first video
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
- Setting `scrapers.amazon.affiliate_links.enabled: false` declares that no affiliate program is in use: the canonicaliser then strips tracking parameters down to `https://www.amazon.com/dp/<ASIN>` and logs at DEBUG instead of WARNING. An explicitly supplied tag still wins over the flag, and `AMAZON_AFFILIATE_LINKS_ENABLED` overrides the YAML field

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
group fall back to FFmpeg without manual intervention. The engine the run
resolves is passed explicitly to the subtitle generator and recorded in the
pipeline state, rather than each consumer re-deriving it from config.
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
- **Burned-in hook overlay**: a short headline is rendered as centre-upper static text on the first 1.5 s (configurable), sized relative to the narration captions, with no per-word reveal. Drawn after the subtitle pass and before the FTC `#ad` corner disclosure so the disclosure stays on top of the z-order. Long text wraps to a configurable maximum number of lines, each held within a configurable fraction of the frame width, and the font shrinks when wrapping alone cannot fit; text that still does not fit at the minimum legible size is truncated with an ellipsis and the truncation is logged. When no text is available the overlay is a silent no-op.
- **Authored hook headline**: the overlay text is a headline written for the screen, generated separately from the spoken script so the hook does not repeat the first line the running captions already show. On a product render it must carry the product category so it reads on its own with the sound off; a topic render uses a separate prompt that names the symptom or the fix instead, and forbids naming anything the script does not cover; a device the script does name may appear. Both are capped at a configurable word count, and the product prompt additionally excludes model and SKU designations. Output that reads as a conversational preamble or a refusal is rejected rather than rendered. When no headline is available the overlay falls back to the first sentence of the spoken script, so the feature degrades rather than blocking a render. The headline is regenerated on re-renders when absent, skipped entirely when the overlay is disabled, and persisted in the pipeline state for downstream analytics.
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

### Stock Visual Media
- Stock footage is fetched from a configured provider and merged into the same visual pool as scraped product media, rather than being a fallback used only when scraped media is missing
- Search terms come from `media_settings.stock_media_keywords`, overridable per profile. A profile that declares no terms inherits the global list; a profile that declares an empty list searches on the product title alone. Two profiles can therefore search different footage within one run, which is what a concurrent visual comparison needs
- A profile controls whether scraped product imagery is used at all (`use_scraped_images`), so a profile can render entirely from stock without code changes. `slideshow_stock` is the bundled profile that does
- A topic supplies its own search terms, and they replace the profile and global lists rather than joining them. The provider concatenates every term into a single query, so combining a topic's words with product-oriented defaults searches for neither
- A profile that draws no visual from the scraped product gathers its footage after the script exists, and searches on phrases derived from the narration. The narration is the only description of what such a video is about; the title states the subject, not what should be on screen while it is explained
- Those phrases are searched one at a time and the results pooled, because the provider concatenates a keyword list into a single query and the library answers a long query with results skewed toward whichever phrase dominates, leaving some phrases unrepresented. Duplicate results across searches are dropped so one photograph cannot appear twice in a render
- Deriving the phrases never blocks a render: no key, a provider failure, or an unusable answer leaves the existing search terms in place
- Fetching is resilient: a provider failure degrades the visual pool rather than failing the render
- A missing provider key is caught at startup rather than mid-render, naming the variable and the profiles that need it, but only for profiles where stock is the whole visual layer. A profile that also draws scraped media degrades as above and still renders, so refusing it would block a working configuration

### Topic Input
- A video can be produced from a topic (a title, a description, optional search terms) with no scraper run and no product directory
- The topic builds the same record the producer consumes for a scraped product, so no pipeline step is topic-aware. Listing-only fields carry nothing rather than a plausible-looking value
- A topics file renders several in turn. A malformed entry fails the run rather than being skipped, since skipping renders fewer videos than requested without saying so
- Topic directories are excluded from producer batch discovery and from a product run's random profile selection, both of which assume scraped imagery. The global batch opts in for a run whose inputs are topics
- Topics are a batch input as well as a producer one, and replace the scraping phase rather than running it
- A topics run draws only from profiles whose visuals come entirely from stock. A profile that draws product imagery is refused before the run starts, because it would gather nothing and fail after the script and the voiceover had been paid for
- Topics cannot be combined with scraper inputs (`--product-ids`, `--keywords`, `--process-all-products`): a topic run skips scraping, so those inputs would be discarded or rendered under the wrong profile

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
- Topic renders select from a separate template pool that answers a question rather than pitching a product. The pool replaces the product pool rather than narrowing it, and the exclusion runs both ways: templates share one directory and the default pool is a glob over it, so a scraped product must not draw a topic template either
- Topic renders use their own narrator profile. The default one is written for someone describing a purchase, down to its call-to-action list, so a topic script would otherwise close by pointing at something to buy that does not exist
- A topic title is not passed through the product-alias heuristic, which trims a listing title to three words and would have the model speak a question fragment as the subject's name
- Every **product** template instructs the LLM to open with a natural conversational hook that carries the long-tail audio keyword (product category, price band, audience cue, pain point) embedded inside speech a person would actually say out loud. The template rule lists six proven hook patterns aligned with `docs/promotional-video-best-practices.md` §1 — price-first reveal, regret/contrarian, POV, outcome-first, numbered teardown, comparison — and names the literal Google-query shape as an anti-pattern. The keyword must land in the first 5 seconds of TTS because TikTok transcribes audio via ASR and indexes the transcript as a primary search signal.
- In a **product** template, line 1 must state a concrete fact, result, or observation about the product. Setup framings ("Today I'll show you", "Let me tell you", "In this video", "I want to share") are anti-patterns — they burn the 1.5-second decision window before the payoff lands.
- Every **product** template instructs the LLM to end the spoken script with one short engagement-bait closing line, right before the CTA, not replacing it. Personal and storytelling templates use a two-option opinion question (comment-fork); analytical and comparison templates use a debatable but defensible spec claim that invites a correction. The closing line drives comments, which feeds the platform algorithm; generic "Comment YES if..." asks are spam-filtered and excluded.
- Analytical templates branch the closing claim on whether the product description states a measurement the script can argue about. The measurement must be quotable verbatim from the description with its unit attached as a whole word, so a unit that appears only inside a longer word does not qualify. Products with such a measurement close with a claim about it; products without close with a material, shape, or use claim carrying no numbers. The spec branch offers no worked closing line to copy, because a worked example is reproduced onto products it does not fit and is the mechanism by which fabricated specs reach the script.
- Topic templates carry a different contract, and the product rules above are not merely absent from them but would be wrong. A topic template states the fix inside the first three seconds rather than building to it, requires the search phrase to be spoken aloud in the first five, asks for one instruction per sentence, forbids inventing a product to recommend, and closes on an honest limit rather than a debatable spec claim
- Scripts name the product the way a person would say it aloud. Model and SKU designations are never spoken. The short alias derived from the listing title is offered as a suggestion rather than an instruction, because it is auto-trimmed and can come out as a fragment carrying a part number; when it does not read as a spoken name, the plain category noun is used instead.
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
- Keywords declare their pillar via a dict keyed by pillar name in `config/scraper.yaml` (`batch.keywords: {value: [...], novelty: [...], utility: [...]}`). Each scraped product carries the source keyword's pillar through to the producer. A flat list is accepted for backward compatibility (no pillar attached). A keyword fitting more than one pillar can appear under each. The keyword-to-pillar mapping describes the configuration, not a particular run, so it is built from the config file whichever source supplies the run's keyword list: a keyword passed on the command line still carries its configured pillar, and an unconfigured one simply has none.
- Script templates are mapped to pillars via a central `pillars` dict under `script_templates` in `config/ai_services.yaml`. A template can appear under multiple pillars when its style works in more than one (e.g., `classic_promo` could land under both value and novelty).
- Deterministic per-product MD5 selection picks within the chosen pillar's templates instead of the full pool.

### Pipeline Behavior
- `--pillar <name>` filters a run to one pillar; without the flag, batch runs balance across all pillars.
- The flag is present on both `src/video/producer/cli.py` and `src/pipeline/global_batch.py` (Module/Batch Alignment Rule).
- Each script prompt is built by stacking three layers, in order: (1) a channel-wide narrator profile (`script_templates.narrator_profile`) that anchors voice, persona, and the anti-AI-tells rules; (2) a per-pillar preamble (`script_templates.pillar_preambles`, or `pillar_preambles_topic` on a topic render) when a pillar is set, nudging the LLM toward that pillar's framing angle; (3) the chosen template's hook structure plus product data. Templates themselves stay pillar-agnostic and channel-agnostic so the same template can serve multiple pillars and personas.
- Platform caption generators (YouTube, TikTok, Instagram) receive the same narrator profile and pillar preamble so captions adopt the video's conversational voice rather than defaulting to SEO copy.
- When a pillar is set, the `{AUDIENCE}` placeholder substitutes the per-pillar audience hint (`script_templates.pillar_audiences[pillar]`, or `pillar_audiences_topic[pillar]` on a topic render) instead of the global `target_audience`. Falls back to the global value when the entry is missing or empty.
- If `--pillar` is given a name that isn't configured in any of `pillars`, `pillar_preambles`, `pillar_audiences`, `pillar_preambles_topic` or `pillar_audiences_topic`, the run logs an info-level hint listing the configured pillars. The run continues; the template filter, preamble injection, and audience override each gracefully no-op.
- The fully-rendered script prompt (narrator profile + pillar preamble + template + product data) is written to `outputs/<asin>/temp/script_prompt.txt` on every run, useful for inspecting what the LLM actually saw.
- The chosen pillar is recorded in `pipeline_state.json`, so a resumed run keeps the pillar an earlier run resolved. It is not carried into the published-products registry; nothing read that column, and a value no consumer reads is a claim the project has to keep true for nothing.

### Prompt Hygiene
- Product titles and descriptions are Unicode-normalized before injection, folding mathematical-alphabet bold codepoints (e.g. Amazon's pseudo-bold section headers) to plain ASCII.
- Em dashes and en dashes in the description are replaced (em dash to a comma, en dash to a hyphen) so the LLM doesn't mimic them in output. Em dashes especially are a strong AI-writing tell.
- Templates receive both the full product title (`{FULL_PRODUCT_NAME}`) and a short alias (`{SHORT_PRODUCT_NAME}`) extracted heuristically from the listing title (a brand-plus-model handle, capped to a few words). Scripts refer to the product by the short alias rather than parsing the SEO-bloated title themselves.
- When the description contains a banned phrase or marketing fluff, the narrator profile instructs the LLM to paraphrase the underlying feature in its own words rather than quoting. Catches Amazon-SEO copy that contains banned-list words ("ultimate", "must-have", "revolutionary", etc.) before they leak into the script.

### Editorial Focus
- Pillar choice drives hook framing and product selection. Subtitle styling and TTS voice stay global so brand voice carries across pillars.
- A `value` video pitches the deal, a `novelty` video pitches discovery, a `utility` video frames the problem and the solution.
- The per-pillar audience hint reinforces the framing: `value` targets budget-conscious shoppers, `novelty` targets curious early discoverers, `utility` targets practical problem-solvers.
- Every **product** script includes one short trade-off or limitation about the product, one sentence max, to keep the brand voice trustworthy rather than purely promotional. The rule lives in each template's `## Rules` block in `src/ai/prompts/scripts/` so the LLM sees it adjacent to the active task and applies it more reliably than when it sat in the channel-wide narrator profile.

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
- Publish via third-party scheduling services (Zernio, formerly Late)
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

### Affiliate Program Literal Phrase
- Render the configured affiliate program identification phrase in the caption body of every published post when enabled
- Disabled by default, including when the configuration section is absent or empty. The phrase asserts membership of the named program, so an unconfigured install must not publish one
- Phrase is placed between the leading disclosure line and the description
- Configurable phrase and program name; defaults to the Amazon Associates identification phrase
- Works in both unified and platform-specific publishing modes
- Applies consistently across the standalone publisher CLI and the global batch pipeline

### Post Analytics
- Day-2 and day-7 cumulative views, plus a durability ratio, stored per published post
- Durability is views after the first 30 days over views within them. At or above 1.0 the post earned more attention after launch than during it
- The scheduler's timeline is cumulative, so a day-N figure is a lookup against one call rather than a scheduled job that must run on the day
- A window the post has not reached yet reports as unknown rather than as the running total, and a post with no views in the durability window reports unknown rather than 0.0. Ranking treats unknown as unmeasurable and sorts it last, never as a zero
- A day-N figure counts every platform or none. Platforms start reporting on their own lag, and a leg's first row carries its lifetime total rather than that day's increment, so a cutoff that some legs had reported by and others had not reports as unknown. The cutoff is recorded on the post, because the sweep that stores the figure is usually earlier than the one that can see the lag, and a later sweep withdraws the number it already kept
- Reports can rank by durability, which answers a different question from ranking by total views or by day-7 views: at day 7 a post that keeps earning and one that spiked and stopped are indistinguishable
- Measuring a post again merges into its stored row field by field rather than replacing it. Past the provider's retention horizon a later reading has *less* history behind it, so a measured figure is never replaced by an absent one, except at a cutoff later found to have straddled a leg's first report, where the stored figure counted only part of the post, and the field recording how far the timeline reached moves with the ratio it dates

### Published Products Registry
- Maintain a registry of all published products in the outputs directory
- Fields: product ID (ASIN), product title, canonical URL, affiliate URL, content-format arm
- Dual format: JSON (machine-readable) and CSV (spreadsheet-friendly)
- Append new entries after each successful publish (no duplicates)
- Republish refreshes the existing row so registry fields reflect the latest publish, not the original. Identical-data calls don't trigger a save.
- Backward-compatible loader, in both directions: a row missing a field the record declares takes that field's default, and keys the record no longer declares are dropped rather than passed on, so removing a column does not make every row written before it unreadable. A row the record cannot build at all costs that row and nothing else — failing the whole load would be worse than raising, since the caller treats an unreadable registry as an empty one and rewrites the file.
- The content-format arm records whether the video came from a topic or a scraped product, so two formats published side by side can be compared later. It is read from the record rather than inferred from the profile or the publish date: a profile is a visual treatment two arms can share, and a date cannot reconstruct an arm that was interleaved, which is the only way to run the comparison fairly.
- Rows written before the arm existed report as unlabelled rather than being counted as either arm, because a comparison that silently absorbs unknown videos into one side is worse than one that shows how many it cannot place.
- The CSV header is derived from the record definition rather than restated, so adding a field cannot fail the registry write.
- Support bulk import from existing scraped data directories
- CLI command to rebuild registry from existing data, merging rediscovered rows into the existing registry rather than replacing it
- Rebuild merges scanned entries into the existing registry; rows whose product directories were cleaned up after publishing stay in the registry
- Each write of the registry renames the existing JSON/CSV file to `<name>.bak` first so a write that drops or corrupts entries can be recovered

---

## Content Metadata Module

### Platform-Specific Optimization
- **Unified mode**: Single metadata set for all platforms
- **Optimized mode**: Platform-tailored titles, descriptions, hashtags
- Character limit validation per platform
- Title and description that exceed the platform's hard cap are trimmed on a word boundary with an ellipsis before reaching the publisher. Hashtag-count violations are logged as warnings; the publisher does not invent or drop tags.
- A per-platform payload carries every field its consumer reads. Where a platform derives a value when none is supplied (a title from the caption's first line, for example), a partially-populated payload is worse than none: the platform silently substitutes its own value and the result looks like working output. Any field added to one side of that contract is added to the other.
- Platforms that accept a distinct video title are sent one. Not sending a title is not neutral, because the platform then derives one from the caption, and the caption leads with the disclosure line.
- Length clamping happens before the per-platform payload is built, so a clamped value cannot be copied in its unclamped form.

### Compliance
- Persistent on-frame disclosure overlay burned into every render that carries a material connection, on every subtitle engine and every subtitle positioning mode. A render path that cannot apply the overlay fails loudly rather than shipping a video without it; silently omitting a required disclosure is the failure this guards against. Fixed corner placement, full-clip duration, sized smaller than narration captions. Configurable text, position, size, color, outline, and background per render. A record that positively shows there is nothing to disclose, a topic with no affiliate link, suppresses the overlay automatically; every ambiguous record still discloses.
- First-line caption disclosure on every platform that carries a material connection, gated on the same decision as the overlay so a caption and a frame cannot disagree about whether a render is promotional. Disclosure leads the caption text on its own line, ahead of the description and hashtag block, satisfying the regulatory requirement for clear and conspicuous placement.
- The producer records the decision and the publisher reads it, rather than each deriving its own. Metadata written before the decision existed is backfilled on the next render, and a file that still lacks it discloses
- Disclosure dedup: when a platform metadata generator emits the disclosure as a hashtag, the published caption renders it once (leading line) rather than twice.
- Platform-policy disclosure tags on every publish payload that targets the relevant platform, carrying the value that render warrants rather than a fixed one. TikTok's commercial-content type is gated on the same decision as the overlay and the caption: `brand_organic` for a render with a material connection, `none` for one without. YouTube's altered-or-synthetic-content tag is opt-in and off by default, since the policy excludes AI narration, AI-written scripts and stock footage; it is configurable rather than removed, because output that does meet the bar is possible.
- Configurable disclosure text per render so language-matched variants are possible without code changes.
- Manual workarounds documented for platform-policy layers the publishing SDK does not expose (YouTube paid-promotion checkbox, Instagram paid-partnership label).
