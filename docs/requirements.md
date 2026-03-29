# Project Requirements

High-level requirements for ContentEngineAI.

---

## Global Requirements

### Configuration System
- **Three-tier precedence**: CLI arguments > Environment variables > YAML files
- CLI arguments override only when **explicitly provided** by the user (use `default=None` in argparse, not hardcoded defaults)
- Environment variables (`.env`) store secrets only—never committed
- YAML files (`config/`) store application settings—safe to commit
- Validate all configuration at startup with clear error messages

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
- **Anchors**: top, center, bottom, above-content, below-content
- **Margin** from anchor edge (default 4%)
- **Horizontal alignment**: left, center (default), right
- Content-aware positioning relative to actual media bounds
- **Platform safe zone**: boundaries avoid TikTok, YouTube Shorts, and Instagram Reels UI overlays (buttons, captions, nav bars). Applies to both ASS and SRT (drawtext) formats. Per-profile overrides supported.
- **Font size**: scales with resolution (range 4-16% of frame height)
- **Two-part mode**: upper (static URL/link) + lower (voiceover transcription)
- **Dynamic repositioning**: both upper and lower subtitles repositioned per visual segment using actual assembler geometry (handles mixed-media profiles where video and images have different bounds)
- **CTA detection**: auto-detects call-to-action phrases in voiceover for styling emphasis
- Style presets: minimal, modern, bold, animated, random
- **ASS effects**: fade, scale_pulse, rotation_bounce, glow, typewriter, karaoke, movement
- One effect applied per video (deterministic by product ID)
- Deterministic randomization seeded by product ID

### Profile System
- **Precedence**: CLI > Profile > Global defaults
- All visual, subtitle, and video settings configurable per profile
- Typed Pydantic models for merged settings
- Deterministic random profile selection per product

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

### TTS Voice Profiles
- Named voice presets with style direction, voice preferences, and text markup
- Multiple TTS providers with automatic fallback
- Style-directed speech (tone, energy, pacing). Profiles favor calm, confident delivery over high-energy pitch.
- Speaking rate and pitch tunable per profile. Rates near 1.0 (natural pace), avoid stacking slow rate + low pitch + "slow" style prompt.
- Inline markup rules for pause insertion at sentence boundaries
- Deterministic profile selection per product for reproducibility
- Configurable profile pool to restrict selection
- CLI override to force a specific profile
- Profile metadata recorded in pipeline output

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

### Duplicate Publish Protection
- Tracks previously published products per platform
- Warns and skips duplicates by default
- `--force` flag to override and republish

### Post-Publication Cleanup
- Remove published product directories after confirmation
- Configurable per-platform enable/disable
- Safety checks verify publication success before deletion
- Require all platforms to succeed before cleanup

### Link-in-Bio Integration
- After publishing a video, add the product's Amazon affiliate link to a link-in-bio page
- Provider-agnostic: swappable between Lnk.Bio, Linktree, Beacons, etc.
- Configurable max links with automatic oldest-link rotation
- Non-blocking: failures never affect video publishing
- Disabled by default, toggled via config

### Published Products Registry
- Maintain a registry of all published products in the outputs directory
- Fields: product ID (ASIN), product title, canonical URL, affiliate URL
- Dual format: JSON (machine-readable) and CSV (spreadsheet-friendly)
- Append new entries after each successful publish (no duplicates)
- Support bulk import from existing scraped data directories
- CLI command to rebuild registry from existing data

---

## Content Metadata Module

### Platform-Specific Optimization
- **Unified mode**: Single metadata set for all platforms
- **Optimized mode**: Platform-tailored titles, descriptions, hashtags
- Character limit validation per platform

### Compliance
- Automatic `#ad` inclusion for FTC compliance
- Proper affiliate disclosure handling
