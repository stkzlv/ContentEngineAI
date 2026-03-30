# Changelog

All notable changes to ContentEngineAI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Platform-aware subtitle safe zones for TikTok, YouTube Shorts, and Instagram Reels (PlatformSafeZone config, per-profile overrides)
- `--product-ids` filter for producer batch mode
- Audio module summary logging
- Voice profiles: soft_intimate, calm_confident, gentle_storyteller
- Gemini voice catalog reference in docs/tts-voice-profiles.md
- Platform safe zone reference in docs/platform-safe-zones.md

### Changed
- Subtitle positioning enforces safe zone on all anchor types and X axis, both ASS and SRT
- Unified SRT/drawtext line splitting to use same word count + char limit rules as ASS
- Subtitle width capped against safe zone boundaries
- Removed YAML re-read from calculate_position() hot path
- Voice profiles tuned to 1.05-1.10 speaking rate (~155-170 WPM) for short-form video retention
- Background music track selection randomized (was always picking shortest eligible track)
- Script templates toned down (challenge_dare, unboxing_reaction, rapid_fire, curiosity_hook)
- Background music queries switched to ambient/chill, volume lowered -20dB to -24dB
- Module summaries unified to consistent format with product IDs, no emojis

### Removed
- Dead constants from video config: MIN_HIGH_RES_IMAGE_*, WHISPER_WORD_LEVEL_TIMING_MIN_CONFIDENCE

## [0.34.1] - 2026-03-19

### Security
- Bump cryptography 45.0.7 → 46.0.5 (CVE-2026-26007: binary EC curve private key leak)
- Bump pillow 11.3.0 → 12.1.1 (OOB write with invalid tile extents)
- Bump werkzeug 3.1.5 → 3.1.6 (Windows `safe_join` device name bypass)

### Dependencies
- Bump nltk 3.9.1 → 3.9.2

## [0.34.0] - 2026-03-17

### Added
- Audio provider platform: pluggable system for background music sourcing with `BaseAudioProvider` ABC, registry, and `AudioManager` chain
- Jamendo Music API provider (CC-licensed tracks, fuzzytags search, configurable queries with random selection)
- Global batch page retry: when products fail media validation, scraper tries next search result pages (configurable `max_retry_pages`)
- `audio_providers` config list in `video_production.yaml` for provider chain ordering
- Public `record_success()`/`record_failure()` methods on `CircuitBreaker`
- 47 new tests (audio provider platform + page retry)
- 23 trending electronics keywords added to scraper batch config

### Changed
- `step_download_music()` refactored from ~100 lines of inline Freesound logic to ~15 lines using AudioManager
- Default audio provider chain: Jamendo (primary) -> Freesound (fallback) -> local files
- Audio timeouts aligned to 15s search / 60s download across all providers
- FreesoundClient wrapped as `FreesoundProvider` adapter (client code untouched)
- Background music volume raised from -24 dB to -20 dB
- Audio provider env vars dynamically read from config (no more hardcoded secret names)

### Fixed
- `FreesoundClient` crash: `AudioSettings.get()` called on Pydantic model instead of dict
- Security CI failing: `setuptools<81` not pinned on cache-hit path (same fix as CI workflow)

## [0.33.0] - 2026-03-12

### Added
- First comment support: post affiliate links as first comment on YouTube and Instagram instead of in captions. Configured via `first_comment` section in `publisher.yaml`
- `--force` flag for `schedule` command to bypass already-published checks
- 15 new tests for first comment builder

### Changed
- Consolidate `batch` command into `schedule --immediate`. Old `batch` CLI is removed
- `schedule` and `calendar` subcommands (`auto`/`list`) are now optional with defaults
- Default `--platform` to youtube/tiktok/instagram across all commands
- Instagram first comment uses product title + "Link in bio!" instead of raw URL (not clickable on Instagram)

### Fixed
- ASIN hashtag (#B0...) missing from posts going through the schedule auto-scheduling path
- First comments not sent when publishing via `schedule auto` (schedule path bypassed `publish_modes.py`)

## [0.32.1] - 2026-03-04

### Fixed
- CLI `--product-ids` no longer picks up YAML keywords (and vice versa). CLI inputs are treated as the complete input set.

### Changed
- Batch pipeline scrapes all inputs in a single Chrome session instead of launching one browser per keyword (~15s saved per keyword)
- Removed `{PRICE}` placeholder from all script templates to avoid stale pricing in videos
- Replaced `money_value` script template with `myth_buster`

## [0.32.0] - 2026-03-04

### Added
- `--clean` flag for global batch pipeline to remove product directories before running
- `scrape-lowpri` and `produce-lowpri` Makefile targets for resource-limited scraping/producing

### Changed
- Switch default Gemini model from `gemini-2.0-flash` to `gemini-2.5-flash-lite`
- Fix `max_tokens` Pydantic default (4096 to 600) to match ai_services.yaml
- Fix `pipeline_timeout_sec` fallback (300 to 900) to match core.yaml

### Removed
- Unused `optimization_settings` section from performance.yaml (background_processing, connection_pooling, async_io, caching)
- Unused `ApiSettings` fields (default_request_timeout_sec, default_retry_attempts, default_retry_delay_sec)
- Unused URL shortener config (fallback_providers, bitly, tinyurl sections)
- Stale keywords from scraper.yaml batch config

## [0.31.0] - 2026-03-03

### Changed
- **Performance**: Convert all f-string logger calls to lazy `%s` formatting in performance module
- **Performance**: Cleanup interval now configurable via `cleanup_interval` param (default: every 10 saves)
- **Performance**: Step completion log level changed from INFO to DEBUG
- **Performance**: `PerformanceMonitor.reset()` accepts `memory_monitor_interval` for config-driven tuning
- **Config**: Wire `performance_monitoring_interval_sec` and `performance_history_cleanup_interval` from config into runtime

### Added
- `PerformanceMonitor.reset()` method for clean state between batch runs
- `PerformanceMonitor.check_thresholds()` for timing/memory warnings after pipeline runs
- `PerformanceHistoryManager.force_cleanup()` for on-demand history pruning
- Corrupt JSONL line handling in history loading (skip and warn instead of crash)
- Percentile stats (p50/p95/p99) in summary and step analysis reports
- `--report-type comparison` for profile-vs-profile performance comparison
- `--report-type regressions` with configurable window/threshold for detecting slowdowns
- Step-level trends in trends report (per-step daily averages)
- `--format csv` export for detailed and trends reports
- `performance_history_cleanup_interval` config field in `OptimizationSettings`
- `perf-trends`, `perf-detailed`, `perf-compare` Makefile targets
- `tests/test_performance_report.py` with 22 tests for report generation
- Expanded `tests/test_performance.py` with HistoryManager, PipelineRunMetrics, and monitor tests

### Removed
- Dead `network_sent`/`network_recv` fields from `PerformanceMetrics` (always zero)
- `sys.path.insert` hack in `tools/performance_report.py`

## [0.30.0] - 2026-02-24

### Changed
- **Config**: `get_profile_merged_settings()` returns typed `MergedProfileSettings` Pydantic model instead of raw dicts
- **Config**: Add `video_vertical_align` field ("center"/"top") for video content positioning via FFmpeg pad expression
- **Scraper**: Validate media counts against producer profile requirements (skip products the profile can't use)
- **Scraper**: Default `products_per_keyword: 1` and `max_products: 50` (was 5 and 1)
- **Config**: Align upper subtitle defaults with YAML profiles (margin 0.04, font_size_scale 0.7)
- **Config**: Move hardcoded LLM values (blocklist, validation thresholds, retry settings) from code into `LLMSettings` and `ai_services.yaml`
- **LLM**: Switch primary provider from OpenRouter to Gemini, OpenRouter becomes fallback
- **LLM**: Rename `openrouter_circuit_breaker` to `llm_circuit_breaker` (provider-agnostic)

### Added
- `MergedSubtitleSettings`, `ProfileInfo`, `MergedProfileSettings` Pydantic models in `visual_models.py`
- `--profile` CLI arg for standalone scraper to align media validation with producer
- `profile_uses_videos` parameter on scraper constructor for profile-aware validation
- `--max-products` and `--products-per-keyword` CLI args for standalone scraper
- **TTS voice profiles**: Configurable voice presets with style prompts, markup rules, and per-profile provider routing
- **Gemini TTS provider**: Style-directed speech via `SynthesisInput(prompt=...)` with automatic fallback to Google Cloud TTS
- `VoiceProfileConfig`, `TextMarkupRule` Pydantic models in `audio_models.py`
- `--voice-profile` CLI override for producer and global batch pipeline
- TTS metadata (profile name, voice name) saved in `pipeline_state.json`
- Inline markup preprocessing: `[short pause]`, `[pause]` inserted at sentence boundaries per profile rules
- Deterministic voice profile selection per product (md5 hash, hex slice `[16:24]`)
- **Dynamic upper subtitle positioning**: content-aware per-segment repositioning using assembler geometry, splits CTA across visual segments
- **Script template system**: 15 distinct prompt templates (curiosity hook, problem-solution, storytelling, comparison, etc.) with deterministic per-product selection via salted md5 hash
- `ScriptTemplateConfig` model: `enabled`, `templates_dir`, `template_pool`, `fixed_template` fields
- `--script-template` CLI override for producer and global batch pipeline
- Script template name saved in `pipeline_state.json` under generate_script metadata
- **Provider fallback chain**: Gemini as primary LLM, automatic fallback to OpenRouter with free model discovery when primary exhausts all models
- `fallback_provider` field on `LLMSettings` (self-referencing Pydantic model)
- `ScriptValidationConfig` model with configurable `min_chars`/`min_words` thresholds
- `model_blocklist`, `min_context_length`, `retry_attempts`/`retry_min_wait_sec`/`retry_max_wait_sec` fields on `LLMSettings`
- `make batch-lowpri` target for resource-constrained batch runs (nice, ionice, memory cap)
- `src/ai/llm_client.py`: shared LLM dispatch layer (Gemini via google-genai SDK, OpenRouter via aiohttp)

### Fixed
- Video content stuck at top of frame when `video_top_position_percent: 0.0` was set (now uses FFmpeg centering)
- Upper subtitle floating in middle of black bar for letterboxed video profiles (now positioned near content edge)
- Unclosed aiohttp session on producer exit (close global connection pool in CLI)
- Fallback provider API key not loaded into secrets dict (global_batch.py, cli.py)
- `os.environ.get()` bypass in platform metadata utilities (now uses secrets dict)
- `update_env_file()` no longer adds new keys to `.env` (only updates existing ones)

### Dependencies
- Bump `google-cloud-texttospeech` from ^2.26.0 to ^2.29.0 (adds `SynthesisInput.prompt` support)

## [0.29.2] - 2026-02-20

### Dependencies
- Update aiohttp 3.11.18 → 3.13.3
- Update pillow 11.2.1 → 11.3.0
- Update torchaudio 2.7.0 → 2.8.0
- Update fonttools 4.58.0 → 4.60.2
- Update transformers 4.51.3 → 4.53.0
- Update authlib 1.6.3 → 1.6.6
- Update marshmallow 4.0.0 → 4.1.2
- Update virtualenv 20.31.2 → 20.36.1
- Regenerate poetry.lock (Poetry 2.2.1 format)

## [0.29.1] - 2026-02-16

### Fixed
- **Correctness**: Remove wrong past-time validation from `PublishResult.__post_init__` (broke deserialized historical results)
- **Correctness**: Make `save_tracking()` atomic via temp-file + rename (prevents corruption on crash)
- **Correctness**: Fix timezone-naive `datetime.now()` in `ScheduleEntry.created_at`
- **Performance**: Hoist `get_accounts()` out of platform loop in batch publisher (N×M → 1 API call)
- **TikTok docs**: Fix caption limit from 2200 to 150 characters (matching `PLATFORM_LIMITS`)

### Changed
- **Logging**: Convert 120+ f-string log calls to lazy `%s`/`%d` format across publisher module
- **Exception handling**: Narrow 15+ bare `except Exception` to specific types (`PublishError`, `OSError`, `TimeoutError`, `ValueError`, `yaml.YAMLError`, `json.JSONDecodeError`)
- **Type safety**: Fix `publisher: object` → `BasePublisher` in `publish_modes.py` using `TYPE_CHECKING`
- **Constants**: Extract `PLATFORM_LIMITS`, `SDK_LIST_PAGE_SIZE`, `MAX_CONCURRENT_CLEANUPS`, `DEFAULT_OUTPUTS_DIR` to `constants.py`
- **DRY**: Extract `_create_publisher_from_config()` (6 call sites), `_call_sdk()` (12+ patterns), split 300-line `publish()` into focused helpers
- **Config**: Simplify `create_default_config_file()` with template string (was 30 `f.write()` calls)

### Added
- Tests for `publish_modes.py` (0% → 98% coverage), `metadata.py` (10% → 62%), `tracking.py` (61% → 94%)

## [0.29.0] - 2026-02-09

### Added
- **Platform-Specific Publishing Mode**: Support both unified (single post, default) and platform-specific (separate posts per platform) publishing modes via `use_platform_specific_content` config flag or `--platform-specific` CLI flag
- **Link-in-Bio Integration**: Automatically add product affiliate links to a link-in-bio page after publishing
  - Provider-agnostic design with Lnk.Bio as first implementation
  - Configurable max links with automatic oldest-link rotation
  - Non-blocking: failures never affect video publishing
  - Enabled by default (`link_in_bio.enabled: true`)
  - CLI flags: `--link-in-bio` / `--no-link-in-bio` to override config
  - Affiliate URL priority: uses `affiliate_link` field, falls back to `url`
  - Image fallback: images array URL first, downloaded local file as fallback
  - Improved logging: INFO for outcomes, DEBUG for API details
- **TikTok Content Settings**: Configurable `TikTokContentSettings` dataclass for content disclosure fields
- **DEFAULT_PLATFORMS constant**: Single source of truth for default platform list
- **Published Products Registry**: Track all published products in JSON and CSV formats
  - Fields: product ID (ASIN), title, canonical URL, affiliate URL
  - Automatically appended after each successful publish (single and batch)
  - CLI command to rebuild registry from existing data (`registry --rebuild`)
  - Supports separate scan and output directories (`--scan-dir`)

### Changed
- **Shared Publishing Helper**: Extract `publish_product()` into `src/publisher/publish_modes.py` for consistent behavior across CLI, batch pipeline, and scheduler
- **max_links default**: Changed from 25 to 0 (unlimited) — no link rotation by default

### Fixed
- **Global Batch Single Post**: Fixed pipeline to create one post per product for all platforms (was creating separate posts per platform)
- **Global Batch Link-in-Bio**: Added link-in-bio, record_publish, and product registry calls to global batch pipeline
- **TikTok Commercial Content Disclosure**: Fix `commercial_content_type` and `is_brand_organic_post` fields unreachable when `use_platform_specific_content: false`
- **Timeout default mismatch**: Align model default (was 30s) with YAML config (120s)
- **Lnk.Bio auth**: Use HTTP Basic Auth and proper User-Agent to bypass Cloudflare
- **Lnk.Bio list endpoint**: Corrected from `/lnks` to `/lnk/list`
- **Config default mismatches**: Sync `immediate_publish` and `link_in_bio.enabled` code defaults with YAML

### Removed
- **backoff_multiplier**: Deprecated field removed from `PublisherConfig` (unused after retry refactor)

## [0.28.0] - 2026-02-08

### Added
- **Scraper URL Support**: Accept full or shortened URLs (tr.ee, amzn.to, etc.) via `--product-ids` or `--input-file`. URLs are navigated directly in the browser and ASIN extracted from the redirected URL.
- **Scraper CLI Options**:
  - `--input-file FILE`: Read product IDs/URLs from a file (one per line), merged with `--product-ids`
  - `--batch-size N`: Process products in sequential batches of N
  - `--output-dir DIR`: Override output directory (e.g. `--output-dir tmp` instead of default `outputs/`)

## [0.27.0] - 2026-02-03

### Added
- **LLM Model Selection Improvements**:
  - Random model selection option (`random_model_selection: true`)
  - Fallback discovery for any free model when configured models fail
  - Blocklist for tiny models (<7B) that produce low-quality output
  - Updated model list with verified working free models (tngtech, arcee-ai, z-ai, nvidia)

- **Centered Image Subtitle Positioning**:
  - Calculate visual bounds from actual image dimensions for centered images
  - Image fallback lookup for parallel pipeline execution

### Fixed
- **Upper Subtitle Positioning**: Fix ABOVE_CONTENT anchor to position relative to visual content top edge instead of frame top

### Refactored
- **Video Producer Module**:
  - Extract `two_part_subtitles.py` (464 lines) for subtitle handling logic
  - Add `constants.py` with Platform enum and visual bounds defaults
  - Add `artifact_registry.py` consolidating 6 duplicate loaders
  - Split `step_generate_description()` into 4 focused helper functions
  - Replace 13 bare `except Exception` with specific exception types
  - Convert f-string logging to lazy format (`%s`)

## [0.26.0] - 2026-01-25

### Added
- **Two-Tier Product Limits**: Granular control over product collection
  - `max_products`: Global cap on total products to collect
  - `products_per_keyword`: Maximum products per individual keyword
  - Processing stops when global limit is reached, even if keywords remain

### Changed
- **CLI Config Precedence**: CLI arguments now only override YAML values when explicitly provided
  - Omitting a CLI flag uses the YAML configuration value
  - Prevents hardcoded defaults from unexpectedly overriding YAML settings

### Refactored
- **Scraper Module**: Split large files into focused modules for maintainability
  - `constants.py`: Centralized magic numbers and filter codes
  - `image_utils.py`: Image validation and URL processing
  - `video_extractor.py`: Video extraction and M3U8 capture
  - `debug_analysis.py`: Debug image analysis utilities
  - `product_extractor.py`: Product data extraction from pages
  - `download_async.py`: Async download operations
  - `download_validators.py`: Download validation logic
- **Structured Logging**: Replaced print statements with logger calls using lazy %-formatting
- **Global State Elimination**: Removed `DEBUG_MODE` global in favor of parameter passing

## [0.25.0] - 2026-01-22

### Added
- **Pipeline Resume Capability**: Continue interrupted pipelines from last checkpoint
  - `PipelineState` dataclass for tracking phase completion and product progress
  - `--resume` CLI flag to continue from last successful phase
  - State persistence to `outputs/.pipeline_state.json` after each phase
  - Graceful handling of corrupted state files (starts fresh)
  - Automatic state file cleanup on successful completion

- **Parallel Platform Publishing**: Concurrent uploads to multiple platforms per video
  - `asyncio.gather()` with `return_exceptions=True` for error isolation
  - Per-platform success/failure tracking with accurate summary statistics
  - Fail-fast check after all platforms processed (not mid-execution)
  - Reduces publishing phase duration when targeting multiple platforms

- **Dry-Run Mode**: Preview pipeline plan without executing
  - `--dry-run` CLI flag validates configuration and shows planned actions
  - Displays products to scrape, profiles to use, platforms to publish
  - Shows API key status and scheduling mode
  - Exits cleanly without executing any pipeline phases

- **JSON Output Format**: Machine-readable pipeline summaries
  - `--output-format json` outputs parseable JSON to stdout
  - Includes ISO timestamps (started_at, completed_at)
  - Contains all statistics, product IDs, and error details
  - Backward compatible (text format remains default)

- **Webhook Notifications**: External monitoring and alerting support
  - Non-blocking POST requests on phase completion and pipeline events
  - Configurable via `webhook` section in `config/pipeline.yaml`
  - Event types: `phase.complete`, `phase.failed`, `pipeline.complete`, `pipeline.failed`
  - Automatic retry with exponential backoff (default: 3 retries)
  - 5-second timeout to prevent pipeline delays
  - URL validation before sending requests

- **Product ID Hashtag**: ASIN/product ID appended as hashtag in post descriptions
  - Enables tracking and discoverability across platforms
  - Added to `PublishMetadata` model with `product_id` field

### Changed
- **Outro Duration**: Renamed `duration_padding_sec` to `outro_duration_sec` for clarity
  - Now clearly indicates purpose: music fade-out time after voiceover ends
  - Default 1.0s provides smooth ending and prevents audio truncation

- **Metadata Generation**: Hashtags now generated in one place only
  - Description field contains text only (no embedded hashtags)
  - Hashtags stored separately in `hashtags` field
  - `format_content()` combines description + hashtags cleanly

### Fixed
- **Duplicate Hashtags**: Fixed hashtags appearing twice in published posts
  - Legacy metadata with embedded hashtags now stripped on load
  - New metadata generation excludes hashtags from description text

- **Voiceover Truncation**: Fixed last word being cut off in videos
  - Increased outro duration from 0.5s to 1.0s
  - Provides buffer for AAC encoding frame alignment

### Documentation
- Updated publisher CLI examples to match current implementation
- Replaced deprecated `--video` flag with positional `product_id` argument

## [0.24.0] - 2026-01-17

### Added
- **Publisher Multi-Account Support**: Route products to different Late.dev accounts
  - `AccountConfig` dataclass with validation (name, api_key, vercel_token, default_platforms)
  - YAML `accounts` section with named accounts and `default_account` selector
  - `--account NAME` CLI flag to switch active account at runtime
  - Backward compatible: single `api_key` at root creates "default" account
  - 25 tests for multi-account functionality

- **Publisher Conflict Resolution**: Automatic scheduling conflict handling
  - `ConflictResolution` dataclass with alternatives sorted by time proximity
  - `find_alternatives()` and `resolve_conflict()` methods in ScheduleManager
  - `--auto-resolve` CLI flag to automatically use first available alternative
  - Configurable via `conflict_alternatives_count` (default: 5)
  - 20 tests for conflict resolution functionality

- **Publisher Retry Queue**: Automatic retry mechanism for failed batch items
  - `--retry-failed` CLI flag to resume failed items
  - Preserves original scheduling for retry attempts

- **Publisher Webhooks**: Real-time status updates without polling
  - WebhookHandler with HMAC-SHA256 signature verification
  - Supports events: `post.scheduled`, `post.published`, `post.failed`, `post.partial`, `account.disconnected`
  - Idempotent event processing with automatic history pruning
  - 28 tests for webhook handling

- **Publisher Documentation**: Comprehensive CLI reference and workflows
  - CLI reference tables for all commands
  - Common workflows section with 5 end-to-end examples
  - Safety guidelines for cleanup operations

- **Publisher Integration Tests**: 42 tests for full publish-schedule-cleanup workflow

### Changed
- **Publisher Configuration**: Fixed timeout default mismatch (30s to 120s)
- **README**: Simplified to reference full documentation

## [0.23.0] - 2026-01-16

### Added
- **Platform Metadata Enhancements**: Five new modules for the platform metadata system
  - `src/ai/platform_metadata/cache.py` - File-based metadata caching with TTL expiration and LRU eviction
  - `src/ai/platform_metadata/ab_testing.py` - Prompt variant selection with deterministic hash-based assignment
  - `src/ai/platform_metadata/batch.py` - Concurrent multi-product processing with semaphore rate limiting
  - `src/ai/platform_metadata/export.py` - Multi-format export (JSON, CSV, YouTube CSV, TikTok, Instagram)
  - `src/ai/platform_metadata/trends.py` - Trend-aware hashtag merging with configurable fallback tags

- **Platform Metadata Tests**: Comprehensive test coverage for enhancement modules
  - `tests/ai/test_metadata_cache.py` - Cache tests (25 tests)
  - `tests/ai/test_ab_testing.py` - A/B testing tests (25 tests)
  - `tests/ai/test_batch_generation.py` - Batch generation tests (25 tests)
  - `tests/ai/test_metadata_export.py` - Export tests (31 tests)
  - `tests/ai/test_trend_aware_hashtags.py` - Trend tests (13 tests)

### Changed
- **Configuration**: Added settings for new metadata modules in `config/ai_services.yaml`
  - Cache settings: TTL, directory, max entries
  - A/B testing: Prompt variants with weights
  - Batch settings: Max concurrent, progress logging
  - Export settings: Formats, encoding, YouTube category/privacy
  - Trend settings: Provider, cache TTL, fallback tags

## [0.22.0] - 2026-01-12

### Added
- **Video Producer Tests**: Comprehensive test coverage for video production modules
  - `tests/video/test_ass_effects.py` - ASS subtitle effects tests (522 lines)
  - `tests/video/test_batch_producer.py` - Batch processing tests (144 lines)
  - `tests/video/test_subtitle_positioning.py` - Subtitle positioning tests (114 lines)
  - `tests/video/test_video_strategies.py` - Video assembly strategy tests (270 lines)
  - `tests/integration/test_producer_integration.py` - Integration tests (163 lines)
  - `tests/audio/test_freesound_client.py` - Freesound client tests (171 lines)
  - `tests/test_tts.py` - SSML generation tests for TTS

- **Video Producer Documentation**: `docs/video-producer.md` - Complete CLI reference guide (346 lines)

### Fixed
- **TTS Last Word Truncation**: Added SSML break tag to prevent voiceover audio cutoff
  - `src/video/tts.py` - Uses SSML with configurable buffer time (default 300ms)
  - `src/video/assembler/audio_builder.py` - Added `apad` filter to extend audio duration
  - `src/video/unified_subtitle_generator.py` - Disabled fade-out on last subtitle segment

### Changed
- **Configuration Documentation**: Added inline documentation to config YAML files
  - `config/ai_services.yaml` - Whisper settings tuning guidance
  - `config/core.yaml` - System timeout documentation
  - `config/performance.yaml` - FFmpeg settings documentation
  - `config/video_production.yaml` - Removed duplicate FFmpeg settings (consolidated to performance.yaml)

## [0.21.0] - 2026-01-10

### Added
- **Platform Detection**: Extensible registry pattern for product ID platform detection
  - `src/scraper/base/platform_detector.py` - Registry with `@register_platform` decorator
  - Amazon ASIN validation (B0/B1 prefix, 10-char alphanumeric)
  - 30 unit tests for platform detection edge cases

- **Scraper Test Suite**: Comprehensive test coverage for scraper modules
  - `tests/scraper/test_platform_detector.py` - Platform detection tests (162 lines)
  - `tests/scraper/test_batch_controller.py` - Batch processing tests (550+ lines)
  - `tests/scraper/test_media_validator.py` - Media validation with FFprobe mocking
  - `tests/integration/test_scraper_integration.py` - End-to-end workflow tests

- **Scraper User Guide**: Comprehensive documentation at `docs/scraper-user-guide.md`

- **Configurable Timeouts**: System timeouts for external commands via `config/core.yaml`
  - FFprobe, xrandr, system_profiler, head_request timeouts

### Changed
- Scraper module version bumped to 2.1.0

## [0.20.0] - 2026-01-06

### Added
- **Network Resilience**: Retry utilities with exponential backoff for network operations
  - `src/utils/retry.py` - `@retry_network` decorator for HTTP requests
  - Automatic retry on 429, 503, 5xx errors and connection timeouts
  - Configurable max attempts, wait times, and backoff multiplier

- **Circuit Breaker Pattern**: Prevent cascade failures from external services
  - `src/utils/circuit_breaker.py` - Pre-configured breakers for Freesound, Pexels, OpenRouter, Google STT, Scraper
  - YAML-based configuration in `config/performance.yaml`
  - States: CLOSED → OPEN → HALF_OPEN with automatic recovery

- **Unified Config Manager**: Three-tier configuration precedence
  - `src/config_manager.py` - CLI arguments > Environment variables > YAML files
  - Type conversion for boolean, int, float from environment strings
  - Dot notation support for nested configuration paths

- **Secret Masking**: Automatic credential protection in logs
  - `src/utils/secrets.py` - Pattern-based secret detection
  - `src/utils/logging_setup.py` - `SecretMaskingFilter` for all log handlers
  - Masks API keys, tokens, passwords before output

- **Claude Code Slash Commands**: Workflow automation commands
  - `.claude/commands/` - commit, bump-version, release, run-linters, update-pr, etc.

### Changed
- **Configuration**: Expanded `.env.example` with all configuration options
- **Documentation**: Updated `docs/configuration.md` with precedence documentation

## [0.19.1] - 2026-01-04

### Changed
- **Documentation**: Reorganized extended docs from root to `docs/` directory
  - Moved 11 documentation files (architecture, configuration, testing, etc.)
  - Updated all internal links in README.md, CONTRIBUTING.md, CLAUDE.md
  - Fixed inaccuracies to match actual codebase state

- **Specs**: Consolidated granular specs into unified module specs
  - 7 unified specs: batch-processing, content-metadata, global-requirements, publisher, scraper, video-producer
  - Added retry logic (tenacity) to global-requirements spec
  - Cleaned up old approval directories and implementation logs

### Removed
- Obsolete compliance tests (~3500 lines)
- Old granular spec directories (freesound-client, late-publisher, etc.)
- Implementation task logs from completed specs

## [0.19.0] - 2025-12-25

### Added
- **Auto-Scheduling with Occupied Slot Detection**: Publisher now queries Late.co API to find unoccupied time slots
  - `global_batch.py` - 8-week lookahead to detect occupied slots via API query (623 lines total, +298 new)
  - Slot normalization at minute precision for accurate comparison
  - Automatic fallback to immediate publishing when all slots occupied
  - Debug logging for publisher config and token loading
  - Integration with global batch pipeline publishing phase

- **Post-Publication Cleanup**: Automatic removal of product directories after successful publish
  - `global_batch.py` - Cleanup logic integrated into publishing phase
  - Verification of multi-platform success before deletion
  - Configurable cleanup settings via `config/publisher.yaml`
  - Smart cleanup respects `require_all_platforms` configuration
  - Cleanup only triggers after ALL configured platforms succeed

- **Global Batch Pipeline Publishing Phase**: Complete 4-phase end-to-end automation
  - Scraping Phase → Handoff Phase → Production Phase → Publishing Phase
  - Auto-scheduling finds first available slot for each product
  - Multi-platform publishing with platform-specific metadata
  - Comprehensive publishing summary with per-platform results
  - Enhanced error handling with detailed failure tracking

### Changed
- **Configuration**: Updated publisher configuration with enhanced validation
  - `config/publisher.yaml` - Added `immediate_publish: false` for auto-scheduling
  - `recurring_schedule.enabled: true` enables slot-based scheduling
  - `cleanup.enabled: true` enables automatic cleanup after publish
  - Enhanced configuration documentation with auto-scheduling examples

- **Documentation**: Comprehensive updates for new features
  - `BATCH_PROCESSING.md` - Updated to 4-phase pipeline architecture (+78 lines)
  - Added publishing examples with auto-scheduling and cleanup
  - Updated YAML configuration section with publishing settings
  - Updated pipeline summary to include publishing phase results
  - `PUBLISHER.md` - Updated features list and auto-scheduling behavior (+22 lines)
  - Changed environment variable from `BLOB_READ_WRITE_TOKEN` to `LATE_VERCEL_TOKEN`
  - Updated auto-scheduling documentation to explain API querying
  - `README.md` - Updated quick start and key features (+23 lines)
  - Added auto-scheduling explanation to batch processing section
  - Updated social media publishing section with cleanup note

### Fixed
- **Environment Variables**: Corrected Vercel token variable name throughout documentation
  - `.env` - Changed from `BLOB_READ_WRITE_TOKEN` to `LATE_VERCEL_TOKEN`
  - Code maintains backward compatibility with old variable name
  - All documentation updated to use new variable name

### Testing
- **Integration Tests**: Comprehensive test coverage for publishing features
  - `test_global_batch_publishing.py` - 419 new lines of integration tests
  - Test auto-scheduling finds first unoccupied slot
  - Test fallback to immediate when all slots occupied
  - Test cleanup removes directory after successful publish
  - Test cleanup preserves directory on partial failure
  - Test Vercel token loaded from environment
  - Coverage for `global_batch.py` improved from 36% to 72%

## [0.18.0] - 2025-12-22

### Added
- **Social Media Publishing**: Complete publishing module for automated video distribution
  - New `src/publisher/` package with modular architecture for platform publishing
  - `base.py` - Abstract publisher interface with error handling (54 lines)
  - `models.py` - Pydantic models for publish metadata, results, and configs (424 lines)
  - `registry.py` - Publisher provider registry with factory pattern (159 lines)
  - `late/client.py` - Late.dev integration with retry logic and rate limiting (1,131 lines)
  - `metadata.py` - Platform metadata loader with fallback support (347 lines)
  - `batch.py` - Batch publisher with stagger delays and progress tracking (531 lines)
  - `config.py` - Three-tier configuration system (CLI → Env → YAML) (434 lines)
  - `late/cli.py` - Command-line interface for publishing operations (1,069 lines)
  - Multi-platform support: YouTube, TikTok, Instagram, Facebook, Twitter, LinkedIn
  - Scheduled publishing with immediate and future posting options
  - Large file support (>4MB) via Vercel CDN integration
  - Exponential backoff retry logic with configurable max retries
  - Rate limit handling with `Retry-After` header support

- **Publisher Scheduling System**: Automated video scheduling with recurring calendar
  - `schedule.py` - Schedule manager with slot allocation (649 lines)
  - `schedule_validator.py` - Schedule validation and conflict detection (256 lines)
  - Recurring schedule configuration with weekly time slots
  - Timezone-aware scheduling (configurable timezone support)
  - Automatic slot allocation across multiple products and platforms
  - Platform-specific metadata integration for scheduled posts
  - Separate posts per platform for customized content
  - Schedule persistence with JSON tracking (`outputs/schedule.json`)
  - Calendar view for visualizing upcoming posts
  - Slot availability validation and conflict prevention

- **Post-Publication Cleanup**: Automated cleanup of published videos
  - `cleanup.py` - Cleanup manager with safety checks (615 lines)
  - Automatic cleanup after successful multi-platform publication
  - Manual cleanup via CLI command
  - Verification of publication success across all platforms
  - Configurable safety options (verify before delete, require all platforms)
  - Dry-run mode for preview before deletion
  - Detailed cleanup reports with file sizes and paths
  - Integration with schedule tracking for status verification

- **CLI Commands**: Publishing, scheduling, and cleanup operations
  - `list-accounts` - List connected social media accounts
  - `single` - Publish single video to one or more platforms
  - `batch` - Batch publish all videos in outputs directory
  - `schedule` - Schedule videos with recurring calendar slots
  - `cleanup` - Remove published videos with safety checks
  - `list-schedule` - View upcoming scheduled posts
  - Platform selection: `--platform youtube --platform tiktok` (repeatable)
  - Scheduling: `--schedule "2025-01-20 14:00:00"` or `--immediate` or `--use-schedule`
  - Debug mode: `--debug` for verbose logging
  - Fail-fast mode: `--fail-fast` to stop on first error

- **Configuration**: Publisher configuration system
  - `config/publisher.yaml` - Publisher settings (defaults, timeouts, retries)
  - Environment variables: `LATE_API_KEY`, `LATE_VERCEL_TOKEN`
  - CLI overrides for all configuration values
  - Stagger delays for batch publishing (30-60s default)
  - Per-platform privacy settings
  - `recurring_schedule` section with weekly time slots
  - Timezone configuration (default: Europe/Berlin)
  - Cleanup configuration (enabled, verify_before_delete, require_all_platforms)

- **Documentation**: Comprehensive user and developer guides
  - `PUBLISHER.md` - 1,251 lines of complete documentation
    - Setup guide with Late.dev account creation
    - CLI usage examples with copy-paste commands
    - Configuration precedence explanation
    - Platform metadata integration guide
    - Batch publishing workflows
    - Publishing schedule and calendar system
    - Post-publication cleanup guide (automatic and manual)
    - Error handling and retry logic
    - Troubleshooting guide for common scenarios
    - API reference for programmatic usage
    - Made large sections collapsible for improved readability
  - Updated `README.md` with publisher section and quick start
  - Added publisher to core documentation table

- **Testing**: Comprehensive test suite (7,000+ lines)
  - `tests/publisher/test_base.py` - Base interface tests (422 lines)
  - `tests/publisher/test_models.py` - Model validation tests (488 lines)
  - `tests/publisher/test_registry.py` - Registry and factory tests (490 lines)
  - `tests/publisher/late/test_client.py` - Client tests with mocking (1,023 lines)
  - `tests/publisher/test_schedule.py` - Schedule manager tests (510 lines)
  - `tests/publisher/test_schedule_manager.py` - Integration tests (660 lines)
  - `tests/publisher/test_schedule_validator.py` - Validation tests (658 lines)
  - `tests/publisher/test_schedule_models.py` - Model tests (378 lines)
  - `tests/publisher/test_cleanup.py` - Cleanup manager tests (650 lines)
  - `tests/integration/test_late_publisher.py` - Real API integration tests (548 lines)
  - `tests/e2e/test_publisher_workflow.py` - End-to-end CLI tests (717 lines)
  - `tests/e2e/test_publisher_schedule_cleanup.py` - E2E workflow tests (1,027 lines)
  - Tests skip gracefully when credentials not available
  - Integration tests require `.env.test` with sandbox credentials
  - E2E tests validate complete workflow: video → metadata → publish → cleanup

### Fixed
- **Type Hints**: Python 3.12 compatibility
  - Changed `callable | None` to `Callable[[int, int], None] | None`
  - Added `from collections.abc import Callable` imports
  - Added `from typing import Any` import to `late/client.py`
  - Fixed type annotation for `platform_results: list[Any]`
  - Fixed type narrowing for `published_urls_list` in status logging
  - Fixed in `src/publisher/base.py` and `src/publisher/late/client.py`

- **Code Formatting**: Line length compliance
  - Fixed 4 line length violations in `schedule.py` (88-character limit)
  - Split long comment lines for readability
  - Fixed f-string concatenation for long log messages
  - Applied Ruff formatting to all publisher code

### Changed
- **Publisher Architecture**: Enhanced for scheduling and cleanup
  - Platform-specific posts now created separately for metadata customization
  - Improved error handling for scheduling conflicts
  - Added post status checking and verification to client

- **Code Quality**: All linting checks passing
  - Fixed all Ruff linting issues (import sorting, line length, docstrings)
  - Fixed all MyPy type annotation errors
  - All checks passing: Ruff, Ruff Format, MyPy, Bandit, Vulture, Safety, Pytest
  - Publisher module security: 0 issues (Bandit scan clean)

- **Documentation Structure**: Improved readability
  - Made large sections collapsible in `PUBLISHER.md` using `<details>` tags
  - Consolidated duplicate code blocks
  - Enhanced markdown structure with proper hierarchy

### Technical
- **New Modules** (13 files, ~7,000 lines):
  - `src/publisher/__init__.py` - Package exports
  - `src/publisher/base.py` - Abstract base (54 lines)
  - `src/publisher/models.py` - Data models (424 lines)
  - `src/publisher/registry.py` - Registry pattern (159 lines)
  - `src/publisher/late/__init__.py` - Late.dev package
  - `src/publisher/late/client.py` - Late client (1,131 lines)
  - `src/publisher/late/cli.py` - CLI interface (1,069 lines)
  - `src/publisher/metadata.py` - Metadata loader (347 lines)
  - `src/publisher/batch.py` - Batch orchestrator (531 lines)
  - `src/publisher/config.py` - Configuration (434 lines)
  - `src/publisher/schedule.py` - Schedule management (649 lines)
  - `src/publisher/schedule_validator.py` - Validation (256 lines)
  - `src/publisher/cleanup.py` - Cleanup management (615 lines)

- **New Tests** (12 files, ~7,000 lines):
  - Unit tests for all publisher modules
  - Integration tests for schedule manager and Late.dev API
  - E2E tests for complete workflows including cleanup
  - High coverage with edge case testing

- **Dependencies**:
  - `late-sdk` - Official Late.dev Python SDK
  - `aiohttp` - Async HTTP client for API calls

## [0.17.0] - 2025-12-16

### Added
- **Platform-Specific Metadata Optimization**: AI-powered content generation for social media
  - New `src/ai/platform_metadata/` package with modular architecture
  - `base.py` - Abstract base generator with template system
  - `models.py` - Pydantic models for metadata and generation configs
  - `utilities.py` - Shared utilities for hashtag and emoji processing
  - `text_formatter.py` - Intelligent text formatting with character limits
  - `youtube.py` - YouTube-optimized metadata (5000-char descriptions)
  - `tiktok.py` - TikTok-optimized metadata (2200-char captions)
  - `instagram.py` - Instagram-optimized metadata (2200-char captions)
  - Multi-platform support via `--target-platform` CLI flag
  - `UPLOAD_INSTRUCTIONS.txt` generation with platform-specific posting guidance
  - Automatic URL shortening and formatting for platform requirements

- **Producer CLI Enhancement**: New `--target-platform` flag
  - Supports `youtube`, `tiktok`, `instagram`, or `multi` (all platforms)
  - Generates optimized metadata per platform requirements
  - Creates ready-to-post instructions with formatted content

### Fixed
- **Circular Import Resolution**: Config module architecture
  - Resolved circular dependency between `video_config.py` and dependent modules
  - Fixed import ordering in subtitle and assembler modules
  - All modules now import correctly without circular reference errors

- **Subtitle Positioning**: Corrected `above_content` anchor logic
  - Fixed inverted positioning calculation for content-aware mode
  - Changed from `visual_bounds.y - margin` to `margin` for top positioning
  - Subtitles now correctly positioned at configured margin from top
  - Updated 3 test cases to match corrected positioning behavior
  - All tests passing: 973/1001 (44.95% coverage)

- **Test Suite Updates**: Enhanced fixture management
  - Updated test fixtures to use config-based positioning values
  - Moved magic numbers to centralized configuration
  - Improved test maintainability and consistency

### Changed
- **Documentation Improvements**: Comprehensive updates
  - Fixed repository URLs from `ContentEngineAI/ContentEngineAI` to `stkzlv/ContentEngineAI`
  - Updated import paths from `video_config` to `config_adapter`
  - Reduced README.md batch section verbosity (saved 24 lines)
  - Added platform metadata feature documentation
  - Improved markdown structure with collapsible sections
  - Moved test reports to `outputs/reports/` directory

- **Configuration Integration**: Platform metadata settings
  - Added platform metadata configuration in `config/ai.yaml`
  - Integrated text formatter with configurable limits
  - URL shortener integration for social media links

### Technical
- **New Modules**:
  - `src/ai/platform_metadata/__init__.py` - Package exports
  - `src/ai/platform_metadata/base.py` - Base generator (175 lines)
  - `src/ai/platform_metadata/models.py` - Data models (95 lines)
  - `src/ai/platform_metadata/utilities.py` - Utilities (68 lines)
  - `src/ai/platform_metadata/text_formatter.py` - Formatter (142 lines)
  - `src/ai/platform_metadata/youtube.py` - YouTube generator (89 lines)
  - `src/ai/platform_metadata/tiktok.py` - TikTok generator (85 lines)
  - `src/ai/platform_metadata/instagram.py` - Instagram generator (85 lines)

- **Code Quality**: All linting checks passing
  - Ruff: Code style and formatting ✓
  - MyPy: Type checking ✓
  - Bandit: Security scanning ✓
  - Vulture: Dead code detection ✓
  - Safety: Dependency vulnerabilities ✓
  - Pytest: 973/1001 tests passing (44.95% coverage) ✓

## [0.16.0] - 2025-12-08

### Changed
- **Assembler Refactoring**: Modular architecture for video assembly
  - Split monolithic `assembler.py` (3,311 lines) into 7 focused modules
  - `core.py` - VideoAssembler orchestrator (~690 lines)
  - `visual_builder.py` - Visual filter chains (~590 lines)
  - `subtitle_builder.py` - Subtitle positioning (~850 lines)
  - `audio_builder.py` - Audio filter chains (~200 lines)
  - `video_strategies.py` - Video mode strategies (~665 lines)
  - `media_inspector.py` - Media file inspection (~170 lines)
  - `subtitle_utils.py` - Subtitle parsing/styling (~280 lines)
  - Improved maintainability and separation of concerns
  - 100% backward compatibility via `__init__.py` re-exports

### Fixed
- **Subtitle Positioning**: Fixed letterboxed video positioning
  - Return actual geometry from `apply_aspect_ratio_mode` for letterbox videos
  - Compute real scaled dimensions and position based on FFmpeg output
  - Prefer actual geometry over config-based positioning
  - Subtitles now correctly positioned relative to letterboxed content
  - Fixes subtitles being placed too far from ultra-wide videos in portrait frames

### Added
- **Assembler Integration Tests**: Basic validation for refactored architecture
  - VideoAssembler initialization test
  - VisualGeometry dataclass tests for letterbox positioning
  - 3 new integration tests added to test suite

## [0.15.0] - 2025-12-05

### Added
- **Global Batch Pipeline**: Unified scrape-then-produce workflow
  - New `src/pipeline/global_batch.py` module (719 lines)
  - Single command for complete batch operations: scraping + video production
  - Inherits all scraper and producer batch features
  - Comprehensive error handling and progress tracking
  - 1,315 tests for end-to-end batch workflows

- **Scraper Batch Mode**: Process multiple products efficiently
  - `BatchController` for orchestrating multi-product scraping
  - Support for product ID lists and keyword searches
  - Configurable search filters (price range, rating, prime-only)
  - Products-per-keyword limit for controlled scraping
  - Deduplication across product IDs and keywords
  - Fail-fast support for early termination on errors
  - Detailed batch summary with media statistics

- **Producer Batch Mode**: Automated video production at scale
  - Batch processing for all scraped product data files
  - Fixed profile or random profile selection per product
  - Deterministic randomization with seed-based selection
  - Configurable profile pools for controlled variety
  - Usage tracking prevents over-selection of profiles
  - Profile pool validation and error handling

### Changed
- **Configuration Architecture**: New batch-specific settings
  - `config/scraper.yaml`: Batch scraping configuration
  - `config/video_production.yaml`: Batch profile settings
  - `config/pipeline.yaml`: Global pipeline configuration
  - Backward-compatible with existing single-product workflows

- **Subtitle Positioning**: Improved visual clarity
  - Increased upper subtitle margin from 0.03 to 0.10
  - Prevents overlap with video content
  - Better separation between upper and lower subtitles

### Fixed
- **Test Suite**: Comprehensive batch testing
  - Fixed batch integration test mock signatures
  - All tests passing: 876 tests (0 failures)
  - Coverage: 46.79% (exceeds 40% minimum target)
  - New test files: `test_batch_controller.py`, `test_batch_integration.py`, `test_global_batch_*.py`

- **Code Quality**: Linting and cleanup
  - Removed unused variable in async context check
  - All linters passing: Ruff, MyPy, Bandit, Vulture, Safety, Pytest

### Documentation
- **BATCH_PROCESSING.md**: Complete user guide for batch operations
- **REQUIREMENTS.md**: Technical specifications for batch features
- **Updated guides**: README.md, TESTING.md, CLAUDE.md with batch commands

### Technical
- **New Modules**:
  - `src/pipeline/__init__.py` - Pipeline package initialization
  - `src/pipeline/config.py` - Pipeline configuration loading
  - `src/pipeline/global_batch.py` - Main orchestration logic
  - `src/scraper/amazon/batch_controller.py` - Batch scraping controller
  - `src/video/producer/utils.py` - Profile selection utilities

- **Test Infrastructure**:
  - 34 scraper batch tests (20 unit + 14 integration)
  - 40 producer batch tests (24 unit + 16 integration)
  - 1,315 global pipeline tests (677 orchestrator + 638 integration)
  - Total: 876 tests passing

## [0.14.0] - 2025-11-27

### Added
- **Pydantic Configuration Models**: Type-safe scraper configuration system
  - Comprehensive Pydantic models in `src/scraper/config_models.py` (19 models, 283 lines)
  - Full validation with Field constraints for all scraper settings
  - Backward-compatible with existing dict-based configuration
  - `load_scraper_config_pydantic()` function for modern config loading
  - Matches video pipeline's configuration architecture

- **Concurrent Download Configuration**: Configurable async download limits
  - `concurrent_image_downloads`: Semaphore limit for image downloads (default: 5)
  - `concurrent_video_downloads`: Semaphore limit for video downloads (default: 3)
  - Moved hardcoded values from downloader.py to config/scraper.yaml
  - Prevents resource exhaustion during high-volume scraping

### Changed
- **Async I/O Architecture**: Converted scraper to async for improved performance
  - `convert_m3u8_to_mp4()` converted to async subprocess execution
  - Added `download_file_async()` helper with aiohttp and retry logic
  - Implemented concurrent downloads with semaphore rate limiting
  - Deprecated `download_file_sync()` in BaseDownloader
  - Maintains Botasaurus compatibility via `asyncio.run()` wrapper

### Fixed
- **Code Quality**: Enhanced type safety and validation
  - All configuration values now validated at startup via Pydantic
  - Eliminated hardcoded concurrency limits
  - Improved error messages for invalid configuration

### Technical
- **Test Infrastructure**: Comprehensive coverage for new systems
  - Added 41 tests for Pydantic config models (100% coverage for config_models.py)
  - Tests for defaults, custom values, and validation constraints
  - Tests for concurrent download configuration
  - Total tests: 805 collected (777 passing, 28 skipped)
  - Coverage: 45.20% (up from 44.10%)

- **Documentation Updates**:
  - Marked SCRAPER_ASYNC_REFACTORING.md as completed
  - Marked SCRAPER_CONFIG_REFACTORING.md as completed
  - Updated TESTING.md with new test statistics

## [0.13.0] - 2025-11-25

### Changed
- **Architecture Refactoring**: Modularized configuration and producer systems
  - Split monolithic `video_config.py` (1150 lines) into specialized modules:
    - `config/core_models.py` - Main VideoConfig and core settings
    - `config/audio_models.py` - TTS, STT, and audio processing
    - `config/visual_models.py` - Video, images, and media settings
    - `config/subtitle_models.py` - Subtitle effects and segmentation
    - `config/constants.py` - Shared constants
  - Split monolithic `producer.py` (2514 lines) into producer package:
    - `producer/cli.py` - Command-line interface
    - `producer/steps.py` - Pipeline step implementations
    - `producer/orchestration.py` - Pipeline execution logic
    - `producer/state.py` - State management
    - `producer/context.py` - Context models
    - `producer/utils.py` - Utility functions
  - Improved subtitle positioning: margins increased from 0.03 to 0.10 for better visibility

### Fixed
- **Code Quality**: Comprehensive linting and cleanup
  - Removed 248 duplicate class definitions across config modules
  - Removed 13 unused constant imports
  - Fixed MD5 hash security warnings with `usedforsecurity=False`
  - Fixed line length violations (88-character limit)
  - All linters passing: Ruff, MyPy, Bandit, Vulture, Safety

### Technical
- **Test Infrastructure**: Updated test suite for new architecture
  - Total tests: 736 passing (28 skipped)
  - Coverage: 45.04% (exceeds 40% minimum target)
  - Updated test imports for modular structure
  - All compliance and integration tests passing

## [0.12.0] - 2025-11-22

### Added
- **M3U8/HLS Video Support**: Native support for M3U8 playlist video extraction
  - FFmpeg-based M3U8 to MP4 conversion with audio stream handling
  - Strict product filtering to exclude related/sponsored products
  - Video muting during scraping for improved performance
  - DEBUG_MODE parameter passing through scraper pipeline
  - 20 comprehensive tests for M3U8 extraction
  - 16 integration tests for video pipeline

- **Product Video Assembly Modes**: Configurable video assembly with aspect ratio handling
  - Multiple assembly modes: product_video_sequential, slideshow_images1, slideshow_images2
  - Automatic aspect ratio detection and constraint enforcement
  - Audio level normalization and mixing
  - 555 tests for video mode assembly
  - 483 tests for video transformations

- **Configurable Video Positioning**: Height-constrained video placement
  - `video_top_position_percent` and `video_content_height_percent` settings
  - Content-aware subtitle positioning using configured video bounds
  - Consistent subtitle placement across all video profiles
  - Enhanced visual bounds calculation for subtitle generation

- **CTA Detection & Synchronization**: Keyword-based call-to-action detection
  - 15 configurable CTA keywords (`link`, `bio`, `visit`, `shop`, etc.)
  - Automatic timing window detection from subtitle text
  - CTA-synchronized upper subtitle display (shows only during CTA moments)
  - Configurable minimum duration threshold and merge gap
  - Centralized configuration in `config/video_production.yaml`

- **Whisper Timeout Configuration**: Adjustable timeout settings for transcription
  - `base_timeout_sec`: Base timeout before audio duration (default: 120s)
  - `duration_multiplier`: Audio duration multiplier (default: 6.0x)
  - `max_timeout_sec`: Maximum timeout cap (default: 900s)
  - Resource monitoring and cleanup options
  - All settings moved from code to `config/ai_services.yaml`

### Changed
- **Subtitle Margin Adjustments**: Fine-tuned two-part subtitle spacing
  - Lower subtitle margin: 0.02 → 0.04 (improved readability)
  - Upper subtitle margin: 0.05 → 0.06 (better visual separation)

- **Video Profile Enhancements**: Extended all product_video_* profiles
  - Added two-part subtitle configuration to all profiles
  - ASS format with randomized fonts, colors, and effects
  - Content-aware positioning enabled across all profiles
  - Subtitle max line length: 38 characters, max words per line: 2

- **Script Generation**: Refined video script prompts
  - Removed price mentions from hook examples
  - Enhanced hook quality guidelines

### Fixed
- **Type Checking**: Resolved MyPy type narrowing errors
  - Fixed 3 indexing errors for optional `profile_settings` (assembler.py:2552, 2750, 3046)
  - Added explicit None checks for type safety

- **Code Quality**: Fixed Ruff linting violations
  - Resolved 3 line length issues (88-character limit)
  - Added missing docstring parameter documentation

- **Content-Aware Positioning**: Improved subtitle placement accuracy
  - Prefer configured video bounds over detected geometry
  - Fallback to geometry detection when config unavailable
  - Consistent positioning for both upper and lower subtitles
  - Better logging for debugging positioning issues

### Technical
- **Test Infrastructure**: Comprehensive test coverage expansion
  - Total tests: 760 (732 passing, 28 skipped)
  - Coverage: 44.16% (exceeds 40% minimum target)
  - Test review completed: All tests verified against current codebase
  - New test categories: M3U8 extraction, video assembly, media validation

- **Media Validation**: Enhanced video extraction validation
  - 411 tests for media validator
  - 194 tests for video extraction validation
  - Strict filtering for product-related content only

- **Configuration System**: Extended video production configuration
  - Video positioning parameters in all profiles
  - Two-part subtitle settings with anchor points
  - Content-aware positioning with visual bounds

## [0.11.0] - 2025-10-28

### Added
- **Freesound OAuth2 Authentication**: Enhanced audio client with production-ready OAuth2 support
  - OAuth2 authorization code flow with PKCE for secure authentication
  - Automatic token refresh and persistence
  - Comprehensive error handling with fallback to local files
  - Interactive setup tool (`tools/freesound_oauth2_setup.py`)
  - 344 integration tests and 755+ unit tests with extensive mocking
  - Attribution tracking for downloaded audio files

- **CTA Detection Configuration System**: Configurable timing validation for subtitle display
  - New `CTADetectionSettings` class in video configuration
  - `min_cta_duration` setting (default: 2.0s) for minimum CTA window validation
  - `fallback_duration` setting (default: 9999.0s) for static subtitle display
  - Prevents blinking subtitles when CTA windows are too short
  - Falls back to full video duration when CTA detection yields insufficient timing

### Changed
- **Background Music Volume**: Reduced from -20.0 dB to -24.0 dB for better voiceover clarity
- **Upper Subtitle Margin**: Adjusted from 0.05 to 0.04 for improved positioning
- **Video Script Prompt**: Enhanced with better hook examples and marketing language exclusions
  - Added concrete hook examples: "I didn't think a $40 gadget could do that"
  - Excluded marketing buzzwords: "Game-changer", "Next-level", "Ultimate solution"

### Fixed
- **Subtitle Timing Bug**: Fixed blinking upper subtitle issue
  - Added minimum duration validation for CTA windows
  - Falls back to full video duration when CTA windows total < 2 seconds
  - Improved logging for CTA detection edge cases

### Technical
- **Configuration Architecture**: Moved hardcoded magic numbers to configuration
  - CTA timing values (2.0s, 9999.0s) now configurable via `config/video_production.yaml`
  - Type-safe configuration with Pydantic models
  - Centralized configuration management for easier maintenance

## [0.10.0] - 2025-10-26

### Added
- **CTA-Based Timing for Upper Subtitles**: Keyword-driven display timing for promotional content
  - Continuous display mode: merges all CTA windows into single period (first to last CTA)
  - Configurable CTA keywords: visit, follow, subscribe, link, check out, shop now
  - Custom URL support via `product_url` field in product data
  - 18 comprehensive tests with 93% coverage
  - Gap threshold configuration for window merging control

### Changed
- **Test Suite Cleanup**: Removed outdated compliance tests
  - Total tests: 627 (down from 630)
  - Removed 3 tests expecting non-existent YAML config structures
  - All 627 tests passing (606 passed, 21 skipped, 0 failed)
  - Coverage maintained at 42.79%
  - Updated TESTING.md with current statistics

### Fixed
- **Code Quality**: Resolved linting issues in CTA detection
  - Fixed MyPy type errors for optional gap_threshold parameter
  - Fixed SubRipTime attribute access type warnings
  - Fixed line length violations in subtitle utilities
  - All linting tools passing (Ruff, MyPy, Bandit, Vulture, Safety)

### Technical
- **CTA Detection Module**: New keyword-based timing window detection
  - `src/video/cta_detector.py`: Core detection and merging logic
  - Integration with subtitle generation pipeline
  - Configurable merge gap threshold (None for continuous mode)
  - REQUIREMENTS.md documentation for CTA system

## [0.9.0] - 2025-10-24

### Added
- **Requirements Compliance Test Suite**: Comprehensive validation of all documented requirements
  - 114 compliance tests across 3 test files
  - Configuration system validation (24 tests): CLI > ENV > YAML precedence, secret isolation
  - Scraper architecture compliance (22 tests): BaseScraper interface, product data extraction, media storage
  - Video production validation (68 tests): subtitle positioning, two-part system, profiles, presets, ASS effects, AI integration
  - All 12 requirements validated with 100% pass rate
  - Test documentation and status reporting in tests/compliance/
  - Pytest compliance marker for isolated test execution

### Changed
- **Test Infrastructure**: Enhanced testing framework
  - Total tests increased from 497 to 611 (114 new compliance tests)
  - All tests passing (592/611, 19 skipped)
  - Added compliance test category to TESTING.md
  - Updated test statistics and documentation

### Technical
- **Quality Assurance**: Progress toward 1.0.0 stability
  - Complete requirements traceability through automated tests
  - Validates configuration precedence, scraper patterns, video features
  - Code inspection approach for complex async provider testing
  - Clear requirement-to-test mapping in compliance README

## [0.8.0] - 2025-10-19

### Added
- **Two-Part Subtitle System**: Display multiple subtitle lines simultaneously
  - Upper subtitle line for affiliate links, product titles, or custom text
  - Lower subtitle line for main script/voiceover content
  - Independent positioning, styling, and effect randomization per line
  - Source field configuration for flexible data mapping
  - 335 comprehensive test cases covering all scenarios
  - Support for visual bounds awareness and margin controls

### Changed
- **Subtitle Configuration Refactoring**: Consolidated to dict-based approach
  - Removed legacy SubtitleSettings Pydantic model (-200 lines)
  - Unified subtitle configuration loaded from config/subtitles.yaml
  - All subtitle access patterns updated to use dict keys
  - Improved configuration flexibility and maintainability
- **Configuration Files**: Enhanced subtitle configuration structure
  - Added two_part_subtitles section with upper/lower line controls
  - New parameters: font_size_scale, style_preset, use_full_duration, randomize_effects
  - Updated video_production.yaml with two-part subtitle examples

### Fixed
- **Code Quality**: Resolved all linting issues
  - Fixed 13 line length violations (E501)
  - Removed duplicate dictionary key (F601)
  - Cleaned up unused variables (F841)
  - Fixed MyPy type errors in assembler and tests
  - All 7 linting tools passing (Ruff, Ruff Format, MyPy, Bandit, Vulture, Safety, Pytest)

### Technical
- **Test Coverage**: Added comprehensive two-part subtitle test suite
  - test_two_part_subtitles.py with 335 lines of tests
  - Tests for positioning, styling, effects, and edge cases
  - Visual bounds integration testing
- **Type Safety**: Improved type annotations for dict-based subtitle settings
- **Code Cleanup**: Removed unreachable code and simplified test logic

## [0.7.0] - 2025-10-13

### Added
- **URL Shortening Integration**: Affiliate link shortening with PicSee.io provider
  - Provider-agnostic registry system supporting multiple URL shortening services
  - Async URL shortening with single and bulk operations
  - Custom alias and branded short domain (BSD) support
  - Exponential backoff retry logic with jitter for API resilience
  - Comprehensive configuration via `config/url_shortener.yaml`
  - Integration with Amazon scraper for automatic affiliate link shortening
  - 7 new retry logic tests ensuring robust error handling
  - Full documentation in configuration comments

### Changed
- **Configuration System**: Enhanced url_shortener.yaml from 59 to 141 lines with comprehensive documentation
  - All retry parameters now configurable (max_retries, retry_delay, backoff_multiplier)
  - PicSee-specific settings separated for multi-provider support
  - Debug logging for retry attempts and configuration values
- **Amazon Scraper**: Updated to load and pass retry configuration to URL shortener
  - Improved logging for URL shortening operations
  - Better error handling for shortening failures

### Fixed
- **Test Suite**: Fixed 5 URL shortener tests using incorrect API response format
  - Changed from v2 bulk API format (`shortLink`) to v1 API format (`picseeUrl`)
  - All 36 URL shortener tests now passing

### Technical
- **Code Quality**: All linting checks passing (Ruff, MyPy, Bandit, Vulture, Safety)
- **Type Safety**: Added explicit type annotations and casts for retry logic
- **Dead Code Detection**: Created Vulture whitelist for async context manager parameters
- **Test Coverage**: Added TestPicseeRetryLogic class with comprehensive retry tests
- **Documentation**: Enhanced configuration comments explaining each setting's purpose and impact

## [0.6.0] - 2025-10-06

### Breaking Changes
- **Removed legacy subtitle configuration system**
  - Removed `positioning_mode`, `alignment`, `margin_v_percent` fields
  - Removed `relative_positioning` and `absolute_positioning` sections
  - Removed `SubtitlePositioningSettings` and `AbsolutePositioningSettings` classes
  - Users must migrate to unified configuration (see `MIGRATION_GUIDE_v0.5_to_v0.6.md`)

- **Fixed ASS effects to enforce exactly 1 effect per video**
  - All presets now use exactly 1 effect (or none for minimal)
  - Random preset selects exactly 1 effect from available effects
  - Removed multi-effect violations per REQUIREMENTS.md

### Added
- **Unified Subtitle Configuration System**
  - Anchor-based positioning with 5 options: `top`, `center`, `bottom`, `above_content`, `below_content`
  - Content-aware positioning via `content_aware` boolean flag
  - 5 style presets: `minimal`, `modern`, `bold`, `animated`, `random`
  - Single configuration interface replaces complex multi-mode system

- **Enhanced Configuration Validation**
  - Effect count validation (enforces max 1 effect per video)
  - Preset-specific validation rules
  - Improved error messages with migration guidance

- **Documentation**
  - `MIGRATION_GUIDE_v0.5_to_v0.6.md`: Step-by-step migration from legacy to unified system
  - Updated `REQUIREMENTS.md` with three-tier configuration precedence
  - Enhanced inline documentation in `config/subtitles.yaml`

### Changed
- **Subtitle Positioning Logic**
  - Unified `_select_effects()` method enforces 1-effect rule
  - Consistent effect application through `self._selected_effects`
  - Removed legacy conversion function `convert_legacy_config()`
  - Replaced with `create_unified_config_from_settings()`

- **Configuration Structure**
  - Absolute mode: `anchor` + `margin` + `content_aware=false`
  - Relative mode: `anchor` + `margin` + `content_aware=true`
  - Simplified preset definitions with explicit effect mapping

### Fixed
- **ASS Effects Violation**: Now enforces exactly 1 effect per video per REQUIREMENTS.md
  - minimal: 0 effects
  - modern: karaoke only
  - bold: fade only
  - animated: movement only
  - random: 1 randomly selected effect

- **Code Simplification**: Removed 150+ lines of legacy code
  - Removed `_add_legacy_structure()` from config adapter
  - Removed legacy result helper functions
  - Simplified validation logic

### Migration
See [`MIGRATION_GUIDE_v0.5_to_v0.6.md`](MIGRATION_GUIDE_v0.5_to_v0.6.md) for complete instructions.

## [0.5.0] - 2025-10-02

### Added
- **Centralized Configuration System**: Media validation settings now centrally managed in config files
  - Added `min_total_media`, `min_images_if_no_video`, `min_images_with_video` to scraper and producer configs
  - Cross-referenced settings between `config/scraper.yaml` and `config/video_production.yaml`
  - Added test verification for config alignment (`test_media_validation_aligns_with_producer`)
- **Enhanced Progress Logging**: Improved scraping visibility with product-level progress tracking
  - Log full ASIN and product title for each scraped product
  - Progress indicators: "Processing product X/Y", "Extracting images/videos for {ASIN}"
  - INFO-level logging for non-debug visibility
- **Centralized Logging Utility**: New `src/utils/logging_setup.py` module provides standardized logging configuration
- **Configuration Audit**: Comprehensive `CONFIG_AUDIT.md` documenting hardcoded values, unused settings, and improvement roadmap
- **Debug Documentation**: Expanded TROUBLESHOOTING.md with debug files reference table and configuration guidance

### Changed
- **Browser Image Display**: Enabled images in browser window (changed `block_images: False`)
- **Media Validation Architecture**: Producer-aligned validation requirements
  - Scraper now validates same requirements as video producer (3 total, 5 images for slideshow, 2 for video mode)
  - Moved hardcoded validation thresholds to configuration files
  - Updated `validate_media_requirements()` to accept config parameter
- **Import Organization**: Reorganized module imports to comply with linting standards (imports at top of file)
- **Debug Mode Logging**: Eliminated 60+ lines of duplicated logging setup code between producer and scraper
- **FFmpeg Logging Logic**: Simplified `_should_create_ffmpeg_logs()` method with clearer fallback behavior

### Fixed
- **Websocket Error Suppression**: Properly suppressed harmless "goodbye" cleanup messages
  - Set `propagate=False` on websocket logger to prevent error propagation
  - Errors no longer appear in console output
- **Linting Issues**: Fixed all Ruff, MyPy, and code quality violations
  - Line length compliance (88 characters)
  - Try-except-pass logging (added debug messages)
  - Docstring completeness for function parameters
  - Type annotations for all functions
- **Headless Mode Issues**: Fixed browser initialization and tab creation bugs
- **Test Suite**: Updated 23 configuration tests to use new centralized settings
- **Logging Configuration**: Producer and scraper now use shared `setup_debug_logging()` function

### Technical
- **Breaking Change**: Configuration structure updated - media validation settings moved from hardcoded values to config files
- **Code Quality**: All linting checks passing (Ruff, MyPy, Bandit, Vulture, Safety)
- **Test Coverage**: 480 tests collected, 470 passing, 41% coverage maintained
- **Configuration Synchronization**: Automated test ensures scraper and producer configs stay aligned

## [0.4.0] - 2025-10-01

### Added
- **Unified Configuration System**: Modular YAML architecture with 6 specialized config files (core, video_production, ai_services, subtitles, performance, scraper)
- **Triple Precedence Configuration**: CLI arguments override environment variables override YAML defaults
- **CLI Configuration Overrides**: Command-line parameters can override any YAML configuration value
- **Environment Variable Support**: All configuration settings can be set via environment variables
- **Configuration Validation**: Enhanced validation with Pydantic models and clear error messages
- **Backward Compatibility Layer**: Adapter classes maintain 100% compatibility with existing code

### Changed
- **Configuration Architecture**: Split monolithic `config/video_producer.yaml` into 6 modular files
- **Complexity Reduction**: 54% reduction in configuration complexity (1,962 → 1,047 lines)
- **Performance Improvement**: 20% faster configuration loading through lazy loading and better caching
- **Documentation Overhaul**: Completely rewritten CONFIGURATION.md with modular system guide
- **Architecture Documentation**: Updated ARCHITECTURE.md with configuration system overview
- **Project Documentation**: Streamlined README.md and consolidated STATUS.md content

### Technical
- **Modular Loading**: Independent loading of configuration modules with dependency resolution
- **Memory Optimization**: Reduced memory footprint through lazy configuration loading
- **Configuration Caching**: Improved caching of parsed configuration values
- **Test Coverage**: Enhanced test suite with configuration validation tests (424 tests maintained)
- **Zero Breaking Changes**: All existing function signatures preserved through adapter pattern

## [0.3.1] - 2025-09-23

### Added
- **RANDOM Preset**: New style preset with deterministic randomization using product-specific seeding for fonts, colors, and single animation effects
- **CLI Style Override**: Added `--preset` command-line argument for easy video styling control (minimal, modern, bold, random)
- **Enhanced Randomization**: Improved font and color randomization system with better effect selection

### Changed
- **Optimized Preset System**: Reduced preset count from 5 to 4 (removed `animated` and `classic`, kept `minimal`, `modern`, `bold`)
- **Effect Limitation**: Limited effects to 1 per preset to prevent visual clutter and rendering issues
- **Improved Documentation**: Updated README.md for simplicity with collapsible sections

### Fixed
- **ASS Effects Application**: Fixed ASS effects not applying by changing condition from >1 to >0 effects
- **Random Effect Selection**: Enabled randomize_effects for RANDOM preset to activate effect system properly
- **Configuration Alignment**: Updated all documentation to match actual 4-preset codebase implementation

### Technical
- **Deterministic Randomization**: RANDOM preset uses product ID-based seeding for consistent per-video styling
- **CLI Integration**: Producer now accepts preset override parameter for flexible styling
- **Test Coverage**: Updated comprehensive test suite to reflect new preset system (424 tests across 27 files)
- **Code Quality**: All quality gates pass with optimized preset system implementation

## [0.3.0] - 2025-09-21

### Added
- **Font and Color Randomization System**: New comprehensive deterministic randomization system for subtitle fonts and colors
- **New Font Manager**: Added `font_color_manager.py` module for centralized font and color management
- **Product-Specific Seeding**: Deterministic font/color selection based on product ID for consistent results
- **Enhanced Subtitle Configuration**: New subtitle settings with font/color randomization options
- **Comprehensive Test Coverage**: Added new test suites for subtitle validation and unified subtitle generation

### Changed
- **Code Quality Improvements**: Fixed 18 linting issues across 6 core files for better maintainability
- **Type Annotations**: Enhanced type checking with proper annotations and MyPy compliance
- **Security Compliance**: Added proper security warning suppressions for non-cryptographic randomization
- **Configuration Enhancement**: Updated video producer configuration with new subtitle randomization options
- **Documentation Updates**: Updated architecture and testing documentation

### Fixed
- **Line Length Issues**: Fixed E501 violations by splitting long debug messages across multiple lines
- **Import Sorting**: Resolved I001 violations with proper import organization
- **Docstring Issues**: Fixed missing parameter descriptions and formatting issues
- **Type Checking**: Resolved MyPy errors with proper SubtitleSettings object usage
- **Constructor Parameters**: Added missing optional parameters to UnifiedSubtitleConfig

### Technical
- **Subtitle Pipeline**: Enhanced subtitle generation with randomization capabilities
- **Performance Monitoring**: Maintained consistent pipeline performance (232-283 seconds)
- **Testing Framework**: All 413 tests pass with improved coverage
- **Code Standards**: Achieved compliance with Ruff, MyPy, Bandit, Vulture, and Safety tools

## [0.2.1] - 2025-09-20

### Fixed
- **Missing Pipeline Step**: Added missing `generate_description` step to pipeline execution - description generation was completely skipped despite having all the code
- **Critical Path Resolution**: Fixed description generator failing due to relative path issues when run from different working directories
- **Producer Cleanup**: Fixed missing `description.txt` and erroneous directories (`~`, `outputs`) in cleanup process with `--clean` flag
- **Whisper Model Caching**: Fixed literal `~` directory creation by properly expanding home directory path with `os.path.expanduser()`
- **Pipeline Reliability**: Ensured producer works correctly regardless of current working directory

### Changed
- Enhanced producer cleanup to remove all temporary and generated files consistently
- Improved path handling throughout the pipeline for better portability
- Updated test documentation to reflect current structure (365 tests across 23 files)
- Updated project status documentation with current capabilities and fixes

### Technical
- Added `generate_description` step to pipeline graph with proper dependency on `generate_script` step
- Made description generator use absolute paths for template loading
- Added proper home directory expansion in Whisper model configuration
- Enhanced producer file cleanup logic with comprehensive file removal
- Improved error handling and path resolution across multiple modules

## [0.2.0] - 2025-09-20

### Added
- **AI-Generated Video Descriptions**: New feature for generating social media descriptions using LLM providers
- New `description_generator.py` module with template-based prompt formatting and hashtag validation
- `DescriptionSettings` configuration class with platform targeting and validation options
- Social media compliance with required #ad hashtag for advertising disclosure
- Integration with video producer pipeline as new `STEP_GENERATE_DESCRIPTION` step
- Comprehensive test suite for description generation functionality

### Changed
- Extended video producer pipeline to include description generation step
- Updated configuration schema to include `description_settings` section
- Enhanced product files structure to include `description.txt` output
- Updated all test fixtures to support new configuration requirements

### Technical
- Added circuit breaker pattern for API resilience in description generation
- Implemented async/await patterns following existing LLM integration standards
- Added Pydantic validation for description settings and content quality
- Extended configuration loading to validate new description settings

## [0.1.2] - 2025-09-18

### Fixed
- Fixed CI test failures by adding FFmpeg to release workflow
- Resolved FFmpeg dependency validation issues in test environment
- Fixed media validator test error message expectations
- Improved test reliability in CI environments

### Changed
- Enhanced subtitle positioning system with improved style presets
- Renamed DYNAMIC subtitle preset to RELATIVE for better clarity
- Added font_width_to_height_ratio configuration to all subtitle style presets
- Updated video producer configuration with enhanced subtitle settings

### Technical
- Added FFmpeg installation to GitHub Actions release workflow
- Improved CI/CD pipeline reliability and test coverage
- Enhanced configuration validation for production environments

## [0.1.1] - 2025-09-17

### Fixed
- Resolved all CI linting and type checking issues
- Fixed MyPy type annotation errors in media validator and assembler modules
- Updated test expectations to match implementation changes
- Fixed hardcoded path issues in test files for better portability
- Improved code style compliance with 88-character line limit

### Changed
- Enhanced debug logging and error handling in assembler module
- Improved test reliability with proper mock configurations

### Technical
- All quality gates now pass: Ruff, MyPy, Bandit, Vulture, Safety, pytest
- GitHub Actions CI pipeline fully functional
- Enhanced type safety and code maintainability

## [0.1.0] - Initial Release

### Added
- Initial open source release
- Complete AI video production pipeline for e-commerce products
- Amazon product scraper with configurable search parameters
- Multi-provider AI service support (OpenRouter, Google Cloud, OpenAI)
- Professional video assembly with FFmpeg
- Audio-synchronized subtitle generation
- Background music integration
- Batch processing capabilities
- Performance monitoring and optimization framework
- Comprehensive test suite with 280+ test cases
- Modular, extensible architecture supporting future platforms

### Technical Features
- **Pipeline Processing**: 6-step modular pipeline with parallel execution
- **Multi-Provider Support**: Fallback mechanisms for reliability
- **Configuration Management**: 100+ customizable parameters via YAML
- **Output Management**: Clean, product-centric directory structure
- **Code Quality**: Comprehensive linting, type checking, and security scanning
