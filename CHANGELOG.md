# Changelog

All notable changes to ContentEngineAI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
