# Tasks Document: Video Producer Module

## Implementation Status

The Video Producer Module is **fully implemented** and production-ready. All core requirements are satisfied including video assembly strategies, subtitle system, Freesound integration, and batch processing. These tasks address enhancements and ensure comprehensive test coverage.

## Tasks

- [x] 1. Video Assembly Strategy Implementation
  - Files: src/video/assembler/video_strategies.py
  - Implemented: VideoAssemblyStrategy ABC, SequentialStrategy, SingleBestStrategy, MixedMediaStrategy, VideoFirstFallbackStrategy
  - VideoStrategyFactory provides strategy instantiation
  - _Requirements: 1, 2, 3, 4, 5, 6_

- [x] 2. Freesound Client with OAuth2
  - File: src/audio/freesound_client.py (708 lines)
  - Implemented: OAuth2 with token refresh, circuit breaker protection, local fallback
  - Three-tier reliability: OAuth2 HQ → API key preview → local files
  - _Requirements: 11, 12, 13, 14, 15, 16_

- [x] 3. Subtitle System with ASS Effects
  - Files: src/video/config/subtitle_models.py, src/video/subtitle/
  - Implemented: Anchor-based positioning, two-part subtitles, style presets
  - ASS effects: karaoke, fade, typewriter, glow, bounce, pulse
  - _Requirements: 7, 8, 9, 10_

- [x] 4. Batch Processing with Profile Randomization
  - File: src/video/producer/cli.py (batch discovery, progress tracking)
  - File: src/video/producer/utils.py (profile selection, usage tracking)
  - Implemented: Product discovery, deterministic randomization, fail-fast mode
  - _Requirements: 17, 18, 19, 20, 21, 22_

- [x] 5. Profile System
  - Files: src/video/config/visual_models.py, src/video/config/core_models.py
  - Implemented: VideoProfile with per-profile overrides
  - CLI > Profile > Global Config precedence
  - _Requirements: 23, 24_

- [x] 6. Pipeline Steps
  - File: src/video/producer/steps.py (54KB)
  - Implemented: gather_visuals, generate_script, create_voiceover, generate_subtitles, download_music, assemble_video, apply_subtitles
  - Step resumability with artifact loading
  - _Requirements: 25, 26_

## Enhancement Tasks

- [ ] 7. Add unit tests for video assembly strategies
  - File: tests/video/test_video_strategies.py (new)
  - Test all four assembly strategies with edge cases
  - Test duration matching, fallback behavior, transition handling
  - Purpose: Ensure assembly strategies work correctly
  - _Leverage: tests/conftest.py, unittest.mock_
  - _Requirements: 1, 2, 3, 4, 5, 6_
  - _Prompt: Role: QA Engineer | Task: Create comprehensive unit tests for VideoAssemblyStrategy classes covering: SequentialStrategy with 0/1/many videos, SingleBestStrategy duration handling, MixedMediaStrategy interleaving, VideoFirstFallbackStrategy cascading | Restrictions: Mock FFmpeg calls, use pytest parametrize, test edge cases (empty inputs, single item) | Success: 100% coverage of video_strategies.py_

- [ ] 8. Add unit tests for Freesound client
  - File: tests/audio/test_freesound_client.py (new)
  - Test OAuth2 token refresh, circuit breaker behavior
  - Test search with duration filtering, download fallback chain
  - Purpose: Ensure Freesound integration is resilient
  - _Leverage: tests/conftest.py, aioresponses for async mocking_
  - _Requirements: 11, 12, 13, 14, 15_
  - _Prompt: Role: QA Engineer | Task: Create unit tests for FreesoundClient covering: OAuth2 token refresh (success, expired, failure), circuit breaker state transitions, search with duration matching, download fallback chain (OAuth2 → API key → local) | Restrictions: Use aioresponses for HTTP mocking, test async methods properly, maintain isolation | Success: All OAuth2 and circuit breaker scenarios tested_

- [ ] 9. Add unit tests for subtitle positioning
  - File: tests/video/test_subtitle_positioning.py (new)
  - Test anchor positions, content-aware positioning
  - Test two-part subtitle configuration
  - Purpose: Ensure subtitle positioning works correctly
  - _Leverage: tests/conftest.py_
  - _Requirements: 7, 8_
  - _Prompt: Role: QA Engineer | Task: Create unit tests for UnifiedSubtitleConfig and positioning logic covering: all anchor positions (top, center, bottom, above/below_content), margin calculations, content-aware positioning with visual bounds, two-part subtitle upper/lower line positioning | Restrictions: Test edge cases (zero margin, full-frame content), use pytest fixtures | Success: All positioning scenarios tested_

- [ ] 10. Add unit tests for batch processing
  - File: tests/video/test_batch_producer.py (new)
  - Test product discovery, profile randomization
  - Test fail-fast behavior, progress tracking
  - Purpose: Ensure batch processing works correctly
  - _Leverage: tests/conftest.py, tempfile for outputs directory_
  - _Requirements: 17, 18, 19, 20, 21_
  - _Prompt: Role: QA Engineer | Task: Create unit tests for batch processing covering: product discovery (valid/invalid data.json, skip directories), profile randomization determinism (same product → same profile), ProfileUsageTracker statistics, fail-fast vs graceful degradation | Restrictions: Use temp directories, mock process_product calls, test progress format | Success: All batch scenarios tested_

- [ ] 11. Add integration test for full video production pipeline
  - File: tests/integration/test_producer_integration.py (new)
  - Test complete pipeline: data.json → final video
  - Use mock HTTP responses for external services
  - Purpose: Verify end-to-end production works correctly
  - _Leverage: tests/conftest.py, pytest-aiohttp_
  - _Requirements: 1-26 (full pipeline)_
  - _Prompt: Role: QA Engineer | Task: Create integration test that mocks external services (LLM, TTS, Freesound) and verifies: media gathering, script generation, voiceover creation, subtitle generation, music download, video assembly, subtitle application | Restrictions: Use temp output directory, mock all network calls, verify output video exists | Success: Full pipeline tested without real API calls_

- [ ] 12. Add ASS effect integration tests
  - File: tests/video/test_ass_effects.py (new)
  - Test karaoke timing, fade effects, style presets
  - Verify generated ASS syntax is valid
  - Purpose: Ensure ASS effects render correctly
  - _Leverage: tests/conftest.py, sample subtitle files_
  - _Requirements: 9, 10_
  - _Prompt: Role: QA Engineer | Task: Create tests for ASS effect generation covering: karaoke word timing (\kf tags), fade in/out (\fad tags), style preset application (minimal vs animated), typewriter character reveal | Restrictions: Validate ASS syntax, test with real subtitle text, verify timing calculations | Success: ASS output validates against spec_

- [ ] 13. Update docs/video-producer.md with comprehensive guide
  - File: docs/video-producer.md (new or modify)
  - Document all CLI options with examples
  - Add batch processing guide, profile configuration
  - Include troubleshooting section
  - Purpose: Provide complete producer usage reference
  - _Leverage: src/video/producer/cli.py for CLI options_
  - _Requirements: 17-22, 23-24_
  - _Prompt: Role: Technical Writer | Task: Create comprehensive video producer documentation with: 1) Quick start examples, 2) CLI reference table (60+ options), 3) Batch processing guide, 4) Profile configuration examples, 5) Assembly mode selection guide, 6) Troubleshooting section (TTS failures, music download issues) | Restrictions: Use existing doc style, keep examples runnable | Success: Users can use all producer features from docs alone_

- [ ] 14. Add profile validation at startup
  - File: src/video/producer/cli.py (modify)
  - Validate profile exists before batch processing starts
  - Validate profile-pool entries against available profiles
  - Purpose: Fail fast with clear error when profile invalid
  - _Leverage: src/video/config/core_models.py_
  - _Requirements: 24_
  - _Prompt: Role: Python Developer | Task: Add profile validation in batch mode startup: verify --batch-profile exists, verify all --profile-pool entries exist, fail with clear error listing available profiles | Restrictions: Validate before processing starts, provide helpful error message with profile suggestions | Success: Invalid profile causes immediate exit with clear message_

- [ ] 15. Add batch summary JSON output
  - File: src/video/producer/cli.py (modify)
  - Add --output-format=json option for machine-readable summary
  - Include profile distribution, timing, error details
  - Purpose: Enable programmatic batch result processing
  - _Leverage: existing BatchSummary dataclass_
  - _Requirements: 22_
  - _Prompt: Role: Python Developer | Task: Add --output-format argument (text/json) that outputs machine-readable batch summary including: total/succeeded/failed/skipped counts, profile distribution map, per-product timing, error messages for failures | Restrictions: Maintain backward compatibility (text default), use json.dumps with indent, include ISO timestamps | Success: JSON output parseable by downstream tools_
