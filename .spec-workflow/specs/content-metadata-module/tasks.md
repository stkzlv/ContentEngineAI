# Tasks Document: Content Metadata Module

## Implementation Status

The Content Metadata Module is **fully implemented** and production-ready. All core requirements are satisfied including platform-specific metadata generation for YouTube, TikTok, and Instagram, LLM-powered generation with prompt templates, validation, and pipeline integration. These tasks reflect the completed implementation and potential enhancements.

## Tasks

### Section 1: Foundation (Completed)

- [x] 1. Data Models
  - File: src/ai/platform_metadata/models.py
  - Implemented: PlatformMetadata dataclass (frozen, immutable)
  - Pydantic settings: YouTubePlatformSettings, TikTokPlatformSettings, InstagramPlatformSettings
  - Top-level PlatformMetadataSettings with target_platform field
  - _Requirements: 1, 5, 9_

- [x] 2. Configuration Schema
  - File: src/video/config/core_models.py, config/video.yaml
  - Implemented: platform_metadata section in VideoConfig
  - Per-platform settings with defaults
  - YAML configuration with inline documentation
  - _Requirements: 9_

- [x] 3. Package Structure
  - File: src/ai/platform_metadata/__init__.py
  - Implemented: Package exports and factory registration
  - Clean import interface for generators
  - _Requirements: 1_

### Section 2: Core Generators (Completed)

- [x] 4. Abstract Base Generator
  - File: src/ai/platform_metadata/base.py
  - Implemented: BasePlatformMetadataGenerator ABC
  - Abstract methods: generate(), validate(), platform_name property
  - Shared utilities: _load_prompt_template(), _format_product_prompt(), _truncate_if_needed()
  - _Requirements: 1, 6_

- [x] 5. Generator Utilities
  - File: src/ai/platform_metadata/utilities.py
  - Implemented: LLM response parsing, hashtag extraction
  - Character counting and truncation helpers
  - _Requirements: 5, 7_

- [x] 6. Text Formatter
  - File: src/ai/platform_metadata/text_formatter.py
  - Implemented: Text formatting utilities for metadata
  - _Requirements: 5_

### Section 3: Platform Generators (Completed)

- [x] 7. YouTube Metadata Generator
  - File: src/ai/platform_metadata/youtube.py
  - Implemented: YouTubeMetadataGenerator with title optimization
  - Title 50-60 chars, description ≤5000 chars, 3-5 hashtags
  - #Shorts tag for vertical videos, SEO keywords
  - _Requirements: 2_

- [x] 8. TikTok Metadata Generator
  - File: src/ai/platform_metadata/tiktok.py
  - Implemented: TikTokMetadataGenerator with SEO-focused captions
  - Caption 100-300 chars optimal, 3-5 niche hashtags
  - Avoids generic tags (#fyp, #viral, #foryoupage)
  - _Requirements: 3_

- [x] 9. Instagram Metadata Generator
  - File: src/ai/platform_metadata/instagram.py
  - Implemented: InstagramMetadataGenerator with dual caption styles
  - Short (3-5 words) or SEO (100-200 chars) mode
  - 15-30 hashtags, emoji support
  - _Requirements: 4_

### Section 4: Factory & Orchestration (Completed)

- [x] 10. Platform Metadata Factory
  - File: src/ai/platform_metadata/__init__.py
  - Implemented: PlatformMetadataFactory with get_generator()
  - Generator registration and lookup by platform name
  - generate_for_platforms() for multi-platform generation
  - _Requirements: 1, 8_

- [x] 11. FTC Compliance
  - Files: All generators
  - Implemented: Automatic #ad hashtag inclusion
  - Validation ensures #ad present in all outputs
  - _Requirements: 1, 7_

### Section 5: Prompt Templates (Completed)

- [x] 12. YouTube Prompt Template
  - File: src/ai/prompts/youtube_metadata.md
  - Implemented: YouTube Shorts optimization prompt
  - Includes SEO guidelines, character limits, output format
  - _Requirements: 2, 6_

- [x] 13. TikTok Prompt Template
  - File: src/ai/prompts/tiktok_metadata.md
  - Implemented: TikTok SEO caption prompt
  - Niche hashtag guidance, avoid generic tags
  - _Requirements: 3, 6_

- [x] 14. Instagram Prompt Template
  - File: src/ai/prompts/instagram_metadata.md
  - Implemented: Instagram Reels optimization prompt
  - Dual caption style support, extensive hashtags
  - _Requirements: 4, 6_

### Section 6: Pipeline Integration (Completed)

- [x] 15. Video Producer Integration
  - File: src/video/producer/steps.py
  - Implemented: Metadata generation in step_generate_description()
  - Conditional generation based on platform_metadata.enabled
  - Returns platform metadata alongside description
  - _Requirements: 10_

### Section 7: Testing (Completed)

- [x] 16. Unit Tests - Models
  - File: tests/ai/platform_metadata/test_models.py
  - Tested: PlatformMetadata creation, serialization, character counts
  - Pydantic settings validation
  - _Requirements: 1, 5_

- [x] 17. Unit Tests - Generators
  - File: tests/ai/platform_metadata/test_generators.py
  - Tested: Each generator with mocked LLM responses
  - Validation logic for all platforms
  - _Requirements: 2, 3, 4, 7_

- [x] 18. Unit Tests - Factory
  - File: tests/ai/platform_metadata/test_factory.py
  - Tested: Generator lookup, multi-platform generation
  - Factory pattern correctness
  - _Requirements: 1_

- [x] 19. Integration Tests
  - File: tests/ai/platform_metadata/test_integration.py
  - Tested: Full generation flow with mock LLM
  - Pipeline integration
  - _Requirements: 1-10_

### Section 8: Documentation (Completed)

- [x] 20. Code Documentation
  - Files: All module files
  - Implemented: Comprehensive docstrings
  - Type hints throughout
  - _Requirements: All_

## Enhancement Tasks

- [x] 21. Add metadata caching
  - File: src/ai/platform_metadata/cache.py (new)
  - Cache generated metadata by product_id + platform
  - Configurable TTL for cache entries
  - Purpose: Avoid regenerating metadata for unchanged products
  - _Leverage: src/ai/platform_metadata/models.py_
  - _Requirements: 1_
  - _Prompt: Role: Python Developer | Task: Add metadata caching layer: cache PlatformMetadata by (product_id, platform) key, configurable TTL, invalidation on product change | Restrictions: Use file-based cache for persistence, handle cache corruption gracefully | Success: Repeated generation requests use cached metadata_
  - **Implementation:**
    - `MetadataCacheSettings` Pydantic model in `models.py` with `enabled`, `ttl_hours`, `cache_dir`, `max_entries`
    - `MetadataCache` class in `cache.py` with file-based JSON storage
    - `CacheEntry` dataclass with TTL expiration and product hash validation
    - Integrated with `PlatformMetadataFactory.generate_multi_platform()` via optional `cache` parameter
    - Graceful corruption handling (removes invalid cache files)
    - 25 unit tests in `tests/ai/test_metadata_cache.py`

- [x] 22. Add A/B testing support for prompts
  - File: src/ai/platform_metadata/ab_testing.py (new)
  - Support multiple prompt variants per platform
  - Track which variant produced metadata
  - Purpose: Enable prompt optimization through experimentation
  - _Leverage: src/ai/prompts/_
  - _Requirements: 6_
  - _Prompt: Role: Python Developer | Task: Add A/B testing for prompt templates: load variant based on config, track variant in metadata, support multiple variants per platform | Restrictions: Deterministic variant selection for reproducibility, log variant used | Success: Can compare metadata quality across prompt variants_
  - **Implementation:**
    - `ABTestingSettings`, `PlatformABConfig`, `PromptVariant` Pydantic models
    - `PromptVariantSelector` class with deterministic hash-based selection
    - `VariantSelection` dataclass for tracking selected variant
    - Added `prompt_variant` field to `PlatformMetadata` model
    - Weighted variant selection for traffic splitting (e.g., 80/20)
    - Configuration in `ai_services.yaml` under `platform_metadata_config.ab_testing`
    - 25 unit tests in `tests/ai/test_ab_testing.py`

- [ ] 23. Add batch metadata generation
  - File: src/ai/platform_metadata/__init__.py (modify)
  - Generate metadata for multiple products in parallel
  - Progress tracking with [N/total] format
  - Purpose: Efficient metadata generation for batch video production
  - _Leverage: asyncio.gather_
  - _Requirements: 1, 6_
  - _Prompt: Role: Python Developer | Task: Add batch metadata generation: generate for multiple products concurrently, track progress, aggregate results | Restrictions: Respect rate limits, maintain per-product error isolation | Success: Batch generation completes faster than sequential_

- [ ] 24. Add metadata export formats
  - File: src/ai/platform_metadata/export.py (new)
  - Export metadata in CSV, JSON, and platform-specific formats
  - Support bulk export for analytics
  - Purpose: Enable external analysis and platform import
  - _Leverage: src/ai/platform_metadata/models.py_
  - _Requirements: 7_
  - _Prompt: Role: Python Developer | Task: Add metadata export: JSON (default), CSV for spreadsheet analysis, platform-specific formats (YouTube CSV, TikTok format) | Restrictions: Maintain data fidelity, handle encoding correctly | Success: Exported metadata importable by target platforms_

- [ ] 25. Add trend-aware hashtag generation
  - File: src/ai/platform_metadata/trends.py (new)
  - Integrate with trend APIs to suggest current hashtags
  - Platform-specific trending tag lookup
  - Purpose: Improve discoverability with trending hashtags
  - _Leverage: External trend APIs_
  - _Requirements: 3, 4_
  - _Prompt: Role: Python Developer | Task: Add trend-aware hashtags: fetch trending tags from platform APIs or third-party services, merge with generated hashtags | Restrictions: Cache trend data, handle API failures gracefully, don't replace all hashtags | Success: Generated metadata includes relevant trending hashtags_

## Testing Checklist

All tests verified and passing:

- [x] 26.1 YouTube metadata generation produces valid output
- [x] 26.2 TikTok metadata generation produces valid output
- [x] 26.3 Instagram metadata generation produces valid output
- [x] 26.4 Character limits enforced for all platforms
- [x] 26.5 #ad hashtag included in all outputs
- [x] 26.6 #Shorts hashtag included for YouTube vertical videos
- [x] 26.7 TikTok avoids generic hashtags
- [x] 26.8 Instagram hashtag count within 15-30 range
- [x] 26.9 Factory returns correct generator per platform
- [x] 26.10 Multi-platform generation works correctly
- [x] 26.11 Validation catches character limit violations
- [x] 26.12 Validation catches missing #ad hashtag
- [x] 26.13 Truncation with ellipsis works correctly
- [x] 26.14 LLM retry logic handles transient failures
- [x] 26.15 Pipeline integration generates metadata when enabled
- [x] 26.16 Unified mode generates single metadata set
- [x] 26.17 Configuration loading from YAML works
- [x] 26.18 Platform disable skips generation
- [x] 26.19 All unit tests pass with good coverage
- [x] 26.20 All integration tests pass reliably
