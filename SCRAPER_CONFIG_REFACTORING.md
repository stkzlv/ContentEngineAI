# Scraper Configuration Refactoring (Deferred)

**Goal:** Unify the configuration system by migrating the scraper's legacy configuration to the modern, Pydantic-based system used by the video pipeline.

## Current State

- `src/scraper/config_adapter.py`: 194 lines, provides backward compatibility via `ScraperConfigAdapter`
- `src/scraper/amazon/config.py`: Uses global `CONFIG` dict from config adapter
- `src/config_manager.py`: Likely has unified config manager implementation
- `config/scraper.yaml`: Consolidated scraper configuration file

## Implementation Steps

1.  **Create Pydantic Models for Scraper Configuration:**
    *   Create `src/scraper/config_models.py` with Pydantic models:
        - `GlobalScraperSettings` (retry, browser, download config)
        - `AmazonScraperConfig` (Amazon-specific settings)
        - `ScraperConfig` (top-level model combining global + platform configs)
    *   Mirror structure from `src/video/video_config.py` for consistency

2.  **Update Scraper to Use Pydantic Models:**
    *   Refactor `src/scraper/amazon/scraper.py` to accept Pydantic config instead of dict
    *   Refactor `src/scraper/amazon/downloader.py` to use typed config
    *   Replace `CONFIG.get()` patterns with typed attribute access
    *   Update `src/scraper/amazon/config.py` to instantiate Pydantic models

3.  **Eliminate the Scraper Config Adapter:**
    *   Verify `src/config_manager.py` can load scraper config with Pydantic
    *   Add `load_scraper_config_pydantic()` method similar to `load_video_config_modular()`
    *   Deprecate `ScraperConfigAdapter` with warnings
    *   Eventually delete `src/scraper/config_adapter.py`

4.  **Update Configuration Files:**
    *   Verify `config/scraper.yaml` aligns with Pydantic model structure
    *   Add validation to catch config errors at startup

5.  **Testing:**
    *   Run `tests/test_scraper_config_enhanced.py` (23 tests)
    *   Verify all scraper tests pass with new config system
    *   Test CLI overrides work correctly with Pydantic models

## Risk Analysis

| Work Item | Risk | Notes |
|-----------|------|-------|
| Config Modernization | 🔴 High | Widespread impact, unclear migration path |

## Implementation Strategy

Create Pydantic models side-by-side with dict config. Keep `config_adapter.py` indefinitely for backward compatibility. Gradual migration with deprecation warnings.

## Implementation Timeline

**Estimated**: 5-7 days
**Status**: Deferred
**Priority**: Low - wait until async refactoring stabilizes
