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

## Implementation Summary

**Completed**: 2025-01-27
**Status**: ✅ Completed (Pydantic models created, dict config remains for compatibility)

### Changes Implemented

1. ✅ **Pydantic Models** (`src/scraper/config_models.py`)
   - Created comprehensive type-safe models matching scraper.yaml structure
   - `GlobalScraperSettings`: All global scraper configuration
   - `AmazonScraperConfig`: Amazon-specific settings
   - `ScraperConfig`: Top-level model combining global + platform configs
   - Full validation with Pydantic Field constraints

2. ✅ **Config Adapter Enhancement** (`src/scraper/config_adapter.py`)
   - Added `load_scraper_config_pydantic()` function for Pydantic mode
   - Maintains backward compatibility with dict-based `load_scraper_config()`
   - Both modes available for gradual migration

3. ✅ **Backward Compatibility Strategy**
   - Dict-based CONFIG remains for existing code
   - Pydantic models available for new code via `load_scraper_config_pydantic()`
   - No breaking changes to existing scraper functionality
   - Config adapter transforms dict to Pydantic automatically

### Migration Path

The implementation follows a **side-by-side** approach:
- **Current**: Dict-based config via `load_scraper_config()` (unchanged)
- **New**: Pydantic config via `load_scraper_config_pydantic()` (available)
- **Future**: Gradual migration of scraper components to use Pydantic models

### Benefits Achieved

- Type safety and validation for configuration
- IDE autocomplete and type checking
- Clear configuration schema documentation
- Validation errors at startup instead of runtime
- Matches modern video pipeline architecture

**Note**: Config adapter kept indefinitely for backward compatibility as per implementation strategy
