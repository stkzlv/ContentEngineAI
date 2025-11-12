# Configuration Cleanup Plan

**Generated**: 2025-11-12
**Status**: Phase 1 Complete (Verified) - Phase 2 Ready
**Last Updated**: 2025-11-12

This document outlines all configuration improvements identified through comprehensive code analysis.

---

## Executive Summary

**Phase 1 Status (COMPLETED & VERIFIED):**
- ✅ 4 of 6 items successfully removed/cleaned
- ⚠️ 2 items kept (preserve_aspect_ratio IS used, attribution_settings required by Pydantic)
- 🔍 Verification report: docs/CONFIG_CLEANUP_VERIFICATION.md

**Issues Found:**
- 15 unused configuration settings (4 removed, 2 kept for valid reasons)
- 5 groups of duplicate settings (all removed)
- 20+ magic numbers hardcoded in source files (Phase 2)
- Missing documentation for critical settings (Phase 4)

**Impact So Far:**
- ✅ ~100 lines of unused config removed
- ✅ Configuration file complexity reduced
- ✅ Clear NOTE comments explain what's in scraper.yaml vs video_production.yaml
- ⏳ Phase 2: Add config settings for magic numbers
- ⏳ Phase 3: Update code to use config values
- ⏳ Phase 4: Add comprehensive documentation

---

## Phase 1: Remove Unused Settings (Safe)

### video_production.yaml

#### 1. ✅ Remove Media Quality Settings (Lines 304-320) - COMPLETED
**Reason**: Never referenced in codebase, duplicates scraper.yaml settings
**Status**: Replaced with NOTE comment pointing to scraper.yaml

```yaml
# COMPLETED - Now shows:
# NOTE: max_images_per_product and max_videos_per_product are configured in scraper.yaml
# NOTE: image_quality and video_quality settings are configured in scraper.yaml
```

**Important**: max_images_per_product and max_videos_per_product ARE used in `src/scraper/amazon/media_extractor.py` but they exist in `config/scraper.yaml` (correct location), not video_production.yaml. No deletion needed.

#### 2. ✅ Remove Stock Media Enabled Flag (Line 326) - COMPLETED
**Reason**: Stock media controlled by profile `use_stock_*` flags
**Status**: Replaced with NOTE comment

```yaml
# COMPLETED - Now shows:
stock_media_settings:
  # NOTE: Stock media usage is controlled by profile flags: use_stock_images, use_stock_videos
  # No global "enabled" flag needed - each profile specifies which media types to use
  pexels_api_key_env_var: "PEXELS_API_KEY"
```

#### 3. ⚠️ Remove Base Profile Video Settings (Lines 342-348) - PARTIALLY COMPLETED
**Reason**: Never used, transition effects controlled by `video_transition_duration` in profiles
**Status**: Most settings removed, but `preserve_aspect_ratio` kept (it IS used)

```yaml
# COMPLETED - Now shows:
base:
  description: "Base profile with common settings for all video types."
  # NOTE: Video settings (transitions, aspect ratio) controlled by profile-level config
  # NOTE: Audio normalization handled automatically by pipeline
  # NOTE: Background music controlled by audio_settings.background_music section
  # NOTE: Subtitle settings controlled by profile-level subtitle_* config
```

**Important**: `preserve_aspect_ratio` remains at line 82 in video_settings section - it IS actively used in profile merging logic (video_config.py lines 1234, 1436-1438). This is correct and should NOT be removed.

#### 4. ✅ Remove Duplicate Audio Settings (Lines 156-163, 278-280) - COMPLETED
**Reason**: Duplicated by `output_audio_*` settings which are actually used
**Status**: Removed and replaced with NOTE comments

```yaml
# COMPLETED - Lines 155-179 now show:
audio_settings:
  # NOTE: Audio encoding settings (sample_rate, channels, bitrate) are handled automatically by FFmpeg
  # Output format is controlled by output_audio_codec and output_audio_bitrate below

  music_volume_db: -24.0
  voiceover_volume_db: 3.0
  audio_mix_duration: "longest"
  output_audio_codec: "aac"
  output_audio_bitrate: "192k"
  music_fade_in_duration: 2.0  # ✓ KEPT - actively used
  music_fade_out_duration: 3.0  # ✓ KEPT - actively used

# Lines 274-275 now show:
background_music:
  enabled: true
  # NOTE: fade_in_duration and fade_out_duration are configured above (lines 170-171)
  # NOTE: Background music always loops to match video duration
```

#### 5. ⚠️ Attribution Settings (Lines 676-679) - KEPT AS MINIMAL STUB
**Reason**: Feature not implemented, but required by Pydantic VideoConfig model
**Status**: Kept minimal stub for validation, cannot be fully removed

```yaml
# CURRENT STATE - Lines 676-679:
# NOTE: Attribution feature not implemented, minimal config for Pydantic validation
attribution_settings:
  enabled: false
  attribution_template: ""
  attribution_entry_template: ""
```

**Important**: AttributionSettings is part of VideoConfig Pydantic model (video_config.py:341-345). Removing from YAML causes validation errors. To fully remove, Pydantic model must be updated to make this field Optional.

#### 6. ✅ Remove Temp Subtitle Filename (Line 28) - COMPLETED
**Reason**: Never used, system uses `subtitles.srt`, `subtitles.ass`, etc.
**Status**: Removed and replaced with NOTE comment

```yaml
# COMPLETED - Line 28 now shows:
# NOTE: temp_subtitle removed - system uses subtitles.srt, subtitles.ass, etc. directly
```

---

## Phase 2: Add Missing Configuration Settings

### video_production.yaml - New Sections

Add these sections to make hardcoded values configurable:

```yaml
# ============================================================================
# FORMAT NORMALIZATION SETTINGS
# ============================================================================
# Controls video format normalization to ensure compatibility
format_normalization:
  # Target frame rate for all videos
  target_fps: 30.0

  # FPS tolerance for determining if re-encoding is needed (0.1 = 10% tolerance)
  fps_tolerance: 0.1

  # Frame rate string format for FFmpeg
  default_fps_string: "30/1"

  # Target video codec (H.264 for maximum compatibility)
  target_codec: "H.264"

  # Target pixel format
  target_pixel_format: "yuv420p"

# ============================================================================
# ASPECT RATIO HANDLING
# ============================================================================
# Controls how aspect ratio differences are handled in smart-scale mode
aspect_ratio:
  # Tolerance for aspect ratio similarity (0.10 = 10%)
  # If content aspect ratio is within 10% of target, use letterbox
  # Otherwise use crop to fill frame
  smart_scale_tolerance: 0.10

# ============================================================================
# TIMING CONSTRAINTS
# ============================================================================
# Minimum duration constraints for video segments
timing:
  # Minimum segment duration in seconds (prevents too-short clips)
  min_segment_duration_sec: 0.1

  # Minimum safe trim duration (provides buffer for timing accuracy)
  min_safe_trim_duration_sec: 0.5

# ============================================================================
# ENHANCED FREESOUND CONFIGURATION
# ============================================================================
# Add these under existing audio_settings:
audio_settings:
  # ... existing settings ...

  # Token refresh configuration
  freesound_token_refresh:
    # Maximum retry attempts for token refresh
    max_retries: 2

    # Timeout for token refresh requests (seconds)
    timeout_sec: 5

    # Exponential backoff configuration
    backoff_base_delay_sec: 0.5
    backoff_multiplier: 2.0

  # Download retry configuration
  freesound_download:
    # Maximum retry attempts for downloads
    max_retries: 2

    # Exponential backoff configuration
    backoff_base_delay_sec: 1.0
    backoff_multiplier: 2.0
```

### subtitles.yaml - New Settings

Add these to the `text_rendering` section:

```yaml
text_rendering:
  # ... existing settings ...

  # Subtitle space calculation multiplier
  # Controls how much vertical space is reserved relative to font height
  subtitle_space_multiplier: 1.3

  # Default subtitle reserved space as fraction of frame height
  # Used when content-aware positioning is disabled
  default_subtitle_reserved_space_percent: 0.15

  # Font offset multiplier for content-aware positioning (landscape videos)
  # In landscape mode, subtitles positioned using: image_bottom + spacing - (font_size * multiplier)
  # This creates tight spacing between content and subtitles
  content_aware_font_offset_multiplier: 5.5

  # Fallback Y position when content-aware positioning fails
  # Position as fraction of frame height (0.80 = 80% down from top)
  fallback_y_position: 0.80
```

### scraper.yaml - New Settings

Add browser and timing configuration:

```yaml
# ============================================================================
# BROWSER CONFIGURATION
# ============================================================================
browser_config:
  # Browser window size as percentage of monitor resolution
  # 0.80 = 80% of screen width/height
  browser_size_percent: 0.80

# ============================================================================
# TIMING CONFIGURATION
# ============================================================================
timing_config:
  # Human-like delay range for interactions (seconds)
  # Random delay chosen between min and max to simulate human behavior
  human_delay_min_sec: 0.5
  human_delay_max_sec: 2.0
```

---

## Phase 3: Update Code to Use Config Values

### High Priority Code Changes

#### 1. src/video/assembler.py

**Line 1718** - Replace hardcoded aspect ratio tolerance:
```python
# BEFORE:
if abs(content_aspect - target_aspect) < 0.10:

# AFTER:
aspect_tolerance = self.config.aspect_ratio.get('smart_scale_tolerance', 0.10)
if abs(content_aspect - target_aspect) < aspect_tolerance:
```

**Line 2830** - Font offset multiplier (already fixed in recent changes):
```python
# CURRENT (good):
font_offset = font_size * 5.5

# FUTURE: Make configurable via subtitles config
from src.video.subtitle_positioning import load_subtitle_config
subtitle_config = load_subtitle_config()
multiplier = subtitle_config.text_rendering.get('content_aware_font_offset_multiplier', 5.5)
font_offset = font_size * multiplier
```

**Lines 1884, 1886** - Remove hardcoded video positioning fallbacks:
```python
# BEFORE:
video_top = getattr(self.config, 'video_top_position_percent', 0.10)
video_height = getattr(self.config, 'video_content_height_percent', 0.75)

# AFTER: Remove fallbacks, require config values
video_top = self.config.video_top_position_percent
video_height = self.config.video_content_height_percent
```

**Lines 801-850** - Use format normalization config:
```python
# BEFORE:
target_fps = 30.0
fps_tolerance = 0.1

# AFTER:
format_norm = self.config.format_normalization
target_fps = format_norm['target_fps']
fps_tolerance = format_norm['fps_tolerance']
```

#### 2. src/audio/freesound_client.py

**Lines 349-418** - Use retry configuration:
```python
# BEFORE:
max_retries = 2
timeout = 5
backoff = 0.5 * (2 ** attempt)

# AFTER:
refresh_config = self.config.audio_settings['freesound_token_refresh']
max_retries = refresh_config['max_retries']
timeout = refresh_config['timeout_sec']
backoff = refresh_config['backoff_base_delay_sec'] * (refresh_config['backoff_multiplier'] ** attempt)
```

#### 3. src/video/subtitle_positioning.py

**Lines 65-94** - Ensure defaults match config:
```python
# Keep Pydantic defaults for validation, but ensure they match config file defaults
# Document that config file values override these
```

---

## Phase 4: Configuration Documentation

### Add Comprehensive Comments

Each configuration section needs:
1. **Purpose**: What this setting controls
2. **Impact**: How changing it affects video output
3. **Valid Range**: Min/max values and units
4. **Default**: What value is used if not specified
5. **Examples**: Common use cases

#### Example Documentation Format:

```yaml
# ============================================================================
# SUBTITLE POSITIONING
# ============================================================================
# Controls how subtitles are positioned relative to video content
#
# Key Concepts:
# - Anchor: Reference point for positioning (bottom, top, below_content, above_content)
# - Margin: Space between content and subtitle as fraction of frame height
# - Content-aware: Dynamic positioning based on actual video/image geometry
#
# Impact on Output:
# - Higher margins = more space between content and subtitles
# - below_content anchor = subtitles positioned dynamically below visible content
# - bottom anchor = fixed position at bottom of frame
#
subtitle_positioning:
  # Anchor point for subtitle positioning
  # Options: "bottom", "top", "below_content", "above_content", "center"
  # - bottom: Fixed position at bottom of frame
  # - below_content: Dynamic position below visible content (requires content-aware)
  # Default: "below_content"
  anchor: "below_content"

  # Margin as fraction of frame height (0.01 = 1% = ~19px for 1920px height)
  # Valid range: 0.00 to 0.20 (0% to 20%)
  # Lower values = tighter spacing, higher values = more gap
  # Default: 0.05 (5%)
  margin: 0.01
```

---

## Implementation Priority

### Immediate (High Risk if Not Done)
1. ✅ **DONE**: Fix portrait vs landscape subtitle positioning (already completed)
2. ✅ **DONE**: Update config for slideshow_images4 profile margin (already completed)

### Short Term (Low Risk, High Value)
3. ✅ **DONE**: Remove unused settings from video_production.yaml (Phase 1 completed)
4. Add missing configuration settings for magic numbers
5. Document all configuration options with comprehensive comments

### Medium Term (Refactoring)
6. Update code to use new config values instead of hardcoded numbers
7. Add validation for config value ranges
8. Create migration guide for users with custom configs

### Long Term (Architecture)
9. Consider splitting video_production.yaml into multiple files by domain
10. Add config schema validation with JSON Schema or Pydantic
11. Create config testing suite to validate all settings are used

---

## Testing Plan

After each phase:

1. **Config Validation Test**
   ```bash
   poetry run python -c "from src.video.video_config import load_config; load_config()"
   ```

2. **Video Generation Test**
   ```bash
   # Test both portrait and landscape profiles
   poetry run python -m src.video.producer outputs/B082F13J55/data.json slideshow_images4 --debug
   poetry run python -m src.video.producer outputs/B0CPSY5HJY/data.json product_video_sequential --debug
   ```

3. **Regression Test**
   - Compare output videos before/after changes
   - Verify subtitle positioning matches expected values
   - Check that all config values are respected

---

## Risk Assessment

### Low Risk Changes (Can implement immediately)
- Removing unused settings
- Adding new config settings with defaults
- Adding documentation comments

### Medium Risk Changes (Require testing)
- Updating code to use config values
- Removing hardcoded fallbacks
- Consolidating duplicate settings

### High Risk Changes (Require careful planning)
- Changing default values that affect output
- Removing settings that might be used in production
- Restructuring config file organization

---

## Conclusion

This cleanup plan addresses:
- **100+ lines** of unnecessary configuration
- **20+ magic numbers** that should be configurable
- **Missing documentation** for critical settings
- **Code maintainability** through centralized configuration

**Next Steps:**
1. ✅ Review this plan with team
2. ✅ Implement Phase 1 (safe removals) - COMPLETED
3. Add new configuration settings (Phase 2)
4. Update code to use config (Phase 3)
5. Add comprehensive documentation (Phase 4)

**Expected Benefits:**
- Cleaner, more maintainable configuration
- No need to modify code to adjust video output
- Better understanding of how settings affect output
- Reduced risk of configuration errors
