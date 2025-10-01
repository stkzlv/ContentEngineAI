# Configuration Audit Report

**Generated**: 2025-10-01
**Purpose**: Document hardcoded values, unused settings, and configuration improvements needed

---

## Executive Summary

This audit identified 25+ hardcoded values that should be moved to configuration, 7 unused settings that should be removed, and 5 duplicate configuration conflicts that need resolution.

### Priority Actions

1. **High Priority**: Remove unused/orphaned configuration settings (reduces confusion)
2. **Medium Priority**: Resolve duplicate configuration conflicts (prevents bugs)
3. **Low Priority**: Move hardcoded values to configuration (improves customization)

---

## 1. Hardcoded Values to Move to Configuration

### AI Services (`src/ai/description_generator.py`)

| Line | Current Value | Proposed Config Path | Impact |
|------|---------------|---------------------|--------|
| 342 | `50` | `llm_validation_settings.min_description_chars` | Description length threshold |
| 356 | `10` | `llm_validation_settings.min_description_words` | Word count threshold |
| 449 | `2` | `llm_validation_settings.description_retry_attempts` | Retry behavior |

**Status**: ✅ Added to `LLMValidationSettings` model in `video_config.py`

### Scraper - Amazon (`src/scraper/amazon/scraper.py`)

| Line | Current Value | Proposed Config Path | Impact |
|------|---------------|---------------------|--------|
| 173 | `1500` | `scraper_config.media.min_high_res_dimension` | Image quality threshold |
| 687 | `3` | `scraper_config.retry.max_attempts` | Retry resilience |
| 688 | `multiplier=1, min=1, max=10` | `scraper_config.retry.backoff_*` | Retry timing |
| 1028 | `":0.0"` | `scraper_config.browser.display_id` | X11 display |

**Status**: ⚠️ Some already in scraper.yaml, needs code update to use config

### Scraper - Downloader (`src/scraper/amazon/downloader.py`)

| Line | Current Value | Proposed Config Path | Impact |
|------|---------------|---------------------|--------|
| 26-27 | `// 2` | `download_config.parallel_browser_divisor` | Download concurrency |
| 30 | `3` | `download_config.max_retry_attempts` | Download retries |
| 148 | `10000` | `media_config.min_high_res_file_size` | File size filter |
| 493 | `30` | `download_config.timeout_sec` | Download timeout |
| 495 | `8192` | `download_config.chunk_size_bytes` | Streaming chunk size |
| 606 | `10` | `system_timeouts.head_request_timeout` | HEAD request timeout |
| 620-621 | User-Agent string | `http_headers.media_download.User-Agent` | Browser identification |
| 632 | Accept header | `http_headers.media_download.Accept` | Image format support |
| 637-652 | Thumbnail dimensions | `media_config.thumbnail_dimensions` | Thumbnail detection |
| 669-680 | HQ dimensions | `media_config.high_quality_dimensions` | Quality tiers |
| 720 | `1000` | `media_config.absolute_min_file_size_bytes` | Safety threshold |
| 731 | `0.7` | `media_config.borderline_threshold_multiplier` | Quality tolerance |

**Status**: ✅ Most already in scraper.yaml, needs code update to use config

### Subtitle Utils (`src/video/subtitle_utils.py`)

| Line | Current Value | Proposed Config Path | Impact |
|------|---------------|---------------------|--------|
| 308 | `(1080, 1920)` | `video_settings.default_frame_size` | Fallback resolution |

**Status**: 🔴 Needs addition to video_config.py

---

## 2. Unused Configuration Settings

### Remove from `config/video_production.yaml`

```yaml
# UNUSED - Never referenced in code
audio_settings:
  normalize_audio: true           # Line never accessed
  apply_compression: false        # Line never accessed
  noise_reduction: false          # Line never accessed
  sound_effects_volume_db: -6     # Line never accessed

# UNUSED PROFILE
video_profiles:
  product_showcase:  # Entire profile defined but never used
    ...
```

### Remove from `config/subtitles.yaml`

```yaml
# UNUSED - Animation system not implemented
subtitle_settings:
  animation_probability: 0.7      # Feature not coded
```

### Remove from `config/performance.yaml`

```yaml
# UNUSED - Entire section orphaned (not in Pydantic model)
network_settings:
  connection:
    max_retries: 3
    timeout: 30
    ...
  # All subsections unused
```

---

## 3. Orphaned Settings (In YAML but Not in Pydantic Model)

### `config/core.yaml`

```yaml
# ORPHANED - Not in VideoConfig model
debug_mode: true                  # Should be CLI-only or in DebugSettings

scraper_output_config:            # Meant for scraper, wrong file
  base_directory: "outputs"
  # Entire section should move to scraper.yaml
```

### `config/performance.yaml`

```yaml
# ORPHANED - Not in VideoConfig model
monitoring:                       # Entire section not in model
  performance:
    enabled: true
  memory:
    enabled: true
  api:
    log_requests: false
  errors:
    auto_report: false

# ORPHANED - Not in FFmpegSettings model
ffmpeg_settings:
  encoding:
    hardware_acceleration: "auto"
    threads: 0
    buffer_size: "512k"
    fast_seek: true
```

**Action**: Either add to Pydantic models or remove from YAML

---

## 4. Duplicate Configuration Conflicts

### Conflict 1: Subtitle Font Color Format

```yaml
# video_production.yaml
subtitle_settings:
  font_color: "#FFFFFF"          # Hex format

# subtitles.yaml
subtitle_settings:
  font_color: "&H00FFFFFF"       # ASS format
```

**Resolution**: Keep ASS format in subtitles.yaml (authoritative for subtitle rendering)

### Conflict 2: Font Directory Path

```yaml
# video_production.yaml
subtitle_settings:
  font_directory: "assets/fonts"

# subtitles.yaml
subtitle_settings:
  font_directory: "static/fonts"
```

**Resolution**: Verify actual directory location, update both to match

### Conflict 3: TTS Model Name

```yaml
# video_production.yaml
tts_config:
  coqui:
    model_name: "tts_models/en/ek1/tacotron2"

# subtitles.yaml
tts_config:
  coqui:
    model_name: "tts_models/en/ljspeech/vits"
```

**Resolution**: Determine primary model, consolidate configuration

### Conflict 4: Google Cloud Voice Criteria

```yaml
# video_production.yaml & ai_services.yaml
tts_config:
  google_cloud:
    voice_selection_criteria:
      - gender: NEUTRAL            # Single criterion

# subtitles.yaml
tts_config:
  google_cloud:
    voice_selection_criteria:     # 8 comprehensive criteria
      - gender: NEUTRAL
      - language_code: en-US
      ... (6 more fallbacks)
```

**Resolution**: Use comprehensive list from subtitles.yaml as authoritative

### Conflict 5: Whisper Model Download Root

```yaml
# video_production.yaml
whisper_settings:
  model_download_root: "~/.cache/whisper_models"

# ai_services.yaml
whisper_settings:
  model_download_root: ""        # Empty string
```

**Resolution**: Use explicit path from video_production.yaml

---

## 5. Missing Pydantic Model Definitions

Settings in YAML but not validated by Pydantic models:

1. `monitoring` section (performance.yaml)
2. `network_settings` section (performance.yaml)
3. FFmpeg `hardware_acceleration`, `threads`, `buffer_size`, `fast_seek`
4. `debug_mode` in core.yaml

**Action**: Either add to Pydantic models or remove from YAML (recommended: remove unused)

---

## 6. Implementation Recommendations

### Phase 1: Cleanup (Low Risk)

1. Remove unused settings from YAML files
2. Remove orphaned settings not in Pydantic models
3. Document removed settings in CHANGELOG

### Phase 2: Resolve Conflicts (Medium Risk)

1. Consolidate duplicate settings (keep authoritative source)
2. Update code to reference single source of truth
3. Add tests to verify configuration loading

### Phase 3: Move Hardcoded Values (Higher Risk)

1. Add new Pydantic fields to models
2. Update YAML files with new settings
3. Update code to read from config instead of hardcoded values
4. Add migration guide for users

### Testing Strategy

```bash
# For each phase:
make lint          # Verify code quality
make test          # Verify functionality
make test-cov      # Verify coverage maintained

# Integration test:
poetry run python -m src.video.producer test_product.json profile_name --debug
```

---

## 7. Configuration Documentation Needs

### Add Comments to Config Files

Priority files needing comprehensive inline comments:

1. ✅ **config/scraper.yaml** - Already excellent documentation
2. 🔴 **config/video_production.yaml** - Needs impact/usage comments
3. 🔴 **config/performance.yaml** - Needs explanation of thresholds
4. 🔴 **config/ai_services.yaml** - Needs model selection guidance
5. 🔴 **config/subtitles.yaml** - Needs positioning/style explanations

### Documentation Format

Follow scraper.yaml example:
- Section headers with visual separators
- Inline comments explaining what each setting does
- Impact statements (how it affects pipeline)
- Valid value ranges and defaults
- Usage notes at end of file

---

## 8. Next Steps

### Immediate Actions (This PR)

- [x] Add LLM validation settings to video_config.py
- [ ] Update config file comments with comprehensive documentation
- [ ] Remove clearly unused settings (audio normalization, product_showcase profile)
- [ ] Resolve duplicate configuration conflicts

### Future Work (Separate PRs)

- [ ] Move all hardcoded values in downloader.py to scraper.yaml
- [ ] Move hardcoded values in description_generator.py to use new LLM settings
- [ ] Add Pydantic models for monitoring/network settings or remove from YAML
- [ ] Create configuration migration guide for users
- [ ] Add configuration validation tests

---

## Appendix A: Configuration File Locations

```
config/
├── core.yaml              # General settings, file paths, logging
├── video_production.yaml  # Video generation, profiles, media
├── performance.yaml       # Optimization, caching, debug
├── ai_services.yaml       # LLM, TTS, STT providers
├── subtitles.yaml         # Subtitle styling, positioning
├── scraper.yaml           # Web scraping, browser, media extraction
└── profiles/
    ├── slideshow_images1.yaml
    ├── slideshow_images2.yaml
    └── ...
```

## Appendix B: Pydantic Model Hierarchy

```python
VideoConfig (src/video/video_config.py)
├── general: GeneralSettings
├── video_settings: VideoSettings
├── audio_settings: AudioSettings
├── subtitle_settings: SubtitleSettings
├── tts_config: TTSConfig
├── whisper_settings: WhisperSettings
├── llm_settings: LLMSettings
├── media_validation_settings: MediaValidationSettings
├── llm_validation_settings: LLMValidationSettings  # New
├── debug_settings: DebugSettings
├── stock_media_config: StockMediaConfig
└── freesound_config: FreesoundConfig
```
