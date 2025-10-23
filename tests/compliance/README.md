# Compliance Testing

Compliance tests verify that ContentEngineAI implements all documented requirements from the [requirements specification](.spec-workflow/specs/requirements-compliance/requirements.md). These tests ensure system reliability, prevent regression, and validate that advertised features work as specified.

## Quick Start

```bash
# Run all compliance tests
poetry run pytest -m compliance

# Run specific test file
poetry run pytest tests/compliance/test_config_compliance.py -v
poetry run pytest tests/compliance/test_video_compliance.py -v

# Run with coverage report
poetry run pytest -m compliance --cov=src --cov-report=html

# Combine with category markers
poetry run pytest -m "compliance and unit" -v
poetry run pytest -m "compliance and integration" -v
```

## Test Structure

```
tests/compliance/
├── README.md                      # This file
├── __init__.py                    # Package marker
├── test_config_compliance.py      # Requirements 1, 11, 12
├── test_scraper_compliance.py     # Requirements 2, 3
└── test_video_compliance.py       # Requirements 4-10
```

### Test Files

**test_config_compliance.py** - Configuration system validation
- Three-tier configuration precedence (CLI > env > YAML)
- Environment variable loading and secret management
- Configuration validation and error handling

**test_scraper_compliance.py** - Scraper architecture and media handling
- Multi-platform scraper architecture (`BaseScraper` compliance)
- Product data extraction and validation
- Media discovery, quality filtering, and storage structure

**test_video_compliance.py** - Video production features
- Dynamic video assembly and timing
- Unified subtitle positioning system
- Two-part subtitle system
- Profile-specific visual settings
- Style preset system (minimal, modern, bold, animated, random)
- ASS effects formatting
- AI service integration and fallbacks

## Requirement Mappings

### Requirement 1: Three-Tier Configuration System
**Tests**: test_config_compliance.py
- `test_req_1_1_cli_overrides_env_and_yaml` - CLI precedence validation
- `test_req_1_2_env_overrides_yaml_defaults` - Environment variable precedence
- `test_req_1_3_yaml_fallback_when_no_cli_or_env` - YAML fallback behavior
- `test_req_1_4_api_keys_only_in_env_file` - Secret security validation
- `test_req_1_5_env_file_loaded_at_runtime` - .env file loading
- `test_req_1_6_env_example_template_exists` - .env.example template
- `test_req_1_7_yaml_references_env_vars` - Environment variable references

### Requirement 2: Multi-Platform Scraper Architecture
**Tests**: test_scraper_compliance.py
- `test_req_2_1_scraper_extends_base_scraper` - BaseScraper inheritance
- `test_req_2_2_product_id_validation` - Product ID format validation
- `test_req_2_3_extracts_essential_data` - Required field extraction
- `test_req_2_4_skips_incomplete_products` - Graceful data handling
- `test_req_2_5_handles_multiple_asins_individually` - Individual product processing
- `test_req_2_6_keyword_search_filters` - Search filter support
- `test_req_2_7_filters_low_quality_media` - Media quality filtering
- `test_req_2_8_stealth_techniques` - Anti-detection features
- `test_req_2_9_graceful_failure_handling` - Error resilience

### Requirement 3: Product Media Discovery and Storage
**Tests**: test_scraper_compliance.py
- `test_req_3_1_media_stored_in_outputs_directory` - Output directory structure
- `test_req_3_2_downloads_high_resolution_images` - Image quality preference
- `test_req_3_3_includes_video_urls` - Video media handling
- `test_req_3_4_excludes_failed_validation_images` - Quality validation
- `test_req_3_5_customizable_output_path` - Path configuration
- `test_req_3_6_cleanup_removes_unexpected_files` - Cleanup functionality

### Requirement 4: Dynamic Video Assembly
**Tests**: test_video_compliance.py
- `test_req_4_1_video_duration_matches_voiceover` - Duration synchronization
- `test_req_4_2_configurable_image_duration` - Per-image timing
- `test_req_4_3_calculates_image_count_from_duration` - Image count calculation
- `test_req_4_4_reuses_images_for_long_voiceover` - Image reuse logic
- `test_req_4_5_applies_smooth_transitions` - Transition effects

### Requirement 5: Unified Subtitle Positioning System
**Tests**: test_video_compliance.py
- `test_req_5_1_supports_anchor_options` - Anchor positioning
- `test_req_5_2_content_aware_dynamic_positioning` - Content-aware mode
- `test_req_5_3_fixed_positioning_when_disabled` - Fixed positioning mode
- `test_req_5_4_margin_as_fraction_of_height` - Margin configuration
- `test_req_5_5_width_constraints_enforced` - Width limits
- `test_req_5_6_repositions_to_avoid_overlap` - Overlap avoidance
- `test_req_5_7_maintains_consistent_spacing` - Spacing consistency

### Requirement 6: Two-Part Subtitle System
**Tests**: test_video_compliance.py
- `test_req_6_1_two_independent_subtitle_lines` - Two-line mode
- `test_req_6_2_upper_line_displays_product_url` - URL display
- `test_req_6_3_customizable_data_source` - Field selection
- `test_req_6_4_upper_line_above_content_positioning` - Upper positioning
- `test_req_6_5_upper_line_persistent_visibility` - Persistent display
- `test_req_6_6_lower_line_timed_subtitles` - Timed synchronization
- `test_req_6_7_lower_line_below_content_positioning` - Lower positioning
- `test_req_6_8_stt_based_timing_synchronization` - STT timing
- `test_req_6_9_independent_styling` - Independent styles
- `test_req_6_10_separate_margin_control` - Margin control
- `test_req_6_11_content_aware_both_lines` - Content-aware for both
- `test_req_6_12_backward_compatible_single_line` - Single-line fallback

### Requirement 7: Profile-Specific Visual Settings
**Tests**: test_video_compliance.py
- `test_req_7_1_all_visual_settings_per_profile` - Profile configuration
- `test_req_7_2_image_positioning_overrides` - Image overrides
- `test_req_7_3_subtitle_settings_overrides` - Subtitle overrides
- `test_req_7_4_profile_merging_precedence` - Merge precedence
- `test_req_7_5_backward_compatibility_global_config` - Global compatibility
- `test_req_7_6_anchor_based_layout_in_profiles` - Anchor support

### Requirement 8: Style Preset System
**Tests**: test_video_compliance.py
- `test_req_8_1_all_five_presets_defined` - 5 preset validation
- `test_req_8_2_minimal_preset_no_effects` - Minimal style
- `test_req_8_3_modern_preset_karaoke_only` - Modern style
- `test_req_8_4_bold_preset_high_contrast_fade` - Bold style
- `test_req_8_5_animated_preset_movement_only` - Animated style
- `test_req_8_6_random_preset_font_color_effect` - Random style
- `test_req_8_7_font_randomization_deterministic` - Font selection
- `test_req_8_8_color_randomization_contrast` - Color coordination
- `test_req_8_9_preset_compatible_all_formats` - Format compatibility

### Requirement 9: ASS Effects System
**Tests**: test_video_compliance.py
- `test_req_9_1_ass_effects_enclosed_in_curly_braces` - Effect formatting
- `test_req_9_2_exactly_one_effect_per_video` - Effect consistency
- `test_req_9_9_karaoke_timing_format` - Karaoke tag format

### Requirement 10: AI Service Integration
**Tests**: test_video_compliance.py
- `test_req_10_1_tts_manager_tries_providers_in_order` - Provider ordering
- `test_req_10_2_voice_selection_criteria_configurable` - Voice selection
- `test_req_10_3_tts_config_validates_providers` - Provider validation
- `test_req_10_1_stt_whisper_primary` - Whisper primary
- `test_req_10_1_stt_google_fallback` - Google Cloud fallback
- `test_req_10_1_stt_circuit_breaker` - Circuit breaker

### Requirement 11: Global Debug Mode
**Tests**: test_config_compliance.py
- `test_req_11_1_debug_flag_enables_verbose_logging` - Debug logging
- `test_req_11_2_invalid_config_clear_errors` - Validation errors
- `test_req_11_3_missing_env_var_error_message` - Environment errors
- `test_req_11_4_graceful_component_failure` - Graceful degradation
- `test_req_11_5_service_fallback_before_failure` - Service fallbacks
- `test_req_11_6_stack_trace_in_debug_mode` - Debug stack traces

### Requirement 12: Configuration Validation
**Tests**: test_config_compliance.py
- `test_req_12_1_pydantic_validation_at_startup` - Pydantic validation
- `test_req_12_2_missing_required_field_error` - Missing field errors
- `test_req_12_3_type_mismatch_error` - Type validation
- `test_req_12_4_invalid_env_var_reference` - Environment reference validation
- `test_req_12_5_yaml_syntax_error_details` - YAML parsing errors
- `test_req_12_6_cli_argument_conflict_error` - CLI validation

## Test Statistics

**Total Compliance Tests**: 114

**By Requirement**:
- Requirement 1 (Config System): 7 tests
- Requirement 2 (Scraper): 9 tests
- Requirement 3 (Media): 6 tests
- Requirement 4 (Assembly): 5 tests
- Requirement 5 (Positioning): 7 tests
- Requirement 6 (Two-Part): 12 tests
- Requirement 7 (Profiles): 6 tests
- Requirement 8 (Presets): 9 tests
- Requirement 9 (ASS): 3 tests
- Requirement 10 (AI): 6 tests
- Requirement 11 (Debug): 6 tests
- Requirement 12 (Validation): 6 tests

**By Category**:
- Unit tests: 72
- Integration tests: 42

## Coverage

Run compliance tests with coverage:

```bash
poetry run pytest -m compliance --cov=src --cov-report=html
open outputs/coverage/index.html  # View coverage report
```

**Coverage Targets**:
- Configuration modules: >90%
- Scraper modules: >80%
- Video production modules: >80%

## Best Practices

1. **Test Isolation**: Each compliance test is independent and can run in any order
2. **Descriptive Names**: Test names follow pattern `test_req_<number>_<subsection>_<description>`
3. **Requirement Traceability**: Each test maps directly to acceptance criteria in requirements.md
4. **Mocking External Services**: AI services, API calls, and file I/O are mocked for speed and reliability
5. **Deterministic Testing**: Random selections use fixed seeds for reproducibility

## Troubleshooting

**Import errors:**
```bash
poetry install  # Ensure package installed in editable mode
```

**Test failures after config changes:**
```bash
# Check if changes maintain backward compatibility
poetry run pytest tests/compliance/test_config_compliance.py -v
```

**Missing fixtures:**
```bash
# Verify conftest.py is present
ls tests/conftest.py
ls tests/compliance/__init__.py
```

## Resources

- **Requirements Specification**: `.spec-workflow/specs/requirements-compliance/requirements.md`
- **Main Testing Guide**: `TESTING.md`
- **Configuration Guide**: `CONFIGURATION.md`
- **Contributing Guide**: `CONTRIBUTING.md`
