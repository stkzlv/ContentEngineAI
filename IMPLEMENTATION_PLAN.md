# Implementation Plan: Remove Legacy Subtitle System & Fix ASS Effects (v0.6.0)

## Objective
1. Remove all legacy subtitle configuration code and fully migrate to the unified subtitle system
2. Fix ASS effects violation to enforce exactly 1 effect per video
3. Ensure unified system supports both absolute and relative positioning modes
4. Simplify codebase and improve maintainability

## Branch Setup
- **Branch**: `feature/remove-legacy-subtitle-system`
- **Type**: Refactor + Feature (MINOR version bump: 0.5.0 → 0.6.0)
  - Breaking change: Removes deprecated legacy subtitle settings
  - Users must migrate to unified config format

## Key Changes Summary

### 1. Configuration System Update
- **REQUIREMENTS.md updated** to reflect three-tier precedence:
  - CLI Arguments (highest) > Environment Variables > YAML Config (lowest)

### 2. Unified Subtitle System
- **Maintains both absolute and relative positioning**:
  - Absolute mode: `anchor` + `margin` with `content_aware=false`
  - Relative mode: `anchor` + `margin` with `content_aware=true`
- **5 style presets** (updated from 4):
  - minimal, modern, bold, animated, random

### 3. ASS Effects Fix
- **Enforce exactly 1 effect per video** (fixes REQUIREMENTS.md violation)
- **Preset effect mapping**:
  - minimal: No effects
  - modern: Karaoke only
  - bold: Fade only
  - animated: Movement only
  - random: One randomly selected from all available

## Current State Analysis

### Legacy Components to Remove

#### 1. **Video Config Models** (`src/video/video_config.py`)
- `SubtitlePositioningSettings` class (lines 91-110) - DEPRECATED
- `AbsolutePositioningSettings` class (lines 112-125) - DEPRECATED
- Legacy fields in `SubtitleSettings`:
  - `positioning_mode` (line 129-132)
  - `alignment` (line 146-150)
  - `margin_v_percent` (line 151-154)
  - `relative_positioning` (line 155)
  - `absolute_positioning` (line 156)
  - All `ass_*` prefixed fields (lines 177-266) - migrate to unified config

#### 2. **Conversion Functions** (`src/video/subtitle_positioning.py`)
- `convert_legacy_config()` function (lines 373-446)
- Remove legacy parameter handling from `get_style_config()`

#### 3. **Integration Points**
- `src/video/subtitle_utils.py` (line 305): Remove `convert_legacy_config` usage
- `src/video/unified_assembler_integration.py` (lines 30-32): Direct unified config usage
- `src/video/config_validator.py` (lines 337-386): Remove legacy validation
- `src/video/assembler.py` (lines 1151-1155): Remove legacy conversion
- `src/video/producer.py` (line 1097): Update to use unified config directly

#### 4. **Result Types** (`src/video/result_types.py`)
- `create_legacy_subtitle_result()` (lines 219-245)
- `extract_legacy_values()` (lines 247-260)

#### 5. **Config Adapter** (`src/video/config_adapter.py`)
- `_add_legacy_structure()` method (lines 77-78)

## Implementation Steps

### Phase 1: Prepare Unified Configuration
1. **Update `SubtitleSettings` model** to use `UnifiedSubtitleConfig` fields directly
   - Add: `anchor`, `margin`, `content_aware`, `style_preset`, `font_size_scale`
   - Remove: `positioning_mode`, `alignment`, `margin_v_percent`, `relative_positioning`, `absolute_positioning`
   - Migrate `ass_*` fields to style preset configuration
   - **Ensure unified system supports both modes**:
     - Absolute: `anchor` + `margin` with `content_aware=false`
     - Relative: `anchor` + `margin` with `content_aware=true`

2. **Update `config/subtitles.yaml`** as primary configuration
   - Ensure all unified parameters are properly documented
   - Remove legacy `relative_positioning` section (lines 20-24)
   - Add clear examples for absolute vs relative modes
   - Document 5 presets: minimal, modern, bold, animated, random

3. **Simplify `config/video_production.yaml`**
   - Remove legacy subtitle settings
   - Reference unified config from `subtitles.yaml`

### Phase 2: Remove Legacy Code
4. **Delete deprecated classes** from `video_config.py`:
   - `SubtitlePositioningSettings`
   - `AbsolutePositioningSettings`

5. **Remove legacy conversion** from `subtitle_positioning.py`:
   - Delete `convert_legacy_config()` function
   - Clean up imports in dependent modules

6. **Update integration points**:
   - `subtitle_utils.py`: Direct `UnifiedSubtitleConfig` instantiation
   - `unified_assembler_integration.py`: Use config directly
   - `assembler.py`: Remove conversion logic
   - `producer.py`: Use unified config from settings

7. **Clean up result types**:
   - Remove `create_legacy_subtitle_result()`
   - Remove `extract_legacy_values()`
   - Update callers to use modern result structure

8. **Simplify config adapter**:
   - Remove `_add_legacy_structure()`
   - Update structure builder

### Phase 3: Fix ASS Effects Violation
9. **Enforce exactly 1 effect per video** (`unified_subtitle_generator.py`):
   - **Current issue**: Line 715 allows `num_effects > 1` via `random.sample()`
   - **Fix**: Force `num_effects = 1` for all presets
   - **Update effect selection logic**:
     ```python
     # OLD (WRONG): chosen_effects = random.sample(available_effects, num_effects)
     # NEW (CORRECT): chosen_effects = [random.choice(available_effects)]
     ```
   - **Preset effect mapping**:
     - minimal: `effects = []` (no effects)
     - modern: `effects = ["karaoke"]` (karaoke only)
     - bold: `effects = ["fade"]` (fade only)
     - animated: `effects = ["movement"]` (movement only)
     - random: Select exactly 1 from all available effects

10. **Update style presets config** (`config/subtitles.yaml`):
    - Ensure each preset defines exactly 1 effect (or none for minimal)
    - Update `random` preset to select 1 effect from full list

### Phase 4: Update Validation
11. **Simplify `config_validator.py`**:
    - Remove legacy parameter detection
    - Remove legacy-to-unified conversion tests
    - Add validation for unified parameters only
    - Add validation for effect limitation (max 1 effect)

12. **Update Pydantic validators** in `SubtitleSettings`:
    - Remove legacy positioning mode validation
    - Add unified parameter validation
    - Add effect count validation

### Phase 5: Testing & Documentation
13. **Update tests**:
    - Remove legacy config tests
    - Update integration tests to use unified config
    - Add tests for 1-effect-per-video enforcement
    - Test both absolute and relative positioning modes
    - Test all 5 presets (minimal, modern, bold, animated, random)
    - Ensure all subtitle tests pass with new structure

14. **Update documentation**:
    - `MIGRATION_GUIDE.md`: Document legacy → unified migration
    - `config/subtitles.yaml`: Add inline examples for absolute/relative modes
    - `ARCHITECTURE.md`: Update subtitle system section
    - `CHANGELOG.md`: Document breaking changes
    - `REQUIREMENTS.md`: Already updated with unified system

15. **Add deprecation warnings** (Optional transitional support):
    - Detect legacy config usage
    - Log warning with migration instructions
    - Provide clear error messages

### Phase 6: Quality Assurance
16. **Run quality checks**:
    ```bash
    make lint          # Code quality
    make test          # Test suite
    make security      # Security scan
    ```

17. **Integration testing**:
    ```bash
    # Test with actual video generation
    poetry run python -m src.video.producer outputs/<ASIN>/data.json slideshow_images1 --debug
    ```

18. **Verify subtitle generation**:
    - Check ASS format output
    - Verify positioning accuracy
    - Test all style presets
    - Validate randomization features

### Phase 7: Version & Release
19. **Update version to 0.6.0** (`pyproject.toml`)
    - MINOR bump due to breaking changes

20. **Update CHANGELOG.md**:
    ```markdown
    ## [0.6.0] - 2025-10-05

    ### Breaking Changes
    - **Removed legacy subtitle configuration system**
      - Removed `positioning_mode`, `alignment`, `margin_v_percent`
      - Removed `relative_positioning` and `absolute_positioning`
      - Removed `SubtitlePositioningSettings` and `AbsolutePositioningSettings`
    - **Fixed ASS effects to enforce 1 effect per video**
      - All presets now use exactly 1 effect (or none for minimal)
      - Random preset selects 1 effect from all available

    ### Added
    - Unified subtitle configuration as primary system
    - Direct `UnifiedSubtitleConfig` integration
    - Simplified configuration structure
    - Both absolute and relative positioning via unified system
    - 5 style presets: minimal, modern, bold, animated, random

    ### Fixed
    - ASS effects violation: Now enforces exactly 1 effect per video
    - Preset effect mapping clearly defined

    ### Migration Guide
    - See `MIGRATION_GUIDE.md` for step-by-step migration
    - Absolute mode: `content_aware=false`
    - Relative mode: `content_aware=true`
    ```

21. **Create migration guide** (`MIGRATION_GUIDE.md`):
    - Document old → new parameter mapping
    - Provide example configurations
    - Include troubleshooting tips

## Files to Modify

### Core Changes (8 files)
1. `src/video/video_config.py` - Remove legacy classes and fields
2. `src/video/subtitle_positioning.py` - Remove conversion function
3. `src/video/subtitle_utils.py` - Direct unified config usage
4. `src/video/unified_assembler_integration.py` - Remove conversion
5. `src/video/assembler.py` - Direct unified config usage
6. `src/video/producer.py` - Update config initialization
7. `src/video/config_validator.py` - Remove legacy validation
8. `src/video/result_types.py` - Remove legacy result functions

### Configuration (2 files)
9. `config/subtitles.yaml` - Add migration examples
10. `config/video_production.yaml` - Remove legacy settings

### Documentation (4 files)
11. `CHANGELOG.md` - Document breaking changes
12. `MIGRATION_GUIDE.md` - Create migration guide
13. `ARCHITECTURE.md` - Update subtitle system docs
14. `pyproject.toml` - Version bump to 0.6.0

### Testing (Update existing tests)
15. `tests/test_subtitle_*.py` - Update for unified config
16. `tests/test_video_config.py` - Remove legacy tests
17. `tests/test_producer*.py` - Update config usage

## Risk Assessment

### High Risk
- **Breaking changes** for existing configurations
  - Mitigation: Provide clear migration guide and examples
  - Mitigation: Add transitional warnings before full removal

### Medium Risk
- **Integration points** may have hidden dependencies
  - Mitigation: Comprehensive testing of video pipeline
  - Mitigation: Check all subtitle generation paths

### Low Risk
- **Test coverage** should catch regressions
  - Mitigation: Run full test suite
  - Mitigation: Manual integration testing

## Success Criteria

✅ All legacy classes and functions removed
✅ No `DEPRECATED` markers in subtitle code
✅ All tests passing with unified config
✅ ASS effects enforced to exactly 1 per video
✅ Both absolute and relative modes working via unified system
✅ All 5 presets (minimal, modern, bold, animated, random) tested
✅ Video generation working with new config
✅ Documentation updated and complete
✅ Migration guide validated
✅ Version bumped to 0.6.0
✅ CHANGELOG updated
✅ CI/CD pipeline passing

## Timeline Estimate

- Phase 1 (Prepare Unified Config): 2-3 hours
- Phase 2 (Remove Legacy Code): 2-3 hours
- Phase 3 (Fix ASS Effects): 1-2 hours
- Phase 4 (Update Validation): 1-2 hours
- Phase 5 (Test/Doc): 2-3 hours
- Phase 6 (QA): 1-2 hours
- Phase 7 (Release): 1 hour

**Total: 10-16 hours**

## Rollback Plan

If issues arise:
1. Revert branch to previous commit
2. Keep legacy system temporarily
3. Add more comprehensive warnings
4. Extend migration period
5. Release as 0.5.1 (non-breaking) instead

## Post-Implementation

- Monitor for issues in production use
- Collect user feedback on unified config
- Consider additional style presets
- Evaluate performance improvements from simplified code
