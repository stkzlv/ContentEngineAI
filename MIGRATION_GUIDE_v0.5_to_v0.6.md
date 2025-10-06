# Migration Guide: Legacy Subtitle System → Unified System (v0.6.0)

## Overview

Version 0.6.0 removes the legacy subtitle positioning system in favor of a unified, anchor-based configuration. This guide helps you migrate existing configurations.

## Breaking Changes

### Removed Configuration Fields

The following fields have been **removed** from `SubtitleSettings`:

| Removed Field | Replacement |
|--------------|-------------|
| `positioning_mode` | Use `content_aware` boolean |
| `alignment` | Use `anchor` + `content_aware` |
| `margin_v_percent` | Use `margin` (0.0-1.0) |
| `relative_positioning` | Merged into unified config |
| `absolute_positioning` | Merged into unified config |

### Removed Classes

- `SubtitlePositioningSettings` → Use `UnifiedSubtitleConfig`
- `AbsolutePositioningSettings` → Use `anchor` + `margin` + `content_aware=false`

## Migration Steps

### Step 1: Update Subtitle Settings

**Old Configuration (DEPRECATED):**
```yaml
subtitle_settings:
  positioning_mode: "relative"  # or "absolute"
  alignment: "bottom_center"
  margin_v_percent: 0.15
  relative_positioning:
    anchor: "below_content"
  absolute_positioning:
    anchor_vertical: "bottom"
```

**New Configuration (v0.6.0+):**
```yaml
subtitle_settings:
  # Positioning
  anchor: "below_content"        # top, center, bottom, above_content, below_content
  margin: 0.1                    # Normalized margin (0.0-1.0)
  content_aware: true            # Relative positioning

  # Styling
  style_preset: "modern"         # minimal, modern, bold, animated, random
  font_size_scale: 1.0
  max_line_length: 25
  horizontal_alignment: "center"
  subtitle_format: "ass"
```

### Step 2: Choose Positioning Mode

#### Absolute Positioning (Fixed Position)
Use when subtitles should stay in one place regardless of visual content:

```yaml
anchor: "bottom"
margin: 0.1
content_aware: false
```

#### Relative Positioning (Content-Aware)
Use when subtitles should avoid overlapping visual elements:

```yaml
anchor: "below_content"
margin: 0.08
content_aware: true
```

### Step 3: Update Style Presets

The system now supports **5 presets**, each with **exactly 1 effect** (or none for minimal):

| Preset | Effect | Description |
|--------|--------|-------------|
| `minimal` | None | Clean, simple styling |
| `modern` | Karaoke | Contemporary look |
| `bold` | Fade | High contrast |
| `animated` | Movement | Full animations |
| `random` | 1 Random | Randomized styling |

**Example:**
```yaml
style_preset: "modern"  # Will use karaoke effect only
```

### Step 4: Update Effect Configuration

**Old (DEPRECATED):**
```yaml
ass_effect_fade: true
ass_effect_karaoke: true        # ❌ Multiple effects violation
ass_effect_scale_pulse: false
```

**New (v0.6.0+):**
```yaml
style_preset: "modern"           # ✅ Uses karaoke effect only
# OR
style_preset: "bold"             # ✅ Uses fade effect only
# OR
style_preset: "random"           # ✅ Selects 1 random effect per video
```

## Configuration Mapping

### Alignment → Anchor Mapping

| Old `alignment` | New `anchor` | `content_aware` |
|----------------|--------------|-----------------|
| `top_center` | `top` | `false` |
| `center` | `center` | `false` |
| `bottom_center` | `bottom` | `false` |
| `above_content` | `above_content` | `true` |
| `below_content` | `below_content` | `true` |

### Positioning Mode → content_aware

| Old `positioning_mode` | New `content_aware` |
|------------------------|---------------------|
| `absolute` | `false` |
| `relative` | `true` |

## Complete Example Migration

### Before (v0.5.x)
```yaml
subtitle_settings:
  enabled: true
  positioning_mode: "relative"
  alignment: "below_content"
  margin_v_percent: 0.15

  relative_positioning:
    anchor: "below_content"
    margin: 0.1

  # Effects
  ass_effect_karaoke: true
  ass_effect_fade: true
  ass_font_randomization: true
```

### After (v0.6.0+)
```yaml
subtitle_settings:
  enabled: true

  # Unified positioning
  anchor: "below_content"
  margin: 0.1
  content_aware: true

  # Unified styling
  style_preset: "modern"          # karaoke effect
  font_size_scale: 1.0
  max_line_length: 25
  horizontal_alignment: "center"
  subtitle_format: "ass"
```

## Profile-Specific Settings

When using video profiles (e.g., `slideshow_images1`), you can override subtitle settings:

```yaml
video_profiles:
  slideshow_images1:
    subtitle_settings:
      anchor: "below_content"
      margin: 0.015
      content_aware: true
      style_preset: "random"
      font_size_scale: 1.1
      max_line_length: 22
      horizontal_alignment: "center"
      subtitle_format: "ass"
```

## Troubleshooting

### Error: "Invalid anchor value"

**Cause:** Using old alignment values
**Fix:** Use one of: `top`, `center`, `bottom`, `above_content`, `below_content`

### Error: "Invalid style preset"

**Cause:** Using non-existent preset
**Fix:** Use one of: `minimal`, `modern`, `bold`, `animated`, `random`

### Warning: "Preset has multiple effects"

**Cause:** Custom preset violates 1-effect rule
**Fix:** Ensure presets have exactly 1 effect (or none for minimal)

```yaml
# ❌ WRONG
custom_preset:
  effects: ["fade", "karaoke"]  # Multiple effects

# ✅ CORRECT
custom_preset:
  effects: ["karaoke"]           # Exactly 1 effect
```

### Subtitles Not Positioning Correctly

**Cause:** `content_aware` mode needs visual geometry data
**Fix:** Ensure visual elements are being tracked, or use absolute positioning:

```yaml
content_aware: false  # Use fixed positioning instead
```

## Configuration Validation

Run validation to check your configuration:

```bash
poetry run python -m src.video.config_validator
```

Or enable debug mode for detailed positioning information:

```bash
poetry run python -m src.video.producer <data.json> <profile> --debug
```

## API Changes (for Developers)

### Import Changes

```python
# Old (DEPRECATED)
from src.video.subtitle_positioning import convert_legacy_config

# New (v0.6.0+)
from src.video.subtitle_positioning import create_unified_config_from_settings
```

### Function Changes

```python
# Old
unified_config = convert_legacy_config(settings_dict)

# New
unified_config = create_unified_config_from_settings(settings_dict)
```

### Result Types

```python
# Old (DEPRECATED)
from src.video.result_types import create_legacy_subtitle_result

# New (v0.6.0+)
from src.video.result_types import SubtitleResult

result = SubtitleResult(
    success=True,
    path=output_path,
    format="ass",
    segments_created=10,
)
```

## Need Help?

- Review `config/subtitles.yaml` for complete examples
- Check `REQUIREMENTS.md` for system specifications
- See `ARCHITECTURE.md` for subtitle system design
- Report issues at https://github.com/anthropics/claude-code/issues

## Rollback Instructions

If you encounter issues, you can temporarily pin to v0.5.x:

```bash
git checkout v0.5.0
poetry install
```

However, we recommend migrating to v0.6.0+ for improved stability and simplified configuration.
