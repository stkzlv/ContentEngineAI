# Subtitle Configuration Cleanup

Audit of the subtitle configuration system produced during the best-practices
alignment pass. Catalogs pre-existing bugs, dead code, and simplification
opportunities. None of these are urgent — the pipeline works today. But the
system has accumulated enough drift that maintenance cost is high and
silent config bugs are easy to introduce.

This document exists to:
1. Record what's broken so future work doesn't re-discover it
2. Rank cleanup opportunities by impact vs effort
3. Sketch concrete before/after for the biggest refactors
4. Mark things that look ugly but should actually stay as-is

**Companion docs**:
[subtitle-best-practices.md](subtitle-best-practices.md) (the recipe),
[pycaps-followups.md](pycaps-followups.md) (pycaps-specific deferred work),
[configuration.md](configuration.md) (general 3-level config system).

---

## TL;DR

All bugs fixed. Four high-value refactors, ~20 dead fields.

| # | Work item | Kind | Status | Effort |
|---|---|---|---|---|
| 1 | ~~Fix duration key namespace + width fraction passthrough~~ | ~~Bug~~ | **FIXED** (commit `c385df1`) | ~~30 min~~ |
| 2 | ~~`UnifiedSubtitleGenerator` constructs fresh `PlatformSafeZone()` instead of reading config~~ | ~~Bug~~ | **FIXED** (PR #65) | ~~15 min~~ |
| 3 | ~~Wire profile `subtitle_safe_zone_*` overrides into `_collect_overrides` field map~~ | ~~Bug~~ | **FIXED** (PR #65) | ~~15 min~~ |
| 4 | Collapse `MergedSubtitleSettings` + `UnifiedSubtitleConfig` into one typed model | **Refactor** | Open | 1-2 days |
| 5 | Nest `two_part_subtitles` as a typed sub-model instead of 14 flat fields | **Refactor** | Open | 0.5 day |
| 6 | Move `style_presets` into the Pydantic model, delete the inline YAML re-read in `get_style_config()` | **Refactor** | Open | 0.5 day |
| 7 | Move font and color pools from Python enums to YAML | **Refactor** | Open | 1 day |
| 8 | ~~Delete ~20 dead YAML keys and Pydantic fields~~ | ~~Cleanup~~ | **DONE** | ~~1 hour~~ |
| 9 | Remove the "Legacy Compatibility Settings" block in `subtitle_settings` | **Cleanup** | Open (gated on #6) | 1 hour |
| 10 | Rename duplicate/confusing keys to canonical names | **Cleanup** | Open (gated on #4) | 1 hour |

Remaining effort: roughly 3 days if everything open is done.

---

## 1. How we got here

The subtitle configuration has evolved through at least three generations:

1. **Legacy monolithic config** — flat keys at the top of `subtitle_settings` (`font_name`, `font_color`, `font_size_percent`, etc.). Consumed by the original drawtext/SRT path.
2. **Unified positioning system** — `UnifiedSubtitleConfig` (`subtitle_positioning.py:78`) with a simpler surface (anchor, margin, style_preset). Introduced alongside ASS support and content-aware positioning.
3. **Preset + randomization system** — `style_presets` section with per-preset font/color/effect bundles, plus `font_color_manager.py` for per-product random selection.

Each generation was layered on top of the previous without removing the old
settings. The result:
- Legacy `font_name` / `font_size_percent` / `outline_thickness` still exist and are still consumed by the drawtext fallback.
- The same concept often has **two or three different names** across layers (e.g., `max_duration` vs `max_subtitle_duration` vs `max_subtitle_duration_sec`).
- Conversion layers hide drift: `MergedSubtitleSettings` → `dict` → `UnifiedSubtitleConfig` via `settings.get("max_duration", 4.5)` silently loses fields whose names don't match.
- Some mid-generation settings were added, then never wired up (`subtitle_safe_zone_*` on VideoProfile, `fallback_y_position` in YAML).

None of this is anyone's fault — it's the natural state of a 2-year-old
rendering pipeline. But it's at the point where adding a new subtitle
knob takes about as long as understanding why a previous knob doesn't work.

---

## 2. The current data flow (as it actually is)

```
config/subtitles.yaml
  subtitle_settings: { ... 40+ keys ... }
  subtitle_effects: { karaoke_*, pulse_*, bounce_*, ... }       ──┐
  text_rendering:                                                │
    safe_zone: { min_x, max_x, min_y, max_y }                    │
    narrow_char_width_factor / wide / space                      │
    (13 more fields — mostly dead)                               │
  style_presets: { minimal, modern, bold, animated, random }    │
  subtitle_segmentation: { min_words_*, fallback_segment_* }    │
  tts_config: { ... }  (not subtitle rendering)                 │
                                                                 │
config/video_production.yaml                                     │
  profiles:                                                      │
    slideshow_images1:                                           │
      subtitle_* flat overrides (30+ nullable fields)            │
      pycaps_* flat overrides (7 fields)                         │
                                                                 │
CLI args (producer/cli.py + pipeline/global_batch.py)            │
  --subtitle-* / --pycaps-*                                      │
                                                                 │
       ┃                                                         │
       ▼                                                         │
load_video_config_modular (config_adapter.py)                    │
       ┃                                                         │
       ▼                                                         │
VideoConfig(                                                     │
  subtitle_settings: dict[str, Any]  ◄────── NOT A PYDANTIC MODEL│
  subtitle_effects: SubtitleEffectsSettings  ◄────────────────────┤
  text_rendering: TextRenderingSettings  ◄────────────────────────┤
  subtitle_segmentation: SubtitleSegmentationSettings  ◄──────────┘
  video_profiles: dict[str, VideoProfile]
)
       ┃
       ▼
config.get_profile_merged_settings(profile_name, cli_overrides)
  _build_subtitle_base(self)       ────► flat dict with field-name remapping
  _collect_overrides(profile, ...) ────► field_map with 20+ entries
  subtitle_data.update(...)
  MergedSubtitleSettings(**subtitle_data)   ◄───── 50-field Pydantic model with
                                                   extra="allow" (anything sneaks through)
  apply CLI dotted overrides (2-level + 3-level for pycaps)
       ┃
       ▼
MergedProfileSettings(video_settings, subtitle_settings, profile)
       ┃
       ▼
step_generate_subtitles(ctx)
       ┃
       ▼
subtitle_dict = subtitle_settings.model_dump()  ◄───── round-trip to dict
       ┃
       ▼
create_unified_subtitles(subtitle_dict, ...)
       ┃
       ▼
create_unified_config_from_settings(subtitle_dict)
       ┃    reads settings.get("max_duration", 4.5) etc.
       ┃    hardcoded defaults inside — drops unrecognized keys
       ▼
UnifiedSubtitleConfig(15 fields)  ◄───── second Pydantic model; partial overlap with MergedSubtitleSettings
       ┃
       ▼
UnifiedSubtitleGenerator(unified_config, frame_size, product_id)
       ┃
       ▼
  .generate_from_timings(...)
       ┃
       │    side channel: get_style_config() does open("config/subtitles.yaml") directly
       │    to read style_presets, ignoring the CLI/profile layers entirely
       │
       │    side channel: FontManager / ColorManager in font_color_manager.py
       │    hold Python enums for the font pool, ignoring YAML
       ▼
SRT/ASS file → SubtitleGraphBuilder (assembler) → FFmpeg filter graph
```

The boxed complexity is real. Every arrow is a place a bug can hide.

---

## 3. Confirmed bugs (fix first)

### 3.1 Duration key namespace + width fraction passthrough — FIXED

**Status**: fixed in commit `c385df1` on `feature/pycaps-subtitle-engine`.

Three sub-issues were resolved together:

1. **`_build_subtitle_base`** now reads `max_duration` as a fallback alongside
   `max_subtitle_duration` / `max_subtitle_duration_sec`. Hardcoded fallbacks
   updated from 4.5/0.4 to 2.5/0.6. Same for `min_duration`.

2. **`_build_subtitle_base`** now passes `max_subtitle_width_fraction` into
   the merged dict (it was missing entirely, so profiles without an explicit
   override got the Pydantic default 0.67 instead of the YAML global 0.80).

3. **`create_unified_config_from_settings`** now reads `max_subtitle_duration`
   as a fallback for `max_duration`, and updated all hardcoded fallback
   defaults to match the best-practice recipe (width 0.80, words/line 3,
   duration 2.5/0.6).

**Verification**: all 9 profiles confirmed passing best-practice checks after
the fix.

### 3.2 `UnifiedSubtitleGenerator` constructs its own safe zone — FIXED

**Status**: fixed in PR #65 on `bugfix/safe-zone-config-passthrough`.

**Where**: `src/video/unified_subtitle_generator.py:565-569`

```python
from src.video.config.core_models import PlatformSafeZone
sz = PlatformSafeZone()  # ← hardcoded defaults, ignores config.text_rendering.safe_zone
safe_zone_width = int(self.frame_size[0] * (sz.max_x - sz.min_x))
```

Nothing reads `self.config.text_rendering` here. If a user overrides
`text_rendering.safe_zone` in YAML, the width clamp inside the generator
still uses the hardcoded fractions (0.046 / 0.778).

**Fix**: either pass the safe zone into `UnifiedSubtitleGenerator.__init__`
alongside `config`, or have it read from a shared singleton. The
subtitle_builder already has a `_get_safe_zone()` helper that does the
right thing — extract it and reuse.

### 3.3 Profile-level safe zone overrides are dead — FIXED

**Status**: fixed in PR #65 on `bugfix/safe-zone-config-passthrough`.

**Where**: `src/video/config/visual_models.py:312-321`

```python
subtitle_safe_zone_min_x: float | None = Field(None, ...)
subtitle_safe_zone_max_x: float | None = Field(None, ...)
subtitle_safe_zone_min_y: float | None = Field(None, ...)
subtitle_safe_zone_max_y: float | None = Field(None, ...)
```

These fields exist on `VideoProfile` but `_collect_overrides` (core_models.py:715)
doesn't include a mapping for them. Profile YAML can set them but they go
nowhere — `MergedSubtitleSettings` doesn't have matching fields either.

**Fix (option A)**: add them to the field_map. But then they need a home on
`MergedSubtitleSettings` too, and something needs to consume them. The
simplest route is to store them on `MergedSubtitleSettings.safe_zone:
PlatformSafeZone` (new nested field) and have subtitle_builder read that
with fallback to `config.text_rendering.safe_zone`.

**Fix (option B)**: delete the dead fields and require users to override
`text_rendering.safe_zone` at the global level. Simpler but reduces per-profile
flexibility.

Either is fine; A is more useful if per-profile safe zones are actually
needed (different platform targets per profile), B if they're just dead
weight.

---

## 4. High-value simplifications

### 4.1 Collapse MergedSubtitleSettings + UnifiedSubtitleConfig into one model

**Current shape**:
- `MergedSubtitleSettings` (`visual_models.py:432`) — 50 fields, `extra="allow"`
- `UnifiedSubtitleConfig` (`subtitle_positioning.py:78`) — 15 fields, strict
- Bridge: `create_unified_config_from_settings(dump)` with hardcoded defaults

The bridge is where bugs like 3.1 hide. It uses `dict.get(key, default)`
which is the weakest possible type discipline: any key rename drops the
value silently, any typo reads the default.

**Target**: one Pydantic model, `SubtitleSettings`, that both the config
layer and the generator operate on directly. No `.model_dump()` → `dict`
→ second Pydantic model round trip.

**Approach**:
```python
# src/video/config/subtitle_settings.py (new module)
class SubtitleSettings(BaseModel):
    """Unified subtitle configuration — single source of truth for both
    config-layer merge output and runtime generator input."""
    model_config = ConfigDict(extra="forbid")  # STRICT — no more extra="allow"

    # --- Engine + format ---
    enabled: bool = True
    subtitle_engine: Literal["ffmpeg", "pycaps"] = "ffmpeg"
    subtitle_format: Literal["srt", "ass"] = "ass"

    # --- Positioning ---
    anchor: PositionAnchor = PositionAnchor.BELOW_CONTENT
    margin: float = Field(0.04, ge=0.0, le=0.5)
    content_aware: bool = True
    horizontal_alignment: Literal["left", "center", "right"] = "center"
    safe_zone: PlatformSafeZone = Field(default_factory=PlatformSafeZone)

    # --- Style ---
    style_preset: StylePreset = StylePreset.MODERN
    font_size_scale: float = Field(1.0, ge=0.5, le=2.0)

    # --- Text formatting ---
    max_line_length: int = Field(38, ge=1)
    max_words_per_line: int = Field(3, ge=0)
    max_subtitle_width_fraction: float = Field(0.80, ge=0.0, le=1.0)
    max_duration: float = Field(2.5, gt=0)
    min_duration: float = Field(0.6, gt=0)

    # --- Randomization ---
    randomize_fonts: bool = False
    randomize_colors: bool = False
    randomize_effects: bool = False
    selected_font: str | None = None
    selected_color_pair: str | None = None

    # --- Nested ---
    pycaps: PycapsSettings | None = None
    two_part: TwoPartSubtitleSettings | None = None
```

**Why `extra="forbid"`**: the current `extra="allow"` silently accepts any
typo in YAML. Switching to `forbid` means YAML typos throw validation
errors at startup instead of being silently dropped at runtime.

**Migration path**:
1. Add `SubtitleSettings` alongside the existing types.
2. Add a one-shot translator `SubtitleSettings.from_legacy_dict(old_dict)`
   that handles the rename (max_subtitle_duration → max_duration, etc.)
   so existing YAML still loads.
3. Port consumers one at a time: start with `UnifiedSubtitleGenerator`
   (the leaf), then `step_generate_subtitles` (the caller), then
   `SubtitleGraphBuilder` (the assembler), then delete `MergedSubtitleSettings`
   and `UnifiedSubtitleConfig` + the bridge.
4. Update profile overrides: see §4.3 for the nested override pattern that
   eliminates the 30-line `_collect_overrides` field_map.

**Effort**: 1-2 days. The translation shim is the time sink.

### 4.2 Nest `two_part_subtitles` as a typed sub-model

**Current**: 14 flat fields on `MergedSubtitleSettings` (`two_part_subtitles_enabled`,
`two_part_subtitles_upper_enabled`, `two_part_subtitles_upper_source_field`, ...)
plus 14 matching flat nullable overrides on `VideoProfile` plus 14 entries
in `_collect_overrides`'s field_map plus a runtime reassembly in
`TwoPartSubtitleHandler.__init__`:

```python
# Current: src/video/producer/two_part_subtitles.py:63
upper_config = {
    "enabled": ss.two_part_subtitles_upper_enabled,
    "source_field": ss.two_part_subtitles_upper_source_field,
    "anchor": ss.two_part_subtitles_upper_anchor,
    ... 8 more ...
}
```

We flatten the hierarchy to store it, then reassemble it to use it. The
worst of both worlds.

**Target**:
```python
class TwoPartSubtitleUpperLine(BaseModel):
    enabled: bool = True
    source_field: str = "shortened_affiliate_link"
    custom_url: str | None = None
    anchor: str = "above_content"
    margin: float = 0.08
    font_size_scale: float = 0.75
    style_preset: str = "minimal"
    use_full_duration: bool = True
    randomize_effects: bool = False
    prefix_replace: str | None = None

class TwoPartSubtitleLowerLine(BaseModel):
    enabled: bool = True
    anchor: str = "below_content"
    margin: float = 0.05

class TwoPartSubtitleSettings(BaseModel):
    enabled: bool = False
    upper: TwoPartSubtitleUpperLine = Field(default_factory=TwoPartSubtitleUpperLine)
    lower: TwoPartSubtitleLowerLine = Field(default_factory=TwoPartSubtitleLowerLine)
```

The YAML already has the nested shape (`subtitle_settings.two_part_subtitles.upper_line.*`).
The flattening is an artifact of how `_build_subtitle_base` expands it.
Undo the flattening and the Pydantic model accepts the YAML shape directly.

`TwoPartSubtitleHandler._parse_config()` becomes:
```python
def _parse_config(self) -> TwoPartSubtitleSettings:
    return self.merged_profile_settings.subtitle_settings.two_part \
        or TwoPartSubtitleSettings()
```

Two lines instead of 30. Profile overrides become one line in the YAML:
```yaml
profiles:
  slideshow_images3:
    subtitle_settings:
      two_part:
        enabled: true
        upper:
          source_field: shortened_affiliate_link
          style_preset: minimal
```

This composes naturally with §4.1 (single `SubtitleSettings` model) and
§4.3 (nested profile overrides).

**Effort**: 0.5 day. The Pydantic models are small; the work is porting
`TwoPartSubtitleHandler` consumers.

### 4.3 Replace flat profile overrides with a nested override pattern

**Current**: 30+ nullable `subtitle_*` fields on `VideoProfile` plus a 20-entry
field_map in `_collect_overrides` plus the bridging logic. Each new
subtitle setting means touching three files and writing a field map entry.

**Target**: one optional field on `VideoProfile` holding a partial
`SubtitleSettings`:
```python
class VideoProfile(BaseModel):
    description: str
    use_scraped_images: bool = False
    ...
    subtitle_settings: PartialSubtitleSettings | None = None
```

Where `PartialSubtitleSettings` is a subclass of `SubtitleSettings` with
every field set to `None` by default. During `get_profile_merged_settings`,
merge the partial into the global:

```python
def merge_subtitle_settings(
    base: SubtitleSettings,
    override: PartialSubtitleSettings | None,
) -> SubtitleSettings:
    if override is None:
        return base
    non_null = {k: v for k, v in override.model_dump().items() if v is not None}
    return base.model_copy(update=non_null)
```

This automatically handles nested sub-models (`pycaps`, `two_part`,
`safe_zone`) because Pydantic's `model_copy(update=...)` replaces whole
sub-objects. For deep merging of nested fields, use recursive logic.

**YAML shape** (profile override):
```yaml
profiles:
  slideshow_images1:
    description: ...
    subtitle_settings:           # <- new nested block
      style_preset: bold
      max_words_per_line: 3
      pycaps:
        template_name: hype
      two_part:
        enabled: true
        upper:
          style_preset: minimal
```

vs the current:
```yaml
profiles:
  slideshow_images1:
    subtitle_style_preset: bold
    subtitle_max_words_per_line: 3
    pycaps_template: hype
    two_part_subtitles_enabled: true
    two_part_subtitles_upper_enabled: true
    two_part_subtitles_upper_style_preset: minimal
    ... 10 more flat keys
```

**Why this is better**:
- New subtitle fields automatically become profile-overrideable with zero field_map maintenance
- Deep hierarchy matches the runtime model's hierarchy
- Type checking applies at YAML load time (with `extra="forbid"`)

**Migration**: add a compatibility shim that reads legacy flat keys and
constructs the nested dict. Remove the shim after all profiles have been
migrated (ideally in one commit).

**Effort**: 1 day including the migration shim.

### 4.4 Move `style_presets` into the Pydantic config layer

**Current**: `get_style_config()` in `subtitle_positioning.py:168` does:
```python
subtitles_config_path = Path("config/subtitles.yaml")
try:
    if subtitles_config_path.exists():
        with open(subtitles_config_path, encoding="utf-8") as f:
            subtitles_data = yaml.safe_load(f)
            style_presets = subtitles_data.get("style_presets", {})
except Exception as e:
    logger.warning(f"Could not load style presets from config: {e}")

if preset_key not in style_presets:
    # fallback to hardcoded dict of presets inside Python (60 lines)
```

Problems:
- **CWD-dependent**: `Path("config/subtitles.yaml")` fails when the process
  runs from anywhere other than the project root.
- **Bypasses CLI overrides**: the preset block is fresh-read on every call,
  skipping the 3-level merge. A user setting `--subtitle-font-color` can't
  touch a preset's `font_color`.
- **Silent YAML fallback**: the hardcoded Python dict is the same 5 presets
  but the values drift over time (different `font_name` in the Python
  fallback vs the YAML version).
- **No validation**: if a YAML preset has a typo (`font_nam: Montserrat`),
  the code silently drops it.

**Target**: add `style_presets: dict[str, StylePresetConfig]` to `VideoConfig`,
let Pydantic validate the YAML at load time, and make `get_style_config()`
a simple dict lookup:

```python
class StylePresetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str = ""
    font_name: str = "Montserrat"
    font_color: str = "&H00FFFFFF"
    outline_color: str = "&H00000000"
    background_color: str | None = None
    bold: bool = True
    outline_thickness: int = 3
    shadow: bool = True
    effects: list[str] = Field(default_factory=list)
    font_width_to_height_ratio: float = 0.5


# On VideoConfig:
style_presets: dict[str, StylePresetConfig] = Field(
    default_factory=lambda: _default_style_presets()
)
```

The hardcoded Python fallback moves into `_default_style_presets()` as the
`default_factory`. `get_style_config()` shrinks to:
```python
def get_style_config(preset: StylePreset, config: UnifiedSubtitleConfig | None = None,
                    product_id: str | None = None) -> dict[str, Any]:
    # Look up preset from the validated config layer
    preset_config = _video_config().style_presets.get(
        preset.value, _video_config().style_presets["modern"]
    )
    base = preset_config.model_dump()
    # Apply randomization (unchanged)
    return base
```

**Effort**: 0.5 day. The biggest win is removing the file I/O on every
render, the second biggest is forcing YAML presets through Pydantic
validation.

### 4.5 Move font and color pools from Python enums to YAML

**Current**: `src/video/font_color_manager.py:22-40` hardcodes:
```python
class FontFamily(str, Enum):
    MONTSERRAT = "Montserrat-Bold.ttf"
    POPPINS = "Poppins-Bold.ttf"
    GABARITO = "Gabarito-Bold.ttf"
    RUBIK = "Rubik-Bold.ttf"
    DM_SERIF = "DMSerifDisplay-Regular.ttf"  # ← violates best practices, tracked in pycaps-followups #7

class ColorPair(str, Enum):
    CLASSIC = "classic"
    HIGH_CONTRAST = "high_contrast"
    VIBRANT = "vibrant"  # ← amateur palette, also tracked
    WARM = "warm"
    MODERN = "modern"
```

Adding a new font or color pair requires editing Python. The pool is
invisible to YAML users. The "best" pairs (classic white/black, yellow
on black) are hardcoded and mixed in with the amateur ones.

**Target**: define the pools in YAML:
```yaml
subtitle_settings:
  font_pool:
    - name: Montserrat
      file: Montserrat-Black.ttf
      system_fallback: Arial
    - name: Inter
      file: Inter-Black.ttf
      system_fallback: sans-serif
    - name: Poppins
      file: Poppins-Black.ttf
      system_fallback: Arial

  color_pool:
    - name: classic
      font_color: "&H00FFFFFF"
      outline_color: "&H00000000"
      description: "White on black stroke (21:1 contrast, WCAG AAA)"
    - name: yellow_on_black
      font_color: "&H0000FFFF"
      outline_color: "&H00000000"
      description: "Highest-converting highlight color (Submagic)"
```

And two Pydantic models on `SubtitleSettings`:
```python
class FontPoolEntry(BaseModel):
    name: str
    file: str
    system_fallback: str = "Arial"

class ColorPoolEntry(BaseModel):
    name: str
    font_color: str
    outline_color: str
    description: str = ""

class SubtitleSettings(BaseModel):
    ...
    font_pool: list[FontPoolEntry] = Field(default_factory=_default_font_pool)
    color_pool: list[ColorPoolEntry] = Field(default_factory=_default_color_pool)
```

`FontManager` and `ColorManager` in `font_color_manager.py` become thin
loaders over these lists. The md5-keyed random selection stays the same.

**Effort**: 1 day. The main work is updating the tests that reference
`FontFamily.MONTSERRAT` etc. — those need to switch to string lookups.

---

## 5. Dead code to delete

These YAML keys and Pydantic fields have no runtime consumers. Deleting
them reduces the surface area by ~25% with zero functional impact.

### 5.1 Dead keys in `subtitle_settings`

Verified by grep across `src/` — zero hits outside the YAML file itself,
the Pydantic model, and test fixtures that mirror the config shape:

| YAML key | Why it's dead |
|---|---|
| `fallback_y_position` | No consumers. Zero grep hits outside config. |
| `subtitle_space_multiplier` | Duplicate of `video_settings.reserved_space_font_multiplier` (the latter is what the assembler reads). |
| `default_subtitle_reserved_space_percent` | Duplicate of `video_settings.default_subtitle_reserved_space`. |
| `content_aware_font_offset_multiplier` | The consumer reads `text_rendering.content_aware_font_offset_multiplier`, not this one. |
| `subtitle_overlap_threshold` | Only referenced in test fixtures. |
| `word_timestamp_pause_threshold` | Only referenced in test fixtures. |
| `subtitle_split_on_punctuation` | Only referenced in test fixtures. |
| `punctuation_marks` | Only referenced in comments. |

### 5.2 Dead fields in `TextRenderingSettings`

`TextRenderingSettings` has 18 fields. Only 5 are actually consumed:

**Used (keep)**:
- `safe_zone` → subtitle_builder._get_safe_zone, subtitle_positioning
- `narrow_char_width_factor` → unified_subtitle_generator:87
- `wide_char_width_factor` → unified_subtitle_generator:89
- `space_char_width_factor` → unified_subtitle_generator:90
- `content_aware_font_offset_multiplier` → subtitle_builder:1057

**Dead (delete or justify)**:
- `default_margin_fraction` — duplicate of subtitle_settings.margin
- `default_font_size_scale` — duplicate of subtitle_settings.font_size_scale
- `max_chars_per_line` — duplicate of subtitle_settings.max_line_length
- `max_subtitle_duration_sec` — duplicate, never read
- `min_subtitle_duration_sec` — duplicate, never read
- `min_safe_y_position` — duplicate of safe_zone.min_y, marked "backward compat" but has no consumers
- `max_safe_y_position` — same
- `center_position_fraction` — hardcoded `0.5`, never overridden
- `left_position_fraction` — duplicate of safe_zone.min_x
- `right_position_fraction` — duplicate of safe_zone.max_x
- `base_font_size_percent` — shadowed by `SUBTITLE_BASE_FONT_SIZE_PERCENT` constant; `get_font_size()` reads the constant, not this field
- `min_font_size` — shadowed by `SUBTITLE_MIN_FONT_SIZE` constant
- `max_font_size` — shadowed by `SUBTITLE_MAX_FONT_SIZE` constant

**Proposed shape** (after cleanup):
```python
class TextRenderingSettings(BaseModel):
    """Text layout tuning knobs for character-width estimation and
    content-aware subtitle positioning."""
    safe_zone: PlatformSafeZone = Field(default_factory=PlatformSafeZone)
    narrow_char_width_factor: float = 0.4
    wide_char_width_factor: float = 1.2
    space_char_width_factor: float = 0.3
    content_aware_font_offset_multiplier: float = 5.5
```

From 18 fields to 5. Everything else moves to `SubtitleSettings` (canonical
names) or to the `constants.py` module (for values that should not be
runtime-configurable).

### 5.3 Dead fields in `SubtitleSegmentationSettings`

Four fields on the model; only one is read:

**Used**:
- `fallback_segment_duration_sec` → unified_subtitle_generator.py:506

**Dead**:
- `min_words_for_sentence_break`
- `min_words_natural_break`
- `min_words_duration_limit`

The unused three were presumably reserved for a segmentation refactor
that never happened. Delete them or wire them up in a follow-up.

### 5.4 Dead effect fields in `SubtitleEffectsSettings`

24 fields total. All currently have consumers, but per
[pycaps-followups.md](pycaps-followups.md) item #8, the `movement`,
`rotation_bounce`, `glow`, and `scale_pulse` effects are anti-patterns
to be removed. After that removal, these fields become dead:

- `pulse_duration_factor`
- `pulse_scale_max`
- `pulse_scale_normal`
- `bounce_duration_factor`
- `bounce_rotation_max`
- `bounce_rotation_min`
- `bounce_rotation_rest`
- `glow_duration_factor`
- `glow_start_color`
- `glow_end_color`
- `movement_distance_pixels`

That's 11 of 24 fields gone. Keep karaoke_*, fade_duration_ms,
typewriter_char_reveal_max_sec, typewriter_min_timing_ms.

### 5.5 The "Legacy Compatibility Settings" block

`subtitle_settings.font_directory`, `font_name`, `font_size_percent`,
`font_width_to_height_ratio`, `font_color`, `outline_color`, `back_color`,
`bold`, `outline_thickness`, `shadow` form a "legacy" block (commented as
such in YAML:254). They duplicate fields that also live in every style
preset.

The duplication is load-bearing: the drawtext/SRT path in
`SubtitleGraphBuilder.build_subtitle_graph` reads the legacy fields
directly as fallbacks when a style preset doesn't cover them. Deleting
them breaks the drawtext path.

**Fix**: have style presets be the single source of truth. The legacy
block's only role is "what to use when no preset is active", which is
really just "the default preset". Merge the legacy block into the `modern`
preset definition and delete the legacy block.

**Effort**: 1-2 hours. The risk is subtle interactions with randomization:
`font_color_manager` overrides presets' font_color at runtime, and the
override path currently writes into the merged settings dict. This needs
a careful pass.

---

## 6. Duplicate/confusing key names — rename to canonical

After 4.1 (collapsed model) these go away automatically, but if the
refactor is deferred, here's the rename plan:

| Current name | Where | Canonical name |
|---|---|---|
| `max_duration` / `max_subtitle_duration` / `max_subtitle_duration_sec` | YAML / profile / TextRendering | `max_duration` |
| `min_duration` / `min_subtitle_duration` / `min_subtitle_duration_sec` | YAML / profile / TextRendering | `min_duration` |
| `margin` / `default_margin_fraction` | YAML / TextRendering | `margin` |
| `font_size_scale` / `default_font_size_scale` | YAML / TextRendering | `font_size_scale` |
| `max_line_length` / `max_chars_per_line` | YAML / TextRendering | `max_line_length` |
| `font_size_percent` / `base_font_size_percent` | YAML / TextRendering | `font_size_percent` |
| `subtitle_format` (srt/ass) + `subtitle_engine` (ffmpeg/pycaps) | YAML | Consider collapsing to `subtitle_engine: srt \| ass \| pycaps` where srt/ass both imply ffmpeg |

The `subtitle_format` / `subtitle_engine` collapse is debatable — they're
orthogonal concepts today (format is output file, engine is renderer),
but in practice pycaps makes format irrelevant and the two-dimension
choice exposes a matrix where only 3 of 4 cells are meaningful
(ffmpeg+srt, ffmpeg+ass, pycaps). A single `subtitle_engine` with three
values captures the intent without the matrix.

---

## 7. Things to leave alone

Not everything that looks weird needs fixing. These are conscious or
pragmatic choices:

- **The 3-level precedence** (YAML → profile → CLI). It's the right design for a CLI-driven pipeline with batch runs. Keep.
- **Modular YAML files** (subtitles.yaml / video_production.yaml / core.yaml / etc.). Splitting by concern is fine. The naive `dict.update()` merge has edge cases but they haven't bitten anyone. Keep.
- **Separate `SubtitleEffectsSettings`** for ASS effect tuning. It's a leaf model with a clear single consumer (the unified generator's ASS output path). Keep — just prune the dead effect fields after follow-up #8.
- **Separate `TextRenderingSettings`** for character-width estimation. The three width factors are a technical detail unrelated to user-facing subtitle settings. Keep (in its slimmed-down form from §5.2).
- **`VideoConfig.subtitle_settings` being a dict[str, Any]**. Yes, it skips Pydantic validation at that layer, but `MergedSubtitleSettings` picks it up. If §4.1 happens, the dict disappears anyway. If §4.1 doesn't happen, this is acceptable.
- **`PlatformSafeZone` as its own model**. Small, focused, reused by multiple code paths. Keep.
- **`font_color_manager.py` as a standalone module**. The randomization logic is non-trivial and shouldn't be inlined. Keep, just refactor its data source per §4.5.

---

## 8. Suggested order of operations

Bug 3.1 (duration + width) is fixed. Remaining work, in order:

1. ~~**Fix bugs 3.1, 3.2, 3.3**~~ — 3.1 done. **Fix bugs 3.2, 3.3**
   (15 min each). The generator's hardcoded safe zone and the orphaned
   profile safe_zone fields.
2. **Delete dead fields (§5.1, §5.2, §5.3)** (1 hour). Pure subtraction,
   no behaviour change. Good confidence builder.
3. **Move `style_presets` into Pydantic (§4.4)** (0.5 day). Removes
   the CWD-dependent side channel and the hardcoded Python fallback.
4. **Nest `two_part_subtitles` (§4.2)** (0.5 day). Independent of the
   big refactor, already has a clean shape in YAML, just needs the
   Pydantic layer to catch up.
5. **Move font/color pools to YAML (§4.5)** (1 day). Unblocks follow-up
   item #7 from `pycaps-followups.md` (remove bad palettes).
6. **Collapse the two Pydantic models (§4.1)** (1-2 days). The biggest
   refactor; do it last when the surrounding surface area is already
   clean. Include the flat to nested profile override migration (§4.3)
   as part of the same PR.
7. **Rename canonical keys (§6)** (1 hour). After §4.1 this is a small
   YAML-only rename sweep.
8. **Delete the legacy block (§5.5)** (1-2 hours). Gated on §4.4: once
   style presets are authoritative, the legacy fields become pure
   duplicates.

Steps 2-8 should be separate PRs, one per numbered item, to keep
reviews tractable.

---

## 9. What NOT to do

A few tempting directions that would make things worse:

- **Don't add a migration tool that rewrites YAML in place.** The config
  files are human-maintained and carry comments. A rewrite tool would
  strip the comments or botch the formatting. Do renames via documented
  migration guide + compatibility shim that accepts both old and new
  names for one release.

- **Don't move subtitle config into a database or dynamic config service.**
  Tempting for the randomization pools (fonts, colors), but the batch
  pipeline runs are short-lived and the reload-on-every-call pattern
  is fine for the current scale.

- **Don't unify `SubtitleEffectsSettings` into the main `SubtitleSettings` model.**
  Effect tuning (karaoke timing, fade duration) is a leaf concern that
  users almost never touch. Keeping it separate keeps the main
  `SubtitleSettings` surface compact.

- **Don't add a fourth Pydantic layer "to handle X".** Every layer is
  a place where values can silently be dropped (see bug 3.1). If
  anything, remove layers.

- **Don't gate the refactor on "getting everything right in one PR".**
  This is a 5-day cleanup in the optimistic case. Do it in small slices
  — each one passes tests on its own, and the system stays green
  throughout.

---

## 10. Appendix — the bug trace, for reference

### Bug 3.1 (duration + width) — BEFORE the fix

```
YAML max_duration: 2.5
Merged model max_subtitle_duration: 2.5   ← profile override reached this layer
Runtime max_duration: 4.5                  ← LOST at the model→dict→UnifiedSubtitleConfig boundary
```

### Bug 3.1 — AFTER the fix (commit c385df1)

```
YAML max_duration: 2.5
Merged model max_subtitle_duration: 2.5
Runtime max_duration: 2.5                  ← now reaches the generator correctly
```

Verified all 9 profiles pass the best-practice checks:

```
✓ base: OK
✓ product_video_mixed: OK
✓ product_video_primary: OK
✓ product_video_sequential: OK
✓ product_video_single: OK
✓ slideshow_images1: OK
✓ slideshow_images2: OK
✓ slideshow_images3: OK
✓ slideshow_images4: OK
```

### Verification one-liner (reusable for future audits)

```bash
poetry run python -c "
import src.video.config as cfg_mod
cfg = cfg_mod.config
from src.video.subtitle_positioning import create_unified_config_from_settings
for name in cfg.video_profiles:
    m = cfg.get_profile_merged_settings(name)
    uc = create_unified_config_from_settings(m.subtitle_settings.model_dump())
    issues = []
    if uc.max_duration != 2.5: issues.append(f'max_dur={uc.max_duration}')
    if uc.min_duration != 0.6: issues.append(f'min_dur={uc.min_duration}')
    if uc.max_subtitle_width_fraction < 0.75: issues.append(f'width={uc.max_subtitle_width_fraction}')
    if uc.max_words_per_line < 3: issues.append(f'wpl={uc.max_words_per_line}')
    tag = '✗' if issues else '✓'
    print(f'{tag} {name}: {\"  \".join(issues) if issues else \"OK\"}')
"
```
