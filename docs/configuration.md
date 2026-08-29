# Configuration Guide

ContentEngineAI uses a **unified modular configuration system** that splits settings across specialized files with CLI overrides and environment variable support. This guide explains all configuration options and how to customize the system for your needs.

> **📖 For batch processing workflows**: See [Batch Processing](batch-processing.md) for complete batch mode usage examples and automation workflows.

## Configuration Overview

ContentEngineAI implements a **triple-precedence configuration system**:

1. **CLI Arguments** (highest priority)
2. **Environment Variables** (medium priority)
3. **YAML Configuration** (default values)

### Three-Tier Precedence in Detail

The configuration system uses a layered approach where each tier can override values from the tier below it:

```
┌─────────────────────────────────────────────────────────────┐
│  CLI Arguments (Highest Priority)                           │
│  --debug --subtitle-anchor top --font-size-scale 1.2       │
├─────────────────────────────────────────────────────────────┤
│  Environment Variables (Medium Priority)                     │
│  CONTENT_ENGINE_DEBUG=true SUBTITLE_ANCHOR=bottom           │
├─────────────────────────────────────────────────────────────┤
│  YAML Configuration (Default Values)                         │
│  config/core.yaml, config/subtitles.yaml, etc.              │
└─────────────────────────────────────────────────────────────┘
```

**Example: Subtitle Anchor Configuration**

```yaml
# config/subtitles.yaml (YAML default)
subtitle_settings:
  anchor: "below_content"
```

```bash
# .env file (environment override)
SUBTITLE_ANCHOR=bottom
```

```bash
# CLI override (highest priority)
poetry run python -m src.video.producer \
  outputs/B0ASIN123/data.json slideshow_images1 \
  --subtitle-anchor top
```

**Result**: Subtitle anchor will be `top` (CLI wins over all).

**Example: Subtitle engine selector (`ffmpeg` vs `pycaps`)**

The same 3-level precedence applies to the subtitle engine. Bundled
`config/subtitles.yaml` selects `pycaps`; the `SubtitleSettings` Pydantic
field default (used when constructing without YAML) is `ffmpeg`.

```yaml
# config/subtitles.yaml (bundled)
subtitle_settings:
  subtitle_engine: "pycaps"
  pycaps:
    template_pool: ["explosive", "word-focus"]
    renderer: "css"
    fallback_policy: "fallback_ffmpeg"  # forks without --with pycaps land here
```

```yaml
# config/video_production.yaml — per-profile override
profiles:
  slideshow_images1:
    subtitle_engine: "ffmpeg"                  # beats YAML default
    subtitle_settings:
      pycaps:
        template_pool: ["explosive"]           # profile-level override
```

```bash
# CLI override (highest)
poetry run python -m src.video.producer outputs/B0ASIN123/data.json \
    slideshow_images1 --subtitle-engine ffmpeg
```

Nested CLI overrides use dotted keys internally
(`subtitle_settings.pycaps.template_name`). See
[docs/pycaps-subtitles.md](pycaps-subtitles.md) for the full pycaps config
reference and install instructions.

**Complete CLI Override Example:**

```bash
# Override multiple configuration values at runtime
poetry run python -m src.video.producer \
  outputs/B0ASIN123/data.json slideshow_images1 \
  --debug \
  --subtitle-anchor below_content \
  --subtitle-margin 0.08 \
  --content-aware \
  --preset modern \
  --font-size-scale 1.1 \
  --max-line-length 35 \
  --target-platform multi
```

This command:
- Enables debug mode (overrides `debug_mode: false` in YAML)
- Sets subtitle positioning (overrides any env var or YAML value)
- Uses content-aware positioning
- Applies modern style preset with custom font scaling
- Generates metadata for all platforms

### Modular Architecture

The configuration system uses **9 specialized files** instead of a monolithic configuration:

- **`config/core.yaml`** - Global settings and output paths
- **`config/video_production.yaml`** - Video pipeline and effects
- **`config/ai_services.yaml`** - TTS, LLM, and AI providers
- **`config/subtitles.yaml`** - Subtitle positioning and styling
- **`config/performance.yaml`** - Resource limits and optimization
- **`config/scraper.yaml`** - Web scraping and browser settings
- **`config/pipeline.yaml`** - Batch processing and global pipeline settings
- **`config/publisher.yaml`** - Social media publishing via Zernio (published via the legacy Late SDK), plus the `analytics` sweep size
- **`config/url_shortener.yaml`** - URL shortening providers and integration

Machine-specific settings for the scheduled analytics sweep are separate, in a
gitignored `deploy/schedule.env` alongside its committed sample. They shape
systemd unit files rather than application behaviour, so they are read before
any of the loading below applies. See [the publisher docs](publisher.md).

### How Configuration Loading Works

1. **Modular Loading**: Each config file is loaded independently
2. **Environment Resolution**: Variables resolved using `api_key_env_var` mappings
3. **CLI Override**: Command-line parameters override YAML values
4. **Validation**: Pydantic models ensure type safety and completeness

**Example:**
```yaml
# In config/ai_services.yaml
llm_settings:
  provider: "gemini"                     # Primary LLM provider (required)
  api_key_env_var: "GEMINI_API_KEY"      # References env var

# In .env file
GEMINI_API_KEY=your-actual-key-here

# CLI override
poetry run python -m src.video.producer --models "gpt-4"
```

## Configuration Files

ContentEngineAI's modular system organizes settings by purpose:

### 1. **Core Configuration** (`config/core.yaml`)
Global settings and output structure:

```yaml
# Base output directory and structure
global_output_directory: "outputs"
debug_mode: false
pipeline_timeout_sec: 900

# System-wide timeouts for command execution and media analysis
system_timeouts:
  ffprobe_timeout: 10
  xrandr_timeout: 5
  system_profiler_timeout: 10
  head_request_timeout: 10

output_structure:
  product_directory_pattern: "{product_id}"
  product_files:
    scraped_data: "data.json"
    script: "script.txt"
    voiceover: "voiceover.wav"
    subtitles: "subtitles.srt"
    final_video: "video_{product_id}_{profile}.mp4"
  global_dirs:
    cache: "cache"
    logs: "logs"
    reports: "reports"
```

### 2. **Video Production Configuration** (`config/video_production.yaml`)
Video pipeline settings and effects:

```yaml
video_settings:
  resolution: [1080, 1920]  # 9:16 vertical format
  frame_rate: 30
  codec: "libx264"

  # Media validation requirements (must match scraper.yaml)
  min_total_media: 3              # Minimum total media files
  min_images_if_no_video: 5       # Minimum images for slideshow mode
  min_images_with_video: 2        # Minimum images when videos available

audio_settings:
  voiceover_volume_db: 0
  music_volume_db: -20
  music_fade_in_duration: 2.0

video_profiles:
  slideshow_images1:
    description: "Image slideshow optimized for product focus"
    use_scraped_images: true
    use_stock_images: false
```

### 3. **AI Services Configuration** (`config/ai_services.yaml`)
LLM and description generation settings (TTS is in `config/subtitles.yaml`):

```yaml
llm_settings:
  provider: "gemini"                     # Primary provider (required)
  api_key_env_var: "GEMINI_API_KEY"
  models: ["gemini-2.5-flash-lite"]
  temperature: 0.7
  # OpenRouter is an optional fallback (OPENROUTER_API_KEY); see the fallback_provider section below

description_settings:
  enabled: true
  target_platform: "multi"  # youtube, tiktok, instagram, or multi
```

### 4. **Subtitle Configuration** (`config/subtitles.yaml`)
Subtitle positioning, styling, TTS settings, and two-part subtitle system:

```yaml
subtitle_settings:
  enabled: true
  anchor: "below_content"
  style_preset: "modern"  # Available: minimal, modern, bold, animated, random
  content_aware: true
  font_directory: "static/fonts"

# Define custom style presets
style_presets:
  minimal:
    font_name: "Poppins"
    effects: []  # No effects for clean look
    bold: false
  modern:
    font_name: "Montserrat"
    effects: ["karaoke"]
    bold: true
  bold:
    font_name: "Rubik"
    effects: ["fade"]
    bold: true
  animated:
    font_name: "Gabarito"
    effects: ["movement"]
    bold: true
  random:
    # Randomly selects from available fonts, colors, and single effect
    effects: ["fade", "scale_pulse", "rotation_bounce", "glow", "typewriter", "karaoke", "movement"]
```

### 5. **Performance Configuration** (`config/performance.yaml`)
Resource limits and optimization:

```yaml
optimization_settings:
  caching:
    enabled: true
    ttl_seconds: 3600
    max_size_mb: 100
  memory:
    max_memory_mb: 2048

api_settings:
  llm:
    timeout: 90
    max_retries: 5
```

### 6. **Scraper Configuration** (`config/scraper.yaml`)
Web scraping and browser settings with type-safe Pydantic models:

```yaml
global_settings:
  debug_mode: false
  output_config:
    base_directory: "outputs"
    file_patterns:
      product_file: "{keyword}_products.json"

  download_config:
    download_timeout: 30              # HTTP download timeout (seconds)
    video_download_timeout: 300       # Video download timeout (seconds)
    concurrent_image_downloads: 5     # Max parallel image downloads
    concurrent_video_downloads: 3     # Max parallel video downloads

  validation_config:
    # Media validation requirements (must match video_production.yaml)
    min_total_media: 3              # Minimum total media files
    min_images_if_no_video: 5       # Minimum images for slideshow mode
    min_images_with_video: 2        # Minimum images when videos available

scrapers:
  amazon:
    enabled: true
    base_url: "https://www.amazon.com"
    max_products: 10
```

**Type Safety (v0.14.0+)**: The scraper uses Pydantic models (`src/scraper/config_models.py`) for configuration validation. Load with:
```python
from src.scraper.config_adapter import load_scraper_config_pydantic
config = load_scraper_config_pydantic()  # Type-safe ScraperConfig instance
```

### 7. **URL Shortener Configuration** (`config/url_shortener.yaml`)
URL shortening for affiliate links. Two providers ship; trade-offs and the Picsee tag-preservation caveat live in [docs/scraper.md](scraper.md#url-shortener).

```yaml
url_shortener:
  enabled: true                    # Enable/disable URL shortening
  provider: bare                   # Default: bare (no-op). Opt-in: picsee.

  api:
    timeout_sec: 30
    max_retries: 3
    retry_delay_sec: 2
    retry_backoff_multiplier: 2

  # Bare (no-op) provider. Returns input unchanged. No API key needed.
  bare: {}

  # Picsee provider. Opt-in via `provider: picsee` plus PICSEE_API_KEY in .env.
  picsee:
    api_key_env_var: PICSEE_API_KEY
    api_base_url: https://api.pics.ee
    custom_domain: stte.psee.io
    max_bulk_size: 100
    bulk_timeout_multiplier: 2.0

  integration:
    shorten_on_scrape: true        # Run shortener during scrape
    fallback_to_original: true     # On shortener failure, keep canonical URL
```

## Core Configuration Sections

<details>
<summary><strong>1. Global Settings</strong></summary>

### 1. Global Settings

```yaml
# Pipeline execution timeout (seconds)
pipeline_timeout_sec: 900

# Logging configuration
logging_level: "INFO"  # DEBUG, INFO, WARNING, ERROR
debug_mode: false

# Inter-product processing delay
inter_product_delay_range: [30, 60]  # Random delay in seconds
```

**Options:**
- `pipeline_timeout_sec`: Maximum time for entire pipeline (default: 900s)
- `logging_level`: Controls verbosity of logging output
- `debug_mode`: Enables detailed tracing and intermediate file retention
- `inter_product_delay_range`: Random delay between processing multiple products

</details>

<details>
<summary><strong>2. Output Directory Structure</strong></summary>

### 2. Output Directory Structure

ContentEngineAI uses a **simplified, product-centric** directory structure that's fully configurable:

```yaml
# Base output directory - all files go under this directory
global_output_directory: "outputs"

# Product-centric structure configuration
output_structure:
  # Product directory pattern
  product_directory_pattern: "{product_id}"
  
  # Files created within each product directory
  product_files:
    scraped_data: "data.json"           # Scraped product data
    script: "script.txt"                # Generated script
    voiceover: "voiceover.wav"          # Generated audio
    subtitles: "subtitles.srt"          # Generated subtitles
    final_video: "video_{product_id}_{profile}.mp4"  # Final video output
    metadata: "metadata.json"           # Pipeline metadata
    ffmpeg_log: "ffmpeg_command.log"    # FFmpeg execution log
  
  # Subdirectories within each product directory
  product_subdirs:
    images: "images"                    # Product images
    videos: "videos"                    # Product videos  
    music: "music"                      # Background music
    temp: "temp"                        # Temporary files
  
  # Global directories (shared across all products)
  global_dirs:
    cache: "cache"                      # API cache, models
    logs: "logs"                        # Application logs
    reports: "reports"                  # Performance reports
    temp: "temp"                        # Global temp files

# Path building configuration
path_config:
  use_product_oriented_structure: true
  
  cleanup:
    remove_temp_on_success: true        # Auto-cleanup temp files
    keep_temp_on_failure: true          # Preserve debug files
    cache_max_age_hours: 168            # 7-day cache TTL
```

### Directory Structure Example

```
outputs/
├── B0DLKB5V35/                    # Product directory
│   ├── data.json                  # Scraped data
│   ├── script.txt                 # Generated script
│   ├── voiceover.wav              # Generated audio
│   ├── subtitles.srt              # Generated subtitles
│   ├── video_B0DLKB5V35_slideshow_images1.mp4 # Final video
│   ├── metadata.json              # Pipeline metadata
│   ├── ffmpeg_command.log         # FFmpeg execution log
│   ├── images/                    # Product images
│   │   ├── B0DLKB5V35_image_1.jpg
│   │   └── ...
│   ├── videos/                    # Product videos
│   │   ├── B0DLKB5V35_video_1.mp4
│   │   └── ...
│   ├── music/                     # Background music
│   └── temp/                      # Temporary files
├── cache/                         # Global cache
│   └── botasaurus/               # Browser cache
├── logs/                          # Global logs
│   ├── producer.log
│   ├── scraper.log
│   └── debug/
└── reports/                       # Global reports
```

**Key Features:**
- ✅ **No File Conflicts**: Each product in separate directory
- ✅ **Centralized Management**: `src/utils/outputs_paths.py` handles all paths
- ✅ **Auto-Cleanup**: Temp files removed on success, preserved on failure
- ✅ **Configurable**: All paths and patterns controlled via YAML
- ✅ **Cross-Module Consistency**: Same structure used by scraper and producer

**Pattern Variables:**
- `{product_id}`: Product identifier (ASIN for Amazon)
- `{profile}`: Video profile name (e.g., "slideshow_images1")
- `{platform}`: Source platform (e.g., "amazon")
- `{timestamp}`: Current timestamp
- `{ext}`: File extension

</details>

<details>
<summary><strong>3. Video Settings</strong></summary>

### 3. Video Settings

```yaml
video_settings:
  # Output specifications
  resolution: [1080, 1920]           # Width x Height (9:16 aspect ratio)
  frame_rate: 30                     # Frames per second
  output_codec: "libx264"            # Video codec
  output_pixel_format: "yuv420p"     # Pixel format for compatibility
  
  # Duration controls
  default_image_duration_sec: 3      # Default duration for images
  min_visual_segment_duration_sec: 2 # Minimum segment duration
  total_duration_limit_sec: 60       # Maximum video length
  
  # Visual positioning
  image_width_percent: 90            # Image width as % of frame
  image_top_position_percent: 15     # Top position as % from top
  
  # Transitions
  transition_duration_sec: 1.0       # Crossfade transition duration
  transition_type: "fade"            # Transition type
  
  # Quality settings
  min_video_file_size_mb: 1          # Minimum output file size
  video_duration_tolerance_sec: 2    # Acceptable duration variance
```

</details>

<details>
<summary><strong>3.1. Overlay Settings (Disclosure and Hook)</strong></summary>

### 3.1. Overlay Settings

Two text overlays are burned into the frame by the assembler. Both are nested under `video_settings`. They are drawn in a fixed order: subtitles first, then the hook, then the disclosure, so the disclosure always stays on top of the z-order.

**Disclosure overlay** — the persistent FTC affiliate marker. Required for affiliate content; disable only for non-affiliate renders.

```yaml
video_settings:
  disclosure_overlay:
    enabled: true                # Skip the overlay (not recommended for affiliate content)
    text: "#ad"                  # Override for non-English renders, e.g. "#publi"
    position: "top-right"        # top-left, top-right, bottom-left, bottom-right
    size_factor: 0.45            # Font size as a fraction of the subtitle font (0.2-1.0)
    font_color: "white"
    outline_color: "black"
    outline_thickness: 3         # 0 disables the outline
    background_enabled: true     # Semi-transparent box behind the text
    background_color: "black@0.5"  # FFmpeg color@alpha
    margin_x_percent: 0.04       # Distance from the horizontal edge (0.0-0.5)
    margin_y_percent: 0.12       # Distance from the vertical edge (0.0-0.5)
```

`size_factor` sits slightly under the FTC's 50-60% guidance band because the corner placement is tighter than a full-width caption; the rendered font is floored at 8px so a small subtitle base can't produce an illegible disclosure. `margin_y_percent` clears the YouTube Shorts top header and the TikTok username strip.

**Hook overlay** — a short headline held on the opening seconds to win the scroll-past decision.

```yaml
video_settings:
  hook_overlay:
    enabled: true                # Bundled config enables it; the model default is false
    duration_sec: 1.5            # How long the hook is held (0.5-3.0)
    size_factor: 1.1             # Font size as a multiple of the subtitle font (1.0-2.5)
    font_color: "white"
    outline_color: "black"
    outline_thickness: 6
    background_enabled: false    # Outline alone usually reads better here
    background_color: "black@0.5"
    margin_y_percent: 0.28       # Distance from the top of the frame (0.0-0.5)
    max_words: 7                 # Word budget, also passed to the headline prompt (3-12)
    max_width_fraction: 0.78     # Max width of one line as a fraction of the frame (0.5-1.0)
    max_lines: 2                 # Wrap limit before the font shrinks to fit (1-2)
```

The overlay text is an authored headline generated separately from the spoken script, so the hook does not repeat the first caption line; when no headline is available it falls back to the script's first sentence. `max_words` does double duty: it is the cap applied to the generated headline and the budget the headline prompt asks for, so raising it changes what the model writes rather than only truncating afterwards.

Fitting is automatic. Text wraps to at most `max_lines` lines, each kept within `max_width_fraction` of the frame width, and the font shrinks when wrapping alone cannot fit. Text that still does not fit at the minimum size is ellipsized and logged. Because the hook is centred, `max_width_fraction: 0.78` leaves roughly an 11% margin on each side, clearing the platform side insets.

**Bundled config differs from the model defaults** for the hook, the same pattern used by the subtitle engine. The Pydantic defaults (`enabled: false`, `size_factor: 1.35`, `max_width_fraction: 0.9`) apply when constructing settings programmatically without YAML; `config/video_production.yaml` ships the tuned values above.

</details>

<details>
<summary><strong>4. Audio Settings</strong></summary>

### 4. Audio Settings

```yaml
audio_settings:
  # Volume controls (in decibels)
  voiceover_volume_db: 0             # Voiceover volume adjustment
  voiceover_volume_boost_db: 3       # Additional voiceover boost
  music_volume_db: -20               # Background music volume
  music_volume_boost_db: 0           # Additional music boost
  music_min_volume_db: -30           # Minimum music volume
  
  # Mixing settings
  audio_mix_duration: "longest"      # How to handle different audio lengths
  
  # Fade effects
  music_fade_in_sec: 2               # Music fade-in duration
  music_fade_out_sec: 3              # Music fade-out duration
```

</details>

<details>
<summary><strong>5. Subtitle Settings (Unified System)</strong></summary>

### 5. Subtitle Settings (Unified System)

ContentEngineAI uses a unified subtitle positioning system that simplifies configuration while providing powerful content-aware positioning capabilities.

```yaml
subtitle_settings:
  enabled: true

  # Unified Positioning
  anchor: "below_content"            # Positioning anchor: top, center, bottom, above_content, below_content
  margin: 0.1                        # Margin as fraction of frame height (0.0-0.5)
  content_aware: true                # Automatically adjust position based on visual content
  horizontal_alignment: "center"     # Text alignment: left, center, right

  # Style Presets (5 production-ready presets)
  style_preset: "modern"             # Style preset: minimal, modern, bold, animated, random
  font_size_scale: 1.0              # Font size multiplier (0.5-2.0)

  # Text Formatting
  max_line_length: 38                # Maximum characters per line
  max_duration: 4.5                  # Maximum duration for subtitle segments (seconds)
  min_duration: 0.4                  # Minimum duration for subtitle segments (seconds)

  # Randomization Options (for 'random' preset)
  randomize_fonts: false             # Use random font selection from curated collection
  randomize_colors: false            # Use random coordinated color combinations
  randomize_effects: false           # Use random animation effects

  # Manual Overrides (Optional)
  selected_font: null                # Override font selection (font family name)
  selected_color_pair: null          # Override color pair selection

  # Advanced Positioning (Optional Override)
  custom_position:                   # Custom position override (advanced users)
    x: 0.5                          # Horizontal position (0.0-1.0 fraction)
    y: 0.8                          # Vertical position (0.0-1.0 fraction)
```

#### Positioning Anchors

- **`top`**: Position at the top of the frame with margin
- **`center`**: Position at the vertical center of the frame
- **`bottom`**: Position at the bottom of the frame with margin
- **`above_content`**: Position above visual content (content-aware)
- **`below_content`**: Position below visual content (content-aware) - **Recommended**

#### Style Presets

- **`minimal`**: Clean, simple styling with no effects (Arial font)
- **`modern`**: Contemporary look with karaoke effect (Montserrat font, bold) - **Default**
- **`bold`**: High contrast styling with fade effect (Gabarito font, bold)
- **`animated`**: Full animations with movement effect (Gabarito font, bold)
- **`random`**: Deterministic randomization with product-specific fonts, colors, and single effect from available pool

#### Two-Part Subtitle System

Enable dual independent subtitle lines for displaying static product information alongside timed voiceover subtitles:

```yaml
subtitle_settings:
  two_part_subtitles:
    enabled: false  # Enable dual subtitle lines

    # Upper line (static product info)
    upper_line:
      enabled: true
      source_field: "product_url"      # Field from data.json
      anchor: "above_content"          # Position anchor
      margin: 0.03                     # Spacing from content
      font_size_scale: 0.8            # Relative to main subtitles
      style_preset: "minimal"          # Style preset

    # Lower line (voiceover subtitles)
    lower_line:
      enabled: true
      anchor: "below_content"
      margin: 0.05
```

**Use Cases:**
- Display shortened affiliate links while showing voiceover subtitles
- Show product titles or custom text independently from speech
- Create two-line subtitle layouts with different styling

**Features:**
- Independent positioning, styling, and effect randomization per line
- Content-aware positioning for both lines
- Source field configuration for flexible data mapping
- Profile-specific configuration support
- **CTA-synchronized timing**: Upper line appears only during CTA moments

</details>

<details>
<summary><strong>6. CTA Detection Configuration</strong></summary>

### 6. CTA Detection Configuration

CTA (Call-to-Action) detection enables synchronized display of promotional content (like affiliate links) during relevant voiceover moments.

```yaml
cta_detection:
  # Minimum total duration for CTA windows
  # If detected CTA < this value, falls back to full video duration
  min_cta_duration: 0.5              # seconds

  # Fallback duration when voiceover unavailable
  fallback_duration: 9999.0          # seconds (effectively full duration)

  # CTA keyword detection
  keywords:
    - "link"
    - "bio"
    - "check out"
    - "visit"
    - "follow"
    - "share"
    - "like"
    - "subscribe"
    - "click"
    - "tap"
    - "swipe"
    - "purchase"
    - "buy"
    - "shop"
    - "get"

  # Matching behavior
  case_sensitive: false              # Case-insensitive keyword matching
  merge_gap_threshold: 0.5           # Merge CTA segments within 0.5s
```

**Use Cases:**
- Display affiliate links only when user hears "check out the link in bio"
- Show promotional URLs during relevant voiceover moments
- Synchronize calls-to-action with speech patterns

**How It Works:**
1. Analyzes lower subtitle text for CTA keywords
2. Detects timing windows where keywords appear
3. Merges nearby segments (within `merge_gap_threshold`)
4. Displays upper subtitle only during detected windows
5. Falls back to full duration if total CTA time < `min_cta_duration`

</details>

<details>
<summary><strong>7. TTS (Text-to-Speech) Configuration</strong></summary>

### 7. TTS (Text-to-Speech) Configuration

**Location**: `config/subtitles.yaml` (under `tts_config` section)

**Note**: The bundled config's current default TTS path is Gemini voice profiles, selected by voice name (Charon, Puck, etc.), configured under `tts_config.voice_profiles`. See [docs/tts-voice-profiles.md](tts-voice-profiles.md). The `google_cloud` (Chirp 3 HD) and `coqui` schema below is the underlying provider config those profiles build on and the fallback path.

```yaml
tts_config:
  # Provider priority order (first = primary)
  provider_order: ["google_cloud"]   # "coqui" needs extra installs; see below

  # Google Cloud TTS settings
  google_cloud:
    model_name: "en-US-Chirp3-HD"
    language_code: "en-US"

    # Voice selection criteria (priority order)
    voice_selection_criteria:
      - { language_code: "en-US", name_contains: "Chirp3", ssml_gender: "FEMALE" }
      - { language_code: "en-US", name_contains: "Chirp3", ssml_gender: "MALE" }
      - { language_code: "en-US", name_contains: "Neural2", ssml_gender: "FEMALE" }

    # Speech parameters
    speaking_rate: 1.0               # Speech rate (0.25-4.0)
    pitch: 0.0                       # Pitch adjustment (-20.0 to 20.0)
    volume_gain_db: 0.0              # Volume adjustment
    api_timeout_sec: 60

  # Coqui TTS settings (local). Kept so the config side of re-enabling stays a
  # one-line change, but coqui-tts is not installed by default and "coqui" is
  # not in provider_order. Re-enabling also needs transformers <5 and torchcodec
  # from the PyTorch CPU index. See docs/troubleshooting.md.
  coqui:
    model_name: "tts_models/en/ljspeech/vits"
    speaker_name: null               # For multi-speaker models
```

</details>

<details>
<summary><strong>7. LLM Settings</strong></summary>

### 7. LLM Settings

```yaml
llm_settings:
  # Primary provider: Gemini
  provider: "gemini"
  api_key_env_var: "GEMINI_API_KEY"
  models:
    - "gemini-2.5-flash-lite"

  # Gemini doesn't use free model discovery
  auto_select_free_model: false
  random_model_selection: false
  fallback_discover_any_free: false

  # Generation parameters
  temperature: 0.7
  max_tokens: 600
  timeout_seconds: 60

  # Retry configuration
  retry_attempts: 5
  retry_min_wait_sec: 2
  retry_max_wait_sec: 30

  # OpenRouter model discovery filters (used by fallback)
  model_blocklist:
    - "liquid/lfm-2.5-1.2b-instruct:free"
    - "liquid/lfm-2.5-1.2b-instruct"
  min_context_length: 8000

  # Script validation thresholds
  script_validation:
    min_chars: 200
    min_words: 50

  # Prompt configuration
  prompt_template_path: "src/ai/prompts/video_script.md"
  target_audience: "Tech-savvy young adults"

  # Fallback provider: activated when primary exhausts all models
  fallback_provider:
    provider: "openrouter"
    api_key_env_var: "OPENROUTER_API_KEY"
    base_url: "https://openrouter.ai/api/v1"
    prompt_template_path: "src/ai/prompts/video_script.md"
    auto_select_free_model: true
    fallback_discover_any_free: true
    max_tokens: 600
    temperature: 0.7
    timeout_seconds: 60
    models:
      - "tngtech/deepseek-r1t2-chimera:free"
      - "arcee-ai/trinity-large-preview:free"
      - "z-ai/glm-4.5-air:free"
      - "nvidia/nemotron-3-nano-30b-a3b:free"
```

**Platform-Specific Metadata Generation:**

ContentEngineAI now supports generating platform-optimized metadata (titles, descriptions, captions, hashtags) tailored for YouTube Shorts, TikTok, and Instagram Reels. This is configured via the `platform_metadata` section in `config/ai_services.yaml`.

See **Section 7.1: Platform Metadata Settings** below for detailed configuration.

</details>

<details>
<summary><strong>7.1. Platform Metadata Settings (YouTube/TikTok/Instagram)</strong></summary>

### 7.1. Platform Metadata Settings

Platform metadata generation creates platform-specific titles, descriptions/captions, hashtags, and keywords optimized for each social media platform's algorithm and best practices (as of 2025).

**Supported Platforms:**
- **YouTube Shorts**: SEO-optimized titles (50-60 chars), descriptions with first 150 chars critical, 3-5 hashtags including #Shorts
- **TikTok**: Search-friendly captions (100-300 chars optimal), 3-5 niche hashtags, avoids generic viral tags
- **Instagram Reels**: Dual caption styles (ultra-short 3-5 words OR SEO 100-200 chars), 15-30 hashtags in caption

**Configuration:**

```yaml
platform_metadata:
  enabled: true
  target_platform: "multi"  # Options: "youtube", "tiktok", "instagram", "multi"

  # YouTube Shorts Configuration
  youtube:
    title_length_min: 50
    title_length_max: 60
    description_length_max: 5000
    description_seo_priority_chars: 150  # First 150 chars are critical for SEO
    hashtag_count_min: 3
    hashtag_count_max: 5
    require_shorts_tag: true              # Always include #Shorts
    require_ad_tag: true                  # Always include #ad

  # TikTok Configuration
  tiktok:
    caption_length_optimal_min: 100
    caption_length_optimal_max: 300
    caption_length_max: 2200              # Hard limit
    hashtag_count_min: 3
    hashtag_count_max: 5
    require_ad_tag: true                  # Always include #ad
    avoid_generic_tags: true              # Avoid #fyp, #foryoupage, #viral
    # Generic tags blacklist
    generic_hashtags:
      - "#fyp"
      - "#foryoupage"
      - "#foryou"
      - "#viral"

  # Instagram Reels Configuration
  instagram:
    caption_style: "seo"                  # Options: "short" (3-5 words) or "seo" (100-200 chars)
    caption_length_short_min: 3           # For short style (word count)
    caption_length_short_max: 5
    caption_length_seo_min: 100           # For SEO style (character count)
    caption_length_seo_max: 200
    hashtag_count_min: 15
    hashtag_count_max: 30
    hashtags_in_caption: true             # CRITICAL: Hashtags must be in caption, not comments (2024+ algorithm)
    require_ad_tag: true                  # Always include #ad
    emoji_enabled: true                   # Allow emojis in captions
```

**Platform Targeting Modes:**

1. **Single Platform Mode** (`target_platform: "youtube"`, `"tiktok"`, or `"instagram"`):
   - Generates metadata for one platform only
   - Optimized for single-platform distribution
   - Faster generation (single API call)

2. **Multi-Platform Mode** (`target_platform: "multi"`):
   - Generates metadata for all three platforms in parallel
   - Saves separate files: `metadata_youtube.json`, `metadata_tiktok.json`, `metadata_instagram.json`
   - Ideal for cross-platform content distribution

**CLI Override:**

You can override the target platform at runtime using the `--target-platform` argument:

```bash
# Generate YouTube-only metadata
poetry run python -m src.video.producer outputs/B0ASIN123/data.json slideshow_images1 --target-platform youtube

# Generate TikTok-only metadata
poetry run python -m src.video.producer outputs/B0ASIN123/data.json slideshow_images1 --target-platform tiktok

# Generate Instagram-only metadata
poetry run python -m src.video.producer outputs/B0ASIN123/data.json slideshow_images1 --target-platform instagram

# Generate for all platforms (default)
poetry run python -m src.video.producer outputs/B0ASIN123/data.json slideshow_images1 --target-platform multi
```

**Best Practices by Platform:**

**YouTube Shorts:**
- **Title**: 50-60 characters, front-load keywords, use numbers and power words
- **Description**: First 150 characters are CRITICAL for SEO - optimize heavily
- **Hashtags**: Always include #Shorts first, then #ad, then 1-3 niche tags
- **Keywords**: 5-10 search terms users actually type

**TikTok:**
- **Caption**: 100-300 characters (optimal), use exact search phrases users type
- **Hashtags**: 3-5 NICHE-SPECIFIC tags only - avoid #fyp, #foryoupage, #viral (provide NO discovery value as of 2024-2025)
- **SEO Focus**: TikTok is now a search engine - use searchable language, not creative hooks

**Instagram Reels:**
- **Caption Style**: Choose between ultra-short (3-5 words, punchy hooks) OR SEO-descriptive (100-200 chars, searchable)
- **Hashtags**: 15-30 hashtags IN THE CAPTION (not comments) - algorithm prioritizes caption hashtags
- **Mix**: Use 5-10 popular tags (100k-1M posts) + 10-15 niche (10k-100k) + 5-10 specific (<10k)

**Understanding Keywords vs Hashtags:**

Platform metadata includes both **hashtags** and **keywords** - they serve different purposes:

| Type | Purpose | Visibility | Where to Use |
|------|---------|------------|--------------|
| **Hashtags** | Content discovery & categorization | Visible in video (clickable) | Add to description or dedicated hashtag field |
| **Keywords** | SEO & search ranking (backend tags) | Hidden from viewers | YouTube Studio "Tags" field (backend only) |

**How to Use Keywords:**

**YouTube Shorts:**
- Keywords are **critical for SEO** - they help YouTube understand and rank your video in search results
- During upload in YouTube Studio, find the "Tags" or "Keywords" section (below description field)
- Copy keywords from `metadata_youtube.json` → Paste as comma-separated tags in YouTube Studio
- Example keywords: `4K mini projector, portable projector, home theater projector, wifi projector, bluetooth projector`
- Use 5-10 keywords that match actual search terms users type
- **Location**: Backend only - viewers never see these tags

**TikTok:**
- Keywords have **limited SEO value** on TikTok (platform primarily uses hashtags and caption text)
- Generated keywords are for reference/analytics only
- TikTok's algorithm analyzes caption text directly for search ranking
- **Don't manually enter** - no keyword field exists in TikTok upload interface

**Instagram:**
- Instagram has **no keyword field**
- Generated keywords are for reference/analytics tracking only
- Instagram's algorithm relies on hashtags and caption text for discovery
- **Don't manually enter** - no backend tags system exists

**Current Limitation:** Keywords are generated and saved in `metadata_{platform}.json` files but are **not included in `UPLOAD_INSTRUCTIONS.txt`**. You must manually open the JSON files to copy keywords for YouTube uploads.

**Output Files:**

When platform metadata generation is enabled, the following files are created in the product directory root (`outputs/{product_id}/`):

```
outputs/B0ASIN123/
├── description.txt              # Legacy unified description (backward compatible)
├── metadata.json               # Unified metadata
├── metadata_youtube.json       # YouTube Shorts metadata
├── metadata_tiktok.json        # TikTok metadata
├── metadata_instagram.json     # Instagram Reels metadata
└── UPLOAD_INSTRUCTIONS.txt     # Human-readable upload guide (all platforms)
```

**Human-Readable Upload Instructions (`UPLOAD_INSTRUCTIONS.txt`):**

The pipeline automatically generates a ready-to-copy text file with formatted posting instructions for all platforms:

```
================================================================================
                    READY-TO-POST SOCIAL MEDIA CONTENT
                        Product: B0ASIN123
                    Video: video_B0ASIN123_slideshow_images1.mp4
                    URL: https://amazon.com/dp/B0ASIN123
================================================================================

📱 ALL PLATFORMS: Upload the same video file to each platform

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📺 YOUTUBE SHORTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TITLE (Copy below):
──────────────────────────────────────────────────────────────────────────────
Best Wireless Earbuds Under $50 - Amazing Sound Quality
──────────────────────────────────────────────────────────────────────────────

DESCRIPTION (Copy below):
──────────────────────────────────────────────────────────────────────────────
Looking for affordable wireless earbuds with premium sound?...
──────────────────────────────────────────────────────────────────────────────

HASHTAGS (Copy below):
──────────────────────────────────────────────────────────────────────────────
#Shorts #WirelessEarbuds #TechReview #ad
──────────────────────────────────────────────────────────────────────────────
```

This file contains copy-paste-ready content for all three platforms (YouTube, TikTok, Instagram) with clear section separators and formatting guidance.

**Metadata JSON Structure:**

Each platform metadata file follows this structure:

```json
{
  "platform": "youtube",
  "title": "Best Wireless Earbuds Under $50 - Amazing Sound Quality",
  "description": "Looking for affordable wireless earbuds with premium sound?...",
  "hashtags": ["#Shorts", "#WirelessEarbuds", "#TechReview", "#ad"],
  "keywords": ["wireless earbuds under 50", "budget wireless earbuds", ...],
  "character_counts": {
    "title": 58,
    "description": 487
  },
  "generated_at": "2025-01-15T12:00:00Z",
  "product_id": "B0ASIN123",
  "validation_status": "valid",
  "validation_messages": []
}
```

**Validation:**

Platform metadata is automatically validated against platform-specific requirements:
- **YouTube**: Title length (50-60 chars), #Shorts tag presence, hashtag count (3-5)
- **TikTok**: Caption length (optimal 100-300, max 2200), no generic hashtags, hashtag count (3-5)
- **Instagram**: Caption style compliance, hashtag count (15-30), hashtags in caption

Validation failures are logged but don't block generation - invalid metadata is saved with `validation_status: "invalid"` and detailed `validation_messages`.

#### Platform Metadata Enhancement Modules (v0.23.0+)

The platform metadata system includes five enhancement modules for production workflows:

**1. Metadata Caching** (`cache`):
```yaml
platform_metadata:
  cache:
    enabled: true
    ttl_hours: 24              # Cache expiration (1-720 hours)
    cache_dir: ".cache/platform_metadata"
    max_entries: 1000          # LRU eviction when exceeded
```

**2. A/B Testing** (`ab_testing`):
```yaml
platform_metadata_config:
  ab_testing:
    enabled: false
    youtube:
      enabled: true
      variants:
        - name: "control"
          template_path: "src/ai/prompts/youtube_metadata.md"
          weight: 50
```

**3. Batch Generation** (`batch`):
```yaml
platform_metadata_config:
  batch:
    enabled: true
    max_concurrent: 3          # Parallel product processing (1-20)
    log_progress: true         # [N/total] format logging
```

**4. Multi-Format Export** (`export`):
```yaml
platform_metadata_config:
  export:
    enabled: true
    default_format: "json"     # json, csv, youtube_csv, tiktok, instagram
    youtube_category: "22"     # People & Blogs
    youtube_privacy: "private"
    csv_encoding: "utf-8-sig"  # Excel compatibility
```

**5. Trend-Aware Hashtags** (`trends`):
```yaml
platform_metadata_config:
  trends:
    enabled: false
    provider: "static"         # static, mock (future: external APIs)
    cache_ttl_hours: 4
    max_trending_tags: 2
    fallback_tags:
      youtube: ["#Shorts", "#Trending", "#Tech"]
      tiktok: ["#ForYou", "#Viral", "#TechTok"]
      instagram: ["#Reels", "#InstaGood", "#Innovation"]
```

**Programmatic Usage**:
```python
from src.ai.platform_metadata import (
    MetadataCache, BatchMetadataGenerator, MetadataExporter,
    TrendAwareHashtagGenerator, PromptVariantSelector
)

# Caching
cache = MetadataCache(cache_settings)
cached = cache.get(product_id, "youtube", product)

# Batch generation
batch_gen = BatchMetadataGenerator(max_concurrent=5)
results = await batch_gen.generate_batch(products, settings, ...)

# Export
exporter = MetadataExporter()
exporter.export_all_formats(metadata_list, output_dir)

# Trends
trend_gen = TrendAwareHashtagGenerator(trend_settings)
tags = await trend_gen.merge_trending_tags("tiktok", existing_tags)
```

</details>

<details>
<summary><strong>7.2. Script Templates and Content Pillars</strong></summary>

### 7.2. Script Templates and Content Pillars

The `script_templates` sub-section of `llm_settings` (in `config/ai_services.yaml`) controls multi-template script generation, channel-wide voice direction, and the content pillars system.

```yaml
llm_settings:
  # ... (other LLM settings; see section 7)
  script_templates:
    enabled: true
    templates_dir: "src/ai/prompts/scripts"
    template_pool: []                 # Empty = use all .md templates in templates_dir
    fixed_template: null              # null = deterministic random per product; or a name to force

    # Pillar -> list of template names. A template can appear under multiple pillars.
    pillars:
      value:
        - classic_promo
        - rapid_fire
        - social_proof
        - myth_buster
        - comparison
        - skeptic_converted
      novelty:
        - classic_promo
        - curiosity_hook
        - secret_reveal
        - story_driven
        - unboxing_reaction
        - skeptic_converted
      utility:
        - classic_promo
        - before_after
        - comparison
        - lifestyle_flex
        - myth_buster
        - problem_solution
        - question_driven
        - challenge_dare

    # Pillar -> preamble string prepended to the LLM prompt at runtime.
    pillar_preambles:
      value: |-
        Pillar context: this video runs under the value pillar. Lean into the deal-pitch angle...
      novelty: |-
        Pillar context: this video runs under the novelty pillar. Lean into the discovery angle...
      utility: |-
        Pillar context: this video runs under the utility pillar. Lean into the problem-solving angle...

    # Pillar -> audience hint substituted into {AUDIENCE} when pillar is set.
    pillar_audiences:
      value: "Budget-conscious shoppers looking for products that punch above their price."
      novelty: "Curious people who enjoy discovering lesser-known products before everyone else."
      utility: "Practical buyers solving a real, recurring problem with the right tool."

    # Channel-wide voice direction prepended above any pillar preamble.
    # Carries banned phrases, word target, anti-AI-tells, persona anchor, and a voice example.
    narrator_profile: |-
      Narrator profile: every video on this channel is voiced by the same person...
      # (full profile — see config/ai_services.yaml in the repo for the shipped default)
```

**Field reference:**

| Field | Type | Purpose |
|---|---|---|
| `enabled` | bool | Master switch. When false, falls back to the single `prompt_template_path`. |
| `templates_dir` | path | Where the `.md` template files live. |
| `template_pool` | list[str] | Names (without `.md`) eligible for selection. Empty = all templates in `templates_dir`. |
| `fixed_template` | str \| null | Force one template for every product. Null = deterministic per-product MD5 selection. |
| `pillars` | dict[str, list[str]] | Pillar name -> templates that fit it. A template can be in multiple pillars. Empty dict disables pillar filtering. |
| `pillar_preambles` | dict[str, str] | Pillar name -> preamble string prepended to the LLM prompt when that pillar is set. Empty dict disables preamble injection. |
| `pillar_audiences` | dict[str, str] | Pillar name -> audience hint substituted into the `{AUDIENCE}` placeholder. Falls back to `target_audience` when missing. |
| `narrator_profile` | str | Channel-wide voice direction prepended to every script prompt. Empty string disables narrator profile injection. |
| `topic_templates` | list[str] | Names eligible when the record came from a topic rather than a scraped product. Replaces the pool rather than narrowing it, and is excluded from the product pool. Empty list disables the split, which renders topics through product templates. |
| `narrator_profile_topic` | str | Voice direction for topic scripts. Empty string falls back to `narrator_profile`, whose call-to-action list points at something to buy. |
| `pillar_preambles_topic` | dict[str, str] | Topic counterpart to `pillar_preambles`, using the same pillar keys. Read instead of the product map on a topic render. **Empty dict falls back to the product map**, which describes a product fixing an annoyance and lands above the topic prompt's rule against naming one. |
| `pillar_audiences_topic` | dict[str, str] | Topic counterpart to `pillar_audiences`, same keys. Empty dict falls back to the product map, which describes buyers and shoppers. |

**Runtime prompt structure:**

```
[narrator_profile]      <- always, when non-empty; narrator_profile_topic
                           instead for a topic render, when set
[pillar_preambles[X]]   <- when --pillar X is set and the entry exists;
                           pillar_preambles_topic[X] instead for a topic
                           render, when that map is non-empty
[template content]      <- selected template, with {FULL_PRODUCT_NAME},
                           {SHORT_PRODUCT_NAME}, {PRODUCT_DESCRIPTION},
                           {AUDIENCE} substituted. Topic templates use
                           {TOPIC_TITLE} and {TOPIC_DETAIL}; all six are
                           always passed, and a template uses what it names.
```

**Selection rules:**

1. If `fixed_template` is set, that template wins.
2. Otherwise, the active pool is `template_pool` (or all templates when empty).
   For a scraped product, anything in `topic_templates` is then removed: the two
   families share one directory and the default pool is a glob over it.
   For a topic render, the pool is replaced by `topic_templates` outright, since
   a product template left reachable renders the topic as an advertisement.
3. When `--pillar <name>` is set and `pillars[name]` exists, the active pool is intersected with `pillars[name]`. If the intersection is empty, the unfiltered pool is used and a warning is logged. On a topic render the intersection is always empty by design, because `pillars` lists product templates and the pool is the topic family, so that case logs at debug and the pillar acts through its preamble and audience only.
4. Selection within the pool is deterministic per product (salted MD5 hash of `<product_id>:script_template`).

**Pillar resolution order:** `--pillar <name>` on `src/video/producer/cli.py` or `src/pipeline/global_batch.py`, then the pillar a previous run of the same product recorded, then the product record's own value, which the scraper attaches from the source keyword's configured group. Unknown pillar names log an info-level hint and gracefully no-op (no template filter, no preamble, no audience override); the run still completes.

See [Requirements](requirements.md) "Content Pillars" for the behavior contract.

</details>

<details>
<summary><strong>8. Stock Media Settings</strong></summary>

### 8. Stock Media Settings

```yaml
stock_media_settings:
  pexels:
    enabled: true
    api_key_env_var: "PEXELS_API_KEY"
    source_name: "Pexels"
    
    # Media preferences
    orientation: "portrait"          # portrait, landscape, square
    size: "large"                    # small, medium, large
    
    # Download settings
    concurrent_downloads: 3          # Parallel downloads
    timeout_sec: 30
    
    # Quality filters
    min_width: 1080                  # Minimum image width
    min_height: 1920                 # Minimum image height
```

#### Search phrases from the script

A profile that draws no visual from the scraped product (`slideshow_stock`)
writes the script first and then asks what to put on screen for it. Configure
this under `llm_settings` in `config/ai_services.yaml`:

```yaml
llm_settings:
  visual_search_terms:
    enabled: true
    max_phrases: 3            # one library search per phrase
    max_words_per_phrase: 5
```

Each phrase is searched on its own and the results pooled, so `max_phrases` is
how many different shots a render draws on. Passing the phrases together would
not work: the provider concatenates a keyword list into a single query, and the
library answers a long query with one page of loosely relevant results skewed
toward whichever phrase dominates. Some phrases go unrepresented entirely, so
the render is short of the shots the script asked for.

Set `enabled: false` to search the topic title and the profile's own keywords
instead. Deriving the phrases never blocks a render either way: no API key, a
provider failure, or an unusable answer leaves the existing search terms in
place. A profile that shows product photography ignores this section.

</details>

<details>
<summary><strong>9. Freesound Audio Settings</strong></summary>

### 9. Freesound Audio Settings

ContentEngineAI uses Freesound.org to automatically download Creative Commons licensed background music that matches your video duration. The system supports both preview downloads (with API key only) and full-quality downloads (with OAuth2 authentication).

#### Quick Start (API Key Only)

For basic usage with preview-quality music:

```yaml
audio_settings:
  # Freesound API key (required)
  freesound_api_key_env_var: "FREESOUND_API_KEY"

  # Search configuration
  freesound_search_query: "upbeat instrumental corporate"
  freesound_filters: "duration:[60 TO 180]"
  freesound_sort: "rating_desc"
  freesound_max_results: 15

  # Timeouts
  freesound_api_timeout_sec: 30
  freesound_download_timeout_sec: 300
```

**Setup:**
1. Get free API key: https://freesound.org/apiv2/apply/
2. Add to `.env`: `FREESOUND_API_KEY=your_api_key_here`
3. System automatically downloads preview-quality MP3s matching video duration

#### OAuth2 Setup (Full-Quality Downloads)

For original quality audio downloads, configure OAuth2 authentication:

**Step 1: Register Your Application**

1. Visit Freesound API registration: https://freesound.org/apiv2/apply/
2. Fill in application details:
   - **Name**: "ContentEngineAI" (or your project name)
   - **Description**: "Automated video production pipeline"
   - **Redirect URI**: `http://localhost:8000/callback` (for local testing)
   - **Accepted Terms**: Check the box to accept Freesound API terms
3. Click "Apply" and wait for approval (usually instant)
4. Note down your **Client ID** and **Client Secret**

**Step 2: Get Refresh Token**

Use the Freesound OAuth2 helper script to obtain a refresh token:

```bash
# Install required dependencies (already in pyproject.toml)
poetry install

# Run OAuth2 authorization flow
poetry run python tools/freesound_oauth2_setup.py \
  --client-id YOUR_CLIENT_ID \
  --client-secret YOUR_CLIENT_SECRET
```

**Script will:**
1. Print authorization URL for Freesound
2. You open URL in browser, log in, and approve access
3. Copy authorization code from redirect URL
4. Paste code into script when prompted
5. Script exchanges code for access + refresh tokens
6. Refresh token printed to console

**Step 3: Configure Environment Variables**

Add OAuth2 credentials to `.env`:

```bash
# Required for all Freesound operations
FREESOUND_API_KEY=your_api_key_here

# Optional - for full-quality downloads
FREESOUND_CLIENT_ID=your_client_id_here
FREESOUND_CLIENT_SECRET=your_client_secret_here
FREESOUND_REFRESH_TOKEN=your_refresh_token_here
```

**Step 4: Verify Configuration**

Test OAuth2 authentication:

```bash
# Test token refresh and download
poetry run python -c "
from src.audio.freesound_client import FreesoundClient
import asyncio
import os

async def test():
    client = FreesoundClient(
        FREESOUND_API_KEY=os.getenv('FREESOUND_API_KEY'),
        FREESOUND_CLIENT_ID=os.getenv('FREESOUND_CLIENT_ID'),
        FREESOUND_CLIENT_SECRET=os.getenv('FREESOUND_CLIENT_SECRET'),
        FREESOUND_REFRESH_TOKEN=os.getenv('FREESOUND_REFRESH_TOKEN')
    )
    success = await client.refresh_oauth_token()
    print('✓ OAuth2 configured correctly' if success else '✗ OAuth2 failed')

asyncio.run(test())
"
```

#### Token Refresh and Persistence

**Automatic Token Management:**
- Access tokens expire after 1 hour (3600 seconds)
- System automatically refreshes tokens 60 seconds before expiration
- New refresh tokens are saved to `.env` file using `dotenv.set_key()`
- No manual intervention required after initial setup

**Token Refresh Configuration:**

```yaml
audio_settings:
  # Token expiration time (Freesound default: 3600s)
  freesound_token_expiry_sec: 3600

  # Refresh buffer - triggers refresh this many seconds before expiry
  # Recommendation: 60s provides safety margin
  freesound_token_refresh_buffer_sec: 60
```

**Manual Token Refresh:**

If refresh token becomes invalid or expires, regenerate using OAuth2 setup script:

```bash
poetry run python tools/freesound_oauth2_setup.py \
  --client-id YOUR_CLIENT_ID \
  --client-secret YOUR_CLIENT_SECRET
```

**Troubleshooting Token Refresh:**

If token refresh fails with timeout errors:

1. **Check network connectivity**: Ensure you can reach `https://freesound.org`
2. **Verify credentials**: Confirm `FREESOUND_CLIENT_ID`, `FREESOUND_CLIENT_SECRET`, and `FREESOUND_REFRESH_TOKEN` are correct in `.env`
3. **Regenerate token**: If refresh token is expired or invalid, run OAuth2 setup script again
4. **Check timeout settings**: Default is 5s - increase if needed in `config/video_production.yaml`:
   ```yaml
   audio_settings:
     freesound_token_refresh:
       timeout_sec: 10  # Increase if network is slow
   ```
5. **Fallback behavior**: System automatically falls back to HQ preview downloads if OAuth2 fails

**Token Storage Location:**
- Primary: `.env` file in project root (automatically updated by system)
- Format: `FREESOUND_REFRESH_TOKEN=your_token_here`
- Auto-update: New refresh tokens are saved automatically using `dotenv.set_key()`

#### Search Configuration

**Basic Search:**

```yaml
audio_settings:
  # Search query (use descriptive keywords)
  freesound_search_query: "upbeat instrumental corporate"

  # Filter by duration range (seconds)
  freesound_filters: "duration:[60 TO 180]"

  # Sort order (rating_desc recommended for quality)
  freesound_sort: "rating_desc"

  # Max results to fetch (10-20 recommended)
  freesound_max_results: 15
```

**Advanced Filtering:**

```yaml
# Multiple filters example
freesound_filters: "duration:[60 TO 180] license:\"Creative Commons 0\""

# Short clips for quick videos
freesound_filters: "duration:[10 TO 30]"

# Public domain only
freesound_filters: "license:\"Creative Commons 0\""

# Multiple tags
freesound_filters: "tag:music tag:background tag:corporate"
```

**Filter Syntax:**
- Duration: `duration:[MIN TO MAX]` (seconds)
- License: `license:"Creative Commons 0"` (exact match)
- Tags: `tag:keyword` (multiple allowed)
- Bitrate: `bitrate:[MIN TO MAX]` (kbps)
- Sample rate: `samplerate:[MIN TO MAX]` (Hz)

**Sort Options:**
- `rating_desc` - Best rated tracks first (recommended)
- `duration_asc` - Shortest tracks first
- `duration_desc` - Longest tracks first
- `created_desc` - Newest tracks first
- `downloads_desc` - Most downloaded first

#### Circuit Breaker Configuration

ContentEngineAI uses a circuit breaker pattern to prevent wasting time on unavailable APIs during batch processing.

**How It Works:**

1. **Closed State (Normal)**: All API calls proceed normally
2. **Open State (Failed)**: After repeated failures, circuit opens and API calls fast-fail
3. **Half-Open State (Testing)**: After timeout, system tests if API recovered

**Configuration:**

Circuit breaker settings are in `src/utils/circuit_breaker.py`:

```python
freesound_circuit_breaker = CircuitBreaker(
    failure_threshold=5,        # Open after 5 consecutive failures
    timeout=60,                 # Stay open for 60 seconds
    recovery_timeout=30,        # Test recovery after 30 seconds
    expected_exception=Exception
)
```

**Tuning Guidelines:**

- **failure_threshold**: Lower = faster fallback, Higher = more API tolerance
  - Recommended: 3-5 for production, 10+ for testing
- **timeout**: How long to skip API after opening
  - Recommended: 60-300 seconds for batch processing
- **recovery_timeout**: How long before testing API recovery
  - Recommended: 30-60 seconds

**Monitoring Circuit State:**

```python
from src.utils.circuit_breaker import freesound_circuit_breaker

# Check current state
print(f"Circuit state: {freesound_circuit_breaker.state}")
print(f"Failure count: {freesound_circuit_breaker.failure_count}")

# Manually reset circuit
freesound_circuit_breaker.reset()
```

#### Fallback Behavior

ContentEngineAI implements a **three-tier fallback system** for music selection:

**Tier 1: OAuth2 Full Downloads** (Best Quality)
- Original format and quality from Freesound
- Requires OAuth2 credentials
- Fallback on failure: Tier 2

**Tier 2: API Key Preview Downloads** (Good Quality)
- MP3 preview quality from Freesound
- Requires only API key
- Fallback on failure: Tier 3

**Tier 3: Local Files** (Guaranteed Availability)
- Uses files from `background_music_paths` config
- Random selection from available files
- Memory-mapped I/O for files >1MB

**Fallback Triggers:**
- OAuth2 credentials missing or invalid → Tier 2
- API timeouts or rate limits → Tier 2/3
- Circuit breaker open → Tier 3 (fast-fail)
- Network errors → Tier 2/3
- No suitable tracks found → Tier 3

**Local Fallback Configuration:**

```yaml
audio_settings:
  background_music_paths:
    - "static/background-music-calm-soft-334182.mp3"
    - "static/background-music-happy-333014.mp3"
    - "static/background-music-upbeat-energetic-333016.mp3"
```

**Add Your Own Music:**
1. Place MP3/WAV files in `static/` directory
2. Add file paths to `background_music_paths` list
3. System randomly selects from available files

</details>

<details>
<summary><strong>10. Speech-to-Text Settings</strong></summary>

### 10. Speech-to-Text Settings

```yaml
# Whisper STT settings (primary)
whisper_settings:
  enabled: true
  model_size: "small"                # tiny, base, small, medium, large
  language: "en"                     # Language code
  device: "cpu"                      # auto, cpu, cuda
  model_device: "cpu"                # Device for model inference
  model_in_memory: false             # Keep model in memory between uses
  fp16: false                        # Use 16-bit floating point
  beam_size: 5                       # Beam search size
  temperature: 0.0                   # Sampling temperature
  compression_ratio_threshold: 2.4   # Detect repetitive text
  logprob_threshold: -1.0            # Filter low-confidence words
  no_speech_threshold: 0.2           # Detect silence vs speech
  condition_on_previous_text: true   # Use context for accuracy

  # Timeout settings (configurable for system performance)
  base_timeout_sec: 120              # Base timeout before audio duration added
  duration_multiplier: 15.0          # Multiplier for audio duration (timeout = base + duration * multiplier)
  max_timeout_sec: 1800              # Maximum timeout (30 minutes)
  timeout_retry_attempts: 1          # Retries after a timeout, each on a wider limit
  timeout_retry_multiplier: 2.0      # How much wider each retry's limit is
  progress_monitor_interval_sec: 30  # Progress monitoring interval
  enable_resource_monitoring: true   # Monitor CPU/memory during transcription
  enable_resource_cleanup: true      # Cleanup resources after processing

# Google Cloud STT settings (fallback)
google_cloud_stt_settings:
  enabled: true
  api_key_env_var: "GOOGLE_APPLICATION_CREDENTIALS"
  language_code: "en-US"
  enable_word_time_offsets: true     # Required for subtitle synchronization
  use_enhanced: true                 # Use enhanced models
  sample_rate_hertz: 16000
  encoding: "LINEAR16"
```

</details>

<details>
<summary><strong>11. FFmpeg Settings</strong></summary>

### 11. FFmpeg Settings

```yaml
ffmpeg_settings:
  # Executable configuration
  ffmpeg_path: "ffmpeg"              # Path to FFmpeg executable
  ffprobe_path: "ffprobe"            # Path to FFprobe executable

  # I/O timeout prevention
  rw_timeout_microseconds: 30000000  # 30 seconds timeout for file operations

  # Filter options
  enable_zoompan: false              # Enable zoom/pan effect on images
  zoompan_duration: 1.0              # Zoom effect duration

  # Debug options
  save_command: true                 # Save FFmpeg command to log file
  show_debug_info: false             # Show debug overlay on video

  # Verification settings
  verify_streams: true               # Verify video/audio streams exist
  verify_duration: true              # Check final video duration
  verify_subtitles: true             # Verify subtitle content
```

</details>

<details>
<summary><strong>12. Video Profiles with Per-Profile Settings</strong></summary>

### 12. Video Profiles with Per-Profile Settings

Video profiles define different strategies for media selection and support per-profile overrides for all visual settings. Each profile can customize image positioning, subtitle styling, and other visual parameters independently.

#### Video Assembly Modes

ContentEngineAI supports multiple video assembly strategies that determine how product videos and images are combined:

**Product Video Profiles** (prioritize product videos):

1. **`product_video_sequential`** - Concatenates all videos sequentially with crossfades
   - **Single video handling**: Loops video multiple times (3x, 4x, etc.) if too short, trims with fade-out if too long
   - **Best for**: Showcasing multiple product angles/demos

2. **`product_video_single`** - Uses longest video with seamless looping
   - **Single video handling**: Loops seamlessly with crossfade transitions at loop points
   - **Best for**: Single-angle product demos with smooth repetition

3. **`product_video_mixed`** - Interleaves videos and images throughout
   - **Single video handling**: Places video at full duration, distributes images around it (images → video → images)
   - **Best for**: Dynamic visual variety with mixed content

4. **`product_video_primary`** - All videos first, then images
   - **Single video handling**: Uses video once (no looping), fills remainder with images, trims if too long
   - **Best for**: Ensuring all video content is shown while meeting duration requirements

**Slideshow Profiles** (image-focused, videos ignored):

1. **`slideshow_images1-4`** - Image-only slideshows with different styling
   - Uses assembly modes: `single_best`, `mixed_media`, or `video_first_fallback`
   - **Video handling**: Ignores product videos entirely, uses only images

```yaml
video_profiles:
  slideshow_images1:
    description: "Image slideshow optimized for product focus"
    use_scraped_images: true
    use_scraped_videos: false
    use_stock_images: false
    use_stock_videos: false
    use_dynamic_image_count: true

    # Profile-Specific Image Settings
    image_width_percent: 0.85         # 85% frame width for product focus

    # Profile-Specific Subtitle Settings (nested override block)
    subtitle_settings:
      randomize_effects: true         # Enable effect randomization

  slideshow_images2:
    description: "Alternative slideshow with different styling"
    use_scraped_images: true
    use_scraped_videos: false
    use_stock_images: false
    use_stock_videos: false
    use_dynamic_image_count: true

    # Image positioning
    image_width_percent: 0.80         # 80% frame width
    image_top_position_percent: 0.15  # Position 15% from top

    # Subtitle overrides — single nested block, only fields that differ from
    # the global subtitle_settings need to be set. Nested sub-blocks (pycaps,
    # two_part_subtitles, safe_zone) deep-merge per-field.
    subtitle_settings:
      anchor: "below_content"
      margin: 0.08
      content_aware: true
      horizontal_alignment: "center"
      style_preset: "minimal"
      font_size_scale: 0.9
      randomize_fonts: true
      randomize_colors: true
      randomize_effects: false
      max_line_length: 28
      max_words_per_line: 3
      max_subtitle_width_fraction: 0.85
      max_duration: 2.5
      min_duration: 0.6
```

**Available Per-Profile Overrides:**

```yaml
# Image Settings (all optional, top-level on the profile)
image_width_percent: 0.85            # Override global image width
image_top_position_percent: 0.15     # Override global image position
image_vertical_align: "top"          # Override global vertical alignment ("top" or "center")
preserve_aspect_ratio: true          # Override aspect ratio setting

# Subtitle Settings — single nested block. Any field on the global
# subtitle_settings can be overridden here except subtitle_format, which is
# global-only because the subtitle file's extension is derived from it; unset
# fields inherit from global.
subtitle_settings:
  anchor: "below_content"            # top, center, bottom, above_content, below_content
  margin: 0.08                       # Margin as fraction of frame height (0.0-0.5)
  content_aware: true
  horizontal_alignment: "center"     # left, center, right
  style_preset: "modern"             # minimal, modern, bold, animated, random
  font_size_scale: 1.1               # 0.5-2.0
  max_line_length: 35
  max_words_per_line: 3
  max_subtitle_width_fraction: 0.85
  max_duration: 2.5                  # Canonical name (was max_subtitle_duration)
  min_duration: 0.6                  # Canonical name (was min_subtitle_duration)
  randomize_fonts: false
  randomize_colors: false
  randomize_effects: false
  pycaps:                            # Nested pycaps overrides (deep-merged)
    template_name: "hype"
    renderer: "css"
  two_part_subtitles:                # Nested two-part overrides (deep-merged)
    enabled: true
    upper_line:
      style_preset: "minimal"
  safe_zone:                         # Nested safe-zone overrides (deep-merged)
    max_y: 0.60                      # Tighter bottom than the 0.651 default
```

**Key Features:**
- **Individual Customization**: Each profile can override any global setting except `subtitle_format`, which is global-only
- **Selective Overrides**: Only specify settings you want to change
- **Fallback System**: Unspecified settings use global defaults
- **Type Safety**: All overrides validated by Pydantic models
- **CLI Override Support**: All profile settings can be overridden via command-line arguments

</details>

## Timeout Configuration

Timeouts are configured in different files based on component:

### 1. System Timeouts (`config/core.yaml`)
Global timeouts for external command execution and basic connectivity checks.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `system_timeouts.ffprobe_timeout` | int | 10 | Timeout for media analysis (seconds) |
| `system_timeouts.xrandr_timeout` | int | 5 | Timeout for monitor detection on Linux (seconds) |
| `system_timeouts.system_profiler_timeout` | int | 10 | Timeout for monitor detection on macOS (seconds) |
| `system_timeouts.head_request_timeout` | int | 10 | Default timeout for HTTP HEAD requests (seconds) |

**Global Pipeline** (`config/core.yaml`):
```yaml
pipeline_timeout_sec: 900  # Total pipeline execution timeout
```

**FFmpeg Operations** (`config/performance.yaml`):
```yaml
ffmpeg_settings:
  final_assembly_timeout_sec: 600  # Video assembly timeout
  rw_timeout_microseconds: 30000000  # I/O timeout (30 seconds)
```

**API Timeouts** (`config/performance.yaml`):
```yaml
api_settings:
  downloads:
    timeout_sec: 30
  tts:
    request_timeout_sec: 60
  stock_media:
    request_timeout_sec: 30
  general:
    default_request_timeout_sec: 15
```

**Whisper STT** (`config/ai_services.yaml`):
```yaml
whisper_settings:
  base_timeout_sec: 120
  duration_multiplier: 15.0
  max_timeout_sec: 1800
  timeout_retry_attempts: 1
```

The limit is derived from audio duration, which says nothing about how fast the
machine transcribes. A contended machine can cross it, and the step sits after
the script and the voiceover, so a timeout discards a render that has already
been paid for. One retry on a widened limit costs far less than that loss; set
`timeout_retry_attempts: 0` to restore the single attempt.

## Environment Variables

ContentEngineAI uses environment variables for sensitive credentials and runtime configuration overrides. Copy `.env.example` to `.env` and configure your values.

### Required API Keys

These secrets are validated at startup. The system will exit with an error if any are missing.

| Variable | Type | Description | Setup URL |
|----------|------|-------------|-----------|
| `GEMINI_API_KEY` | string | Google Gemini API key for LLM script/description generation | https://aistudio.google.com/apikey |
| `PEXELS_API_KEY` | string | Pexels API key for stock images/videos | https://www.pexels.com/api/ |
| `FREESOUND_API_KEY` | string | Freesound API key for background music | https://freesound.org/apiv2/apply/ |

### Optional API Keys

These enhance functionality but are not required for basic operation.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OPENROUTER_API_KEY` | string | None | OpenRouter API key for LLM fallback when Gemini fails |
| `GOOGLE_APPLICATION_CREDENTIALS` | path | None | Path to Google Cloud service account JSON for TTS |
| `LATE_API_KEY` | string | None | Zernio API key for social media publishing, still named for the legacy Late SDK (alt: `PUBLISHER_API_KEY`) |
| `PICSEE_API_KEY` | string | None | Picsee API key for URL shortening |
| `AMAZON_ASSOCIATE_TAG` | string | None | Amazon Associates affiliate tag for monetization |
| `AMAZON_AFFILIATE_LINKS_ENABLED` | bool | `true` | Set false when no affiliate program is in use. Overrides `scrapers.amazon.affiliate_links.enabled` |
| `LNKBIO_CLIENT_ID` | string | None | Lnk.Bio OAuth2 client ID for link-in-bio |
| `LNKBIO_CLIENT_SECRET` | string | None | Lnk.Bio OAuth2 client secret |

### Freesound OAuth2 (Full-Quality Downloads)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `FREESOUND_CLIENT_ID` | string | None | OAuth2 client ID for full-quality audio |
| `FREESOUND_CLIENT_SECRET` | string | None | OAuth2 client secret |
| `FREESOUND_REFRESH_TOKEN` | string | None | OAuth2 refresh token (auto-updated by system) |

### Runtime Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CONTENT_ENGINE_DEBUG` | bool | false | Enable debug mode (alt: `DEBUG_MODE`) |
| `CONTENT_ENGINE_OUTPUT` | string | outputs | Base output directory (alt: `OUTPUTS_DIR`) |
| `CONTENT_ENGINE_TIMEOUT` | int | 300 | Pipeline timeout in seconds |
| `FFMPEG_THREADS` | int | 0 | FFmpeg threads (0 = auto-detect) |

### Subtitle Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SUBTITLE_ANCHOR` | string | bottom | Anchor: top, center, bottom, above_content, below_content |
| `SUBTITLE_MARGIN` | float | 0.05 | Margin from anchor (0.0-0.5 fraction of frame height) |
| `SUBTITLE_CONTENT_AWARE` | bool | true | Enable content-aware positioning |
| `SUBTITLE_STYLE_PRESET` | string | modern | Style preset: minimal, modern, bold, animated, random |
| `SUBTITLE_FONT_SIZE_SCALE` | float | 1.0 | Font size multiplier (0.5-2.0) |
| `SUBTITLE_ALIGNMENT` | string | center | Text alignment: left, center, right |
| `SUBTITLE_MAX_WIDTH_FRACTION` | float | 0.9 | Max subtitle width (0.0-1.0 fraction) |
| `SUBTITLE_MAX_LINE_LENGTH` | int | 42 | Maximum characters per line |
| `SUBTITLE_MAX_WORDS_PER_LINE` | int | 8 | Maximum words per line (0 to disable) |
| `SUBTITLE_MAX_DURATION` | float | 5.0 | Maximum subtitle duration (seconds) |
| `SUBTITLE_MIN_DURATION` | float | 1.0 | Minimum subtitle duration (seconds) |
| `SUBTITLE_RANDOMIZE_FONTS` | bool | false | Enable random font selection |
| `SUBTITLE_RANDOMIZE_COLORS` | bool | false | Enable random color selection |
| `SUBTITLE_RANDOMIZE_EFFECTS` | bool | false | Enable random effect selection |

### Publishing Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LATE_VERCEL_TOKEN` | string | None | Vercel Blob token for large uploads (alt: `BLOB_READ_WRITE_TOKEN`) |
| `PUBLISHER_DEFAULT_PLATFORMS` | string | None | Comma-separated platforms: youtube,tiktok,instagram |
| `PUBLISHER_IMMEDIATE` | bool | false | Publish immediately without scheduling |
| `PUBLISHER_MAX_RETRIES` | int | 3 | Maximum retry attempts for failed publishes |
| `PUBLISHER_TIMEOUT` | int | 300 | Request timeout in seconds |
| `PUBLISHER_PROVIDER` | string | late | Publishing service provider |
| `PUBLISHER_PRIVACY_YOUTUBE` | string | None | YouTube privacy: public, private, unlisted |
| `PUBLISHER_PRIVACY_TIKTOK` | string | None | TikTok privacy setting |
| `PUBLISHER_PRIVACY_INSTAGRAM` | string | None | Instagram privacy setting |

### Advanced Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OPENROUTER_BASE_URL` | string | https://openrouter.ai/api/v1 | Custom OpenRouter API URL |
| `PEXELS_BASE_URL` | string | https://api.pexels.com/v1 | Custom Pexels API URL |
| `SCRAPFLY_PROXY` | string | None | Proxy for scraping (format: http://USER:PASS@host:port) |
| `COQUI_TTS_GPU` | bool | false | Enable GPU acceleration for Coqui TTS (only applies if you installed coqui-tts yourself) |

### Environment Variable Validation

At startup, `validate_required_secrets()` checks all required API keys:

```python
from src.config_manager import get_unified_config_manager

manager = get_unified_config_manager()
result = manager.validate_required_secrets(exit_on_missing=True)

# Result contains:
# - result.valid: True if all required secrets present
# - result.missing_required: List of missing required secrets
# - result.missing_optional: List of missing optional secrets
# - result.present: List of configured secrets
```

**Example `.env` file:**

```bash
# Required
GEMINI_API_KEY=your-gemini-key
PEXELS_API_KEY=your-pexels-key
FREESOUND_API_KEY=your-freesound-key

# Optional - LLM fallback
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Optional - TTS
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Optional - Publishing
LATE_API_KEY=your-late-api-key
PICSEE_API_KEY=your-picsee-key
AMAZON_ASSOCIATE_TAG=your-tag-20

# Optional - Link-in-Bio
LNKBIO_CLIENT_ID=your-client-id
LNKBIO_CLIENT_SECRET=your-client-secret

# Runtime overrides
CONTENT_ENGINE_DEBUG=false
SUBTITLE_STYLE_PRESET=modern
```

## Performance Tuning

Performance settings are configured in `config/performance.yaml`:

### Optimization Settings

```yaml
optimization_settings:
  # Background processing
  background_processing:
    enabled: true
    max_workers: 4
    queue_size: 100

  # Connection pooling
  connection_pooling:
    enabled: true
    max_connections: 20
    connection_timeout: 30

  # Async I/O
  async_io:
    enabled: true
    chunk_size: 8192
    max_concurrent: 10

  # Caching
  caching:
    enabled: true
    ttl_seconds: 3600       # 1 hour
    max_size_mb: 100

  # Memory optimization
  memory:
    gc_threshold: 0.8
    max_memory_mb: 2048
    max_image_size_mb: 50
    mmap_threshold_bytes: 1048576  # 1MB
```

### Download Settings

```yaml
api_settings:
  downloads:
    timeout_sec: 30
    retry_attempts: 3
    max_concurrent_downloads: 5
    chunk_size_bytes: 1048576   # 1MB
    max_file_size_mb: 50
```

## CLI Override Arguments

Profile settings can be overridden at runtime using command-line arguments with highest precedence:

```bash
# Image positioning overrides
--image-width-percent 0.75           # Override image width (0.0-1.0)
--image-top-position-percent 0.20    # Override top position (0.0-1.0)

# Subtitle positioning overrides
--subtitle-anchor below_content      # Override anchor point
--subtitle-margin 0.10               # Override margin (0.0-0.5)
--content-aware                      # Enable content-aware mode
--no-content-aware                   # Disable content-aware mode

# Style and formatting overrides
--preset minimal                     # Override style preset
--font-size-scale 0.8                # Override font size scale
--max-line-length 30                 # Override max line length

# Example: Override slideshow_images2 settings
poetry run python -m src.video.producer \
  outputs/B0BTYCRJSS/data.json \
  slideshow_images2 \
  --image-top-position-percent 0.30 \
  --preset bold \
  --debug
```

**CLI Precedence**: CLI args > Profile settings > Global YAML configuration

## Customization Examples

### Creating Custom Profiles

```yaml
video_profiles:
  my_custom_profile:
    description: "Custom profile for my use case"
    use_scraped_images: true
    use_scraped_videos: false
    use_stock_images: true
    use_stock_videos: true
    stock_image_count: 5
    stock_video_count: 2
```

### Custom TTS Voice Selection

**Current System (September 2025)**: Uses prioritized voice selection criteria for Chirp 3 HD voices:

```yaml
# In config/subtitles.yaml
tts_config:
  google_cloud:
    # Priority-based voice selection (Chirp 3 HD → Chirp → Neural2 → Standard)
    voice_selection_criteria:
      # Primary: Chirp 3 HD voices (highest quality)
      - { language_code: "en-US", name_contains: "Chirp3", ssml_gender: "FEMALE" }
      - { language_code: "en-US", name_contains: "Chirp3", ssml_gender: "MALE" }
      # Secondary: Any Chirp voices if Chirp 3 not available
      - { language_code: "en-US", name_contains: "Chirp", ssml_gender: "FEMALE" }
      - { language_code: "en-US", name_contains: "Chirp", ssml_gender: "MALE" }
      # Tertiary: High-quality Neural2 voices
      - { language_code: "en-US", name_contains: "Neural2", ssml_gender: "FEMALE" }
      - { language_code: "en-US", name_contains: "Neural2", ssml_gender: "MALE" }
      # Final fallback: Any US English voice
      - { language_code: "en-US", ssml_gender: "FEMALE" }
      - { language_code: "en-US", ssml_gender: "MALE" }
```

### Custom Subtitle Styling

Styling (font, colors, outline, shadow, effects) is owned by style presets.
Add or override an entry under the top-level `style_presets` block:

```yaml
style_presets:
  brand:
    description: "Brand-colored preset"
    font_name: "Montserrat"
    font_color: "&H000035FF"         # ASS &HAABBGGRR (orange)
    outline_color: "&H00FFFFFF"      # White outline
    background_color: null           # No background box
    bold: true
    outline_thickness: 3
    shadow: true
    effects: ["karaoke"]
    font_width_to_height_ratio: 0.5
```

Then select the preset in `subtitle_settings` (and tune layout knobs at
the same level — colors stay in the preset):

```yaml
subtitle_settings:
  style_preset: "brand"
  horizontal_alignment: "center"
  margin: 0.1                        # Fraction of frame height
  max_line_length: 30
```

## Configuration Validation

The system uses Pydantic models for validation:

```python
# Check configuration validity
poetry run python -c "
from src.video.config_adapter import load_video_config_modular
config = load_video_config_modular()
print('✓ Configuration is valid')
"
```

Common validation errors:
- **Invalid timeout values**: Must be positive numbers
- **Missing required fields**: Check for typos in field names
- **Invalid enum values**: Check allowed values for gender, alignment, etc.
- **Path validation**: Ensure paths exist and are accessible

## Video Extraction Behavior

ContentEngineAI intentionally extracts **only 1 product video per product** by default to avoid competitor content.

**Why**: Amazon mixes official product videos with competitor videos, user reviews, and sponsored content throughout product pages. Reliably distinguishing between these types is difficult, so the scraper extracts only the first video from the main gallery (typically the official product video).

**Additional videos** visible on Amazon pages are located in:
- "Videos for this product" widget (requires tab interaction)
- A+ Content sections
- Customer review sections

These load dynamically and aren't extracted to avoid bot detection and false positives.

**Configuration** (`config/scraper.yaml`):
```yaml
scrapers:
  amazon:
    max_videos_per_product: 10       # Maximum to extract
    enable_m3u8_monitoring: false    # Network monitoring
```

**To extract multiple videos**: Enable `enable_m3u8_monitoring`, modify the extraction logic in `media_extractor.py` to click additional thumbnails/tabs, and implement validation to filter competitor content.

## Scraper Configuration

ContentEngineAI includes an Amazon product scraper with advanced filtering capabilities. The scraper configuration is managed in `config/scraper.yaml`.

### Basic Scraper Settings

```yaml
global_settings:
  cleanup_on_start: true        # Clean output directory on start
  retries: 3                   # Number of retry attempts
  delay_range: [1, 3]          # Random delay between operations (seconds)
  download_concurrency: 10     # Max simultaneous downloads
  high_res_min_sl_size: 1500   # Minimum size for high-res images (pixels)
  
  timeouts:
    navigation: 30000          # Page navigation timeout (ms)
    selector: 15000           # Element selector timeout (ms)
    page_load: 60000          # Full page load timeout (ms)
    download: 60              # Media download timeout (seconds)

scrapers:
  amazon:
    enabled: true
    base_url: "https://www.amazon.com"
    keywords: ["wireless earbuds"]
    max_products: 3
    associate_tag: ""  # Set via AMAZON_ASSOCIATE_TAG env var
    affiliate_links:
      enabled: true    # false = no program in use; strip tracking params, don't warn
                       # Overridable via AMAZON_AFFILIATE_LINKS_ENABLED
```

### Advanced Search Parameters

The scraper supports multiple search filtering options via CLI parameters:

#### Price Filtering
```bash
# Filter products by price range
--min-price 15.0 --max-price 100.0
```

#### Quality Filtering  
```bash
# Filter by minimum rating (1-5 stars)
--min-rating 4
```

#### Shipping Filters
```bash
# Prime eligible items only
--prime-only

# Free shipping items only
--free-shipping
```

#### Brand Filtering
```bash
# Filter by specific brands
--brands Apple Samsung Sony
```

#### Sort Options
```bash
# Sort results by price, reviews, date, etc.
--sort price-low             # Price: low to high
--sort price-high            # Price: high to low
--sort rating                # Best reviews first
--sort newest                # Newest first
--sort featured              # Featured items
--sort relevance             # Default relevance (default)
```

### Complete Example

```bash
# Advanced search with multiple filters
poetry run python -m src.scraper.amazon.scraper \
  --keywords "wireless headphones" \
  --min-price 25.0 --max-price 150.0 \
  --min-rating 4 --prime-only \
  --brands Sony Bose Apple \
  --sort rating --debug --clean
```

### Scraper Selectors

The scraper uses CSS selectors to extract product information. These are configured in `config/scraper.yaml`:

```yaml
css_selectors:
  # Product title selectors (in priority order - first match wins)
  product_title_selectors:
    - "#productTitle"
    - "h1.a-size-large"
    - ".product-title"
    - "h1[data-automation-id='product-title']"

  # Search result card selector
  search_result_card: "div[data-component-type='s-search-result']"

# ASIN validation patterns
asin_patterns:
  modern_asin_pattern: "^B0[A-Z0-9]{8}$"   # B0 + 8 chars
  legacy_asin_pattern: "^[A-Z0-9]{10}$"    # 10 chars
  url_asin_pattern: "/dp/([A-Z0-9]{10})"   # URL extraction
```

## Configuration Best Practices

1. **Environment Variables**: Always use environment variables for sensitive data
2. **Timeouts**: Set realistic timeouts based on your system performance
3. **Provider Order**: List providers in order of preference
4. **Testing**: Test configuration changes in debug mode first
5. **Documentation**: Comment complex or custom configurations
6. **Backup**: Keep backup copies of working configurations

## Recent Configuration Updates (v0.1.0+)

### Per-Profile Settings Feature (Major Update)

**Profile-Specific Overrides**: All image positioning, sizing, subtitle positioning, fonts, and colors can now be configured per video profile. This enables:
- **Custom styling per use case**: Product-focused vs stock media profiles with different visual approaches
- **Content-aware positioning**: Subtitles automatically avoid overlapping with visual content
- **Selective customization**: Override only the settings you need, inheriting global defaults for others
- **Type-safe configuration**: All overrides validated through Pydantic models

**Implementation**: Uses configuration merging pattern where profile settings override global defaults selectively. See Video Profiles section above for complete examples.

### Additional Configuration Settings

The following settings were also added to eliminate magic numbers and improve configurability:

#### Pipeline Settings
```yaml
# Duration padding added to prevent audio cutoff in seconds
# Added to voiceover duration to ensure complete audio playback
duration_padding_sec: 0.5
```

#### Video Settings
```yaml
video_settings:
  # Font size limits for subtitle text rendering
  subtitle_min_font_size: 16    # Minimum readable font size in pixels
  subtitle_max_font_size: 100   # Maximum font size to prevent overflow
```

#### Audio Settings
```yaml
audio_settings:
  # User agent string for HTTP requests
  user_agent: "ContentEngineAI/1.0"
```

#### Subtitle Settings
```yaml
subtitle_settings:
  # Fade in/out duration for subtitle transitions (milliseconds)
  fade_duration_ms: 300

  # Probability of applying random animation effects (0.0-1.0)
  animation_probability: 0.3
```

#### CTA Detection Settings
```yaml
# In config/video_production.yaml
cta_detection:
  # Minimum total duration (seconds) for detected CTA windows
  # If total CTA duration < this value, fall back to full video duration
  min_cta_duration: 0.5

  # Fallback duration (seconds) when voiceover duration unavailable
  # Used as placeholder for static subtitles
  fallback_duration: 9999.0
```

**Purpose**: Validates CTA timing windows for upper subtitle display to prevent blinking subtitles.

**Key Settings**:
- `min_cta_duration`: Minimum acceptable total duration for CTA windows (default: 0.5s)
- `fallback_duration`: Large duration used when voiceover unavailable (default: 9999.0s)

**Behavior**: When detected CTA windows are shorter than `min_cta_duration`, the system falls back to displaying the upper subtitle for the full video duration instead of just during brief CTA moments.

#### LLM Settings
```yaml
llm_settings:
  # Script validation thresholds (nested under script_validation)
  script_validation:
    min_chars: 200    # Minimum character count for valid scripts
    min_words: 50     # Minimum word count for valid scripts
```

#### Text Processing
```yaml
text_processing:
  # Speaking rate for subtitle timing estimation (words per second)
  speaking_rate_words_per_sec: 2.5
```

#### Optimization Settings
```yaml
optimization_settings:
  # Background task cache TTL (10 minutes)
  background_processing_cache_ttl_sec: 600
  
  # Memory usage threshold for subtitle generation (GB)
  memory_threshold_gb: 8.0
```

#### Scraper Settings (New Section)
```yaml
scraper_settings:
  # Default monitor resolution for browser windows
  default_monitor_width: 1920
  default_monitor_height: 1080
  
  # Browser window positioning timeout (seconds)
  window_setup_timeout_sec: 10
```

### Configuration Usage in Code

For developers working with configurations:

```python
# Access configuration values with fallbacks
duration = ctx.config.duration_padding_sec
fade_duration = getattr(config, 'fade_duration_ms', 300)
speaking_rate = (
    self.config.speaking_rate_words_per_sec
    if hasattr(self.config, 'speaking_rate_words_per_sec')
    else 2.5  # Fallback to default
)
```

### Migration Notes

- **No Action Required**: Existing configurations continue to work unchanged
- **Backward Compatible**: All new settings have sensible defaults
- **Optional Customization**: Review new settings for optimization opportunities

## Troubleshooting Configuration

### Missing Required API Keys

**Symptom**: Application exits immediately with message about missing secrets.

```
ERROR: Missing required environment variable: GEMINI_API_KEY
       Set this in your .env file or environment.
       Get your API key at: https://aistudio.google.com/apikey
```

**Solution**:
1. Copy `.env.example` to `.env`: `cp .env.example .env`
2. Edit `.env` and add your API keys
3. Required keys: `GEMINI_API_KEY`, `PEXELS_API_KEY`, `FREESOUND_API_KEY` (`OPENROUTER_API_KEY` is an optional fallback)

**Verify secrets are loaded**:
```bash
# Check which secrets are configured
poetry run python -c "
from src.config_manager import UnifiedConfigManager
mgr = UnifiedConfigManager()
result = mgr.validate_required_secrets(exit_on_missing=False)
print('Valid:', result.valid)
print('Missing required:', [s.name for s in result.missing_required])
print('Missing optional:', [s.name for s in result.missing_optional])
"
```

### Configuration Precedence Issues

**Symptom**: Setting a value but it's being overridden by another source.

**Solution**: Remember the precedence order (CLI > ENV > YAML):
```bash
# See what's actually being used
poetry run python -m src.video.producer \
  outputs/B0TEST123/data.json slideshow_images1 \
  --debug 2>&1 | grep -i "config\|setting\|using"
```

**Common precedence mistakes**:
- Setting `SUBTITLE_ANCHOR=top` in `.env` but passing `--subtitle-anchor bottom` on CLI (CLI wins)
- Editing `config/subtitles.yaml` but forgetting environment variable is set (ENV wins)
- Having value in both primary and alternative env var names (primary wins)

### YAML Syntax Errors

**Symptom**: `yaml.scanner.ScannerError` or configuration values are wrong.

**Solution**:
```bash
# Validate YAML syntax
poetry run python -c "
import yaml
from pathlib import Path
for f in Path('config').glob('*.yaml'):
    try:
        yaml.safe_load(f.read_text())
        print(f'✓ {f.name}')
    except yaml.YAMLError as e:
        print(f'✗ {f.name}: {e}')
"
```

**Common YAML mistakes**:
- Inconsistent indentation (use 2 spaces, not tabs)
- Missing quotes around strings with special characters
- Incorrect boolean format (`true`/`false`, not `True`/`False`)

### Environment Variable Type Conversion

**Symptom**: Value is wrong type (string instead of bool, etc).

**Environment variables are always strings**. The system converts them:
- Booleans: `true`, `false`, `1`, `0`, `yes`, `no` (case-insensitive)
- Numbers: Parsed automatically (`"42"` → `42`, `"3.14"` → `3.14`)
- Lists: Comma-separated (`"a,b,c"` → `["a", "b", "c"]`)

**Verify type conversion**:
```bash
# Check how a value is being interpreted
CONTENT_ENGINE_DEBUG=true poetry run python -c "
from src.config_manager import UnifiedConfigManager
mgr = UnifiedConfigManager()
print('debug_mode:', mgr.debug_mode, type(mgr.debug_mode))
"
```

### Subtitle Configuration Issues

**Symptom**: Subtitles don't appear correctly or positioning is wrong.

**Verify subtitle settings**:
```bash
poetry run python -c "
from src.video.config_adapter import load_video_config_modular
config = load_video_config_modular()
print('Anchor:', config.subtitle_settings.anchor)
print('Margin:', config.subtitle_settings.margin)
print('Content-aware:', config.subtitle_settings.content_aware_positioning)
print('Max width:', config.subtitle_settings.max_width_fraction)
"
```

**Common subtitle issues**:
- **Subtitles cut off**: Increase `SUBTITLE_MAX_WIDTH_FRACTION` (default 0.9)
- **Text too small/large**: Adjust `SUBTITLE_FONT_SIZE_SCALE` (0.5-2.0)
- **Wrong position**: Check `SUBTITLE_ANCHOR` value (top/center/bottom/above_content/below_content)
- **Overlapping content**: Enable `SUBTITLE_CONTENT_AWARE=true`

### Publishing Configuration Issues

**Symptom**: Publishing fails or uploads timeout.

**Check publishing configuration**:
```bash
poetry run python -c "
import os
keys = ['LATE_API_KEY', 'PUBLISHER_API_KEY', 'LATE_VERCEL_TOKEN', 'PUBLISHER_TIMEOUT']
for k in keys:
    v = os.environ.get(k)
    print(f'{k}: {\"set\" if v else \"not set\"}')"
```

**Common publishing issues**:
- **Timeout on large files**: Increase `PUBLISHER_TIMEOUT` (default 300 seconds)
- **Upload fails**: Set `LATE_VERCEL_TOKEN` for Vercel Blob uploads
- **Wrong platform**: Check `PUBLISHER_DEFAULT_PLATFORMS` value

### Google TTS Not Working

**Symptom**: Falls back to local TTS or TTS fails entirely.

**Check Google credentials**:
```bash
# Verify credentials file exists and is valid JSON
poetry run python -c "
import os
import json
creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
if not creds_path:
    print('GOOGLE_APPLICATION_CREDENTIALS not set (using local TTS)')
elif not os.path.exists(creds_path):
    print(f'File not found: {creds_path}')
else:
    with open(creds_path) as f:
        data = json.load(f)
        print(f'Project: {data.get(\"project_id\", \"unknown\")}')
        print(f'Type: {data.get(\"type\", \"unknown\")}')
"
```

### Debug Mode

**Enable comprehensive debugging**:
```bash
# Via environment variable
CONTENT_ENGINE_DEBUG=true poetry run python -m src.video.producer ...

# Via CLI flag
poetry run python -m src.video.producer outputs/B0TEST/data.json profile --debug

# Check if debug is active
poetry run python -c "
from src.config_manager import UnifiedConfigManager
print('Debug mode:', UnifiedConfigManager().debug_mode)
"
```

### Configuration File Locations

**Can't find configuration files?**
```bash
# List all configuration files
find config -name "*.yaml" -o -name "*.yml" | sort

# Show configuration loading order
poetry run python -c "
from pathlib import Path
print('YAML configs:')
for f in sorted(Path('config').glob('*.yaml')):
    print(f'  {f}')
"
```

**Default configuration file locations**:
- `config/core.yaml` - Core pipeline settings
- `config/subtitles.yaml` - Subtitle styling and positioning
- `config/video_production.yaml` - Video production settings and profile definitions
- `config/scraper.yaml` - Scraper settings

### Reset to Defaults

**Start fresh with default configuration**:
```bash
# Backup current .env
cp .env .env.backup

# Create fresh .env from example
cp .env.example .env

# Edit with your API keys
nano .env  # or your preferred editor
```

For more troubleshooting help, see [Troubleshooting](troubleshooting.md).
