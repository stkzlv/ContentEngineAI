# Configuration Guide

ContentEngineAI uses a **unified modular configuration system** that splits settings across specialized files with CLI overrides and environment variable support. This guide explains all configuration options and how to customize the system for your needs.

## Configuration Overview

ContentEngineAI implements a **triple-precedence configuration system**:

1. **CLI Arguments** (highest priority)
2. **Environment Variables** (medium priority)
3. **YAML Configuration** (default values)

### Modular Architecture

The configuration system uses **7 specialized files** instead of a monolithic configuration:

- **`config/core.yaml`** - Global settings and output paths
- **`config/video_production.yaml`** - Video pipeline and effects
- **`config/ai_services.yaml`** - TTS, LLM, and AI providers
- **`config/subtitles.yaml`** - Subtitle positioning and styling
- **`config/performance.yaml`** - Resource limits and optimization
- **`config/scraper.yaml`** - Web scraping and browser settings
- **`config/url_shortener.yaml`** - URL shortening providers and integration

### How Configuration Loading Works

1. **Modular Loading**: Each config file is loaded independently
2. **Environment Resolution**: Variables resolved using `api_key_env_var` mappings
3. **CLI Override**: Command-line parameters override YAML values
4. **Validation**: Pydantic models ensure type safety and completeness

**Example:**
```yaml
# In config/ai_services.yaml
llm_settings:
  api_key_env_var: "OPENROUTER_API_KEY"  # References env var
  models: ["anthropic/claude-3-haiku"]   # Direct config value

# In .env file
OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here

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
  output_codec: "libx264"

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
TTS, LLM, and AI provider settings:

```yaml
tts_config:
  providers:
    - google_cloud_tts
    - coqui_tts
  google_cloud_tts:
    api_key_env_var: "GOOGLE_APPLICATION_CREDENTIALS"
    voice_selection_criteria:
      - { language_code: "en-US", name_contains: "Chirp3" }

llm_settings:
  api_key_env_var: "OPENROUTER_API_KEY"
  models: ["anthropic/claude-3-haiku"]
  temperature: 0.7
```

### 4. **Subtitle Configuration** (`config/subtitles.yaml`)
Subtitle positioning, styling, and two-part subtitle system:

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
performance_settings:
  max_concurrent_downloads: 5
  memory_limit_mb: 2048
  cache_ttl_hours: 24

timeout_settings:
  api_timeout_sec: 30
  download_timeout_sec: 60
  video_processing_timeout_sec: 300
```

### 6. **Scraper Configuration** (`config/scraper.yaml`)
Web scraping and browser settings:

```yaml
scraper_settings:
  debug_mode: true
  headless: false
  timeout_sec: 30

  output_config:
    base_directory: "outputs"
    file_patterns:
      product_file: "{keyword}_products.json"

global_settings:
  validation_config:
    # Media validation requirements (must match video_production.yaml)
    min_total_media: 3              # Minimum total media files
    min_images_if_no_video: 5       # Minimum images for slideshow mode
    min_images_with_video: 2        # Minimum images when videos available

amazon_settings:
  max_results: 10
  skip_unavailable: true
  prime_only: false
```

### 7. **URL Shortener Configuration** (`config/url_shortener.yaml`)
URL shortening providers for affiliate links:

```yaml
url_shortener:
  enabled: true                    # Enable/disable URL shortening
  provider: picsee                 # Primary provider: picsee, bitly, tinyurl

  # Fallback providers (tried in order if primary fails)
  fallback_providers:
    - bitly
    - tinyurl

  # API configuration
  api:
    timeout_sec: 30
    max_retries: 3
    retry_delay_sec: 2
    retry_backoff_multiplier: 2

  # PicSee-specific settings
  picsee:
    api_key_env_var: PICSEE_API_KEY
    custom_domain: stte.psee.io    # Optional branded short domain
    max_bulk_size: 100

  # Integration settings
  integration:
    shorten_on_scrape: true        # Auto-shorten during scraping
    include_in_descriptions: true  # Include in video descriptions
    fallback_to_original: true     # Use original URL if shortening fails
    enable_caching: true           # Cache shortened URLs
    cache_ttl_hours: 168           # 7-day cache TTL
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
    final_video: "video_{profile}.mp4"  # Final video output
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
│   ├── video_slideshow_images1.mp4 # Final video
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
      custom_style: null               # Uses main subtitle_settings
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

</details>

<details>
<summary><strong>6. TTS (Text-to-Speech) Configuration</strong></summary>

### 6. TTS (Text-to-Speech) Configuration

```yaml
tts_config:
  # Provider priority order (first = primary)
  providers:
    - google_cloud_tts
    - coqui_tts
  
  # Google Cloud TTS settings
  google_cloud_tts:
    enabled: true
    api_key_env_var: "GOOGLE_APPLICATION_CREDENTIALS"
    
    # Voice selection criteria
    language_code: "en-US"
    gender: "NEUTRAL"                # NEUTRAL, MALE, FEMALE
    voice_name_pattern: "Wavenet"    # Prefer Wavenet voices
    
    # Speech parameters
    speaking_rate: 1.0               # Speech rate (0.25-4.0)
    pitch: 0.0                       # Pitch adjustment (-20.0 to 20.0)
    volume_gain_db: 0.0              # Volume adjustment
    
    # Timeouts and retries
    timeout_sec: 30
    max_retries: 3
  
  # Coqui TTS settings (local/fallback)
  coqui_tts:
    enabled: true
    model_name: "tts_models/en/ljspeech/tacotron2-DDC"
    device: "auto"                   # auto, cpu, cuda
    speaker_idx: null                # For multi-speaker models
    timeout_sec: 60
```

</details>

<details>
<summary><strong>7. LLM Settings</strong></summary>

### 7. LLM Settings

```yaml
llm_settings:
  # API configuration
  api_base_url: "https://openrouter.ai/api/v1"
  api_key_env_var: "OPENROUTER_API_KEY"
  
  # Model selection with fallbacks
  models:
    - "anthropic/claude-3-haiku"     # Primary model
    - "openai/gpt-3.5-turbo"        # Fallback model
    - "meta-llama/llama-3-8b"       # Second fallback
  
  # Generation parameters
  temperature: 0.7                   # Creativity (0.0-2.0)
  max_tokens: 500                    # Maximum response length
  
  # Prompt configuration
  prompt_template_path: "src/ai/prompts/video_script.md"
  target_audience: "general consumers"
  
  # Timeouts and retries
  timeout_sec: 30
  max_retries: 3
```

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

If refresh token becomes invalid, regenerate using OAuth2 setup script:

```bash
poetry run python tools/freesound_oauth2_setup.py \
  --client-id YOUR_CLIENT_ID \
  --client-secret YOUR_CLIENT_SECRET
```

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
# Whisper STT settings (primary, fixed September 2025)
whisper_settings:
  enabled: true
  model_size: "small"                # tiny, base, small, medium, large (default: small for quality/speed balance)
  language: "en"                     # Language code
  device: "auto"                     # auto, cpu, cuda
  compute_type: "float16"            # float16, int8, int8_float16
  word_timestamps: true              # Enable word-level timing (required for perfect subtitle sync)
  
# Google Cloud STT settings (fallback, implemented September 2025)
google_cloud_stt_settings:
  enabled: true
  api_key_env_var: "GOOGLE_APPLICATION_CREDENTIALS"
  language_code: "en-US"
  enable_word_time_offsets: true     # Required for audio-based subtitle synchronization
  use_enhanced: true                 # Use enhanced models when available
  sample_rate_hertz: 16000
  encoding: "LINEAR16"
  enable_word_time_offsets: true
  use_enhanced: true
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

    # Profile-Specific Subtitle Settings
    subtitle_randomize_effects: true  # Enable effect randomization

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

    # Subtitle positioning
    subtitle_anchor: "below_content"  # Position below images
    subtitle_margin: 0.08             # 8% gap below content
    subtitle_content_aware: true      # Dynamic positioning
    subtitle_horizontal_alignment: "center"

    # Subtitle styling
    subtitle_style_preset: "minimal"  # Clean minimal style
    subtitle_font_size_scale: 0.9     # 10% smaller font
    subtitle_randomize_fonts: true
    subtitle_randomize_colors: true
    subtitle_randomize_effects: false

    # Text formatting
    subtitle_max_line_length: 28
    subtitle_max_words_per_line: 3
    subtitle_max_subtitle_width_fraction: 0.85
    subtitle_max_duration: 4.0
    subtitle_min_duration: 0.5
```

**Available Per-Profile Overrides:**

```yaml
# Image Settings (all optional)
image_width_percent: 0.85            # Override global image width
image_top_position_percent: 0.15     # Override global image position
preserve_aspect_ratio: true          # Override aspect ratio setting

# Subtitle Settings (all optional)
subtitle_anchor: "below_content"     # Override positioning anchor
subtitle_margin: 0.08                # Override margin from anchor
subtitle_content_aware: true         # Override content-aware positioning
subtitle_style_preset: "modern"     # Override style preset (minimal, modern, bold, animated, random)
subtitle_font_size_scale: 1.1        # Override font size scaling
subtitle_max_line_length: 35         # Override line length limit
subtitle_max_words_per_line: 3       # Override max words per line
subtitle_max_subtitle_width_fraction: 0.85  # Override max subtitle width
subtitle_max_duration: 4.5           # Override max subtitle duration
subtitle_min_duration: 0.4           # Override min subtitle duration
subtitle_horizontal_alignment: "center"
subtitle_randomize_fonts: false      # Override font randomization
subtitle_randomize_colors: false     # Override color randomization
subtitle_randomize_effects: false    # Override effect randomization
```

**Key Features:**
- **Individual Customization**: Each profile can override any global setting
- **Selective Overrides**: Only specify settings you want to change
- **Fallback System**: Unspecified settings use global defaults
- **Type Safety**: All overrides validated by Pydantic models
- **CLI Override Support**: All profile settings can be overridden via command-line arguments

</details>

## Timeout Configuration

All pipeline operations have configurable timeouts:

```yaml
# Global pipeline timeout
pipeline_timeout_sec: 900

# Component-specific timeouts
download_timeout_sec: 60           # HTTP downloads
audio_processing_timeout_sec: 120  # TTS, music processing
video_processing_timeout_sec: 300  # FFmpeg operations
api_timeout_sec: 30                # General API calls
file_operation_timeout_sec: 60     # File I/O operations
cleanup_delay_sec: 5               # Cleanup delay

# Provider-specific timeouts (see individual provider sections)
```

## Environment Variables

Sensitive information is stored in environment variables:

```bash
# Required API Keys
OPENROUTER_API_KEY="your_openrouter_key"
PEXELS_API_KEY="your_pexels_key"
FREESOUND_API_KEY="your_freesound_key"

# Optional Google Cloud
GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"

# Optional Freesound OAuth2
FREESOUND_CLIENT_ID="your_client_id"
FREESOUND_CLIENT_SECRET="your_client_secret"
FREESOUND_REFRESH_TOKEN="your_refresh_token"

# Optional URL Shortening
PICSEE_API_KEY="your_picsee_api_key"        # For PicSee URL shortener
# BITLY_API_KEY="your_bitly_key"            # Future: Bitly support
# TINYURL_API_KEY="your_tinyurl_key"        # Future: TinyURL support
```

## Performance Tuning

### Concurrency Settings

```yaml
# Global concurrency limits
max_concurrent_downloads: 5
max_concurrent_api_calls: 3

# Component-specific limits
stock_media_concurrent_downloads: 3
tts_concurrent_requests: 2
subtitle_concurrent_processing: 1
```

### Caching Configuration

```yaml
cache_settings:
  enabled: true
  cache_dir: "outputs/cache"
  default_ttl_hours: 24             # Time-to-live for cached items
  
  # Cache categories
  media_metadata_ttl_hours: 168     # 1 week for media metadata
  api_response_ttl_hours: 24        # 1 day for API responses
  tts_cache_ttl_hours: 720          # 30 days for TTS results
  
  # Cache size limits
  max_cache_size_mb: 1000           # Maximum cache size
  cleanup_threshold_percent: 80      # Cleanup when 80% full
```

### Memory Management

```yaml
memory_settings:
  # Memory-mapped I/O thresholds
  mmap_threshold_mb: 1               # Use mmap for files >1MB
  max_memory_usage_mb: 2048          # Maximum memory usage
  
  # Connection pooling
  http_pool_connections: 10          # HTTP connection pool size
  http_pool_maxsize: 20              # Maximum connections per host
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
--subtitle-content-aware true        # Override content-aware mode

# Style and formatting overrides
--preset minimal                     # Override style preset
--subtitle-font-size-scale 0.8       # Override font size scale
--subtitle-max-line-length 30        # Override max line length

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
tts_config:
  google_cloud_tts:
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

```yaml
subtitle_settings:
  # Brand colors
  font_color: "#FF6B35"              # Brand orange
  outline_color: "#FFFFFF"           # White outline
  back_color: "#00000000"            # No background
  
  # Custom positioning
  alignment: "center"                # Center alignment
  margin_v_percent: 20               # Higher position
  
  # Custom segmentation
  max_line_length: 30                # Shorter lines
  split_on_punctuation: false        # Don't split on punctuation
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
    associate_tag: "your-associate-tag-20"
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
--sort price-asc-rank        # Price: low to high
--sort price-desc-rank       # Price: high to low
--sort review-rank           # Best reviews first
--sort date-desc-rank        # Newest first
--sort featured-rank         # Featured items
--sort relevanceblender      # Default relevance (default)
```

### Complete Example

```bash
# Advanced search with multiple filters
poetry run python -m src.scraper.amazon.scraper \
  --keywords "wireless headphones" \
  --min-price 25.0 --max-price 150.0 \
  --min-rating 4 --prime-only \
  --brands Sony Bose Apple \
  --sort review-rank --debug --clean
```

### Scraper Selectors

The scraper uses CSS selectors to extract product information. These are configured in `scraper.yaml`:

```yaml
selectors:
  product_card: '[data-component-type="s-search-result"]'
  serp_product_link: '.s-title-instructions-style a.a-link-normal'
  product_title: '#productTitle'
  price: '.a-price .a-offscreen'
  # ... more selectors
  
  # Alternative selectors as fallbacks
  alternative_selectors:
    product_title:
      - '.title .a-size-large'
    price:
      - '.a-price-whole'
      - '.a-price-current .a-offscreen'
    # ... more alternatives
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
cta_detection:
  # Minimum total duration (seconds) for detected CTA windows
  # If total CTA duration < this value, fall back to full video duration
  min_cta_duration: 2.0

  # Fallback duration (seconds) when voiceover duration unavailable
  # Used as placeholder for static subtitles
  fallback_duration: 9999.0
```

**Purpose**: Validates CTA timing windows for upper subtitle display to prevent blinking subtitles.

**Key Settings**:
- `min_cta_duration`: Minimum acceptable total duration for CTA windows (default: 2.0s)
- `fallback_duration`: Large duration used when voiceover unavailable (default: 9999.0s)

**Behavior**: When detected CTA windows are shorter than `min_cta_duration`, the system falls back to displaying the upper subtitle for the full video duration instead of just during brief CTA moments.

#### LLM Settings
```yaml
llm_settings:
  # Script validation thresholds
  min_script_chars: 200    # Minimum character count for valid scripts
  min_script_words: 50     # Minimum word count for valid scripts
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

Common issues and solutions:

**Configuration Won't Load:**
```bash
# Check YAML syntax
poetry run python -c "from src.video.config_adapter import load_video_config_modular; load_video_config_modular()"
```

**Environment Variables Not Found:**
```bash
# Check environment variables
poetry run python -c "import os; print([k for k in os.environ if 'API_KEY' in k])"
```

**Invalid Paths:**
- Ensure all file paths exist and are accessible
- Use absolute paths when possible
- Check permissions on directories

For more troubleshooting help, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).