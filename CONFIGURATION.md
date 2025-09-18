# Configuration Guide

ContentEngineAI is highly configurable through a central YAML file (`config/video_producer.yaml`). This guide explains all configuration options and how to customize the system for your needs.

## Configuration Overview

ContentEngineAI uses a **dual configuration system** that combines YAML files and environment variables:

- **YAML files** for human-readable settings and application logic
- **Environment variables** for sensitive information (API keys, credentials)
- **Pydantic models** for validation and type safety
- **Hierarchical structure** organized by component

### How Dual Configuration Works

1. **YAML Configuration** (`config/*.yaml`):
   - Contains all application settings, timeouts, preferences
   - References environment variables using `api_key_env_var` fields
   - Safe to commit to version control

2. **Environment Variables** (`.env` file):
   - Contains sensitive data like API keys and credentials
   - Never committed to version control (gitignored)
   - Loaded at runtime and injected into YAML configuration

3. **Runtime Resolution**:
   - YAML config is loaded first
   - Environment variables are resolved using `api_key_env_var` mappings
   - Missing environment variables cause clear error messages

**Example:**
```yaml
# In config/video_producer.yaml
llm_settings:
  api_key_env_var: "OPENROUTER_API_KEY"  # References env var
  models: ["anthropic/claude-3-haiku"]   # Direct config value

# In .env file
OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here
```

## Configuration Files

ContentEngineAI uses two main configuration files:

### 1. **Video Producer Configuration** (`config/video_producer.yaml`)
Controls video generation pipeline settings:

```yaml
# Base output directory
global_output_directory: "outputs"

# Product-centric output structure
output_structure:
  product_directory_pattern: "{product_id}"
  product_files:
    scraped_data: "data.json"
    script: "script.txt"
    voiceover: "voiceover.wav"
    subtitles: "subtitles.srt"
    final_video: "video_{profile}.mp4"
  global_dirs:
    cache: "cache"
    logs: "logs" 
    reports: "reports"

# Video processing settings
video_production:
  resolution: [1080, 1920]  # 9:16 vertical format
  frame_rate: 30
  output_codec: "libx264"
```

### 2. **Scraper Configuration** (`config/scrapers.yaml`)
Controls web scraping behavior:

```yaml
global_settings:
  debug_mode: true  # Show browser, save debug files
  
  output_config:
    base_directory: "outputs"
    file_patterns:
      product_file: "{keyword}_products.json"
      image_file: "{asin}_image_{index}.{ext}"
      video_file: "{asin}_video_{index}.{ext}"

amazon:
  default_search_params:
    max_results: 10
    skip_unavailable: true
    prime_only: false
  
  browser_config:
    headless: false
    timeout_sec: 30

```

## Core Configuration Sections

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

### 5. Subtitle Settings (Unified System)

ContentEngineAI uses a unified subtitle positioning system that simplifies configuration while providing powerful content-aware positioning capabilities.

```yaml
subtitle_settings:
  enabled: true
  
  # Unified Positioning (NEW: Simplified from legacy multi-mode system)
  anchor: "below_content"            # Positioning anchor: top, center, bottom, above_content, below_content
  margin: 0.1                        # Margin as fraction of frame height (0.0-0.5)
  content_aware: true                # Automatically adjust position based on visual content
  horizontal_alignment: "center"     # Text alignment: left, center, right
  
  # Style Presets (NEW: Professional presets replace manual styling)
  style_preset: "modern"             # Style preset: minimal, modern, relative, classic, bold
  font_size_scale: 1.0              # Font size multiplier (0.5-2.0)
  
  # Text Formatting
  max_line_length: 38                # Maximum characters per line
  max_duration: 4.5                  # Maximum duration for subtitle segments (seconds)
  min_duration: 0.4                  # Minimum duration for subtitle segments (seconds)
  font_width_to_height_ratio: 0.6    # Font width-to-height ratio for pixel-based width calculation
  
  # Visual Customization (Optional)
  randomize_colors: false            # Use random color combinations for variety
  randomize_effects: false           # Use random animation effects for engagement
  
  # Advanced Positioning (Optional Override)
  custom_position:                   # Custom position override (advanced users)
    x: 0.5                          # Horizontal position (0.0-1.0 fraction)
    y: 0.8                          # Vertical position (0.0-1.0 fraction)
  
  # Legacy Compatibility (Automatic Conversion)
  positioning_mode: "absolute"         # Legacy setting - automatically converted to unified format
```

#### Positioning Anchors Explained

- **`top`**: Position at the top of the frame with margin
- **`center`**: Position at the vertical center of the frame
- **`bottom`**: Position at the bottom of the frame with margin
- **`above_content`**: Position above visual content (content-aware)
- **`below_content`**: Position below visual content (content-aware) - **Recommended**

**Content-Aware Positioning**: When enabled, the system analyzes each image's dimensions and positioning to calculate optimal subtitle placement that avoids visual overlaps. This creates two subtitle files:
- `subtitles.ass`: Standard positioning
- `subtitles_content_aware.ass`: Dynamic positioning based on content analysis

#### Style Presets Explained

- **`minimal`**: Clean, simple styling with no effects
- **`modern`**: Contemporary look with subtle effects and background
- **`relative`**: Animated effects with karaoke highlighting and scaling
- **`classic`**: Traditional subtitle styling for formal content
- **`bold`**: High contrast, bold styling for attention-grabbing content

#### Migration from Legacy Configuration

The system automatically converts legacy positioning modes:
- `"absolute"` → `bottom` anchor with `content_aware: false`
- `"relative"` → `below_content` anchor with `content_aware: true`
- `"absolute"` → `bottom` anchor with custom position if specified

> **📝 Note**: The unified system provides the same visual results as the legacy multi-mode system while using 62% fewer configuration parameters.

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

### 9. Freesound Audio Settings

```yaml
freesound_settings:
  # API configuration
  api_key_env_var: "FREESOUND_API_KEY"
  
  # OAuth2 settings (optional, for full downloads)
  client_id_env_var: "FREESOUND_CLIENT_ID"
  client_secret_env_var: "FREESOUND_CLIENT_SECRET"
  refresh_token_env_var: "FREESOUND_REFRESH_TOKEN"
  
  # Search parameters
  search_query: "upbeat commercial background"
  duration_range: [10, 180]         # Min/max duration in seconds
  
  # Quality preferences
  bitrate_preference: "preview"      # preview, mp3_hq, original
  
  # Fallback settings
  local_fallback_dir: "assets/music"
  use_local_fallback: true
```

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

### 12. Video Profiles with Per-Profile Settings

Video profiles define different strategies for media selection and now support **per-profile overrides** for all visual settings. Each profile can customize image positioning, subtitle styling, and other visual parameters independently.

```yaml
video_profiles:
  slideshow_images1:
    description: "Image slideshow optimized for product focus"
    use_scraped_images: true
    use_scraped_videos: false
    use_stock_images: false
    use_stock_videos: false
    stock_image_count: 0
    stock_video_count: 0
    use_dynamic_image_count: true

    # ---- Profile-Specific Image Settings ----
    image_width_percent: 0.85         # 85% frame width for product focus
    image_top_position_percent: 0.15  # Position 15% from top
    preserve_aspect_ratio: true       # Maintain image proportions

    # ---- Profile-Specific Subtitle Settings ----
    subtitle_anchor: "below_content"  # Position below images
    subtitle_margin: 0.08             # 8% gap below content
    subtitle_content_aware: true      # Dynamic positioning
    subtitle_style_preset: "modern"  # Modern styling
    subtitle_font_size_scale: 1.1     # 10% larger font
    subtitle_max_line_length: 35      # Shorter lines
    subtitle_horizontal_alignment: "center"

  scraped_videos_then_images:
    description: "Mixed media with videos prioritized"
    use_scraped_images: true
    use_scraped_videos: true
    use_stock_images: false
    use_stock_videos: false
    stock_image_count: 0
    stock_video_count: 0
    use_dynamic_image_count: true

    # ---- Profile-Specific Settings ----
    image_width_percent: 0.9          # Larger for mixed media
    image_top_position_percent: 0.1   # Higher positioning
    subtitle_anchor: "below_content"
    subtitle_margin: 0.06             # Tighter spacing
    subtitle_style_preset: "bold"     # Bold styling for videos
    subtitle_font_size_scale: 1.0
    subtitle_max_line_length: 40

  stock_focus_short:
    description: "Stock media focus with classic styling"
    use_scraped_images: false
    use_scraped_videos: false
    use_stock_images: true
    use_stock_videos: true
    stock_image_count: 2
    stock_video_count: 1
    use_dynamic_image_count: false

    # ---- Profile-Specific Settings ----
    image_width_percent: 0.8          # Smaller for stock content
    image_top_position_percent: 0.2   # Lower positioning
    subtitle_anchor: "bottom"         # Fixed bottom positioning
    subtitle_margin: 0.1
    subtitle_content_aware: false     # No dynamic positioning
    subtitle_style_preset: "classic" # Traditional styling
    subtitle_font_size_scale: 0.9     # Smaller font
    subtitle_max_line_length: 42
```

#### Per-Profile Settings Architecture

**Key Features:**
- **Individual Customization**: Each profile can override any global setting
- **Selective Overrides**: Only specify settings you want to change
- **Fallback System**: Unspecified settings use global defaults
- **Type Safety**: All overrides validated by Pydantic models

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
subtitle_style_preset: "modern"     # Override style preset
subtitle_font_size_scale: 1.1        # Override font size scaling
subtitle_max_line_length: 35         # Override line length limit
subtitle_horizontal_alignment: "center"
subtitle_randomize_colors: false
subtitle_randomize_effects: false

# Legacy subtitle positioning support
subtitle_positioning:
  anchor: "below_content"
  margin: 0.08
  content_aware: true
```

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
from src.video.video_config import load_config
config = load_config('config/video_producer.yaml')
print('✓ Configuration is valid')
"
```

Common validation errors:
- **Invalid timeout values**: Must be positive numbers
- **Missing required fields**: Check for typos in field names
- **Invalid enum values**: Check allowed values for gender, alignment, etc.
- **Path validation**: Ensure paths exist and are accessible

## Scraper Configuration

ContentEngineAI includes an Amazon product scraper with advanced filtering capabilities. The scraper configuration is managed in `config/scrapers.yaml`.

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

The scraper uses CSS selectors to extract product information. These are configured in `scrapers.yaml`:

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
poetry run python -c "import yaml; yaml.safe_load(open('config/video_producer.yaml'))"
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