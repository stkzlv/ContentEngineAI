# Project Requirements

## Configuration System

ContentEngineAI **MUST** use a three-tier configuration system with precedence:

### Configuration Precedence (Highest to Lowest)
1. **CLI Arguments** - Runtime command-line flags (highest priority)
2. **Environment Variables (`.env`)** - API keys, credentials, secrets
3. **YAML Files (`config/`)** - Application settings, preferences, timeouts (lowest priority)

### 1. CLI Arguments
- Override all other configuration sources
- Enable runtime customization without changing files
- Examples: `--debug`, `--preset`, `--subtitle-format`, `--ass-karaoke`

### 2. Environment Variables (`.env`)
- API keys, credentials, secrets only
- Never committed (gitignored)
- Loaded at runtime and injected into YAML config
- Referenced in YAML via `api_key_env_var` fields

### 3. YAML Files (`config/`)
- Application settings, preferences, timeouts
- Safe to commit to version control
- Provide default values when CLI/env not specified

### Security Rules
- ✅ **DO**: Store secrets in `.env` file
- ✅ **DO**: Provide `.env.example` template
- ❌ **DON'T**: Put API keys in YAML files
- ❌ **DON'T**: Commit `.env` to git

## Scraper Requirements

### Multi-Platform Architecture
- Implement modular, multi-platform scraper (Amazon first)
- Separate core logic from platform-specific implementations
- Support direct product ID lookups and keyword searches

### Product Discovery & Media
- Extract key data: title, price, description, ID, ratings, review count
- Download high-resolution images and videos
- **Video Detection & Download**:
  - Extract MP4 video URLs from product pages
  - Filter for product-specific videos using ASIN matching
  - Validate video accessibility with HEAD requests
  - Download highest quality video streams available
  - Support VDP (Video Detail Page) extraction
  - Store videos in `outputs/{product_id}/videos/` directory
  - Track downloaded video paths in product data
- Filter out low-quality images and invalid file types
- Handle multiple ASINs individually (not in single search query)

### Search & Filtering
- Support keyword-based searches with filters (price, rating, shipping, brands)
- Include sorting options and regional redirect handling
- Validate product IDs against standard formats
- Skip products lacking essential data

### Stealth & Human Simulation
- Implement stealth techniques to evade detection
- Simulate human-like interactions when necessary
- Handle failures gracefully without halting entirely

### Output Management
- Store media in dedicated directories per product ID
- Use configurable "outputs" directory structure
- Continue processing until specified number of products with media collected

## Video Producer Requirements

### Dynamic Video Assembly
- Adjust video duration to voiceover track length
- Show images 2-3 seconds each (configurable) with transitions
- Dynamically select image count based on voiceover duration
- Reuse images if needed to match voiceover length

### Product Video Assembly

ContentEngineAI **MUST** support flexible product video assembly with multiple modes, aspect ratio handling, and audio normalization.

#### Video Assembly Modes

Product videos can be assembled using four configurable modes:

1. **Sequential Mode** (`video_assembly_mode: "sequential"`):
   - Concatenate all product videos end-to-end in order
   - Calculate total video duration from all clips
   - If total duration < voiceover: Loop last video or add images to fill remaining time
   - If total duration > voiceover: Trim last video with fade-out to match exactly
   - Apply crossfade transitions between consecutive video clips

2. **Single Best Mode** (`video_assembly_mode: "single_best"`):
   - Select longest product video as primary content
   - Loop video seamlessly to match voiceover duration
   - Apply crossfade transitions between loop iterations
   - Ensure smooth playback without visible loop points

3. **Mixed Media Mode** (`video_assembly_mode: "mixed_media"`):
   - Interleave product videos and images throughout timeline
   - Calculate optimal placement using duration-based algorithm
   - Distribute videos evenly across voiceover duration
   - Fill gaps between videos with product images (2-3s each)
   - Apply consistent crossfade transitions between all media types
   - Maintain visual variety and engagement

4. **Video-First Fallback Mode** (`video_assembly_mode: "video_first_fallback"`):
   - Use all available product videos first in sequence
   - Calculate remaining duration after all videos played
   - Add product images for remaining time if needed
   - Prioritize video content, use images only as supplementary content
   - Apply transitions between videos and at video-to-image boundary

#### Aspect Ratio Handling

Product videos **MUST** support configurable aspect ratio handling per profile:

1. **Letterbox Mode** (`video_aspect_mode: "letterbox"`):
   - Maintain original video aspect ratio
   - Scale video to fit within 9:16 frame
   - Add black padding (letterboxing) to fill remaining space
   - Center video vertically and horizontally
   - FFmpeg filter: `scale=w:h:force_original_aspect_ratio=decrease,pad=1080:1920:(1080-iw)/2:(1920-ih)/2:black`

2. **Crop-to-Fit Mode** (`video_aspect_mode: "crop_to_fit"`):
   - Scale video to completely fill 9:16 frame
   - Crop edges to eliminate black bars
   - Center crop to preserve main subject
   - May lose peripheral content
   - FFmpeg filter: `scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920`

3. **Smart Scale Mode** (`video_aspect_mode: "smart_scale"`):
   - Analyze video aspect ratio automatically
   - Apply crop if video aspect ratio close to 9:16 (within 10% difference)
   - Apply letterbox if aspect ratio significantly different
   - Optimize for visual quality and content preservation
   - Decision algorithm: `abs(video_ratio - target_ratio) / target_ratio < 0.1` → crop, else letterbox

#### Audio Normalization

Product videos **MUST** support configurable audio handling:

1. **Complete Removal** (`video_audio_handling: "remove"`):
   - Strip all original audio tracks from product videos
   - Output contains only voiceover and background music
   - FFmpeg flag: `-an` on video inputs
   - Default behavior for most profiles

2. **Mixed Audio** (`video_audio_handling: "mixed"`):
   - Preserve original video audio at reduced volume
   - Apply configurable volume adjustment (default: -30dB)
   - Mix original audio with voiceover and background music
   - FFmpeg filter: `volume=-30dB` on video audio stream
   - Configurable via `video_original_volume: -30` setting (range: -60 to 0 dB)
   - Useful for ambient sound or product demonstration audio

#### Format Normalization

All product videos **MUST** be normalized to consistent format:

- **Codec**: H.264 (libx264) for maximum compatibility
- **Frame Rate**: 30fps (match global frame_rate setting)
- **Pixel Format**: yuv420p (required for broad device support)
- **Resolution**: Scale to match profile resolution (default: 1080x1920)
- **Bitrate**: Automatic based on content (CRF 23 default)

**Pre-processing Pipeline**:
- Detect incompatible formats requiring transcoding
- Transcode videos with non-H.264 codecs
- Normalize frame rates to target FPS
- Convert pixel formats to yuv420p
- Cache normalized videos to avoid re-processing

#### Duration Matching Algorithm

Product videos **MUST** match voiceover duration precisely (±1 second tolerance):

1. **Calculate Required Duration**: Extract voiceover audio duration as target
2. **Select Clips** (mode-dependent):
   - Sequential: Use all videos in order
   - Single Best: Select longest video
   - Mixed Media: Distribute videos across timeline with images
   - Video-First: Use all videos, add images if needed
3. **Adjust Duration**:
   - If too short: Loop videos with crossfade or add images
   - If too long: Trim last video with fade-out effect
4. **Apply Transitions**: Add crossfade between all clips (default: 0.5s)
5. **Verify Duration**: Ensure final video matches voiceover ±1s

#### Transition System

Product videos **MUST** support smooth transitions:

1. **Video-to-Video Transitions**:
   - Crossfade between consecutive video clips
   - Configurable duration: `video_transition_duration: 0.5` (seconds)
   - Apply to all video boundaries in sequential/mixed modes
   - FFmpeg xfade filter: `xfade=transition=fade:duration=0.5:offset=X`

2. **Video-to-Image Transitions**:
   - Apply same crossfade style as video-to-video
   - Match transition duration with image-to-image transitions
   - Handle aspect ratio changes smoothly
   - Maintain consistent visual flow

3. **Loop Transitions** (Single Best mode):
   - Seamless crossfade at loop point
   - Prevent visible "jump" or discontinuity
   - Create infinite loop effect

#### Media Validation Requirements

Product video assembly **MUST** validate media availability:

- **Minimum Requirements**: ≥1 product video to enable video-first profiles
- **Fallback Strategy**: If no videos available, fall back to image-only profiles
- **Duration Validation**: Warn if total video duration significantly shorter than voiceover
- **Quality Checks**: Validate video files are readable and not corrupted

#### Profile Configuration Examples

```yaml
video_profiles:
  # Sequential video assembly with letterbox
  product_video_sequential:
    use_scraped_videos: true
    use_scraped_images: true
    video_assembly_mode: "sequential"
    video_aspect_mode: "letterbox"
    video_audio_handling: "remove"
    video_transition_duration: 0.5

  # Single video loop with crop-to-fit
  product_video_single:
    use_scraped_videos: true
    use_scraped_images: false
    video_assembly_mode: "single_best"
    video_aspect_mode: "crop_to_fit"
    video_audio_handling: "mixed"
    video_original_volume: -30
    video_transition_duration: 0.8

  # Mixed media with smart scaling
  product_video_mixed:
    use_scraped_videos: true
    use_scraped_images: true
    video_assembly_mode: "mixed_media"
    video_aspect_mode: "smart_scale"
    video_audio_handling: "remove"
    image_display_duration: 2.5
    video_transition_duration: 0.5

  # Video-first with image fallback
  product_video_primary:
    use_scraped_videos: true
    use_scraped_images: true
    video_assembly_mode: "video_first_fallback"
    video_aspect_mode: "letterbox"
    video_audio_handling: "remove"
    video_transition_duration: 0.5
```

#### Implementation Requirements

- **Profile System Integration**: All video settings configurable per profile
- **Backward Compatibility**: Existing image-only profiles unaffected
- **CLI Overrides**: Support runtime override of video assembly mode
- **Validation**: Validate configuration at startup with clear error messages
- **Performance**: Efficient FFmpeg filter chains for minimal processing time
- **Error Handling**: Graceful degradation if video processing fails

### Subtitle System - Unified Positioning
- **Unified Anchor System**: Single flexible positioning approach with anchor-based layout
- **Anchor Options**: `top`, `center`, `bottom`, `above_content`, `below_content`
- **Content-Aware Mode**: Automatic position adjustment based on visual content boundaries
- **Absolute Mode**: Fixed positioning using anchor + margin (content_aware=false)
- **Relative Mode**: Dynamic positioning relative to image boundaries (content_aware=true)
- **Margin Control**: Configurable spacing as fraction of frame height (0.0-0.5)
- **Text Constraints**: Ensure subtitle width doesn't exceed image width
- **Spacing Consistency**: Maintain consistent spacing between content and subtitles

### Two-Part Subtitle System
- **Independent Dual Lines**: Display two independent subtitle lines simultaneously
- **Upper Line (Product Link/Business URL)**:
  - Display shortened product URL from `data.json` by default
  - Source field configurable per profile (e.g., `product_url`, `product_link`, custom field)
  - **Custom URL Override**: Support custom business URLs (e.g., social links, landing pages) via `custom_url` field
  - Positioned above image using `above_content` anchor
  - **Timing Modes**:
    - `use_full_duration: true` - Always visible throughout video (static display)
    - `use_full_duration: false` - Display only during CTA (Call-To-Action) moments
- **Lower Line (Voiceover Subtitles)**:
  - Standard timed subtitles synchronized to voiceover audio
  - Positioned below image using `below_content` anchor
  - Uses existing subtitle generation system (STT-based timing)
- **CTA Detection System**:
  - Keyword-based detection of call-to-action moments in voiceover
  - Default keywords: "link", "bio", "check out", "visit", "follow", "share", "like", "subscribe", "click", "tap", "swipe", "purchase", "buy", "shop", "get"
  - Configurable keywords, case sensitivity, and merge gap threshold
  - **Continuous Display Mode**: Merge all CTA windows into single continuous period from first to last CTA
  - Automatic timing synchronization with lower subtitle content
  - Fallback to full duration if no CTA moments detected
- **Profile Configuration**:
  - Enable/disable two-part mode per video profile
  - Configure data source for upper line (default: shortened product URL)
  - Custom URL field to override product URLs with business links
  - Independent styling for upper and lower lines
  - Separate margin/positioning control for each line
  - CTA timing control via `use_full_duration` setting
- **Content-Aware Positioning**: Both lines adjust position based on visual content boundaries
- **Backward Compatibility**: Single-line subtitle mode remains default when two-part disabled

### Profile-Specific Settings
- **All visual settings MUST be configurable per video profile**
- Image positioning and sizing settings (width, position, aspect ratio)
- **Video assembly settings** (mode, aspect ratio, audio handling, transitions)
- Subtitle positioning, styling, fonts, colors, and effects
- Profile settings override global defaults through merging system
- Maintain backward compatibility with existing global configuration
- Support unified subtitle positioning system with anchor-based layout
- Support product video configuration per profile

### Font & Color Management
- **Style Preset System**: 5 predefined presets (minimal, modern, bold, animated, random)
- **Preset Descriptions**:
  - `minimal`: Clean, simple styling with no effects
  - `modern`: Contemporary look with subtle effects (karaoke)
  - `bold`: High contrast, bold styling with fade effects
  - `animated`: Full animations with movement effects
  - `random`: Randomized font, colors, and single animation effect
- **Random Preset Features**: Randomized font selection, color pairs, and single animation effect
- **Font Randomization**: Selection from curated collection with deterministic seeding per video
- **Color Randomization**: Coordinated text/outline color combinations with proper contrast
- **System Integration**: Full compatibility with ASS, SRT, and FFmpeg rendering

### ASS Effects System
- **Per-video effect consistency**: Effects MUST be selected once per video, not per subtitle segment
- **Proper ASS formatting**: All ASS override codes MUST be enclosed in curly braces `{}` to prevent literal text display
- **Effect Limitation**: Exactly 1 effect per video to prevent visual clutter and rendering issues
- **Effect Variety**: Support scale_pulse, rotation_bounce, glow, typewriter, karaoke, fade, and movement effects
- **Random Effect Selection**: RANDOM preset selects exactly 1 effect from all available effects using product ID seeding
- **Preset Effect Mapping**:
  - `minimal`: No effects
  - `modern`: Karaoke only
  - `bold`: Fade only
  - `animated`: Movement only
  - `random`: One randomly selected effect from all available
- **Karaoke timing**: Implement word-by-word highlighting with proper `\k` tag formatting in centiseconds
- **Visual consistency**: Maintain coherent animation style throughout individual videos
- **FFmpeg compatibility**: Ensure all ASS effects render correctly through FFmpeg's libass library

### AI Service Integration
- Auto-select AI models from OpenRouter API
- Default to free models with config fallback
- Prioritize Google Cloud Chirp 3 HD voices for TTS
- Hide skipped voices in logs (even debug mode)

### Stock Background Music Integration

ContentEngineAI **MUST** support a modular multi-platform stock music system with graceful degradation.

#### Platform Architecture
- **Modular Design**: Platform-agnostic music client interface for multiple stock audio providers
- **Primary Platform**: Freesound.org integration (community-driven, Creative Commons licensed)
- **Extensibility**: Support for future platforms (AudioJungle, Epidemic Sound, Artlist, etc.)
- **Fallback Strategy**: Hierarchical fallback from online platforms to local stock files

#### Music Selection & Discovery
- **Dynamic Duration Matching**: Search for tracks matching voiceover duration (±tolerance)
- **Fallback Search**: Broader search if duration-specific queries yield no results
- **Quality Filtering**: Filter by ratings, duration ranges, licenses, and file quality
- **Smart Selection**: Sort and select best-matching track from search results

#### Download & Authentication
- **OAuth2 Support**: Full quality downloads with OAuth2 token management
- **API Key Fallback**: Preview quality downloads when OAuth2 unavailable
- **Token Refresh**: Automatic access token refresh with `.env` persistence
- **Preview Downloads**: Lower quality MP3 previews for rapid prototyping

#### Resilience & Error Handling
- **Circuit Breaker Pattern**: Fast-fail on repeated API failures with exponential backoff
- **Timeout Management**: Configurable timeouts for search (30s) and download (300s) operations
- **Retry Logic**: Limited retries (2 attempts) with exponential backoff on transient failures
- **Session Management**: Automatic session recovery on connection failures
- **Graceful Degradation**: Fallback to local stock files if all online sources fail

#### Configuration Requirements
- **Three-Tier Config**: API keys in `.env`, settings in YAML, runtime overrides via CLI
- **Search Configuration**: Configurable search query, filters, sort order, max results
- **Duration Constraints**: Min/max duration ranges for music track filtering
- **Performance Tuning**: Configurable timeouts, chunk sizes, buffer durations

#### Local Fallback System
- **Static Music Library**: 3+ pre-selected local music files for offline/fallback use
- **Random Selection**: Pick random local track when online sources unavailable
- **Licensing Compliance**: Only include properly licensed music in repository
- **Efficient I/O**: Use memory-mapped file copying for large audio files (>1MB)

#### Attribution & Licensing
- **License Tracking**: Store license type, author, URL for each downloaded track
- **Attribution Data**: Include source, name, author, license in video metadata
- **Creative Commons**: Support CC0, CC-BY, CC-BY-SA, CC-BY-NC license types
- **Compliance**: Ensure proper attribution per license requirements

#### Audio Processing Integration
- **Format Support**: MP3, WAV, FLAC, AAC input formats
- **Volume Normalization**: Automatic volume adjustment relative to voiceover (-20dB default)
- **Fade Effects**: Configurable fade-in (2s) and fade-out (3s) durations
- **Loop Support**: Loop shorter tracks to match video duration if needed
- **Mix Duration**: Adjust music to match voiceover length (longest strategy)

#### Performance & Optimization
- **Async Operations**: Full async/await support with `aiohttp` for concurrent downloads
- **Connection Pooling**: Reuse HTTP sessions across multiple requests
- **Parallel Processing**: Download multiple tracks concurrently when needed
- **Memory Efficiency**: Stream large files in chunks (32KB default) to avoid memory spikes
- **Cache Strategy**: Store downloaded tracks in product-specific directories

#### Multi-Platform Extensibility Design
- **Abstract Interface**: Define `BaseStockMusicClient` interface for all platforms
- **Platform Registry**: Decorator-based registration system for new platforms
- **Factory Pattern**: `StockMusicFactory` to instantiate platform-specific clients
- **Unified API**: Consistent search/download interface across all platforms
- **Platform Detection**: Automatic platform selection based on available credentials

#### Future Platform Support
When adding new stock music platforms (AudioJungle, Epidemic Sound, Artlist):
- Implement `BaseStockMusicClient` interface with platform-specific logic
- Register platform with `@register_stock_music_provider` decorator
- Add platform-specific config section to `audio_settings` in YAML
- Store platform credentials in `.env` with `{PLATFORM}_API_KEY` naming
- Maintain backward compatibility with existing Freesound integration
- Support platform-specific search filters and quality tiers

### Output Management
- Fully configurable "outputs" directory structure
- Implement cleanup function to remove unexpected files/directories
- Maintain organized file structure per product ID

## Global Requirements

### Configuration & CLI
- Three-tier precedence: CLI args > env vars > YAML config
- CLI arguments override all other configuration sources
- Environment variables for all major configuration settings
- Global debug mode across all components
- Validate configuration at startup with clear error messages

### Error Handling & Resilience
- Continue processing on individual failures
- Graceful degradation when services unavailable
- Clear error messages for missing environment variables
