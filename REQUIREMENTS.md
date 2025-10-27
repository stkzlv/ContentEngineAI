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
- Subtitle positioning, styling, fonts, colors, and effects
- Profile settings override global defaults through merging system
- Maintain backward compatibility with existing global configuration
- Support unified subtitle positioning system with anchor-based layout

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
