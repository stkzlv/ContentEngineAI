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

### Output Management
- Fully configurable "outputs" directory structure
- Implement cleanup function to remove unexpected files/directories
- Maintain organized file structure per product ID

## Global Requirements

### Configuration & CLI
- Three-tier precedence: CLI args > env vars > YAML config
- CLI arguments override all other configuration sources
- Global debug mode across all components
- Validate configuration at startup with clear error messages

### Error Handling & Resilience
- Continue processing on individual failures
- Graceful degradation when services unavailable
- Clear error messages for missing environment variables
