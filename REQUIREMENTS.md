# Project Requirements

## Configuration System

ContentEngineAI **MUST** use a dual configuration system:

### 1. YAML Files (`config/`)
- Application settings, preferences, timeouts
- Safe to commit to version control
- Reference environment variables via `api_key_env_var` fields

### 2. Environment Variables (`.env`)
- API keys, credentials, secrets only
- Never committed (gitignored)
- Loaded at runtime and injected into YAML config

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

### Subtitle System
- Support both absolute and relative subtitle positioning modes
- **Absolute Mode**: Configurable image scaling and subtitle position
- **Relative Mode**: Calculate subtitle position relative to image boundaries
- Ensure subtitle width doesn't exceed image width
- Maintain consistent spacing between image bottom and subtitle top

### Profile-Specific Settings
- **All visual settings MUST be configurable per video profile**
- Image positioning and sizing settings (width, position, aspect ratio)
- Subtitle positioning, styling, fonts, colors, and effects
- Profile settings override global defaults through merging system
- Maintain backward compatibility with existing global configuration
- Support unified subtitle positioning system with anchor-based layout

### Font & Color Management
- Random font selection from curated list of 5 popular video fonts
- Random color selection from 5 coordinated text/outline color pairs
- Verify font system availability with fallback options
- Ensure compatibility with SRT, ASS, and FFmpeg

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
- CLI arguments override config file settings
- Global debug mode across all components
- Validate configuration at startup with clear error messages

### Error Handling & Resilience
- Continue processing on individual failures
- Graceful degradation when services unavailable
- Clear error messages for missing environment variables