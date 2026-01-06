# Project Requirements

High-level requirements for ContentEngineAI. Implementation details are in specs (`.spec-workflow/specs/`).

---

## Global Requirements

### Configuration System
- **Three-tier precedence**: CLI arguments > Environment variables > YAML files
- CLI arguments enable runtime customization without file changes
- Environment variables (`.env`) store secrets only—never committed
- YAML files (`config/`) store application settings—safe to commit
- Validate all configuration at startup with clear error messages

### Security
- Store API keys and secrets in `.env` file only
- Provide `.env.example` template for required variables
- Never commit `.env` or put secrets in YAML files

### Error Handling & Resilience
- Continue processing on individual item failures (graceful degradation)
- Retry transient network failures with exponential backoff (timeouts, rate limits)
- Circuit breaker pattern to prevent cascading failures from unavailable services
- Provide clear error messages for missing configuration
- Support fail-fast mode via CLI flag when strict behavior needed

### Logging & Monitoring
- Global debug mode across all components
- Progress tracking with `[N/total]` format for batch operations
- Summary reports at end of each pipeline phase

### Documentation Standards
- Required root files: README.md, CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md, CHANGELOG.md, LICENSE
- Use GitHub-Flavored Markdown (GFM)
- Update documentation in same PR as code changes
- Use relative paths for internal links
- Provide working code examples with context and expected output

### Documentation Structure
- Extended documentation organized in `docs/` directory
- README.md contains: project title, description, key features, quick start, links to detailed docs
- CHANGELOG.md follows Keep a Changelog format (Added, Changed, Deprecated, Removed, Fixed, Security)

### Outputs Directory
- Centralized `outputs/` directory for all pipeline artifacts
- Per-product directories: `outputs/<product_id>/` with `data.json`, `images/`, `videos/`
- Global directories: `cache/`, `logs/`, `reports/` for shared resources
- Customizable via `--outputs-dir` CLI flag
- Automatic cleanup of published products (configurable)

### Async Concurrency Control
- Semaphore-based limits for resource-intensive operations (ffmpeg, I/O, network)
- Configurable concurrency limits per operation type
- Prevent system overload during batch processing

### Resource Cleanup
- Use context managers for connections, file handles, temporary resources
- Automatic cleanup of partial files on failure
- Proper connection pool management for HTTP sessions

### Validation Framework
- Pydantic models for configuration and data structures
- Custom validators for domain-specific rules (product IDs, URLs, file formats)
- Early validation with clear error messages

### Performance Metrics
- Time tracking for critical operations (scraping, video assembly, API calls)
- Success/failure rate monitoring
- Resource usage patterns for optimization

---

## Scraper Module

### Multi-Platform Architecture
- Modular design with platform-specific implementations (Amazon first)
- Separate core scraping logic from platform adapters
- Support both direct product ID lookups and keyword searches

### Product Discovery & Media Extraction
- Extract product data: title, price, description, ID, ratings, reviews
- Download high-resolution images and videos
- Filter out low-quality or invalid media
- Store media in dedicated directories per product ID

### Search & Filtering
- Keyword-based searches with filters (price, rating, shipping, brands)
- Sorting options and regional redirect handling
- Product ID validation against platform formats
- Skip products lacking essential data

### Stealth & Human Simulation
- Implement detection evasion techniques
- Simulate human-like browsing patterns
- Handle failures gracefully without halting batch

### Batch Mode
- Process lists of product IDs and/or keywords
- Sequential processing with rate limiting
- Deduplication across multiple sources
- Summary reporting of success/failure counts

---

## Video Producer Module

### Video Assembly
- Adjust video duration to match voiceover length
- Display images with configurable duration (2-3 seconds)
- Smooth transitions between media elements
- Reuse images if needed to fill remaining time

### Product Video Support
- **Assembly modes**: sequential, single-best, mixed-media, video-first-fallback
- **Aspect ratio handling**: letterbox, crop-to-fit, smart-scale
- **Audio handling**: remove original audio or mix at reduced volume
- Normalize all videos to consistent format (H.264, 30fps, yuv420p)
- Match final duration to voiceover (±1 second tolerance)

### Subtitle System
- **Unified anchor-based positioning**: top, center, bottom, above/below content
- Content-aware mode adjusts position based on visual boundaries
- Two-part subtitle support (upper: product URL, lower: voiceover text)
- CTA detection for timed URL display
- Per-profile styling and positioning

### Profile System
- All visual settings configurable per video profile
- Profile settings override global defaults
- Backward compatibility with existing configurations
- Runtime profile selection via CLI

### Font & Color Management
- Style presets: minimal, modern, bold, animated, random
- Deterministic randomization using product ID as seed
- Coordinated color combinations with proper contrast

### ASS Effects
- One effect per video for visual consistency
- Support: scale, rotation, glow, typewriter, karaoke, fade, movement
- Proper ASS formatting for FFmpeg compatibility

### AI Service Integration
- Auto-select AI models from OpenRouter API
- Default to free models with fallback
- Google Cloud TTS with voice prioritization

### Stock Background Music
- Multi-platform music client (Freesound primary)
- Dynamic duration matching to voiceover
- OAuth2 for full quality, API key for previews
- Fallback to local stock files if online sources fail
- Proper attribution and license tracking

### Batch Mode
- Automatic product discovery from outputs directory
- Random or fixed profile selection
- Inter-product delays for rate limiting
- Comprehensive summary reporting

---

## Batch Processing Module

### Global Pipeline
- **End-to-end execution**: scraping → video production → publishing
- Single command for complete workflow
- Phase isolation: failures don't cascade between phases
- Unified configuration with CLI overrides

### Execution Phases
1. **Scraping**: Collect product data, images, videos
2. **Handoff**: Filter products by media availability
3. **Production**: Generate videos with selected profile
4. **Publishing**: Upload and schedule to platforms

### Profile Randomization
- Random profile selection per product from configured pool
- Deterministic seeding for reproducibility
- Profile compatibility checking with skip on mismatch

### Summary Reporting
- Per-phase counts (success, failure, skipped)
- Overall pipeline statistics
- Profile usage distribution

---

## Publisher Module

### Service Integration
- Publish via third-party scheduling services (e.g., Late.dev)
- Multi-platform support: YouTube, TikTok, Instagram
- API-based upload and status tracking

### Scheduling
- Immediate or future publish times with timezone support
- Recurring schedule slots (e.g., "Monday 9am, Wednesday 2pm")
- Auto-scheduling to first available slot
- Batch scheduling of multiple videos

### Post-Publication Cleanup
- Remove published product directories after confirmation
- Configurable per-platform enable/disable
- Safety checks verify publication success before deletion
- Require all platforms to succeed before cleanup

---

## Content Metadata Module

### Platform-Specific Optimization
- **Unified mode**: Single metadata set for all platforms
- **Optimized mode**: Platform-tailored titles, descriptions, hashtags
- Character limit validation per platform

### Compliance
- Automatic `#ad` inclusion for FTC compliance
- Proper affiliate disclosure handling
