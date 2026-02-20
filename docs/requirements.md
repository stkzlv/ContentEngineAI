# Project Requirements

High-level requirements for ContentEngineAI.

---

## Global Requirements

### Configuration System
- **Three-tier precedence**: CLI arguments > Environment variables > YAML files
- CLI arguments override only when **explicitly provided** by the user (use `default=None` in argparse, not hardcoded defaults)
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

### Product Limits
- **`max_products`**: Total products to collect across all keywords (global cap)
- **`products_per_keyword`**: Maximum products to scrape per individual keyword
- Both limits must be consistent across all configurations (pipeline, scraper, CLI)
- Processing continues through keyword list until `max_products` is reached
- Early termination when global limit is hit, even if keywords remain

---

## Video Producer Module

### Video Assembly
- Adjust video duration to match voiceover length
- Display images with configurable duration (2-3 seconds)
- Smooth transitions (crossfade) between media elements
- Reuse images if needed to fill remaining time

### Image Positioning
- Width as percentage of frame (default 100%)
- **Vertical alignment**: center (default) or top with configurable offset
- Always centered horizontally
- Preserve aspect ratio by default
- Reserve space below for subtitles (default 15%)

### Video Positioning
- Top offset as percentage of frame (default 10%)
- Content height as percentage of frame (default 75%)
- **Aspect modes**: letterbox (black bars), crop-to-fit (fill frame), smart-scale (auto-select)
- Smart-scale uses 10% tolerance to choose between letterbox and crop
- **Assembly modes**: sequential, single-best, mixed-media, video-first-fallback
- **Audio handling**: remove original audio or mix at reduced volume
- Normalize to consistent format (H.264, 30fps, yuv420p)
- Match final duration to voiceover (±1 second tolerance)

### Subtitle Positioning
- **Anchors**: top, center, bottom, above-content, below-content
- Content-aware mode positions relative to actual media bounds
- Margin as percentage of frame height (default 10%)
- Horizontal alignment: left, center (default), right
- Font size scales with frame height (default 4%, min 16px, max 100px)
- Safe zone: max vertical position 95% to stay readable

### Two-Part Subtitles
- Upper line: static product URL or affiliate link
- Lower line: voiceover transcription (word-by-word timing)
- Independent positioning and styling per line
- CTA detection triggers timed URL display

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
- Fully automated OAuth2 token setup: headless browser (Playwright) handles login + authorize + code capture, no manual steps
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

### Link-in-Bio Integration
- After publishing a video, add the product's Amazon affiliate link to a link-in-bio page
- Provider-agnostic: swappable between Lnk.Bio, Linktree, Beacons, etc.
- Configurable max links with automatic oldest-link rotation
- Non-blocking: failures never affect video publishing
- Disabled by default, toggled via config

### Published Products Registry
- Maintain a registry of all published products in the outputs directory
- Fields: product ID (ASIN), product title, canonical URL, affiliate URL
- Dual format: JSON (machine-readable) and CSV (spreadsheet-friendly)
- Append new entries after each successful publish (no duplicates)
- Support bulk import from existing scraped data directories
- CLI command to rebuild registry from existing data

---

## Content Metadata Module

### Platform-Specific Optimization
- **Unified mode**: Single metadata set for all platforms
- **Optimized mode**: Platform-tailored titles, descriptions, hashtags
- Character limit validation per platform

### Compliance
- Automatic `#ad` inclusion for FTC compliance
- Proper affiliate disclosure handling
