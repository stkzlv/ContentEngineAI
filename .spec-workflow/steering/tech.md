# Technology Stack

## Project Type

**Async-first Python CLI pipeline** for automated video production with batch processing capabilities. Designed as a command-line orchestrator that coordinates AI services, media APIs, and FFmpeg video assembly into a cohesive production workflow.

## Core Technologies

### Primary Language(s)
- **Language**: Python 3.12
- **Runtime**: CPython 3.12+ (strict version constraint <3.13)
- **Package Management**: Poetry 1.0+
- **Async Framework**: asyncio with aiohttp and aiofiles for I/O operations

### Key Dependencies/Libraries

**AI & Language Models:**
- **OpenRouter**: LLM provider for script and description generation (primary)
- **OpenAI Whisper** (20240930): Speech-to-text for subtitle timing extraction
- **Google Cloud Text-to-Speech** (^2.26.0): Premium TTS with Chirp 3 HD voices
- **Google Cloud Speech** (^2.32.0): STT fallback with word-level timestamps
- **Coqui TTS** (^0.26.0): Local TTS fallback provider

**Media Processing:**
- **FFmpeg** (external binary): Video assembly, filtering, subtitle rendering, audio mixing
- **ffmpeg-python** (^0.2.0): Python wrapper for FFmpeg command construction
- **Pillow**: Image manipulation and validation

**Web Scraping:**
- **Botasaurus** (^4.0.88): Browser automation with anti-detection features
- **Playwright** (^1.55.0): Headless browser engine for product scraping
- **BeautifulSoup4** (^4.13.4): HTML parsing and data extraction

**Media APIs:**
- **pexels-api-py** (^0.0.5): Stock images and videos
- **freesound-api** (^1.1): Background music downloads

**HTTP & Networking:**
- **aiohttp** (^3.11.18): Async HTTP client with connection pooling
- **httpx** (^0.27.0): Modern HTTP client with async support
- **tenacity** (^9.1.2): Retry logic with exponential backoff

**Configuration & Data:**
- **PyYAML** (^6.0.1): YAML configuration parsing
- **Pydantic**: Type-safe configuration validation
- **python-dotenv** (^1.0.1): Environment variable management
- **pysrt** (^1.1.2): SRT subtitle file parsing and generation

### Application Architecture

**Dependency-Aware Pipeline with Parallel Execution:**

```
Step 1: gather_visuals        (independent)
    ↓
Step 2: generate_script        (depends: gather_visuals)
    ↓
Step 3: generate_description   (depends: generate_script)
    ↓
Step 4: create_voiceover       (depends: generate_script)
    ↓  ↓
    ↓  ├─→ Step 5a: generate_subtitles  (parallel)
    ↓  └─→ Step 5b: download_music      (parallel)
    ↓       ↓
    └───────┴→ Step 6: assemble_video
```

**Key Architectural Patterns:**
- **Pipeline Graph**: Topological sorting for automatic dependency resolution
- **Provider Fallbacks**: Multi-level fallback chains for TTS, STT, and LLM
- **Async/Await**: All I/O operations non-blocking
- **Context Management**: Immutable pipeline state passed through steps
- **Plugin Architecture**: Base interfaces for extensible providers

### Data Storage

- **Primary Storage**: File system (JSON product data, MP4 videos, media assets)
- **Caching**:
  - Multi-level TTL cache for API responses (LLM, TTS, STT, stock media)
  - Media metadata cache (FFprobe results)
  - File-based persistence with thread safety
- **Data Formats**:
  - JSON (product data, performance metrics, configuration overrides)
  - JSONL (historical performance logs)
  - SRT/ASS (subtitle formats)
  - MP4 (final video output)
  - YAML (configuration files)

### External Integrations

**APIs:**
- **OpenRouter**: LLM inference with multiple model fallbacks
- **Google Cloud Platform**: TTS (Chirp 3 HD) and STT services
- **Pexels**: Stock media (photos and videos) with rate limiting
- **Freesound**: Creative Commons music downloads
- **PicSee**: URL shortening for affiliate links (with fallback support)

**Protocols:**
- **HTTP/REST**: All API communication
- **WebSocket**: Dashboard real-time updates (localhost:3000)
- **File System**: Media storage and intermediate artifacts

**Authentication:**
- **API Keys**: Environment variable-based (OpenRouter, Google Cloud, Pexels, Freesound, PicSee)
- **Service Account JSON**: Google Cloud authentication
- **No OAuth**: All integrations use static credentials

### Monitoring & Dashboard Technologies

- **Dashboard Framework**: Web-based (spec-workflow MCP server, localhost:3000)
- **Real-time Communication**: WebSocket for approval workflows and status updates
- **Visualization**:
  - CLI-based performance reports (structured text output)
  - JSON/JSONL metrics for programmatic analysis
- **State Management**: File system as source of truth
  - Approval requests stored in `.spec-workflow/approvals/`
  - Performance history in `outputs/performance_history.jsonl`
  - Pipeline artifacts in `outputs/<ASIN>/`

## Development Environment

### Build & Development Tools
- **Build System**: Make (Makefile with 20+ targets)
- **Package Management**: Poetry with lock file for reproducibility
- **Development Workflow**:
  - Watch mode not required (batch processing focus)
  - Debug mode preserves intermediate files
  - Step-specific execution for targeted debugging

**Common Make Targets:**
```bash
make install        # Install all dependencies
make dev-setup      # Complete development environment setup
make quick-check    # Ruff + MyPy (fast quality gate)
make full-check     # Lint + security + test-cov (comprehensive)
make lint           # 7-tool quality suite
make test           # Full test suite
make security       # Vulnerability scanning
```

### Code Quality Tools

**Static Analysis:**
- **Ruff** (^0.5.0): Fast Python linter and formatter (replaces Flake8, isort, Black for linting)
- **MyPy** (^1.10.1): Static type checking with strict configurations
- **Bandit** (^1.7.8): Security vulnerability scanning
- **Vulture** (^2.13): Dead code detection (80% confidence threshold)
- **Safety** (^3.0.0): Dependency vulnerability checking

**Formatting:**
- **Black** (^25.1.0): Code formatting (88-character line limit)
- **Ruff**: Integrated formatter for consistency

**Testing Framework:**
- **Pytest** (^8.3.2): Unit and integration testing
- **pytest-asyncio** (^0.24.0): Async test support
- **pytest-cov** (^5.0.0): Coverage reporting (40% minimum, targeting 90%/80%)
- **pytest-xdist** (^3.6.0): Parallel test execution
- **pytest-timeout** (^2.3.1): Test timeout management (300s default)
- **aioresponses** (^0.7.6): Mock async HTTP requests
- **pytest-mock** (^3.14.0): Enhanced mocking capabilities

**Coverage Targets:**
- Unit tests: >90%
- Integration tests: >80%
- Current minimum: 40% (enforced)

**Documentation:**
- Markdown-based (18 documentation files)
- Auto-generated performance reports via `tools/performance_report.py`

### Version Control & Collaboration

- **VCS**: Git with GitHub remote
- **Branching Strategy**: GitHub Flow
  - `main` branch for production-ready code
  - Feature branches: `feature/*`, `bugfix/*`, `hotfix/*`, `docs/*`
  - No long-lived development branches
- **Code Review Process**:
  - Pull requests required for all changes
  - CI/CD checks must pass (lint, test, security)
  - Squash merge for clean history
  - Conventional commit messages (`feat:`, `fix:`, `docs:`, `chore:`)

**CI/CD Workflows:**
- **CI**: Lint + test + coverage on push/PR to main
- **Security**: Weekly vulnerability scans + PR checks
- **Release**: Version tag-triggered builds

### Dashboard Development

- **Live Reload**: Not applicable (MCP server managed externally)
- **Port Management**: Fixed port 3000 for spec-workflow dashboard
- **Multi-Instance Support**: Single dashboard per project directory
- **Dashboard Type**: Web-based approval workflow UI (spec-workflow MCP)

## Deployment & Distribution

- **Target Platform(s)**: Linux, macOS (development tested on Linux 6.8.0)
- **Distribution Method**:
  - Git clone + Poetry install
  - No pre-built binaries or package manager distribution
  - Source distribution via GitHub
- **Installation Requirements**:
  - Python 3.12+ (strict)
  - FFmpeg (system-installed binary)
  - Poetry 1.0+
  - Playwright browsers (via `poetry run playwright install`)
  - API keys: OpenRouter, Google Cloud, Pexels, Freesound (optional: PicSee)
- **Update Mechanism**: Git pull + `poetry install`

## Technical Requirements & Constraints

### Performance Requirements

- **First Video Generation**: <60 seconds from ASIN to final MP4
- **Pipeline Throughput**: 26% faster with parallel execution (1.35x speedup)
- **Memory Usage**: Monitored via PerformanceMonitor, no hard limits
- **Startup Time**: TTS model preloading during pipeline initialization
- **API Latency**: Retry logic with exponential backoff (configurable timeouts)

**Benchmarks:**
- Subtitle generation: <10 seconds for 30-second video
- Script generation: <5 seconds (LLM API call)
- Video assembly: <20 seconds (FFmpeg processing)

### Compatibility Requirements

**Platform Support:**
- **Operating Systems**: Linux (primary), macOS (secondary), Windows (untested)
- **Architectures**: x86_64 (required for some dependencies)
- **Python Versions**: 3.12 only (strict constraint <3.13)

**Dependency Versions:**
- Poetry lock file ensures reproducible builds
- Google Cloud libraries: Pin to 2.x major versions
- Whisper: Pin to 20240930 release
- Ruff, MyPy, Pytest: Allow minor version updates via `^` semver

**Standards Compliance:**
- PEP 8: Python code style (enforced via Ruff)
- SRT Subtitle Format: Standard timing and formatting
- MP4 Container: H.264 video, AAC audio (social media compatible)

### Security & Compliance

**Security Requirements:**
- **API Key Management**: Environment variables only, never committed
- **Secret Scanning**: Bandit checks for hardcoded secrets
- **Dependency Scanning**: Safety checks for known vulnerabilities (weekly + PR)
- **Input Validation**: Pydantic models for configuration, Playwright stealth for scraping
- **Subprocess Safety**: FFmpeg command injection prevention via ffmpeg-python

**Compliance Standards:**
- **MIT License**: Open source, permissive use
- **Attribution Required**: Auto-generated files for stock media (Pexels, Freesound)
- **GDPR**: Not applicable (no user data collection)

**Threat Model:**
- **Primary Risks**: API key exposure, dependency vulnerabilities, subprocess injection
- **Mitigations**: Environment variables, Safety scanning, parameterized commands

### Scalability & Reliability

**Expected Load:**
- **Typical Usage**: 1-100 videos per batch
- **API Rate Limits**: Pexels (200 req/hour), Freesound (60 req/minute)
- **Concurrent Requests**: Semaphore-based concurrency control (configurable limits)

**Availability Requirements:**
- **Uptime**: Not applicable (local CLI tool)
- **Provider Fallbacks**: TTS (2 providers), STT (2 providers), LLM (multiple models)
- **Retry Logic**: Exponential backoff for transient failures
- **Disaster Recovery**: Debug mode preserves intermediate artifacts

**Growth Projections:**
- **Horizontal Scaling**: Run multiple instances with different ASINs
- **Future Cloud Deployment**: Remote queue management, multi-user teams
- **Platform Expansion**: Additional e-commerce platforms (eBay, Walmart, Shopify)

## Technical Decisions & Rationale

### Decision Log

1. **Python 3.12 Strict Constraint**:
   - **Why**: Leverage modern async syntax, type hints, and performance improvements
   - **Trade-offs**: Limited to CPython 3.12 users, must upgrade when 3.13 support needed
   - **Alternatives Considered**: Support 3.11-3.13 range (rejected due to dependency compatibility)

2. **FFmpeg Over Native Python Video Libraries**:
   - **Why**: Industry-standard quality, complex filter graphs, subtitle rendering
   - **Trade-offs**: External binary dependency, subprocess overhead
   - **Alternatives Considered**: MoviePy (too slow), OpenCV (limited subtitle support)

3. **Async/Await Throughout Pipeline**:
   - **Why**: Non-blocking I/O for API calls, media downloads, subprocess execution
   - **Trade-offs**: Increased code complexity, debugging challenges
   - **Evaluation Criteria**: 26% performance improvement, scalability for batch processing

4. **Botasaurus for Web Scraping**:
   - **Why**: Built-in anti-detection, Playwright integration, stealth techniques
   - **Trade-offs**: Less mainstream than Selenium, smaller community
   - **Alternatives Considered**: Playwright alone (no anti-detection), Selenium (slower)

5. **Modular YAML Configuration (6 files)**:
   - **Why**: 20% faster loading, better maintainability, 1,429 lines split logically
   - **Trade-offs**: Migration complexity (100% backward compatibility required)
   - **Previous System**: Monolithic `video_config.py` (replaced in v0.6.0)

6. **OpenRouter as Primary LLM Provider**:
   - **Why**: Multi-model access, fallback support, cost-effective
   - **Trade-offs**: Additional API dependency vs direct OpenAI/Anthropic
   - **Evaluation Criteria**: Model flexibility, pricing, reliability

7. **Google Chirp 3 HD for TTS**:
   - **Why**: Premium quality, natural prosody, social media optimization
   - **Trade-offs**: API costs vs local Coqui TTS (lower quality)
   - **Fallback Chain**: Google Cloud → Coqui TTS

8. **Dependency-Aware Pipeline Graph**:
   - **Why**: Automatic parallelization, 26% speedup, maintainability
   - **Trade-offs**: Upfront implementation complexity
   - **Impact**: Steps 5a (subtitles) and 5b (music) run concurrently

## Known Limitations

- **Single Instance Dashboard**: Cannot run multiple spec-workflow dashboards simultaneously (port 3000 conflict)
  - **Impact**: Multi-project workflows require stopping/restarting dashboard
  - **Future Solution**: Dynamic port allocation, multi-instance support

- **FFmpeg Binary Dependency**: External installation required
  - **Impact**: Installation friction, version compatibility issues
  - **Why It Exists**: No pure-Python alternative with equivalent quality
  - **Future**: Consider bundling FFmpeg binary for supported platforms

- **Python 3.12 Only**: Strict version constraint
  - **Impact**: Limits adoption to users with Python 3.12 installed
  - **Why**: Dependency compatibility, modern syntax requirements
  - **Future**: Expand to 3.12-3.13 range when dependencies support it

- **Amazon-Only Scraping**: Multi-platform architecture exists but only Amazon implemented
  - **Impact**: Cannot scrape eBay, Walmart, Shopify out of the box
  - **Future**: Implement additional platform scrapers extending BaseScraper

- **No Distributed Processing**: Single-machine execution only
  - **Impact**: Limited by local resources, no cloud scaling
  - **Why**: Local CLI tool design, no distributed queue system
  - **Future**: Add remote queue support, multi-user coordination

- **Test Coverage Below Target**: 40% minimum enforced, targeting 90%/80%
  - **Impact**: Potential bugs in edge cases, refactoring risks
  - **Why**: Rapid feature development prioritized over test coverage
  - **Timeline**: Increase coverage before 1.0.0 release

- **Subtitle Positioning Limited to Pixel Analysis**: Content-aware positioning based on visual analysis only
  - **Impact**: May occasionally overlap with dynamic content or complex scenes
  - **Future**: ML-based scene understanding, object detection integration
