# ContentEngineAI Architecture

This document provides a comprehensive overview of the ContentEngineAI architecture, including system design, component interactions, and technical implementation details.

## System Overview

ContentEngineAI is a modular, async-first pipeline system designed for automated video production. The architecture follows an eight-step workflow with parallel execution capabilities and comprehensive error handling.

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   AI Services   │    │  Media Sources  │
│                 │    │                 │    │                 │
│ • Amazon Pages  │    │ • Gemini (LLM)  │    │ • Jamendo       │
│ • Product Data  │    │ • Gemini TTS    │    │ • Freesound     │
│ • Images/Videos │    │ • Whisper STT   │    │ • Pexels/Local  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                  ┌─────────────────────────────┐
                  │      Pipeline Engine       │
                  │                             │
                  │  ┌─────────────────────────┐│
                  │  │    Step Orchestrator    ││
                  │  └─────────────────────────┘│
                  │  ┌─────────────────────────┐│
                  │  │   Dependency Manager    ││
                  │  └─────────────────────────┘│
                  │  ┌─────────────────────────┐│
                  │  │  Performance Monitor    ││
                  │  └─────────────────────────┘│
                  └─────────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Video Assembly  │    │  Configuration  │    │     Output      │
│                 │    │                 │    │                 │
│ • FFmpeg        │    │ • YAML Config   │    │ • MP4 Videos    │
│ • Filters       │    │ • Pydantic      │    │ • Logs          │
│ • Subtitles     │    │ • Validation    │    │ • Attribution   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Pipeline Architecture

### Core Pipeline Flow

The video production follows a dependency-aware pipeline with parallel execution:

```
Step 1: Gather Visuals
    ├── Scrape Product Data
    ├── Download Product Media
    └── Fetch Stock Media (Pexels)
         │
Step 2: Generate Script (LLM: Gemini primary, OpenRouter fallback)
         │
Step 3: Generate Description (LLM: Gemini primary, OpenRouter fallback)
         │
Step 4: Create Voiceover (TTS: Gemini primary, Google Cloud fallback)
         │
         ├── Step 5a: Generate Subtitles (Whisper STT)
         └── Step 5b: Download Music (Jamendo primary, Freesound fallback)
                 │
         Step 6: Assemble Video (FFmpeg)
                 │
         Step 7: Burn Pycaps Subtitles (pycaps engine; skipped when engine is ffmpeg)
```

**Key Features:**
- **Parallel Execution**: Steps 5a and 5b run concurrently after step 4
- **Dependency Management**: Automatic handling of step dependencies
- **Resume Capability**: Individual step execution for debugging
- **Performance Monitoring**: Built-in metrics collection

<details>
<summary><b>Core Packages Structure</b> (click to expand)</summary>

```
src/
├── video/                      # Central orchestration & video processing
│   ├── producer/              # Main pipeline orchestrator package
│   │   ├── cli.py             # CLI argument parsing
│   │   ├── context.py         # Pipeline context management
│   │   ├── orchestration.py   # Step orchestration logic
│   │   ├── state.py           # Pipeline state tracking
│   │   ├── steps.py           # Individual pipeline steps
│   │   └── utils.py           # Producer utilities
│   ├── config/                # Pydantic configuration models (v0.14.0+)
│   │   ├── core_models.py     # Core video settings
│   │   ├── audio_models.py    # Audio/TTS configuration
│   │   ├── visual_models.py   # Visual effects settings
│   │   ├── subtitle_models.py # Subtitle configuration
│   │   ├── llm_settings.py    # LLM provider settings
│   │   └── constants.py       # Configuration constants
│   ├── assembler/             # FFmpeg-based video assembly (modular)
│   │   ├── core.py            # VideoAssembler orchestrator
│   │   ├── visual_builder.py  # Visual filter chains
│   │   ├── subtitle_builder.py # Subtitle positioning
│   │   ├── audio_builder.py   # Audio filter chains
│   │   ├── video_strategies.py # Video mode strategies
│   │   ├── media_inspector.py # Media file inspection
│   │   └── subtitle_utils.py  # Subtitle parsing/styling
│   ├── config_adapter.py      # Backward-compatible config loader
│   ├── config_validator.py    # Configuration validation utilities
│   ├── cta_detector.py        # Call-to-action detection in scripts
│   ├── font_color_manager.py  # Font and color management
│   ├── pipeline_graph.py      # Dependency-aware execution framework
│   ├── result_types.py        # Pipeline result type definitions
│   ├── stock_media.py         # Stock media fetching (Pexels)
│   ├── stt_functions.py       # Speech-to-text (Whisper, Google Cloud STT)
│   ├── subtitle_positioning.py # Subtitle position calculations
│   ├── subtitle_utils.py      # Subtitle generation utilities
│   ├── subtitle_validation.py # Subtitle validation logic
│   ├── tts.py                 # Text-to-speech with provider fallbacks
│   ├── subtitle_timing_smoother.py # Post-processes Whisper word timings
│   ├── unified_subtitle_generator.py  # FFmpeg ASS/SRT subtitle generation
│   └── pycaps_engine/         # Animated-caption engine (bundled default)
│       ├── renderer.py        # Pycaps render + content-aware layout
│       ├── gemini_llm.py      # Gemini adapter for AI word tagging
│       └── transcript_adapter.py # Whisper transcript to pycaps format
│
├── ai/                        # AI & LLM integration
│   ├── llm_client.py          # Shared LLM dispatch (Gemini, OpenRouter)
│   ├── script_generator.py    # Script generation with provider fallback
│   ├── description_generator.py # Social media description generation
│   ├── platform_metadata/     # Platform-specific metadata (v0.17.0+)
│   │   ├── base.py            # Base metadata generator interface
│   │   ├── youtube.py         # YouTube metadata generation
│   │   ├── tiktok.py          # TikTok caption generation
│   │   ├── instagram.py       # Instagram caption generation
│   │   ├── models.py          # Metadata data models
│   │   ├── utilities.py       # Shared utilities
│   │   └── text_formatter.py  # Platform text formatting
│   └── prompts/              # LLM prompt templates
│
├── scraper/                   # Multi-platform data collection architecture
│   ├── base/                 # Platform-agnostic foundation (5 modules)
│   │   ├── models.py         # Base product data models & registry
│   │   ├── config.py         # Multi-platform configuration manager
│   │   ├── utils.py          # Shared utility functions
│   │   ├── downloader.py     # Base async download logic
│   │   └── browser_utils.py  # Shared browser utilities
│   ├── amazon/               # Amazon implementation (12 modules)
│   │   ├── scraper.py        # Main orchestrator (extends BaseScraper)
│   │   ├── batch_controller.py # Batch scraping orchestration
│   │   ├── browser_functions.py # Browser automation logic
│   │   ├── botasaurus_output.py # Botasaurus output handling
│   │   ├── config.py         # Amazon configuration management
│   │   ├── downloader.py     # Async media downloads with semaphore rate limiting
│   │   ├── media_extractor.py   # Image/video extraction
│   │   ├── media_validator.py   # Media file validation
│   │   ├── models.py         # Amazon-specific models
│   │   ├── search_builder.py # Search URL construction
│   │   └── utils.py          # Amazon utility functions
│   ├── config_models.py      # Pydantic models for type-safe config (v0.14.0+)
│   ├── config_adapter.py     # Backward-compatible config loader
│   └── __init__.py           # ScraperFactory & platform registry
│
├── audio/                     # Background-music provider platform
│   ├── base.py               # BaseAudioProvider ABC
│   ├── registry.py           # AudioProviderRegistry (decorator-based)
│   ├── manager.py            # AudioManager: runs the provider chain
│   ├── jamendo_provider.py   # Jamendo download (primary)
│   ├── freesound_provider.py # Freesound download (fallback)
│   └── freesound_client.py   # Freesound API client (wrapped by the provider)
│
├── utils/                     # Performance optimization & utilities
│   ├── performance.py         # Metrics collection & monitoring
│   ├── async_io.py           # Async subprocess management
│   ├── connection_pool.py    # HTTP connection pooling
│   ├── memory_mapped_io.py   # Memory-mapped file operations
│   ├── caching.py            # Multi-level caching system
│   ├── background_processing.py # Background task management
│   ├── script_sanitizer.py   # Text processing utilities
│   └── url_shortener/        # URL shortening abstraction layer
│       ├── base.py           # Base interfaces and models
│       ├── picsee.py         # PicSee API implementation
│       ├── registry.py       # Provider registry and factory
│       └── __init__.py       # Public API exports
│
├── publisher/                 # Social media publishing (v0.18.0+)
│   ├── base.py               # Base publisher interface
│   ├── batch.py              # Batch publishing orchestration
│   ├── cleanup.py            # Post-publish cleanup utilities
│   ├── config.py             # Publisher configuration
│   ├── constants.py          # Shared constants (limits, defaults)
│   ├── metadata.py           # Metadata resolution logic
│   ├── models.py             # Publisher data models
│   ├── publish_modes.py      # Unified/platform-specific publish helper
│   ├── registry.py           # Platform registry and factory
│   ├── schedule.py           # Scheduling utilities
│   ├── schedule_validator.py # Schedule validation
│   ├── tracking.py           # Publish status tracking (atomic writes)
│   ├── product_registry.py   # Published products registry (JSON + CSV)
│   ├── webhooks.py           # Zernio webhook event handling
│   ├── late/                 # Zernio integration (formerly Late)
│   │   ├── client.py         # Zernio API client (late-sdk)
│   │   └── cli.py            # Zernio publisher CLI
│   └── link_in_bio/          # Link-in-bio integration
│       ├── base.py           # Provider interface
│       ├── lnkbio.py         # Lnk.Bio provider
│       └── manager.py        # Orchestration and fallback logic
│
└── pipeline/                  # Batch processing orchestration
    ├── config.py             # Pipeline configuration
    └── global_batch.py       # Unified scrape + produce pipeline
```

</details>

## Component Details

### 1. Pipeline Engine (`src/video/producer/`)

**Purpose**: Orchestrates the entire video production workflow.

**Key Responsibilities:**
- Manages eight-step pipeline execution
- Handles pipeline context and state
- Creates directory structures
- Implements configurable delays between products
- Provides step-specific execution for debugging
- Complete cleanup of producer-generated files with --clean flag

**Architecture Pattern:**
- **Async/Await**: All operations are async for better concurrency
- **Context Management**: Pipeline context preserves state across steps
- **Error Handling**: Comprehensive error handling with graceful degradation
- **Logging**: Dual logging (console + file) with structured output

### 2. Dependency Management (`src/video/pipeline_graph.py`)

**Purpose**: Manages step dependencies and enables parallel execution.

**Key Features:**
- **Topological Sorting**: Ensures correct execution order
- **Dependency Resolution**: Automatically determines which steps can run in parallel
- **Resource Management**: Manages concurrent execution limits
- **Performance Optimization**: Enables 26% faster pipeline execution

**Technical Implementation:**
```python
# Dependency Graph Definition, as declared by
# src/video/producer/orchestration.py::step_dependencies.
# The first two edges depend on the profile: a profile that draws no visual
# from the scraped product writes the script first, so its stock search can
# use terms taken from the narration.
dependencies = {
    'gather_visuals': [],                        # ['generate_script'] when script-first
    'generate_script': ['gather_visuals'],       # [] when script-first
    'generate_description': ['generate_script'], # + 'gather_visuals' when script-first
    'create_voiceover': ['generate_script'],     # + 'gather_visuals' when script-first
    'generate_subtitles': ['create_voiceover'],  # Can run in parallel
    'download_music': ['create_voiceover'],      # Can run in parallel
    'assemble_video': ['generate_subtitles', 'download_music', 'gather_visuals'],
    'burn_pycaps_subtitles': ['assemble_video'], # No-op unless engine is pycaps
}
```

### 3. Video Assembly (`src/video/assembler/`)

**Purpose**: Combines all elements into final MP4 video using FFmpeg with intelligent video assembly strategies.

**Modular Architecture** (refactored from 3,311-line monolith):
- **`core.py`** - VideoAssembler orchestrator (~690 lines)
- **`visual_builder.py`** - Visual filter chains (~590 lines)
- **`subtitle_builder.py`** - Subtitle positioning (~850 lines)
- **`audio_builder.py`** - Audio filter chains (~130 lines)
- **`video_strategies.py`** - Video mode strategies (~665 lines)
- **`media_inspector.py`** - Media file inspection (~170 lines)
- **`subtitle_utils.py`** - Subtitle parsing/styling (~280 lines)

**Core Functionality:**
- **Media Analysis**: Async extraction of dimensions and durations
- **Video Assembly Modes**: Four configurable strategies for video-first content
- **Aspect Ratio Handling**: Letterbox, crop-to-fit, and smart-scale modes with actual geometry tracking
- **Audio Sources**: Voiceover and background music only; source video audio is dropped
- **Filter Graph Construction**: Dynamic FFmpeg filter generation via specialized builders
- **Subtitle Rendering**: Content-aware positioning with letterbox geometry support
- **Audio Mixing**: Fixed-level multi-track mixing with per-track volume
- **Verification**: Post-assembly quality checks

#### Video Assembly Modes

ContentEngineAI supports **4 video assembly modes** optimized for different content styles:

**1. Sequential Mode** (`video_assembly_mode: "sequential"`)
- Concatenates all product videos end-to-end with crossfade transitions
- Loops videos if total duration < voiceover length
- Adds images to fill remaining time if needed
- **Best for**: Showcasing multiple product angles/demos

**2. Single Best Mode** (`video_assembly_mode: "single_best"`)
- Selects the longest video and loops it seamlessly
- Creates smooth infinite loop effect with crossfade at loop point
- **Best for**: Single-angle product demonstrations with clean looping

**3. Mixed Media Mode** (`video_assembly_mode: "mixed_media"`)
- Interleaves videos and images throughout the timeline
- Distributes videos evenly across duration
- Fills gaps between videos with images
- **Best for**: Dynamic visual variety mixing motion and static content

**4. Video-First Fallback Mode** (`video_assembly_mode: "video_first_fallback"`)
- Plays all product videos first (priority content)
- Fills remaining duration with images
- **Best for**: Ensuring videos are always shown while using images as filler

#### Aspect Ratio Handling

**Letterbox Mode** (`video_aspect_mode: "letterbox"`)
```
Original: 16:9 landscape video
Target:   9:16 vertical frame
Result:   Video centered with black bars (preserves aspect ratio)
```

**Crop-to-Fit Mode** (`video_aspect_mode: "crop-to-fit"`)
```
Original: 16:9 landscape video
Target:   9:16 vertical frame
Result:   Video scaled to fill frame, edges cropped (centers crop region)
```

**Smart-Scale Mode** (`video_aspect_mode: "smart-scale"`)
```
Automatically chooses between letterbox and crop based on aspect ratio difference:
- ≤10% difference → Use crop-to-fit (minimal distortion)
- >10% difference → Use letterbox (preserve content)
```

#### Audio Handling

Product video audio is not carried into the render. The narration is the
message, and the source audio on a marketing or stock clip is a licensed music
bed or a second voice, which is a platform audio-match risk with no audible
benefit under a voiceover.

**FFmpeg Integration:**
- **Complex Filters**: Dynamic filter graph construction
- **Crossfade Transitions**: Smooth visual transitions (configurable duration)
- **Aspect Ratio Transformations**: scale, pad, crop filters with smart positioning
- **Format Normalization**: Auto-conversion to H.264/30fps/yuv420p for compatibility
- **Subtitle Styling**: Font, color, positioning customization
- **Multi-Track Audio**: amix filter with volume normalization

### 4. AI Integration (`src/ai/`)

**Purpose**: Generates promotional scripts and descriptions using LLM providers.

**Provider Architecture:**
- **Primary Provider**: Gemini via `google-genai` SDK
- **Fallback Provider**: OpenRouter via aiohttp (OpenAI chat/completions format)
- **Dispatch Layer**: `llm_client.py` routes calls based on `settings.provider`
- **Fallback Chain**: Primary exhausts all models, then `fallback_provider` settings activate with separate API key, models, and discovery
- **Retry Logic**: Exponential backoff with configurable limits per provider

**Features:**
- **Provider Fallback**: Configurable via `llm_settings.fallback_provider` in YAML
- **Free Model Discovery**: OpenRouter auto-selects free models, filtered by blocklist and context length
- **Script Templates**: 15 prompt styles with deterministic per-product selection
- **Script Sanitization**: Removes emojis, hashtags, formatting issues
- **Configurable Validation**: min_chars/min_words thresholds in `script_validation`

### 5. Media Processing

#### Stock Media (`src/video/stock_media.py`)
- **API Integration**: Pexels API with rate limiting
- **Query Optimization**: Keyword-based search with caching
- **Attribution Tracking**: Automatic attribution file generation
- **Concurrent Downloads**: Semaphore-based concurrency control

#### Background Music (`src/audio/`)
- **Provider Platform**: `BaseAudioProvider` ABC + `AudioProviderRegistry` + `AudioManager`, the same chain pattern used by the publisher module
- **Provider Chain**: Jamendo (primary) then Freesound, with local files as the last resort; first successful download wins
- **Configuration**: `audio_providers` list in `config/video_production.yaml`, tried in order
- **Jamendo**: `client_id` auth, `fuzzytags` search for genre/mood, downloads over HTTP/2 via curl (its CDN blocks HTTP/1.1)
- **Freesound**: `FreesoundProvider` wraps the existing `FreesoundClient`; OAuth2 for full quality, API key for previews

#### TTS Engine (`src/video/tts.py`)
- **Primary Provider**: Gemini TTS via the `google.cloud.texttospeech` SDK; falls back to Google Cloud TTS on failure. Coqui TTS is supported but not installed: the code and config stay in place, and the provider self-disables when the package is absent
- **Voice Selection**: Configurable voice profiles (provider, voice criteria, style)
- **Async Generation**: Non-blocking TTS with timeout handling
- **Caching**: Client and model caching for performance

#### Subtitle Generation (`src/video/pycaps_engine/`, `src/video/unified_subtitle_generator.py`, `src/video/stt_functions.py`)
- **Two Engines**: the bundled default is the pycaps engine (animated captions rendered per word); the FFmpeg ASS/SRT burn is the fallback. Selected via `subtitle_settings.subtitle_engine`
- **Pycaps Engine**: runs as a post-assembly burn step (`src/video/pycaps_engine/renderer.py`), consumes the raw Whisper transcript, positions captions with a content-aware layout, and supports optional Gemini AI word tagging
- **STT**: Whisper (primary) with word-level timing extraction; a timing smoother post-processes the word timestamps before either engine
- **Content-Aware Positioning**: Dynamic subtitle placement that analyzes visual content to avoid overlaps
- **Configurable Video/Subtitle Layout**: Per-profile control of video positioning and subtitle gaps
  - `video_top_position_percent`: Vertical video start position (default: 10% from top)
  - `video_content_height_percent`: Video height as frame percentage (default: 75%)
  - `subtitle_settings.margin`: Gap between content and subtitles (bundled config: 4%)
- **Segmentation Logic**: Smart text splitting with natural boundaries based on actual speech timing

### 6. URL Shortening System (`src/utils/url_shortener/`)

**Purpose**: Provider-agnostic URL shortening for affiliate links with fallback support.

**Architecture Pattern:**
- **Base Interface**: Abstract base class for all providers
- **Provider Registry**: Factory pattern for provider instantiation
- **Async-First Design**: Non-blocking HTTP requests
- **Fallback Chain**: Automatic provider switching on failures

**Features:**
- **Multi-Provider Support**: PicSee (implemented), Bitly/TinyURL (planned)
- **Retry Logic**: Exponential backoff with configurable attempts
- **Response Caching**: TTL-based caching to avoid redundant API calls
- **Custom Domains**: Branded short domains (BSD) support
- **Bulk Operations**: Batch shortening for efficiency (PicSee)
- **Integration Points**: Scraper (automatic), video descriptions (optional)

**Implementation Details:**
```python
# Provider interface
class BaseURLShortener(ABC):
    @abstractmethod
    async def shorten(self, url: str, custom_alias: str | None = None) -> ShortenedURL

    @abstractmethod
    async def shorten_bulk(self, urls: list[str]) -> list[ShortenedURL]
```

**Data Flow:**
```
Affiliate Link → URL Shortener → [Primary Provider]
                                     ↓ (on failure)
                                 [Fallback Provider]
                                     ↓ (on failure)
                                 [Original URL]
```

### 7. Video Processing and Extraction

#### Overview

ContentEngineAI's Amazon scraper includes comprehensive video detection, extraction, validation, and metadata capture capabilities. The system reliably identifies product-specific videos from Amazon pages, downloads them with robust error handling, and extracts detailed metadata for use in the video production pipeline.

#### Video Extraction Flow

```
Product Page → Multi-Method Extraction → URL Validation → Download → FFprobe Metadata → Storage
     │                    │                      │             │              │              │
     │         ┌──────────┴──────────┐          │             │              │              │
     │         │                     │          │             │              │              │
     │    Script Data       Video Elements   HEAD Request  Streaming     Duration      videos/ dir
     │    ASIN Matching     VDP Navigation   (1KB test)    Download    Resolution     Relative paths
     │    Quality Filter                                   300s timeout  Codec info    in data.json
```

#### Multi-Method Video Extraction

The scraper employs a three-tier extraction strategy to maximize video discovery:

**Method 1: Script Data Extraction**
- Parses `window.P.register()` JavaScript blocks for video URLs
- Identifies product videos via ASIN matching in JSON metadata
- Filters video URLs from structured product data
- Prioritizes highest quality versions available

**Method 2: Video Element Detection**
- Scans DOM for `<video>` elements and sources
- Extracts MP4 URLs from video player configurations
- Validates URLs against Amazon CDN domains

**Method 3: VDP (Video Detail Page) Navigation**
- Follows VDP links for high-resolution video streams
- Extracts video data from dedicated video pages
- Captures multi-angle and detailed product views

#### Video Metadata Extraction (`src/scraper/amazon/media_validator.py`)

**Purpose**: Extract comprehensive video metadata using FFprobe for pipeline decision-making.

**Implementation**:
```python
def extract_video_metadata(file_path: Path) -> dict[str, Any] | None:
    """
    Extract video metadata using FFprobe.

    Returns:
        {
            'duration': float,        # Video duration in seconds
            'width': int,            # Video width in pixels
            'height': int,           # Video height in pixels
            'codec': str,            # Video codec (h264, vp9, etc.)
            'format': str,           # Container format (mp4, webm, etc.)
            'bitrate': int,          # Bitrate in bits per second
            'has_audio': bool        # Audio stream presence
        }
    """
```

**Features**:
- **FFprobe Integration**: Uses FFprobe for comprehensive metadata extraction
- **Graceful Degradation**: Returns `None` if FFprobe unavailable or video corrupted
- **Structured Output**: Provides standardized metadata dict for all videos
- **Error Handling**: Logs warnings but doesn't fail validation on metadata errors

#### Video Validation and Quality Filtering

**URL Validation** (`src/scraper/amazon/media_extractor.py:1261-1286`):
- HEAD request validation (1KB range) before full download
- Amazon CDN domain verification for security
- Accessibility checks to filter broken links
- Random delay (0.5-1.5s) to mimic human behavior

**Quality Thresholds**:
- **Minimum Resolution**: 640px (width or height)
- **Minimum Duration**: 1.0 seconds
- **File Format**: MP4 containers only
- **Domain Whitelist**: Amazon CDN domains only

**Enhanced Validation** (`verify_video_file()`):
```python
def verify_video_file(file_path: Path) -> tuple[bool, str, dict[str, Any]]:
    """
    Validate video file and extract metadata.

    Returns:
        (is_valid, reason, metadata_dict)
    """
```

Returns validation status, reason, and metadata in single call for efficient pipeline integration.

#### Robust Download Handling

**Extended Timeouts**:
- **Images**: 30 seconds timeout
- **Videos**: 300 seconds timeout (configurable)
- **Retry Logic**: 2 retry attempts with exponential backoff

**Streaming Downloads**:
- Chunk-based streaming (8KB chunks) for memory efficiency
- Progress tracking for large files
- Graceful handling of network interruptions

**Error Recovery**:
- Automatic retry with exponential backoff
- Continues processing other videos on single failure
- Product processing succeeds even if all videos fail
- Detailed error logging with actionable messages

#### Video Storage Organization

**Directory Structure**:
```
outputs/{ASIN}/
├── data.json                 # Product data with video paths
├── images/                   # Product images
│   ├── image_0.jpg
│   └── image_1.jpg
└── videos/                   # Product videos
    ├── video_0.mp4          # First extracted video
    ├── video_1.mp4          # Second extracted video
    └── video_N.mp4          # Additional videos
```

**Naming Convention**:
- Sequential indexing: `video_{index}.mp4`
- Relative paths stored in `data.json`
- Automatic directory creation if missing

**Product Data Integration**:
```json
{
  "asin": "B0BTYCRJSS",
  "videos": [
    "https://m.media-amazon.com/video1.mp4",
    "https://m.media-amazon.com/video2.mp4"
  ],
  "downloaded_videos": [
    "videos/video_0.mp4",
    "videos/video_1.mp4"
  ]
}
```

#### Configuration Options

**Video Processing** (`config/scraper.yaml:video_config`):
```yaml
video_config:
  min_dimension: 640              # Minimum width/height (pixels)
  min_duration: 1.0               # Minimum duration (seconds)
  max_videos_per_product: 10      # Download limit per product
  mute_video_tabs: true           # Prevent audio during extraction
  enable_metadata_extraction: true # FFprobe metadata extraction
```

**Download Settings** (`config/scraper.yaml:download_config`):
```yaml
download_config:
  download_timeout: 30            # Image timeout (seconds)
  video_download_timeout: 300     # Video timeout (seconds)
  retry_video_downloads: 2        # Retry attempts for videos
  download_chunk_size: 8192       # Streaming chunk size (bytes)
  validation_range_bytes: "0-1023" # HEAD request range
```

**Rate Limiting** (`config/scraper.yaml:rate_limiting`):
```yaml
rate_limiting:
  video_validation_delay: [0.5, 1.5]  # Random delay range (seconds)
```

#### Troubleshooting Video Processing

**Problem**: Videos not detected on product page

**Solutions**:
1. Enable debug mode to see extraction attempts:
   ```bash
   poetry run python -m src.scraper.amazon.scraper --keywords "ASIN" --debug
   ```
2. Check if product page actually has videos (not all products have video content)
3. Review logs for JavaScript parsing errors or ASIN matching issues
4. Verify network connectivity to Amazon CDN

**Problem**: Video downloads timing out

**Solutions**:
1. Increase timeout in `config/scraper.yaml`:
   ```yaml
   download_config:
     video_download_timeout: 600  # Increase to 10 minutes
   ```
2. Check network speed and stability
3. Verify retry settings are enabled:
   ```yaml
   download_config:
     retry_video_downloads: 2
   ```

**Problem**: FFprobe metadata extraction failing

**Solutions**:
1. Verify FFmpeg/FFprobe installation:
   ```bash
   ffprobe -version
   ```
2. Check video file integrity:
   ```bash
   ffprobe -v error outputs/{ASIN}/videos/video_0.mp4
   ```
3. Disable metadata extraction if FFprobe unavailable:
   ```yaml
   video_config:
     enable_metadata_extraction: false
   ```

**Problem**: Low-quality videos being downloaded

**Solutions**:
1. Increase quality thresholds:
   ```yaml
   video_config:
     min_dimension: 1280  # Require 720p minimum
     min_duration: 5.0    # Require longer videos
   ```
2. Review validation logs to see why videos passed filtering
3. Check if higher quality versions available on product page

**Problem**: Video processing errors causing product failures

**Solutions**:
1. Check error logs for specific failure reasons
2. Verify graceful degradation is working (product should succeed without videos)
3. Review retry logic configuration:
   ```yaml
   retry_config:
     default_max_retries: 3
     base_delay: 1.0
     backoff_factor: 2.0
   ```

#### Performance Characteristics

**Video Extraction Performance**:
- URL extraction: <5 seconds per product
- Concurrent downloads: Max 3 simultaneous videos
- Average video download: 30-90 seconds (depends on file size and network)
- Metadata extraction: <2 seconds per video (FFprobe)

**Resource Usage**:
- Memory: Streaming downloads prevent memory spikes
- Network: Chunk-based transfers minimize bandwidth waste
- CPU: Minimal (FFprobe is lightweight)

### 8. Amazon Scraping Features

#### Search Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--min-price` | Minimum price filter | `--min-price 10.99` |
| `--max-price` | Maximum price filter | `--max-price 99.99` |
| `--min-rating` | Minimum rating (1-5 stars) | `--min-rating 4` |
| `--prime-only` | Prime eligible items only | `--prime-only` |
| `--free-shipping` | Free shipping items only | `--free-shipping` |
| `--brands` | Filter by brand names | `--brands Apple Samsung Sony` |
| `--sort` | Sort order | `--sort price-asc-rank` |

#### Sort Options

- `relevanceblender` (default) - Amazon's relevance algorithm
- `price-asc-rank` - Price low to high
- `price-desc-rank` - Price high to low
- `review-rank` - Best reviews first
- `date-desc-rank` - Newest first
- `featured-rank` - Featured items first

### 9. Multi-Platform Web Scraping Architecture

#### **Platform Registry System (`src/scraper/__init__.py`)**

**Purpose**: Factory pattern for unified platform access and extensibility.

**Key Components:**
- **ScraperFactory**: Creates platform-specific scrapers via factory pattern
- **ScraperRegistry**: Auto-discovery and registration of platform implementations
- **MultiPlatformScraper**: Unified interface for all e-commerce platforms

```python
# Unified platform access
scraper = ScraperFactory.create_scraper('amazon')
products = await scraper.scrape_products(['wireless headphones'])

# Platform auto-discovery
available_platforms = ScraperRegistry.get_available_platforms()
# Returns: [Platform.AMAZON, Platform.EBAY, Platform.WALMART, ...]
```

**Note**: only the Amazon scraper is implemented today. The eBay/Walmart entries above are illustrative placeholders showing how the registry extends to new platforms, not shipping code.

#### **Base Scraper Interface (`src/scraper/base/models.py`)**

**Purpose**: Platform-agnostic foundation for all e-commerce scrapers.

**Abstract Interface:**
```python
class BaseScraper(ABC):
    @abstractmethod
    async def scrape_products(self, keywords: List[str]) -> List[BaseProductData]:
        """Scrape products based on search keywords"""
        
    @abstractmethod
    def validate_product_id(self, product_id: str) -> bool:
        """Validate platform-specific product identifiers"""
```

#### **Amazon Implementation (`src/scraper/amazon/scraper.py`)**

**Purpose**: Amazon-specific scraper extending the base interface.

**Technical Implementation:**
- **BaseScraper Extension**: Implements multi-platform interface
- **Playwright Integration**: Headless browser automation with Botasaurus
- **Stealth Techniques**: Anti-detection measures and browser fingerprinting
- **11-Module Architecture**: Modular design for maintainability
- **Media Extraction**: High-resolution images and videos with validation
- **Advanced Search**: Complex filtering with price, rating, brand, and shipping options

## Performance Optimization Architecture

ContentEngineAI implements five optimization categories: pipeline parallelization, I/O optimization, multi-level caching, resource management, and background processing. See the "Performance Optimization" section in `docs/development.md` for the full breakdown of each category and its implementation.

## Performance Monitoring

### Metrics Collection (`src/utils/performance.py`)

**Real-Time Tracking:**
- Step-by-step timing and resource usage
- Memory usage and CPU utilization monitoring
- Historical data persistence (JSONL format)

**Monitoring Components:**
- `PerformanceMonitor`: Real-time metrics collection
- `PerformanceHistoryManager`: Historical data management
- Cross-session analysis and trend detection

**Reporting Tools:**
```bash
make perf-report                    # Quick summary
poetry run python tools/performance_report.py --report-type detailed
poetry run python tools/performance_report.py --report-type trends
```

## Configuration Architecture

### Unified Configuration System

ContentEngineAI uses a **modular configuration architecture** that replaced the original monolithic system while maintaining 100% backward compatibility.

<details>
<summary><strong>System Overview</strong></summary>

**Design Principles:**
- **Modular YAML Files**: 9 specialized files
- **Triple Precedence**: CLI overrides > Environment variables > YAML defaults
- **Zero Breaking Changes**: Existing function signatures preserved through adapters
- **Production Ready**: Environment variable support for all settings

**Configuration Files:**
| File | Purpose | Key Sections |
|------|---------|--------------|
| `config/core.yaml` | Global settings | Output paths, debug, timeouts |
| `config/video_production.yaml` | Video pipeline | Resolution, effects, profiles |
| `config/ai_services.yaml` | AI providers | TTS, LLM, description generation |
| `config/subtitles.yaml` | Subtitle system | Positioning, styles, effects |
| `config/performance.yaml` | Resource limits | Memory, concurrency, optimization |
| `config/scraper.yaml` | Web scraping | Browser, timing, validation, async downloads |
| `config/pipeline.yaml` | Batch processing | Global batch settings, fail-fast mode |
| `config/publisher.yaml` | Social publishing | Zernio integration, platform settings |
| `config/url_shortener.yaml` | URL shortening | Provider settings, affiliate links |

**Type-Safe Configuration (v0.14.0+):**
- **Video Pipeline**: Pydantic models in `src/video/config/` (core, audio, visual, subtitle models)
- **Scraper System**: Pydantic models in `src/scraper/config_models.py` (19 models, 283 lines)
- **Validation**: Field constraints ensure type safety and valid ranges at startup
- **Backward Compatible**: Dict-based config adapter maintains legacy support

**Performance Improvements:**
- **20% faster** configuration loading
- **Reduced memory footprint** through lazy loading
- **Better caching** of parsed configuration values

</details>

### Backward Compatibility Layer

The original configuration system is preserved through `config_adapter.py`:

**Key Configuration Areas:**
- **Timeout Management**: All pipeline timeouts configurable
- **Provider Settings**: API configurations and fallback orders
- **Media Processing**: Video/audio quality and processing parameters
- **Performance Tuning**: Concurrency limits and optimization settings

### Directory Structure Management

**Features:**
- **Flexible Patterns**: Configurable directory structures
- **Dynamic Path Generation**: Product ID and timestamp-based paths
- **Cleanup Integration**: Automated cleanup of unexpected files
- **Pattern Validation**: Expected vs unexpected file location tracking

## Data Flow Architecture

### Pipeline Data Context

```python
@dataclass
class PipelineContext:
    product: ProductData
    config: VideoConfig
    profile: VideoProfile
    temp_dir: Path
    visuals_info: VisualsInfo
    script: Optional[str] = None
    voiceover_path: Optional[Path] = None
    subtitles_path: Optional[Path] = None
    music_path: Optional[Path] = None
    final_video_path: Optional[Path] = None
```

**State Management:**
- Immutable data structures where possible
- Context preservation across async operations
- Structured error propagation
- Debug state serialization

### Media Pipeline Flow

```
Product Data → Visuals Gathering → Script Generation → TTS Generation
                                                           ↓
Final Video ← Video Assembly ← Music Download + Subtitle Generation
```

**Data Transformations:**
1. **Raw HTML** → **Structured ProductData** (Pydantic models)
2. **Product Features** → **Promotional Script** (LLM processing)
3. **Script Text** → **Audio + Timings** (TTS with word-level timestamps)
4. **Audio + Timings** → **SRT Subtitles** (STT with segmentation)
5. **All Components** → **Final MP4** (FFmpeg assembly)

## Error Handling Architecture

### Multi-Level Error Handling

**Level 1: Provider Fallbacks**
- LLM: Gemini (primary) -> OpenRouter (fallback)
- TTS: Gemini TTS (primary) -> Google Cloud TTS (fallback)
- STT: Whisper -> script-based fallback
- Music: Jamendo (primary) -> Freesound -> local files

**Level 2: Retry Logic**
- Exponential backoff for transient failures
- Configurable retry limits and timeouts
- Circuit breaker patterns for persistent failures

**Level 3: Graceful Degradation**
- Continue pipeline with reduced functionality
- Skip optional components (music, subtitles)
- Generate attribution files for partial success

**Level 4: Comprehensive Logging**
- Structured error messages with context
- Debug mode with intermediate file preservation
- Performance impact tracking for failures

## Extensibility Architecture

### Plugin Architecture

**Provider Interface Pattern:**
All external service integrations follow a common interface:

```python
class BaseProvider(ABC):
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize provider with configuration"""
        
    @abstractmethod  
    async def process(self, input_data: Any) -> Any:
        """Process input and return result"""
        
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources"""
```

### Adding New Components

**New Media Sources:**
1. Implement `BaseMediaProvider` interface
2. Add configuration section to `src/video/config/` models
3. Register provider in media fetching pipeline
4. Add attribution tracking support

**New AI Providers:**
1. Implement provider interface (TTS, STT, LLM)
2. Add to provider fallback chain
3. Update configuration validation
4. Add performance monitoring hooks

**New Pipeline Steps:**
1. Define step function with async signature
2. Add to dependency graph in `pipeline_graph.py`
3. Update configuration and validation
4. Add performance monitoring and error handling

## Key Technologies

- **🐍 Python 3.12**: Modern async/await patterns
- **🎥 FFmpeg**: Professional video processing
- **🤖 AI Services**: Gemini (LLM + TTS, primary), OpenRouter and Google Cloud (fallbacks), OpenAI Whisper (STT)
- **🌐 Web Scraping**: Playwright with stealth techniques (Amazon only today)
- **📱 Media APIs**: Jamendo and Freesound (music), Pexels (stock images/videos)
- **⚙️ Configuration**: YAML + Pydantic validation
- **🧪 Testing**: Pytest with async support

## Acknowledgments

- **OpenAI Whisper** for speech-to-text capabilities
- **Google Gemini** for script generation and TTS, with Google Cloud as fallback
- **Pexels** for stock media content
- **Jamendo** and **Freesound** for background music
- **FFmpeg** for video processing excellence

This architecture enables ContentEngineAI to be highly extensible while maintaining performance, reliability, and maintainability across all components.