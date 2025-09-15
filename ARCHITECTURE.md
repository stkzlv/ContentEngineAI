# ContentEngineAI Architecture

This document provides a comprehensive overview of the ContentEngineAI architecture, including system design, component interactions, and technical implementation details.

## System Overview

ContentEngineAI is a modular, async-first pipeline system designed for automated video production. The architecture follows a six-step workflow with parallel execution capabilities and comprehensive error handling.

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   AI Services   │    │  Media Sources  │
│                 │    │                 │    │                 │
│ • Amazon Pages  │    │ • OpenRouter    │    │ • Pexels API    │
│ • Product Data  │    │ • Google Cloud  │    │ • Freesound     │
│ • Images/Videos │    │ • Local Models  │    │ • Local Files   │
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
Step 2: Generate Script (LLM via OpenRouter)
         │
Step 3: Create Voiceover (TTS: Google Cloud/Coqui)
         │
         ├── Step 4a: Generate Subtitles (Whisper/Google STT)
         └── Step 4b: Download Music (Freesound)
                 │
         Step 5: Assemble Video (FFmpeg)
```

**Key Features:**
- **Parallel Execution**: Steps 4a and 4b run concurrently after step 3
- **Dependency Management**: Automatic handling of step dependencies
- **Resume Capability**: Individual step execution for debugging
- **Performance Monitoring**: Built-in metrics collection

### Core Packages Structure

```
src/
├── video/                      # Central orchestration & video processing  
│   ├── producer.py            # Main pipeline orchestrator
│   ├── assembler.py           # FFmpeg-based video assembly
│   ├── unified_subtitle_generator.py  # Unified subtitle generation system
│   ├── stt_functions.py       # Speech-to-text (Whisper, Google Cloud STT)
│   ├── subtitle_utils.py      # Subtitle generation utilities and coordination
│   ├── tts.py                 # Text-to-speech with provider fallbacks
│   ├── stock_media.py         # Stock media fetching (Pexels)
│   ├── video_config.py        # Pydantic configuration models
│   └── pipeline_graph.py      # Dependency-aware execution framework
│
├── ai/                        # AI & LLM integration
│   ├── script_generator.py    # Script generation via OpenRouter
│   └── prompts/              # LLM prompt templates
│
├── scraper/                   # Multi-platform data collection architecture
│   ├── base/                 # Platform-agnostic foundation
│   │   ├── models.py         # Base product data models & registry
│   │   ├── config.py         # Multi-platform configuration manager
│   │   ├── utils.py          # Shared utility functions
│   │   ├── downloader.py     # Base download logic
│   │   └── browser_utils.py  # Shared browser utilities
│   ├── amazon/               # Amazon implementation (14 modules)
│   │   ├── scraper.py        # Main orchestrator (extends BaseScraper)
│   │   ├── browser_functions.py # Browser automation logic
│   │   ├── media_extractor.py   # Image/video extraction
│   │   ├── downloader.py     # Media download functionality
│   │   ├── models.py         # Amazon-specific models
│   │   ├── config.py         # Amazon configuration management
│   │   ├── utils.py          # Amazon utility functions
│   │   └── search_builder.py # Search URL construction
│   └── __init__.py           # ScraperFactory & platform registry
│
├── audio/                     # Audio processing components
│   └── freesound_client.py   # Music download from Freesound
│
└── utils/                     # Performance optimization & utilities
    ├── performance.py         # Metrics collection & monitoring
    ├── async_io.py           # Async subprocess management
    ├── connection_pool.py    # HTTP connection pooling
    ├── memory_mapped_io.py   # Memory-mapped file operations
    ├── caching.py            # Multi-level caching system
    ├── background_processing.py # Background task management
    └── script_sanitizer.py   # Text processing utilities
```

## Component Details

### 1. Pipeline Engine (`src/video/producer.py`)

**Purpose**: Orchestrates the entire video production workflow.

**Key Responsibilities:**
- Manages six-step pipeline execution
- Handles pipeline context and state
- Creates directory structures
- Implements configurable delays between products
- Provides step-specific execution for debugging

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
# Dependency Graph Definition
dependencies = {
    'gather_visuals': [],
    'generate_script': ['gather_visuals'],
    'create_voiceover': ['generate_script'],
    'generate_subtitles': ['create_voiceover'],  # Can run in parallel
    'download_music': ['create_voiceover'],     # Can run in parallel
    'assemble_video': ['generate_subtitles', 'download_music']
}
```

### 3. Video Assembly (`src/video/assembler.py`)

**Purpose**: Combines all elements into final MP4 video using FFmpeg.

**Core Functionality:**
- **Media Analysis**: Async extraction of dimensions and durations
- **Filter Graph Construction**: Dynamic FFmpeg filter generation
- **Subtitle Rendering**: Styled subtitle embedding with customization
- **Audio Mixing**: Multi-track audio with volume control
- **Verification**: Post-assembly quality checks

**FFmpeg Integration:**
- **Complex Filters**: Dynamic filter graph construction
- **Crossfade Transitions**: Smooth visual transitions
- **Aspect Ratio Preservation**: Smart scaling and positioning
- **Subtitle Styling**: Font, color, positioning customization

### 4. AI Integration (`src/ai/script_generator.py`)

**Purpose**: Generates promotional scripts using LLM providers.

**Provider Architecture:**
- **Primary Provider**: OpenRouter API with multiple model support
- **Fallback Mechanism**: Automatic model switching on failures
- **Retry Logic**: Exponential backoff with configurable limits
- **Prompt Engineering**: Template-based prompt construction

**Features:**
- **Model Auto-Selection**: Prioritized model list with fallbacks
- **Script Sanitization**: Removes emojis, hashtags, formatting issues
- **Token Management**: Efficient API usage with caching
- **Error Handling**: Graceful degradation with detailed logging

### 5. Media Processing

#### Stock Media (`src/video/stock_media.py`)
- **API Integration**: Pexels API with rate limiting
- **Query Optimization**: Keyword-based search with caching
- **Attribution Tracking**: Automatic attribution file generation
- **Concurrent Downloads**: Semaphore-based concurrency control

#### TTS Engine (`src/video/tts.py`)
- **Multi-Provider**: Google Cloud TTS, Coqui TTS with fallbacks
- **Voice Selection**: Configurable criteria (language, gender, name)
- **Async Generation**: Non-blocking TTS with timeout handling
- **Caching**: Client and model caching for performance

#### Subtitle Generation (`src/video/unified_subtitle_generator.py`, `src/video/stt_functions.py`)
- **Multi-Provider STT**: Whisper (primary), Google Cloud STT with word-level timing extraction
- **Audio-Based Synchronization**: Perfect timing via actual voiceover transcription (fixed September 2025)
- **Unified Generation System**: Single path for both ASS and SRT formats with content-aware positioning
- **Segmentation Logic**: Smart text splitting with natural boundaries based on actual speech timing
- **ASS/SRT Generation**: Professional subtitle formats with positioning effects

### 6. Amazon Scraping Features

### Search Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--min-price` | Minimum price filter | `--min-price 10.99` |
| `--max-price` | Maximum price filter | `--max-price 99.99` |
| `--min-rating` | Minimum rating (1-5 stars) | `--min-rating 4` |
| `--prime-only` | Prime eligible items only | `--prime-only` |
| `--free-shipping` | Free shipping items only | `--free-shipping` |
| `--brands` | Filter by brand names | `--brands Apple Samsung Sony` |
| `--sort` | Sort order | `--sort price-asc-rank` |

### Sort Options

- `relevanceblender` (default) - Amazon's relevance algorithm
- `price-asc-rank` - Price low to high
- `price-desc-rank` - Price high to low
- `review-rank` - Best reviews first
- `date-desc-rank` - Newest first
- `featured-rank` - Featured items first

### 7. Multi-Platform Web Scraping Architecture

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
- **14-Module Architecture**: Modular design for maintainability
- **Media Extraction**: High-resolution images and videos with validation
- **Advanced Search**: Complex filtering with price, rating, brand, and shipping options

## Performance Optimization Architecture

ContentEngineAI implements five major optimization categories:

### 1. Pipeline Parallelization

**Implementation**: `PipelineGraph` manages dependency-aware parallel execution
**Impact**: 1.35x speedup (26% reduction in pipeline time)
**Technical Details**:
- Topological sorting for execution order
- Concurrent subtitle generation and music download
- Resource allocation optimization

### 2. I/O Operations Optimization

**Implementation**: `AsyncIOManager` for non-blocking operations
**Components**:
- Async subprocess execution with timeouts
- Semaphore-based concurrency control
- Proper resource cleanup and management

### 3. Multi-Level Caching System

**Implementation**: `CacheManager` with TTL support
**Cache Types**:
- Media metadata cache (eliminates redundant ffprobe calls)
- API response cache (LLM, stock media, TTS)
- File-based persistence with thread safety

### 4. Resource Management

**Implementation**: Global connection pooling and memory mapping
**Components**:
- `ConnectionPool`: HTTP session management
- `MemoryMappedIO`: Efficient large file operations
- Resource lifecycle management

### 5. Background Processing

**Implementation**: `BackgroundProcessor` for preloading
**Features**:
- TTS model warming during pipeline startup
- Stock media pre-fetching based on keywords
- Background task lifecycle management

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

### Configuration System (`src/video/video_config.py`)

**Design Principles:**
- **YAML-Based**: Human-readable configuration files
- **Pydantic Validation**: Type-safe configuration with automatic validation
- **Environment Integration**: Secure handling of sensitive information
- **Hierarchical Structure**: Organized settings by component

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
- TTS: Google Cloud → Coqui TTS
- STT: Whisper → Google Cloud STT → Script-based fallback
- LLM: Primary model → Secondary models

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
2. Add configuration section to `video_config.py`
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
- **🤖 AI Services**: OpenRouter, Google Cloud, OpenAI Whisper
- **🌐 Web Scraping**: Playwright with stealth techniques
- **📱 Media APIs**: Pexels (images/videos), Freesound (audio)
- **⚙️ Configuration**: YAML + Pydantic validation
- **🧪 Testing**: Pytest with async support

## Acknowledgments

- **OpenAI Whisper** for speech-to-text capabilities
- **Google Cloud** for TTS and STT services
- **Pexels** for stock media content
- **Freesound** for background music
- **FFmpeg** for video processing excellence

This architecture enables ContentEngineAI to be highly extensible while maintaining performance, reliability, and maintainability across all components.