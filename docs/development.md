# Development Guide

This guide provides detailed information for developers working on ContentEngineAI, including architecture details, performance optimizations, and development best practices.

## Development Environment Setup

**📖 Complete installation guide**: [Installation](installation.md)

```bash
# Quick setup for developers
poetry install --with dev && make install-dev
```

### Optional: pycaps subtitle engine

The `pycaps` subtitle engine is selected by default in `config/subtitles.yaml`
but its dependencies are an optional Poetry group (`pycaps`). The default
`poetry install` skips the group to keep the footprint small; the bundled
`pycaps.fallback_policy: fallback_ffmpeg` then degrades to the FFmpeg path
silently. Install the group locally to actually use the pycaps engine:

```bash
poetry install --with pycaps
poetry run playwright install chromium   # CSS renderer (default). Skip for pictex-only.
```

On Ubuntu 26.04 prefix the chromium install with
`PLAYWRIGHT_HOST_PLATFORM_OVERRIDE=ubuntu24.04-x64` (no 26.04 build yet), and
wrap CSS-renderer runs in `xvfb-run -a` on Wayland. See
[docs/pycaps-subtitles.md](pycaps-subtitles.md) for the full reference.

## Code Quality Standards

**📖 Complete guide**: [Linting](linting.md) • [Testing](testing.md)

```bash
make lint          # Complete quality check (7 tools)
make test          # Full test suite with coverage
make security      # Security vulnerability scan
```

## Architecture Deep Dive

**📖 Complete architecture guide**: [Architecture](architecture.md)

ContentEngineAI uses a **dependency-aware pipeline** with parallel execution achieving **26% faster execution** through intelligent dependency management and concurrent processing.

### Performance Optimization System

ContentEngineAI implements **5 major optimization categories**:

#### 1. Pipeline Parallelization Framework

**Component:** `src/video/pipeline_graph.py`

**Implementation:**
```python
class PipelineGraph:
    def __init__(self, dependencies: Dict[str, List[str]]):
        self.dependencies = dependencies
        self.execution_order = self._topological_sort()
    
    async def execute_parallel(self, steps: Dict[str, Callable]):
        """Execute steps in parallel where dependencies allow"""
        # Implementation enables concurrent subtitle + music download
```

**Performance Impact:** 1.35x speedup (saves ~87 seconds per run)

#### 2. I/O Operations Optimization

**Component:** `src/utils/async_io.py`

**Features:**
- **Async subprocess management** with timeout control
- **Non-blocking FFmpeg operations** with proper cleanup
- **Media probing** with async ffprobe calls
- **Proper error handling** and timeout management

```python
# Async FFmpeg execution with timeout
async def async_run_ffmpeg(
    cmd: list[str],
    timeout_sec: float = 300.0,
    log_path: Path | None = None,
) -> tuple[bool, str, str]:
    """Run FFmpeg command asynchronously with timeout."""
    process = await asyncio.create_subprocess_exec(*cmd, ...)
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_sec)
    return process.returncode == 0, stdout_str, stderr_str

# Async media probing
async def async_probe_media(file_path: Path, timeout_sec: float = 30.0) -> dict | None:
    """Probe media file asynchronously to get metadata."""
```

#### 3. Multi-Level Caching System

**Component:** `src/utils/caching.py`

**Cache Types:**
- **Media metadata cache**: Eliminates redundant ffprobe calls
- **File-based persistence** with TTL support
- **Automatic expiration** and cleanup

```python
class PersistentCache:
    """Persistent file-based cache for API responses and metadata."""

    def __init__(self, cache_dir: Path, ttl_seconds: int = 3600):
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds

    def get(self, key: str) -> Any | None:
        """Get cached value if it exists and hasn't expired."""

    def set(self, key: str, value: Any) -> None:
        """Store value in cache with timestamp."""

# Global helper functions for media metadata caching
def cache_media_metadata(file_path: Path, metadata: dict[str, Any]) -> None:
    """Cache media file metadata."""

def get_cached_media_metadata(file_path: Path) -> dict[str, Any] | None:
    """Get cached media file metadata."""
```

#### 4. Resource Management Optimization

**Components:** 
- `src/utils/connection_pool.py`: HTTP connection pooling
- `src/utils/memory_mapped_io.py`: Memory-mapped file operations

**Features:**
- **Global HTTP session management** with persistent connections
- **Memory-mapped I/O** for large files (>1MB)
- **Resource lifecycle management** with proper cleanup
- **Connection reuse** across API calls

#### 5. Background Processing Framework

**Component:** `src/utils/background_processing.py`

**Capabilities:**
- **TTS model warming** during pipeline startup
- **Stock media pre-fetching** based on product keywords
- **Background task management** with lifecycle control
- **Resource preloading** for reduced latency

### Performance Monitoring

**Component:** `src/utils/performance.py`

**Features:**
- **Real-time metrics collection** for all pipeline steps
- **Historical data persistence** (JSONL format, 100 runs)
- **Cross-session analysis** and trend detection
- **Resource usage tracking** (memory, CPU, I/O)

**Usage:**
```bash
# Quick performance summary
make perf-report

# Detailed analysis
poetry run python tools/performance_report.py --report-type detailed --limit 20

# Product-specific trends
poetry run python tools/performance_report.py --report-type trends --product-id B0BTYCRJSS

# Export for external analysis
poetry run python tools/performance_report.py --output metrics.json
```

## Component Development

### Adding New Pipeline Steps

1. **Define Step Function:**
```python
async def my_new_step(context: PipelineContext) -> PipelineContext:
    """New pipeline step implementation"""
    # Your implementation here
    return context
```

2. **Update Pipeline Graph:**
```python
# In src/video/pipeline_graph.py
PIPELINE_DEPENDENCIES = {
    'my_new_step': ['previous_step'],  # Define dependencies
    'next_step': ['my_new_step'],      # Steps that depend on this
}
```

3. **Add Configuration:**
```python
# In src/video/config/core_models.py
class MyNewStepConfig(BaseModel):
    enabled: bool = True
    timeout_sec: int = 60
    # Additional settings
```

4. **Add Performance Monitoring:**
```python
async def my_new_step(context: PipelineContext) -> PipelineContext:
    async with context.performance_monitor.track_step('my_new_step'):
        # Implementation
        pass
```

### Adding New E-Commerce Platforms

**Multi-Platform Scraper Architecture:**
ContentEngineAI uses a modular, extensible architecture for supporting multiple e-commerce platforms.

#### **1. Implement BaseScraper Interface:**
```python
from src.scraper.base.models import (
    BaseScraper, BaseProductData, Platform, register_scraper
)

@register_scraper(Platform.EBAY)  # Auto-registration with platform registry
class EbayScraper(BaseScraper):
    async def scrape_products(self, keywords: list[str]) -> list[BaseProductData]:
        """eBay-specific product scraping implementation"""
        pass

    def validate_product_id(self, product_id: str) -> bool:
        """Validate eBay item ID format (12 digits)"""
        return re.match(r'^[0-9]{12}$', product_id) is not None
```

#### **2. Create Platform-Specific Modules:**
```
src/scraper/ebay/
├── __init__.py              # Public API interface
├── scraper.py              # Main orchestrator (extends BaseScraper) 
├── browser_functions.py    # eBay browser automation
├── media_extractor.py      # eBay image/video extraction
├── models.py              # eBay-specific data models
├── config.py              # eBay configuration management
└── utils.py               # eBay utility functions
```

#### **3. Add Platform Configuration:**
```yaml
# config/scraper.yaml
platforms:
  ebay:
    enabled: true
    base_url: "https://www.ebay.com"
    max_products: 5
    platform_specific:
      item_id_pattern: "^[0-9]{12}$"
      search_filters:
        condition_codes:
          new: "1000"
          used: "3000"
```

#### **4. Use Factory Pattern:**
```python
from src.scraper import ScraperFactory

# Automatic platform detection and creation
ebay_scraper = ScraperFactory.create_scraper('ebay')
products = await ebay_scraper.scrape_products(['smartphones'])

# Multi-platform access
for platform in ['amazon', 'ebay', 'walmart']:
    scraper = ScraperFactory.create_scraper(platform)
    results = await scraper.scrape_products(['wireless headphones'])
```

### Adding New Provider Integrations

**Base Provider Interface:**
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

**Example TTS Provider:**
```python
class MyTTSProvider(BaseProvider):
    async def initialize(self, config: Dict[str, Any]) -> None:
        self.api_key = config.get('api_key')
        self.timeout = config.get('timeout_sec', 30)
    
    async def process(self, text: str) -> Tuple[bytes, Optional[List[float]]]:
        """Convert text to speech, return audio and timings"""
        # Implementation
        pass
```

### Adding New Media Sources

1. **Implement Media Provider:**
```python
class MyMediaProvider:
    async def search_media(self, query: str, count: int) -> List[MediaItem]:
        """Search for media items"""
        pass
    
    async def download_media(self, item: MediaItem, output_path: Path) -> Path:
        """Download media item"""
        pass
```

2. **Add Attribution Support:**
```python
class MediaItem:
    url: str
    title: str
    author: str
    source: str
    license_info: str
    attribution_required: bool
```

3. **Update Configuration:**
```yaml
stock_media_settings:
  my_provider:
    enabled: true
    api_key_env_var: "MY_PROVIDER_API_KEY"
    concurrent_downloads: 3
```

## Testing

**📖 Complete testing guide**: [Testing](testing.md)

```bash
make test          # Run all tests
make test-cov      # Run tests with coverage report
make test-parallel # Run tests in parallel
```

## Usage Examples

**📖 Complete usage guides**: [README.md](../README.md) • [Batch Processing](batch-processing.md)

```bash
# Single product
poetry run python -m src.video.producer outputs/ASIN/data.json slideshow_images1 --debug

# Batch processing
poetry run python -m src.video.producer --batch --batch-profile slideshow_images1 --debug
```

## Video Assembly

**📖 Complete video configuration**: [Configuration](configuration.md#video-assembly-settings)

ContentEngineAI supports 4 video assembly modes: `sequential`, `single_best`, `mixed_media`, `video_first_fallback`. See [Configuration](configuration.md) for profile details and CLI overrides.

## Debugging and Development Tools

### Debug Mode

```bash
# Enable comprehensive debugging
poetry run python -m src.video.producer products.json profile --debug

# Run specific step
poetry run python -m src.video.producer products.json profile --debug --step generate_script

# Batch processing debugging
poetry run python -m src.video.producer --batch --batch-profile slideshow_images1 --debug

# Batch processing with fail-fast for debugging
poetry run python -m src.video.producer --batch --batch-profile slideshow_images1 --fail-fast --debug
```

**Debug Features:**
- **Intermediate file preservation** in `outputs/temp/`
- **FFmpeg command logging** for video assembly debugging
- **Structured console logging** with timestamps
- **Performance metrics** for each step
- **API request/response logging**

### Performance Profiling

```python
# Add performance tracking to your code
async def my_function():
    async with performance_monitor.track_operation('my_operation'):
        # Your code here
        pass
```

### Logging Best Practices

```python
import logging

logger = logging.getLogger(__name__)

async def my_function():
    logger.info("Starting operation with param=%s", param)
    
    try:
        result = await expensive_operation()
        logger.debug("Operation completed successfully, result=%s", result)
        return result
    except Exception as e:
        logger.error("Operation failed: %s", str(e), exc_info=True)
        raise
```

## Configuration Development

**📖 Complete configuration guide**: [Configuration](configuration.md)

Configuration is managed through modular YAML files in `config/` with Pydantic validation in `src/video/config/` (core_models.py, visual_models.py, audio_models.py, subtitle_models.py).

## Performance Optimization Guidelines

### Async Best Practices

```python
# Good: Use asyncio.gather for concurrent operations
async def process_multiple_items(items):
    tasks = [process_item(item) for item in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

# Good: Use semaphores for rate limiting
async def download_with_limit(urls, max_concurrent=3):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def download_one(url):
        async with semaphore:
            return await download(url)
    
    tasks = [download_one(url) for url in urls]
    return await asyncio.gather(*tasks)
```

### Memory Management

```python
# Use memory-mapped I/O for large files
from src.utils.memory_mapped_io import is_file_suitable_for_mmap, copy_file_mmap

if is_file_suitable_for_mmap(file_path, min_size=1024*1024):
    copy_file_mmap(source, destination)
else:
    shutil.copy(source, destination)
```

### Connection Pooling

```python
# Use global connection pools
from src.utils.connection_pool import get_http_session

async def api_call(url):
    session = await get_http_session()
    async with session.get(url) as response:
        return await response.json()
```

### Caching

```python
# Cache expensive operations using PersistentCache
from pathlib import Path
from src.utils.caching import PersistentCache

cache = PersistentCache(Path("outputs/cache"), ttl_seconds=3600)

def expensive_operation(param):
    cache_key = f"expensive_op_{hash(param)}"

    # Check cache first
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    # Compute and cache result
    result = compute_expensive_result(param)
    cache.set(cache_key, result)
    return result
```

## Release Process

### Version Management

1. **Update Version:**
```bash
# Update pyproject.toml version
poetry version patch  # or minor, major
```

2. **Update Documentation:**
- Update README.md with new features
- Update CHANGELOG.md with changes
- Update configuration examples if needed

3. **Run Full Test Suite:**
```bash
make lint
make test
make security
make vulture
```

### Performance Benchmarking

Before releases, run performance benchmarks:

```bash
# Generate baseline performance report
poetry run python tools/performance_report.py --output baseline.json

# Run test pipelines
poetry run python -m src.video.producer test_products.json slideshow_images1

# Run batch processing benchmarks  
poetry run python -m src.video.producer --batch --batch-profile slideshow_images1

# Compare performance
poetry run python tools/performance_report.py --compare baseline.json
```

## Contributing

**📖 Complete contributing guide**: [CONTRIBUTING.md](../CONTRIBUTING.md)

Follow GitHub Flow with feature branches, comprehensive testing, and code quality checks before PRs.

## Advanced Topics

### Custom FFmpeg Filters

```python
# Adding custom FFmpeg filters
def build_custom_filter(input_specs: List[InputSpec]) -> str:
    """Build custom FFmpeg filter graph"""
    filters = []
    
    # Add your custom filter logic
    filters.append(f"[0:v]scale=1080:1920[scaled]")
    filters.append(f"[scaled]pad=1080:1920:(ow-iw)/2:(oh-ih)/2[padded]")
    
    return ";".join(filters)
```

### Provider Plugin System

```python
# Creating pluggable providers
class ProviderRegistry:
    _providers = {}
    
    @classmethod
    def register(cls, name: str, provider_class: Type[BaseProvider]):
        cls._providers[name] = provider_class
    
    @classmethod
    def get_provider(cls, name: str) -> BaseProvider:
        return cls._providers[name]()

# Usage
@ProviderRegistry.register('my_tts')
class MyTTSProvider(BaseProvider):
    pass
```

### Background Task Management

```python
# Advanced background processing
from src.utils.background_processing import BackgroundProcessor

async def with_background_tasks():
    async with BackgroundProcessor() as bg:
        # Start background tasks
        task = await bg.start_task(
            task_id="preload_models",
            name="Model Preloading",
            coro_func=preload_models,
        )

        # Do main work
        result = await main_processing()

        # Wait for background tasks if needed
        if task:
            await bg.wait_for_task(task.task_id)

        return result
```

## Configuration Development Guidelines

### Working with Configuration Settings

When adding new configuration options, follow these patterns:

**Adding New Settings:**
```python
# 1. Add to Pydantic model in src/video/config/core_models.py
class MySettings(BaseModel):
    my_new_setting: float = Field(default=2.5, description="Description of setting")

# 2. Add to YAML with comprehensive comments
my_settings:
  # Clear description of what this controls and its impact
  # Examples: 2.5 = typical value, 1.0 = conservative, 5.0 = aggressive
  my_new_setting: 2.5
```

**Using Configuration in Code:**
```python
# Always provide fallbacks for new settings
def use_config_setting(self, config):
    # Method 1: hasattr check with fallback
    my_value = (
        config.my_settings.my_new_setting
        if hasattr(config.my_settings, 'my_new_setting')
        else 2.5  # Fallback to default
    )
    
    # Method 2: getattr with fallback
    my_value = getattr(config.my_settings, 'my_new_setting', 2.5)
    
    # Method 3: Direct access (only for required settings)
    my_value = config.my_settings.my_new_setting
```

### Configuration Best Practices

1. **Backward Compatibility**: Always provide sensible defaults for new settings
2. **Documentation**: Add comprehensive comments explaining purpose and impact
3. **Validation**: Use Pydantic Field constraints for validation
4. **Fallbacks**: Implement graceful fallbacks for missing settings
5. **Testing**: Test configuration changes with existing workflows

**Example Configuration Addition:**
```python
# In src/video/config/core_models.py
class VideoSettings(BaseModel):
    # Existing settings...
    
    # NEW: Duration padding to prevent audio cutoff
    duration_padding_sec: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        description="Padding added to video duration to prevent audio cutoff"
    )
```

### Configuration Migration

When removing or changing settings:

```yaml
# Mark removed settings with NOTE comments
# NOTE: old_setting_name removed as unused in codebase

# For renamed settings, document the migration
new_setting_name: 300  # Previously fade_duration, renamed for clarity
```

### Debugging Configuration Issues

```python
# Configuration debugging utilities
import logging

def debug_config_access(config, setting_path: str):
    """Debug configuration access patterns"""
    try:
        value = getattr(config, setting_path)
        logging.debug(f"Config {setting_path} = {value}")
        return value
    except AttributeError:
        logging.warning(f"Config {setting_path} not found, using fallback")
        return None
```

This development guide provides the foundation for contributing to and extending ContentEngineAI. For specific implementation details, refer to the source code and inline documentation.