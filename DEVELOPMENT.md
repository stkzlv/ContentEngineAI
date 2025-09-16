# Development Guide

This guide provides detailed information for developers working on ContentEngineAI, including architecture details, performance optimizations, and development best practices.

## Development Environment Setup

### Prerequisites

- Python 3.12+
- Poetry for dependency management
- FFmpeg installed and in PATH
- Git for version control

### Quick Setup

```bash
# Clone and setup
git clone https://github.com/ContentEngineAI/ContentEngineAI.git
cd ContentEngineAI

# Install dependencies
poetry install --with dev

# Install pre-commit hooks
make install-dev
make pre-commit

# Install Playwright browsers
poetry run playwright install

# Verify setup
make lint
make test
```

## Code Quality Standards

ContentEngineAI maintains high code quality standards with comprehensive linting and testing:

### Linting Tools

```bash
# Run all quality checks
make lint

# Individual tools
make format        # Ruff formatting
make typecheck     # MyPy type checking  
make security      # Bandit security scanning
make vulture       # Dead code detection
make safety        # Dependency vulnerability checking
```

**📖 Complete linting documentation**: [LINTING.md](LINTING.md)

### Configuration

**Ruff Configuration:**
- Line length: 88 characters
- Format: Modern Python with type annotations
- Rules: Comprehensive linting with security checks

**MyPy Configuration:**
- Pragmatic settings for development efficiency
- Allows functions without type hints
- Ignores third-party library issues

**Testing Framework:**
- **Pytest** with async support
- **280 test cases** across 20 files
- **Coverage reporting** with branch coverage
- **Parallel execution** for faster testing

### Pre-commit Hooks

Automatic quality checks before commits:

```bash
# Install hooks
make pre-commit

# Manual run
poetry run pre-commit run --all-files
```

## Architecture Deep Dive

### Pipeline Architecture

ContentEngineAI uses a **dependency-aware pipeline** with parallel execution:

```python
# Traditional Sequential Flow:
# gather_visuals → generate_script → create_voiceover → generate_subtitles → download_music → assemble_video

# Optimized Parallel Flow:
# gather_visuals → generate_script → create_voiceover → [generate_subtitles + download_music] → assemble_video
```

**Key Benefits:**
- **26% faster execution** through parallel processing
- **Dependency management** ensures correct execution order
- **Resume capability** for debugging specific steps
- **Resource optimization** with proper concurrency limits

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
- **Semaphore-based concurrency** limits
- **Proper resource cleanup** and error handling
- **Non-blocking FFmpeg operations**

```python
class AsyncIOManager:
    async def run_subprocess_async(self, command: List[str], timeout: int):
        """Non-blocking subprocess execution with timeout"""
        async with self.semaphore:
            process = await asyncio.create_subprocess_exec(*command)
            await asyncio.wait_for(process.wait(), timeout=timeout)
```

#### 3. Multi-Level Caching System

**Component:** `src/utils/caching.py`

**Cache Types:**
- **Media metadata cache**: Eliminates redundant ffprobe calls
- **API response cache**: LLM responses, stock media searches
- **TTS cache**: Expensive TTS operations
- **File-based persistence** with TTL support

```python
class CacheManager:
    def __init__(self, cache_dir: Path, default_ttl: int = 3600):
        self.cache_dir = cache_dir
        self.default_ttl = default_ttl
        self._cache_lock = threading.Lock()
    
    async def get_or_compute(self, key: str, compute_func: Callable, ttl: int = None):
        """Get cached value or compute and cache it"""
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
# In src/video/video_config.py
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
from src.scraper.base.models import BaseScraper, BaseProductData
from src.scraper.base.config import register_scraper, Platform

@register_scraper(Platform.EBAY)  # Auto-registration with platform registry
class EbayScraper(BaseScraper):
    async def scrape_products(self, keywords: List[str]) -> List[BaseProductData]:
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
# config/scrapers.yaml
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

## Testing Strategy

### Test Structure

```
tests/
├── conftest.py                 # Pytest configuration and fixtures
├── test_*.py                   # Individual component tests
└── README.md                   # Testing documentation
```

### Test Categories

**Unit Tests** (`@pytest.mark.unit`):
- Test individual functions and classes in isolation
- Use mocks for external dependencies
- Target: >90% coverage

**Integration Tests** (`@pytest.mark.integration`):
- Test component interactions
- Use mocked external services
- Target: >80% coverage

**End-to-End Tests** (`@pytest.mark.e2e`):
- Test complete pipeline workflows
- Use real or staging environments
- Focus on critical user journeys

### Writing Tests

**Basic Test Structure:**
```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.unit
async def test_my_function():
    """Test description"""
    # Arrange
    mock_dependency = AsyncMock()
    
    # Act
    result = await my_function(mock_dependency)
    
    # Assert
    assert result == expected_value
    mock_dependency.assert_called_once()
```

**Async Testing:**
```python
@pytest.mark.asyncio
async def test_async_function():
    """Test async function"""
    async with AsyncMock() as mock:
        result = await async_function(mock)
        assert result is not None
```

**Mocking External APIs:**
```python
@patch('src.ai.script_generator.httpx.AsyncClient')
async def test_script_generation(mock_client):
    """Test LLM script generation"""
    mock_response = AsyncMock()
    mock_response.json.return_value = {"choices": [{"message": {"content": "Test script"}}]}
    mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
    
    result = await generate_script(product_data, config)
    assert "Test script" in result
```

### Running Tests

```bash
# All tests
make test

# Specific test types
poetry run pytest -m unit
poetry run pytest -m integration
poetry run pytest -m e2e

# With coverage
poetry run pytest --cov=src --cov-report=html

# Parallel execution
poetry run pytest -n auto

# Specific test file
poetry run pytest tests/test_assembler.py -v
```

**📖 Complete testing documentation**: [TESTING.md](TESTING.md)

## Usage Examples

### Single Product Processing

```bash
# Process single product from data file
poetry run python -m src.video.producer outputs/B0BTYCRJSS/data.json slideshow_images

# Process specific product from list
poetry run python -m src.video.producer products.json slideshow_images --product-index 0

# Custom profile
poetry run python -m src.video.producer data.json my_custom_profile
```

### Batch Processing

ContentEngineAI supports **automatic batch processing** of all products in the outputs directory:

```bash
# Process all products with slideshow_images profile
poetry run python -m src.video.producer --batch --batch-profile slideshow_images --debug

# Batch processing with fail-fast (stop on first error)
poetry run python -m src.video.producer --batch --batch-profile dynamic_mix --fail-fast --debug

# Process products from custom directory
poetry run python -m src.video.producer --batch --batch-profile slideshow_images --outputs-dir /path/to/outputs --debug

# Batch processing with clean runs
poetry run python -m src.video.producer --batch --batch-profile slideshow_images --clean --debug
```

**Batch Processing Features:**
- ✅ **Auto-Discovery**: Automatically finds all products with `data.json` files
- ✅ **Progress Tracking**: Shows "[1/3] Processing product: B08TEST123"
- ✅ **Error Resilience**: Continues processing other products if one fails
- ✅ **Fail-Fast Option**: `--fail-fast` stops on first error for debugging
- ✅ **Comprehensive Reporting**: Final summary with success/failure/skip counts

### Advanced Scraper Usage

```bash
# Budget headphones under $50 with 4+ star ratings
poetry run python -m src.scraper.amazon.scraper \
  --keywords "headphones" --max-price 50 --min-rating 4

# Premium smartphones from specific brands
poetry run python -m src.scraper.amazon.scraper \
  --keywords "smartphone" --brands Apple Samsung \
  --min-price 200 --sort price-desc-rank

# Prime-eligible electronics sorted by best reviews
poetry run python -m src.scraper.amazon.scraper \
  --keywords "electronics" --prime-only --sort review-rank
```

## Performance Metrics

### Pipeline Performance

| Metric | Value |
|--------|-------|
| **Pipeline Architecture** | Parallel execution with intelligent dependency management |
| **Test Coverage** | 280+ test cases across comprehensive test suite |
| **Configuration Options** | 100+ customizable parameters |
| **Supported Providers** | 10+ AI/media service integrations |
| **Processing Time** | Typically 5-8 minutes per video (varies by complexity) |
| **Parallel Speedup** | 1.35x faster execution (26% reduction) |

### Project Statistics

- **🧪 Testing**: 280 test cases across 20 files
- **📦 Dependencies**: 50+ Python packages managed with Poetry
- **🔍 Code Quality**: Ruff, MyPy, Bandit, Vulture, Safety
- **🏗️ Architecture**: Modular, async-first, provider-abstracted design

## Debugging and Development Tools

### Debug Mode

```bash
# Enable comprehensive debugging
poetry run python -m src.video.producer products.json profile --debug

# Run specific step
poetry run python -m src.video.producer products.json profile --debug --step generate_script

# Batch processing debugging
poetry run python -m src.video.producer --batch --batch-profile slideshow_images --debug

# Batch processing with fail-fast for debugging
poetry run python -m src.video.producer --batch --batch-profile slideshow_images --fail-fast --debug
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

### Adding New Configuration Options

1. **Update Pydantic Models:**
```python
# In src/video/video_config.py
class MyComponentConfig(BaseModel):
    enabled: bool = True
    timeout_sec: int = 30
    api_key_env_var: str = "MY_API_KEY"
    
    @validator('timeout_sec')
    def validate_timeout(cls, v):
        if v <= 0:
            raise ValueError('timeout_sec must be positive')
        return v
```

2. **Add to Main Config:**
```python
class VideoConfig(BaseModel):
    my_component: MyComponentConfig = MyComponentConfig()
```

3. **Update YAML Schema:**
```yaml
my_component:
  enabled: true
  timeout_sec: 60
  api_key_env_var: "MY_API_KEY"
```

### Configuration Validation

```python
# Test configuration loading
def test_config_loading():
    config = load_config('config/video_producer.yaml')
    assert config.my_component.enabled is True
    assert config.my_component.timeout_sec > 0
```

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
# Cache expensive operations
from src.utils.caching import get_cache_manager

cache = get_cache_manager()

async def expensive_operation(param):
    cache_key = f"expensive_op_{hash(param)}"
    
    async def compute():
        # Expensive computation
        return result
    
    return await cache.get_or_compute(cache_key, compute, ttl=3600)
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
poetry run python -m src.video.producer test_products.json slideshow_images

# Run batch processing benchmarks  
poetry run python -m src.video.producer --batch --batch-profile slideshow_images

# Compare performance
poetry run python tools/performance_report.py --compare baseline.json
```

## Contributing Guidelines

### Code Style

- **Naming**: `snake_case` for variables/functions, `PascalCase` for classes
- **Type Hints**: Use modern Python typing (`dict[str, Any]`, `| None`)
- **Docstrings**: Comprehensive docstrings for public functions
- **Comments**: Minimal inline comments, prefer self-explanatory code

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes, following code style
# Run tests and linting
make lint
make test

# Commit with conventional commit format
git commit -m "feat: Add your feature description"

# Push and create PR
git push origin feature/your-feature-name
```

### Pull Request Guidelines

- **Clear title and description**
- **Reference related issues**
- **Include tests for new functionality**
- **Update documentation as needed**
- **Ensure CI passes**

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
        task_id = await bg.submit_task(preload_models)
        
        # Do main work
        result = await main_processing()
        
        # Wait for background tasks if needed
        await bg.wait_for_task(task_id)
        
        return result
```

## Configuration Development Guidelines

### Working with Configuration Settings

When adding new configuration options, follow these patterns:

**Adding New Settings:**
```python
# 1. Add to Pydantic model in src/video/video_config.py
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
# In video_config.py
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