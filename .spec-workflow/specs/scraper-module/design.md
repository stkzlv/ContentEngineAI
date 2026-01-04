# Design Document: Scraper Module

## Overview

The Scraper Module provides a platform-agnostic architecture for extracting product data, images, and videos from e-commerce platforms. It supports both single product lookups and batch operations with keyword-based discovery, implementing robust anti-detection measures and graceful error handling.

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLI Entry Point                                 │
│                    src/scraper/amazon/scraper.py (main)                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│  │ --product-ids   │  │ --keywords      │  │ --filters (price, rating)  │ │
│  └────────┬────────┘  └────────┬────────┘  └──────────────┬──────────────┘ │
└───────────┼────────────────────┼──────────────────────────┼─────────────────┘
            │                    │                          │
            ▼                    ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Batch Controller Layer                              │
│                  src/scraper/amazon/batch_controller.py                      │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  BatchController                                                      │  │
│  │  ├── run_batch() -> BatchSummary                                      │  │
│  │  ├── _process_product_ids() -> list[ProductResult]                    │  │
│  │  ├── _process_keywords() -> list[ProductResult]                       │  │
│  │  └── _deduplicate_products() -> list[ProductData]                     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Platform Scraper Layer                              │
│  ┌────────────────────────┐    ┌────────────────────────────────────────┐  │
│  │    ScraperRegistry     │───▶│   BotasaurusAmazonScraper              │  │
│  │  @register_scraper()   │    │   src/scraper/amazon/scraper.py        │  │
│  └────────────────────────┘    │   ├── scrape_products_unified()        │  │
│                                │   ├── _search_products()               │  │
│                                │   └── _scrape_product_page()           │  │
│                                └────────────────────────────────────────┘  │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
            ┌────────────────────────┼────────────────────────┐
            ▼                        ▼                        ▼
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│  Media Extraction    │  │  Media Download      │  │  Media Validation    │
│  media_extractor.py  │  │  downloader.py       │  │  media_validator.py  │
│  ├── extract_images()│  │  ├── download_media()│  │  ├── verify_image()  │
│  ├── extract_videos()│  │  ├── download_video()│  │  ├── verify_video()  │
│  └── filter_by_asin()│  │  └── async streaming │  │  └── extract_meta()  │
└──────────────────────┘  └──────────────────────┘  └──────────────────────┘
```

### Data Flow

```
Input Processing Flow:
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│ CLI Args   │───▶│ Batch      │───▶│ Deduplicate│───▶│ Product    │
│ or YAML    │    │ Controller │    │ Products   │    │ Queue      │
└────────────┘    └────────────┘    └────────────┘    └────────────┘

Scraping Flow (per product):
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│ Product    │───▶│ Scrape     │───▶│ Extract    │───▶│ Download   │
│ Page       │    │ Data       │    │ Media URLs │    │ Media      │
└────────────┘    └────────────┘    └────────────┘    └────────────┘
                                                            │
                                                            ▼
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│ Summary    │◀───│ Save       │◀───│ Validate   │◀───│ Extract    │
│ Report     │    │ data.json  │    │ Media      │    │ Metadata   │
└────────────┘    └────────────┘    └────────────┘    └────────────┘
```

## Detailed Design

### 1. Base Scraper Architecture

**Location**: `src/scraper/base/models.py`

**Purpose**: Platform-agnostic abstract base class for all scrapers.

**Class Structure**:
```python
class BaseScraper(ABC):
    """Abstract base class for platform-specific scrapers."""

    @abstractmethod
    def scrape(self, product_id: str) -> ProductData:
        """Scrape a single product by ID."""
        ...

    @abstractmethod
    def search(self, keyword: str, filters: SearchFilters) -> list[ProductData]:
        """Search for products by keyword with optional filters."""
        ...

    @abstractmethod
    def validate_product_id(self, product_id: str) -> bool:
        """Validate product ID format for this platform."""
        ...


class ScraperRegistry:
    """Registry for platform-specific scrapers."""

    _scrapers: dict[str, type[BaseScraper]] = {}

    @classmethod
    def register(cls, platform: str) -> Callable:
        """Decorator to register a scraper for a platform."""
        def decorator(scraper_cls: type[BaseScraper]) -> type[BaseScraper]:
            cls._scrapers[platform] = scraper_cls
            return scraper_cls
        return decorator

    @classmethod
    def get_scraper(cls, platform: str) -> BaseScraper:
        """Get scraper instance for platform."""
        ...
```

### 2. Batch Controller

**Location**: `src/scraper/amazon/batch_controller.py`

**Purpose**: Orchestrates batch processing of multiple products.

**Class Structure**:
```python
@dataclass
class BatchConfig:
    product_ids: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    min_price: float | None = None
    max_price: float | None = None
    min_rating: float | None = None
    prime_only: bool = False
    max_products_per_keyword: int = 10
    fail_fast: bool = False
    debug_mode: bool = False
    clean_mode: bool = False


@dataclass
class BatchSummary:
    total_attempted: int
    successful: list[str]
    failed: list[tuple[str, str]]  # (product_id, error_message)
    skipped: list[tuple[str, str]]  # (product_id, reason)
    duration_seconds: float


class BatchController:
    """Orchestrates batch scraping operations."""

    def __init__(self, config: BatchConfig, scraper: BaseScraper) -> None:
        self.config = config
        self.scraper = scraper
        self.processed_ids: set[str] = set()

    def run_batch(self) -> BatchSummary:
        """Execute batch processing and return summary."""
        ...

    def _process_product_ids(self) -> list[ProductResult]:
        """Process explicit product ID list."""
        ...

    def _process_keywords(self) -> list[ProductResult]:
        """Process keyword searches with filters."""
        ...

    def _deduplicate_products(self, products: list[str]) -> list[str]:
        """Remove duplicate product IDs."""
        ...
```

### 3. Amazon Scraper Implementation

**Location**: `src/scraper/amazon/scraper.py`

**Purpose**: Amazon-specific implementation using Botasaurus framework.

**Key Components**:
```python
@ScraperRegistry.register("amazon")
class BotasaurusAmazonScraper(BaseScraper):
    """Amazon scraper using Botasaurus for anti-detection."""

    def __init__(self, config: ScraperConfig) -> None:
        self.config = config
        self.session = self._create_session()

    def scrape(self, product_id: str) -> ProductData:
        """Scrape Amazon product page."""
        ...

    def search(self, keyword: str, filters: SearchFilters) -> list[ProductData]:
        """Search Amazon for products."""
        ...

    @botasaurus.browser(
        bypass_cloudflare=True,
        block_images_and_css=False,
        headless=True,
    )
    def _scrape_product_page(self, driver: Driver, url: str) -> dict:
        """Scrape product page with anti-detection."""
        ...
```

### 4. Media Extraction Pipeline

**Location**: `src/scraper/amazon/media_extractor.py`

**Purpose**: Extract image and video URLs from product pages.

**Video URL Extraction**:
```python
class VideoURLExtractor:
    """Extracts video URLs from product pages."""

    def extract_video_urls(self, page_content: str, product_id: str) -> list[VideoURL]:
        """Extract all video URLs from page content."""
        urls = []
        urls.extend(self._extract_mp4_urls(page_content))
        urls.extend(self._extract_m3u8_urls(page_content))
        urls.extend(self._extract_vdp_urls(page_content))
        return self._filter_by_product_id(urls, product_id)

    def _extract_mp4_urls(self, content: str) -> list[str]:
        """Extract direct MP4 URLs using regex patterns."""
        patterns = [
            r'https?://[^"\s]+\.mp4[^"\s]*',
            r'"url"\s*:\s*"([^"]+\.mp4[^"]*)"',
        ]
        ...

    def _extract_vdp_urls(self, content: str) -> list[str]:
        """Extract Video Detail Page URLs for high-quality variants."""
        ...

    def _filter_by_product_id(self, urls: list[str], product_id: str) -> list[str]:
        """Filter URLs to only include product-relevant videos."""
        ...
```

### 5. Media Downloader

**Location**: `src/scraper/amazon/downloader.py`

**Purpose**: Download media files with retry logic and progress tracking.

**Download Pipeline**:
```python
class MediaDownloader:
    """Handles media download with retry and validation."""

    async def download_media_async(
        self,
        product_id: str,
        image_urls: list[str],
        video_urls: list[str],
        output_dir: Path,
    ) -> DownloadResult:
        """Download all media files asynchronously."""
        ...

    async def download_single_video(
        self,
        url: str,
        output_path: Path,
        timeout: int = 60,
    ) -> bool:
        """Download single video with streaming and progress."""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return False
                with open(output_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        f.write(chunk)
        return True

    def _validate_url(self, url: str) -> bool:
        """Validate URL accessibility via HEAD request."""
        ...
```

### 6. Media Validator

**Location**: `src/scraper/amazon/media_validator.py`

**Purpose**: Validate downloaded media files and extract metadata.

**Validation Pipeline**:
```python
class MediaValidator:
    """Validates media files and extracts metadata."""

    def verify_image(self, path: Path) -> ImageValidation:
        """Verify image integrity and dimensions using PIL."""
        try:
            with Image.open(path) as img:
                img.verify()
                width, height = img.size
                return ImageValidation(
                    valid=True,
                    width=width,
                    height=height,
                    format=img.format,
                )
        except Exception as e:
            return ImageValidation(valid=False, error=str(e))

    def verify_video(self, path: Path) -> VideoValidation:
        """Verify video integrity using FFprobe."""
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_format', '-show_streams',
                 '-print_format', 'json', str(path)],
                capture_output=True, text=True
            )
            metadata = json.loads(result.stdout)
            return self._parse_video_metadata(metadata)
        except Exception as e:
            return VideoValidation(valid=False, error=str(e))

    def extract_video_metadata(self, path: Path) -> VideoMetadata:
        """Extract detailed video metadata."""
        ...
```

## File Structure

```
src/scraper/
├── base/
│   ├── __init__.py
│   └── models.py              # BaseScraper, ScraperRegistry, ProductData
├── amazon/
│   ├── __init__.py
│   ├── scraper.py             # BotasaurusAmazonScraper, CLI entry
│   ├── batch_controller.py    # BatchController, BatchConfig, BatchSummary
│   ├── downloader.py          # MediaDownloader, async download
│   ├── media_extractor.py     # VideoURLExtractor, ImageExtractor
│   ├── media_validator.py     # MediaValidator, FFprobe integration
│   └── selectors.py           # CSS selectors for Amazon pages
└── utils/
    ├── __init__.py
    └── retry.py               # Retry decorators with backoff
```

## Integration Points

### CLI Interface

| Flag | Environment Variable | Effect |
|------|---------------------|--------|
| `--product-ids` | PRODUCT_IDS | List of product IDs to scrape |
| `--keywords` | KEYWORDS | Keywords for product search |
| `--min-price` | MIN_PRICE | Minimum price filter |
| `--max-price` | MAX_PRICE | Maximum price filter |
| `--min-rating` | MIN_RATING | Minimum rating filter (0-5) |
| `--prime-only` | PRIME_ONLY | Filter to Prime products |
| `--max-products` | MAX_PRODUCTS | Max products per keyword |
| `--fail-fast` | FAIL_FAST | Stop on first error |
| `--debug` | DEBUG_MODE | Enable debug logging |
| `--clean` | CLEAN_MODE | Delete existing data first |

### Output Format

**data.json Structure**:
```json
{
  "product_id": "B0ASIN123",
  "title": "Product Title",
  "price": 29.99,
  "currency": "USD",
  "description": "Product description...",
  "rating": 4.5,
  "review_count": 1234,
  "images": ["images/abc123.jpg", "images/def456.jpg"],
  "videos": ["videos/vid001.mp4"],
  "downloaded_videos": [
    {
      "path": "videos/vid001.mp4",
      "duration": 45.2,
      "width": 1920,
      "height": 1080,
      "codec": "h264"
    }
  ],
  "scraped_at": "2024-01-15T10:30:00Z"
}
```

## Error Handling Strategy

### Error Categories

1. **Recoverable Errors** (log and retry):
   - Network timeouts
   - Rate limiting responses
   - Temporary page load failures

2. **Non-Recoverable Errors** (skip product):
   - Product not found (404)
   - Invalid product ID format
   - CAPTCHA detection (after retries)

3. **Fatal Errors** (stop batch if fail-fast):
   - Invalid configuration
   - Authentication failures
   - Filesystem errors

### Retry Strategy

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
)
def scrape_with_retry(self, url: str) -> dict:
    """Scrape with exponential backoff retry."""
    ...
```

## Security Considerations

### Anti-Detection Measures

1. **Browser Fingerprint**: Randomized user agents and viewport sizes
2. **Request Timing**: Human-like delays between requests
3. **Session Management**: Cookie persistence across requests
4. **Cloudflare Bypass**: Botasaurus bypass_cloudflare option

### Data Safety

1. **URL Validation**: Sanitize and validate all URLs before fetching
2. **File Type Verification**: Verify downloaded files match expected types
3. **No Credential Storage**: Credentials never stored in output files

## Testing Strategy

### Unit Tests

- `test_batch_controller.py`: Batch processing logic
- `test_media_extractor.py`: URL extraction patterns
- `test_media_validator.py`: Validation logic
- `test_downloader.py`: Download with mocked network

### Integration Tests

- `test_scraper_integration.py`: End-to-end scraping
- `test_batch_integration.py`: Full batch workflow

## Dependencies

| Dependency | Purpose | Version |
|------------|---------|---------|
| botasaurus | Browser automation with anti-detection | ^4.0 |
| aiohttp | Async HTTP client | ^3.9 |
| Pillow | Image validation | ^10.0 |
| pyyaml | YAML configuration | ^6.0 |

## Alternatives Considered

### Browser Automation

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| Selenium | Well-known, flexible | Heavy, easily detected | Not chosen |
| Playwright | Modern, fast | No anti-detection built-in | Not chosen |
| Botasaurus | Anti-detection, Cloudflare bypass | Newer project | **Chosen** |

Rationale: Botasaurus provides built-in anti-detection and Cloudflare bypass, critical for reliable scraping.

### Async Downloads

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| requests (sync) | Simple | Slow for multiple files | Not chosen |
| httpx | Modern, async | Extra dependency | Not chosen |
| aiohttp | Lightweight, async | More complex | **Chosen** |

Rationale: aiohttp provides efficient async downloads needed for batch media operations.
