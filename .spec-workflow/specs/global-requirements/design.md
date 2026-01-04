# Design Document: Global Requirements

## Overview

This design documents the cross-cutting infrastructure for configuration management, error handling, logging, and resilience patterns used across all ContentEngineAI modules. The implementation follows industry best practices for Python applications and enables consistent behavior across scraping, video production, batch processing, and publishing phases.

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Application Layer                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Scraper   │  │  Producer   │  │  Publisher  │  │   Pipeline  │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
└─────────┼────────────────┼────────────────┼────────────────┼───────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Global Infrastructure Layer                         │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    UnifiedConfigManager                           │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐                  │  │
│  │  │ CLI Args   │─▶│ Env Vars   │─▶│ YAML Files │  (precedence)    │  │
│  │  └────────────┘  └────────────┘  └────────────┘                  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │
│  │  LoggingSetup   │  │  CircuitBreaker │  │  ProgressTracker        │ │
│  │  - Debug mode   │  │  - CLOSED state │  │  - [N/total] format     │ │
│  │  - Dual output  │  │  - OPEN state   │  │  - Summary reports      │ │
│  │  - File/Console │  │  - HALF_OPEN    │  │  - Phase summaries      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Configuration Loading Flow:
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│ YAML Files │───▶│ Env Vars   │───▶│ CLI Args   │───▶│ Validated  │
│ (defaults) │    │ (override) │    │ (override) │    │ Config     │
└────────────┘    └────────────┘    └────────────┘    └────────────┘

Error Handling Flow:
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│ Operation  │───▶│ Try/Catch  │───▶│ Circuit    │───▶│ Log + Skip │
│ Attempt    │    │ Handler    │    │ Breaker    │    │ or Fail    │
└────────────┘    └────────────┘    └────────────┘    └────────────┘
```

## Detailed Design

### 1. UnifiedConfigManager

**Location**: `src/config_manager.py`

**Purpose**: Centralized configuration loading with three-tier precedence system.

**Class Structure**:
```python
class UnifiedConfigManager:
    """Manages configuration with CLI > ENV > YAML precedence."""

    def __init__(
        self,
        yaml_config: dict[str, Any] | None = None,
        cli_args: dict[str, Any] | None = None,
    ) -> None:
        self.yaml_config = yaml_config or {}
        self.cli_args = cli_args or {}
        self.final_config: dict[str, Any] = {}

    def apply_precedence_rules(self) -> dict[str, Any]:
        """Apply CLI > ENV > YAML precedence and return merged config."""
        ...

    def _apply_env_overrides(self, config: dict[str, Any]) -> dict[str, Any]:
        """Override config values with environment variables."""
        ...

    def _apply_cli_overrides(self, config: dict[str, Any]) -> dict[str, Any]:
        """Override config values with CLI arguments."""
        ...

    def _set_nested_value(
        self, config: dict, path: str, value: str, expected_type: type | None = None
    ) -> None:
        """Set nested config value with automatic type conversion."""
        ...
```

**Environment Variable Mappings**:
| Environment Variable | Config Path | Type |
|---------------------|-------------|------|
| DEBUG_MODE | video.debug_mode | bool |
| OPENROUTER_API_KEY | llm.api_key | str |
| LATE_API_KEY | publisher.api_key | str |
| TTS_ENABLED | tts.enabled | bool |
| VOICEOVER_PROVIDER | voiceover.provider | str |

**Type Conversion Rules**:
- `"true"/"false"` → `bool`
- Numeric strings → `int` or `float`
- Comma-separated → `list[str]`
- All others → `str`

### 2. Retry Logic

**Location**: `src/utils/retry.py`

**Purpose**: Retry transient failures with exponential backoff before circuit breaker trips.

**Library**: `tenacity`

**Decorator Signature**:
```python
from tenacity import (
    retry, stop_after_attempt, wait_exponential_jitter,
    retry_if_exception_type, before_sleep_log
)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=30),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def retry_network(func):
    """Decorator for network operations with retry logic."""
    ...
```

**Retryable Exceptions**:
- `requests.Timeout`, `requests.ConnectionError`
- `httpx.TimeoutException`, `httpx.ConnectError`
- HTTP 429 (Rate Limited), HTTP 503 (Service Unavailable)

**Non-Retryable** (fail immediately):
- HTTP 4xx (except 429) - client errors
- `AuthenticationError` - invalid credentials
- `ValidationError` - bad request data

**Integration Order**:
```
Circuit Breaker → Retry → Actual Call

@circuit_breaker
@retry_network
def call_external_api():
    ...
```

This order ensures:
1. Circuit breaker prevents retry storms when service is down
2. Retries handle transient failures within healthy service
3. Repeated failures trip the circuit breaker

### 3. CircuitBreaker

**Location**: `src/utils/circuit_breaker.py`

**Purpose**: Prevent cascading failures when external services are unavailable.

**State Machine**:
```
                  success
    ┌───────────────────────────┐
    │                           │
    ▼           failure         │
┌────────┐  (threshold) ┌──────────┐  timeout  ┌───────────┐
│ CLOSED │─────────────▶│   OPEN   │──────────▶│ HALF_OPEN │
└────────┘              └──────────┘           └───────────┘
    ▲                                               │
    │              success                          │
    └───────────────────────────────────────────────┘
                        │
                        │ failure
                        ▼
                   ┌──────────┐
                   │   OPEN   │
                   └──────────┘
```

**Class Structure**:
```python
class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Fast-fail mode
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> None: ...

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        ...

    async def call_async(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute async function with circuit breaker protection."""
        ...
```

**Pre-configured Breakers**:
| Breaker Name | Failure Threshold | Timeout | Usage |
|-------------|-------------------|---------|-------|
| google_stt_circuit_breaker | 3 | 60s | Text-to-speech |
| freesound_circuit_breaker | 3 | 30s | Background music |
| pexels_circuit_breaker | 3 | 30s | Stock media |
| openrouter_circuit_breaker | 5 | 60s | LLM API calls |

### 4. Logging Setup

**Location**: `src/utils/logging_setup.py`

**Purpose**: Centralized logging with debug mode and dual output.

**Function Signature**:
```python
def setup_debug_logging(
    log_file: Path,
    debug_mode: bool = False,
    verbose: bool = False,
    component_name: str = "ContentEngineAI",
) -> None:
    """Configure logging with console and file handlers."""
```

**Output Formats**:
| Mode | Console Format | File Format |
|------|---------------|-------------|
| Normal (INFO) | `%(levelname)s: %(message)s` | `%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s` |
| Debug | `%(levelname)s: %(name)s - %(message)s` | Same as above |
| Verbose | `%(asctime)s - %(levelname)s - %(message)s` | Same as above |

**Third-Party Logger Suppression**:
Loggers set to WARNING level: `numba`, `websocket`, `httpx`, `httpcore`, `google`, `asyncio`, `urllib3`, `selenium`

### 5. Progress Tracking

**Pattern**: Integrated into batch controllers and orchestrators.

**Format**: `[N/total] Action: identifier`

**Examples**:
```
[1/10] Processing product: B0ASIN123
[2/10] SKIPPED: Insufficient media for B0ASIN456
[3/10] SUCCESS: Video created for B0ASIN789
[4/10] FAILED: API error for B0ASINXYZ
```

### 6. Summary Reports

**Structure**:
```python
@dataclass
class PhaseSummary:
    phase_name: str
    total: int
    successful: int
    failed: int
    skipped: int
    duration_seconds: float
    failed_items: list[tuple[str, str]]  # (id, error_message)
```

**Output Format**:
```
═══════════════════════════════════════════════════════════════
                    PHASE SUMMARY: Scraping
═══════════════════════════════════════════════════════════════
Total:      10
Successful: 8
Failed:     1
Skipped:    1
Duration:   45.2s

Failed Items:
  - B0ASIN123: Network timeout
═══════════════════════════════════════════════════════════════
```

## File Structure

```
src/
├── config_manager.py           # UnifiedConfigManager
├── utils/
│   ├── logging_setup.py        # setup_debug_logging()
│   └── circuit_breaker.py      # CircuitBreaker, CircuitState
├── scraper/
│   └── amazon/
│       └── batch_controller.py # BatchController with fail-fast
├── pipeline/
│   ├── config.py               # GlobalBatchConfig, PhaseSummary
│   └── global_batch.py         # GlobalPipelineOrchestrator
└── video/
    └── producer/
        └── batch_producer.py   # BatchProducer with progress
```

## Integration Points

### Component Integration

| Component | Uses Config | Uses Logging | Uses Circuit Breaker |
|-----------|------------|--------------|---------------------|
| Scraper | ✓ scraper_filters | ✓ debug_mode | ✓ (network ops) |
| Producer | ✓ video settings | ✓ debug_mode | ✓ (TTS, stock media) |
| Publisher | ✓ platforms, schedule | ✓ debug_mode | ✓ (API calls) |
| Pipeline | ✓ all above | ✓ debug_mode | ✓ (orchestration) |

### CLI Flag Integration

| Flag | Environment Variable | Effect |
|------|---------------------|--------|
| `--debug` | DEBUG_MODE=true | Enable DEBUG logging |
| `--fail-fast` | FAIL_FAST=true | Stop on first error |
| `--profile X` | VIDEO_PROFILE=X | Set video profile |
| `--clean` | CLEAN_MODE=true | Clean before run |

## Error Handling Strategy

### Error Categories

1. **Recoverable Errors** (log and continue):
   - Network timeouts
   - Missing optional media
   - Rate limiting (with backoff)

2. **Non-Recoverable Errors** (skip item):
   - Invalid product ID
   - Missing required data
   - Authentication failures

3. **Fatal Errors** (stop pipeline):
   - Invalid configuration
   - Missing required secrets
   - Filesystem errors

### Fail-Fast vs Graceful Degradation

```python
# Graceful degradation (default)
for product_id in product_ids:
    try:
        process_product(product_id)
        summary.successful += 1
    except ProductError as e:
        logger.error(f"[{i}/{total}] FAILED: {product_id} - {e}")
        summary.failed += 1
        summary.failed_items.append((product_id, str(e)))
        continue  # Process next item

# Fail-fast mode
for product_id in product_ids:
    try:
        process_product(product_id)
    except ProductError as e:
        logger.error(f"[{i}/{total}] FAILED: {product_id} - {e}")
        logger.error(f"Fail-fast enabled. {total - i} items pending.")
        raise  # Stop immediately
```

## Security Considerations

### Secrets Handling

1. **Storage**: All secrets in `.env` file only
2. **Loading**: Via `os.environ.get()` with fallback names
3. **Logging**: Masked in output (first/last 4 chars)
4. **Validation**: Check at startup, fail with clear message

### Secret Masking

```python
def mask_secret(value: str) -> str:
    """Mask secret for logging, showing first/last 4 chars."""
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}...{value[-4:]}"
```

## Testing Strategy

### Unit Tests

- `test_config_manager.py`: Precedence rules, type conversion
- `test_circuit_breaker.py`: State transitions, recovery
- `test_logging_setup.py`: Handler configuration, levels

### Integration Tests

- `test_global_batch_integration.py`: Full pipeline with config
- `test_batch_controller.py`: Fail-fast behavior

## Documentation Structure

### Required Root Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview, quick start guide, key features |
| `CONTRIBUTING.md` | Development workflow, coding standards, PR process |
| `CODE_OF_CONDUCT.md` | Community standards and expectations |
| `SECURITY.md` | Security policy and vulnerability reporting |
| `CHANGELOG.md` | Version history (Keep a Changelog format) |
| `LICENSE` | MIT license |

### Documentation Directory Structure

```
docs/
├── installation.md      # Setup and prerequisites
├── configuration.md     # Config reference (YAML, ENV, CLI)
├── development.md       # Architecture, debugging, contributing
├── troubleshooting.md   # Common issues and solutions
├── versioning.md        # Semantic versioning, release process
└── api/                 # API documentation (if applicable)
```

### Documentation Standards

1. **Format**: GitHub-Flavored Markdown (GFM)
2. **Language**: Plain language with defined technical terms
3. **Code Examples**: Working examples with context and expected output
4. **Internal Links**: Use relative paths
5. **Updates**: Documentation updated in same PR as code changes

### CHANGELOG Format (Keep a Changelog)

```markdown
## [Unreleased]

## [0.18.0] - 2025-01-04
### Added
- New feature description

### Changed
- Modified behavior description

### Fixed
- Bug fix description

### Deprecated
- Feature deprecation notice

### Removed
- Removed feature description

### Security
- Security fix description
```

## Outputs Directory Structure

### Directory Layout

```
outputs/
├── <product_id>/              # Per-product directories (e.g., B0ASIN123)
│   ├── data.json              # Required: Product data from scraper
│   ├── images/                # Product images
│   ├── videos/                # Product videos (source)
│   ├── music/                 # Background music (optional)
│   ├── temp/                  # Intermediate files during processing
│   ├── metadata.json          # Platform metadata (after production)
│   ├── video_<id>_<profile>.mp4  # Generated video files
│   └── <id>_media_validation_report.json  # Media validation
├── cache/                     # Global cache directory
│   └── botasaurus/            # Scraper browser cache
├── logs/                      # Log files
├── reports/                   # Performance and summary reports
├── temp/                      # Global temporary files
└── performance_history/       # Historical performance data
```

### Path Management Utilities

**Location**: `src/utils/outputs_paths.py`

| Function | Purpose |
|----------|---------|
| `get_outputs_root()` | Get root outputs directory |
| `get_product_directory(product_id)` | Get/create product directory |
| `get_product_images_directory(product_id)` | Get/create images subdirectory |
| `get_product_videos_directory(product_id)` | Get/create videos subdirectory |
| `get_cache_directory()` | Get global cache directory |
| `get_logs_directory()` | Get global logs directory |
| `get_reports_directory()` | Get global reports directory |
| `validate_outputs_structure()` | Validate directory structure |
| `cleanup_invalid_outputs()` | Remove invalid product directories |

### Validation Rules

- **Valid product directory**: Must have `data.json` and at least one media subdirectory (`images/` or `videos/`)
- **Valid product ID**: 8-15 alphanumeric characters (matches ASIN format)
- **Expected global directories**: `cache/`, `logs/`, `reports/`

## Dependencies

| Dependency | Purpose | Version |
|------------|---------|---------|
| pyyaml | YAML config loading | ^6.0 |
| python-dotenv | .env file loading | ^1.0 |
| tenacity | Retry with backoff | ^9.0 |
| (stdlib) | logging, os, dataclasses | Python 3.11+ |

## Alternatives Considered

### Configuration Management

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| Dynaconf | Full-featured, layered | Heavy dependency | Not chosen |
| Pydantic Settings | Type-safe | Requires pydantic | Not chosen |
| Custom UnifiedConfigManager | Lightweight, tailored | Manual maintenance | **Chosen** |

Rationale: Custom solution provides exactly the three-tier precedence needed without external dependencies.

### Circuit Breaker

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| pybreaker | Well-tested | External dependency | Not chosen |
| tenacity | Retry-focused | Not a circuit breaker | Not chosen |
| Custom CircuitBreaker | Tailored, async support | Manual maintenance | **Chosen** |

Rationale: Custom solution provides both sync and async support with pre-configured instances for each service.

### Retry Logic

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| tenacity | Full-featured, maintained | External dependency | **Chosen** |
| Custom retry | No dependencies | Manual backoff/jitter | Not chosen |
| urllib3 Retry | Built into requests | Limited to HTTP only | Not chosen |

Rationale: tenacity is the industry standard, supports async, and provides exponential backoff with jitter out of the box.
