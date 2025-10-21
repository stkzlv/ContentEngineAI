# Project Structure

## Directory Organization

```
ContentEngineAI/
├── src/                           # Source code (modular architecture)
│   ├── video/                    # Video production pipeline (core orchestration)
│   ├── scraper/                  # Multi-platform e-commerce scrapers
│   ├── ai/                       # LLM integration (script, description)
│   ├── audio/                    # Music and audio processing
│   ├── utils/                    # Performance, caching, utilities
│   └── config_*.py               # Configuration management adapters
│
├── config/                        # Modular YAML configuration (6 files, 1,429 lines)
│   ├── core.yaml                 # Global settings (output paths, debug, timeouts)
│   ├── video_production.yaml     # Video pipeline (resolution, effects, profiles)
│   ├── ai_services.yaml          # AI providers (TTS, LLM, description)
│   ├── subtitles.yaml            # Subtitle system (positioning, styles, effects)
│   ├── performance.yaml          # Resource limits (memory, concurrency)
│   ├── scraper.yaml              # Web scraping (browser, timing, validation)
│   └── url_shortener.yaml        # URL shortening providers
│
├── tests/                         # Test suite (pytest with async support)
│   ├── test_*.py                 # Unit and integration tests (40 test files)
│   └── conftest.py               # Pytest fixtures and configuration
│
├── tools/                         # Utilities and development scripts
│   ├── performance_report.py     # Performance metrics analysis
│   ├── cleanup_outputs.py        # Output directory management
│   └── lint.py                   # Code quality automation
│
├── static/                        # Static assets (fonts, music)
│   ├── fonts/                    # TTF fonts (Montserrat, Poppins, Rubik, etc.)
│   └── *.mp3                     # Default background music tracks
│
├── typings/                       # Type stubs for untyped dependencies
│   ├── aiohttp/
│   ├── botasaurus/
│   ├── pysrt/
│   └── tenacity/
│
├── docs/                          # Documentation and migration guides
│   └── archive/                  # Historical documentation
│
├── outputs/                       # Generated artifacts (gitignored)
│   ├── <ASIN>/                   # Per-product output directories
│   │   ├── data.json             # Scraped product data
│   │   ├── images/               # Downloaded product images
│   │   ├── script.txt            # Generated promotional script
│   │   ├── voiceover.mp3         # TTS audio
│   │   ├── subtitles.srt         # Subtitle file
│   │   ├── music.mp3             # Background music
│   │   ├── video.mp4             # Final assembled video
│   │   └── attribution.txt       # Stock media attribution
│   └── performance_history.jsonl # Historical performance metrics
│
├── .spec-workflow/                # Spec workflow MCP (steering docs, approvals)
│   ├── steering/                 # Project steering documents
│   ├── templates/                # Document templates
│   └── approvals/                # Pending approval requests
│
└── [Root documentation files]     # 18 markdown files (README, ARCHITECTURE, etc.)
```

**Key Organizational Principles:**
- **Modular by Subsystem**: Video, scraper, AI, audio, utils are separate packages
- **Configuration Separation**: YAML configs isolated from code, hot-reloadable
- **Outputs Isolation**: All generated artifacts in `outputs/`, easy cleanup
- **Static Assets**: Fonts and default music centralized in `static/`
- **Type Safety**: Custom type stubs in `typings/` for untyped dependencies

## Naming Conventions

### Files

**Python Modules:**
- **Core Modules**: `snake_case` (e.g., `script_generator.py`, `unified_subtitle_generator.py`)
- **Adapters/Integrations**: `*_adapter.py`, `*_integration.py` (e.g., `config_adapter.py`)
- **Utilities**: `*_utils.py`, `*_client.py` (e.g., `subtitle_utils.py`, `freesound_client.py`)
- **Tests**: `test_*.py` (e.g., `test_assembler.py`, `test_unified_config_system.py`)

**Configuration:**
- **YAML Config**: `snake_case.yaml` (e.g., `ai_services.yaml`, `video_production.yaml`)
- **Environment**: `.env` (never committed)

**Documentation:**
- **Main Docs**: `UPPERCASE.md` (e.g., `README.md`, `ARCHITECTURE.md`, `CONFIGURATION.md`)
- **Migration Guides**: `MIGRATION_GUIDE_v{old}_to_v{new}.md`

### Code

**Classes/Types:**
- **PascalCase**: `ProductData`, `VideoConfig`, `PipelineContext`, `BaseScraper`
- **Exception Classes**: `*Error` suffix (e.g., `ValidationError`, `ConfigurationError`)
- **Dataclasses**: `@dataclass` decorator for immutable data structures

**Functions/Methods:**
- **snake_case**: `generate_script()`, `assemble_video()`, `download_music()`
- **Async Functions**: Prefix with `async def`, no naming distinction (e.g., `async def fetch_stock_media()`)
- **Private Methods**: Leading underscore `_method_name()` (e.g., `_validate_config()`)

**Constants:**
- **UPPER_SNAKE_CASE**: `DEFAULT_TIMEOUT`, `MAX_RETRIES`, `API_BASE_URL`
- **Type Aliases**: PascalCase (e.g., `PathLike = str | Path`)

**Variables:**
- **snake_case**: `product_data`, `voiceover_path`, `subtitle_config`
- **Private Attributes**: Leading underscore `_cache`, `_session`

## Import Patterns

### Import Order (Enforced by Ruff)

1. **Future imports**: `from __future__ import annotations`
2. **Standard library**: `import asyncio`, `from pathlib import Path`
3. **Third-party**: `import aiohttp`, `from pydantic import BaseModel`
4. **First-party**: `from src.video import assembler`, `from src.scraper.base import BaseScraper`
5. **Local/relative**: `from .utils import sanitize_script`

**Example:**
```python
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import aiohttp
from pydantic import BaseModel

from src.utils.caching import CacheManager
from src.video.video_config import VideoConfig

from .subtitle_utils import generate_subtitles
```

### Module/Package Organization

**Absolute Imports Preferred:**
- Use `from src.video.assembler import assemble_video` (absolute)
- Avoid `from ..video.assembler import assemble_video` (relative, use only within same package)

**Package `__init__.py` Files:**
- Minimal exports, re-export public API only
- Example: `src/scraper/__init__.py` exports `ScraperFactory`, `BaseScraper`, `Platform`
- Avoid importing entire submodules in `__init__.py` to prevent circular dependencies

**Dependency Management:**
- Poetry for Python packages (`pyproject.toml`, `poetry.lock`)
- External binaries: FFmpeg (system-installed), Playwright browsers (`playwright install`)

## Code Structure Patterns

### Module/Class Organization

**Standard File Structure:**
```python
# 1. Docstring (module purpose)
"""
Video assembly module for FFmpeg-based video production.
"""

# 2. Future imports
from __future__ import annotations

# 3. Imports (standard → third-party → first-party → local)
import asyncio
from pathlib import Path

import ffmpeg
from pydantic import BaseModel

from src.utils.performance import PerformanceMonitor

# 4. Constants and configuration
DEFAULT_TIMEOUT = 300
MAX_CONCURRENT_JOBS = 4

# 5. Type definitions
VideoPath = Path | str

# 6. Main implementation (classes, functions)
class VideoAssembler:
    def __init__(self, config: VideoConfig):
        self.config = config

    async def assemble(self, context: PipelineContext) -> Path:
        # Implementation
        pass

# 7. Helper/utility functions (private if possible)
def _validate_media_file(path: Path) -> bool:
    # Implementation
    pass

# 8. No explicit exports (__all__ not used unless necessary)
```

### Function/Method Organization

**Standard Function Pattern:**
```python
async def generate_subtitles(
    audio_path: Path,
    script: str,
    config: SubtitleConfig,
) -> Path:
    """
    Generate subtitle file from audio using STT.

    Args:
        audio_path: Path to voiceover audio file
        script: Original script text for fallback
        config: Subtitle configuration

    Returns:
        Path to generated SRT file

    Raises:
        SubtitleGenerationError: If STT fails and fallback unavailable
    """
    # 1. Input validation
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # 2. Core logic
    try:
        segments = await _transcribe_audio(audio_path, config.stt_provider)
    except Exception as e:
        # 3. Error handling with fallback
        logger.warning(f"STT failed: {e}, using script-based fallback")
        segments = _generate_segments_from_script(script)

    # 4. Final processing and return
    subtitle_path = _write_srt_file(segments, audio_path.parent)
    return subtitle_path
```

**Key Principles:**
1. Input validation early (fail fast)
2. Core logic with clear control flow
3. Error handling with fallbacks where appropriate
4. Single responsibility (one function does one thing well)

### File Organization Principles

**One Primary Class Per File:**
- `src/video/assembler.py` → `VideoAssembler` class
- `src/scraper/amazon/scraper.py` → `AmazonScraper` class
- Exception: Small related classes (e.g., `Result` types in `result_types.py`)

**Utility Modules:**
- Group related helper functions (e.g., `subtitle_utils.py`, `script_sanitizer.py`)
- No class required if purely functional

**Public API Clarity:**
- Public functions/classes at module level
- Private helpers prefixed with `_`
- Complex modules export via `__init__.py` (e.g., `src/scraper/__init__.py`)

## Code Organization Principles

1. **Single Responsibility**: Each module has one clear purpose
   - `assembler.py`: Video assembly only
   - `script_generator.py`: Script generation only
   - `tts.py`: Text-to-speech only

2. **Modularity**: Reusable components with clear interfaces
   - `BaseScraper` abstract class for multi-platform scrapers
   - `BaseURLShortener` for provider-agnostic URL shortening

3. **Testability**: Async functions with dependency injection
   - Pass `config` objects explicitly, not global state
   - Mock external APIs via `aioresponses` or `pytest-mock`

4. **Consistency**: Follow established patterns
   - All pipeline steps: `async def step_name(context: PipelineContext) -> None`
   - All providers: Fallback chains with retry logic
   - All config: Pydantic models with validation

## Module Boundaries

### Core vs Extensions

**Core Pipeline (`src/video/producer.py`):**
- Orchestrates 7-step workflow
- Manages PipelineContext state
- **Depends on**: `assembler`, `tts`, `stock_media`, `pipeline_graph`
- **Independent of**: Specific scraper implementation, URL shortener provider

**Scrapers (`src/scraper/`):**
- Platform-specific implementations (Amazon, eBay, Walmart)
- **Depends on**: `base/` abstract interfaces only
- **Independent of**: Video production pipeline (can be used standalone)

**AI Services (`src/ai/`):**
- LLM integrations for script and description generation
- **Depends on**: `utils/caching`, `utils/script_sanitizer`
- **Independent of**: Video assembly, scraping

### Public API vs Internal

**Public API (exposed via `__init__.py`):**
- `src/scraper/__init__.py`: `ScraperFactory`, `BaseScraper`, `Platform`
- `src/utils/url_shortener/__init__.py`: `URLShortenerFactory`, `BaseURLShortener`

**Internal Implementation:**
- `src/video/subtitle_positioning.py`: Content-aware positioning logic (not exported)
- `src/scraper/amazon/browser_functions.py`: Playwright automation (not exported)

### Platform-Specific vs Cross-Platform

**Cross-Platform Base (`src/scraper/base/`):**
- `BaseScraper`, `BaseProductData`, `Platform` enum
- Shared utilities: `browser_utils.py`, `downloader.py`, `utils.py`

**Platform-Specific (`src/scraper/amazon/`):**
- Amazon-specific models, config, browser logic
- 11 modules extending base architecture

### Stable vs Experimental

**Stable (v0.8.0):**
- Core pipeline (Steps 1-6)
- Amazon scraping
- OpenRouter LLM, Google Cloud TTS/STT
- Modular YAML configuration

**Pre-Production (not yet stable):**
- Multi-platform scraping (only Amazon implemented)
- URL shortening (PicSee only, Bitly/TinyURL planned)
- Content-aware subtitle positioning (new in v0.8.0, under refinement)

### Dependencies Direction

**Allowed:**
- `src/video/producer.py` → `src/video/assembler.py` → `src/utils/performance.py`
- `src/scraper/amazon/scraper.py` → `src/scraper/base/models.py`
- `src/ai/script_generator.py` → `src/utils/caching.py`

**Forbidden (circular):**
- `src/utils/*` should NOT import from `src/video/` or `src/scraper/`
- `src/scraper/base/` should NOT import from `src/scraper/amazon/`

## Code Size Guidelines

**File Size:**
- **Ideal**: <500 lines
- **Maximum**: 1,000 lines (refactor if exceeded)
- **Current exceptions**: `scraper.py` modules (~800 lines due to multi-step workflows)

**Function/Method Size:**
- **Ideal**: <50 lines
- **Maximum**: 100 lines (split into helpers if exceeded)
- **Exception**: FFmpeg filter graph construction (complex, well-documented)

**Class Complexity:**
- **Ideal**: <10 public methods
- **Cyclomatic Complexity**: Keep methods simple, avoid deep nesting

**Nesting Depth:**
- **Maximum**: 3 levels (if/for/with)
- **Prefer early returns** over deep nesting:
  ```python
  # Good
  if not condition:
      return None
  process_data()

  # Avoid
  if condition:
      process_data()
  ```

## Dashboard/Monitoring Structure

### Spec Workflow MCP Dashboard (External)

**Structure:**
```
.spec-workflow/
├── steering/              # Project steering documents (product, tech, structure)
├── specs/                 # Feature specifications (requirements, design, tasks)
├── templates/             # Document templates (auto-populated)
├── user-templates/        # Custom templates (optional)
└── approvals/             # Pending approval requests (transient, cleaned up)
```

**Separation of Concerns:**
- Dashboard runs independently (localhost:3000)
- No code dependencies in `src/` (MCP server external)
- File-system based state (`.spec-workflow/` directory)
- WebSocket for real-time updates

### Performance Monitoring (Built-in)

**CLI-Based:**
```bash
make perf-report                                    # Quick summary
poetry run python tools/performance_report.py       # Detailed analysis
```

**Data Storage:**
- `outputs/performance_history.jsonl`: Per-run metrics (JSONL format)
- In-memory during pipeline execution (`PerformanceMonitor` class)

**Isolation:**
- `src/utils/performance.py`: Metrics collection (no UI dependencies)
- `tools/performance_report.py`: Analysis and reporting (CLI tool)

## Documentation Standards

**Public APIs:**
- **Required**: Docstrings for all public classes, functions, methods
- **Format**: Google-style docstrings with Args, Returns, Raises sections
- **Type Hints**: Required for all function signatures

**Complex Logic:**
- **Inline comments**: Explain "why", not "what"
- **Example**: FFmpeg filter graph construction has step-by-step comments

**Module READMEs:**
- **Not used**: Prefer comprehensive `ARCHITECTURE.md` and code docstrings
- **Exception**: `docs/` directory for migration guides

**Language-Specific Conventions:**
- **PEP 257**: Docstring conventions (enforced by Ruff `D` rules)
- **PEP 484**: Type hints (checked by MyPy)
- **Ruff D-rules**: Partial enforcement (D100-D107 ignored for brevity)

**Documentation Files (18 total):**
- **User-facing**: README, INSTALL, CONFIGURATION, TROUBLESHOOTING
- **Developer-facing**: ARCHITECTURE, DEVELOPMENT, TESTING, LINTING, CONTRIBUTING
- **Project management**: CHANGELOG, VERSIONING, MIGRATION_GUIDE_*
- **AI prompts**: `src/ai/prompts/*.md` (LLM prompt templates)
