# ContentEngineAI Project Memory

## 1. Project Overview

**ContentEngineAI** is an automated, AI-powered video production pipeline designed for e-commerce platforms. It transforms product data into promotional videos by orchestrating a 7-step workflow: gathering visuals, generating scripts (LLM), creating voiceovers (TTS), producing subtitles (STT), downloading music, and assembling the final video (FFmpeg).

**Goal**: To provide a modular, extensible, and high-performance system for batch-producing vertical (9:16) social media content from e-commerce product pages (primarily Amazon).

## 2. System Design & Architecture

The project follows an **async-first, modular architecture** with parallel execution capabilities.

### Core Pipeline (`src/video/producer/`)
*   **Orchestrator**: Manages the dependency-aware execution of pipeline steps.
*   **Parallelism**: Steps like subtitle generation and music download run concurrently.
*   **Resume Capability**: Supports individual step execution for debugging.

### Key Components
*   **Scraper Module** (`src/scraper/`): Multi-platform architecture (base + Amazon implementation) using **Botasaurus** (Playwright) for stealth scraping.
*   **AI Integration** (`src/ai/`): Uses **OpenRouter** for scripts and descriptions; **Gemini TTS** with **Google Cloud TTS** as fallback for voiceovers; **OpenAI Whisper** for subtitles.
*   **Video Assembly** (`src/video/assembler/`): **FFmpeg**-based assembly with support for dynamic filter graphs, content-aware cropping, and multi-track audio mixing.
*   **Configuration**: Centralized YAML-based config (`config/`) backed by **Pydantic** models (`src/video/config/`, `src/scraper/config_models.py`) for validation.

### Data Flow
`Product Page (URL)` -> `Scraper` -> `Structured Data` -> `AI Processing (Script/TTS)` -> `Asset Gathering (Media/Music)` -> `FFmpeg Assembly` -> `Final Video (.mp4)`

## 3. Context & Dependencies

### Core Technologies
*   **Language**: Python 3.12+ (Type hints required)
*   **Dependency Management**: Poetry
*   **Video Processing**: FFmpeg (must be installed system-wide)
*   **Browser Automation**: Botasaurus / Playwright
*   **Testing**: Pytest (Asyncio, xdist for parallel execution)

### Critical Libraries
*   `pydantic`: Extensive configuration validation and data modeling.
*   `aiohttp`: Async network requests.
*   `ffmpeg-python`: Python bindings for FFmpeg.
*   `tenacity`: Retry logic for resilient API calls.
*   `botasaurus`: Anti-detection scraping framework.

## 4. Team Norms & Development Workflow

### GitHub Flow
1.  **Branching**: Create feature branches from `main` (`feature/name`, `bugfix/name`).
2.  **Commits**: Use **imperative mood** ("Add feature", not "Added").
    *   **CRITICAL**: Never mention Gemini, Claude Code, or AI tools in commit messages or PR descriptions.
3.  **PRs**: Squash merge to `main`. Ensure all CI checks pass (`make full-check`).

### Documentation
*   Update `GEMINI.md` (this file) if architectural patterns change.
*   Maintain `docs/` for user-facing documentation.
*   Adhere to **Semantic Versioning** (`docs/versioning.md`).

## 5. Coding Standards

*   **Style**: Adhere to `ruff` formatting (88 chars).
*   **Typing**: Modern Python typing (`dict[str, Any]`, `list[int]`, `val: int | None = None`).
*   **Naming**:
    *   Functions/Variables: `snake_case`
    *   Classes: `PascalCase`
    *   Constants: `UPPER_CASE`
*   **Error Handling**: specific exceptions (avoid bare `except Exception`), structured logging.
*   **Async**: Prefer `async/await` for all I/O bound operations.

## 6. Essential Commands

### Core Workflows
```bash
# Generate video for a single product (debug mode)
poetry run python -m src.scraper.amazon.scraper --keywords <ASIN> --debug --clean
poetry run python -m src.video.producer outputs/<ASIN>/data.json slideshow_images1 --debug

# Batch Scraping (Keywords with filters)
poetry run python -m src.scraper.amazon.scraper --keywords "wireless earbuds" --min-price 20 --min-rating 4.0 --debug

# Global Batch Pipeline (Scrape + Produce)
poetry run python -m src.pipeline.global_batch --keywords "smart watch" --max-products 5 --profile slideshow_images1 --debug
```

### Quality Assurance
```bash
make dev-setup     # Setup environment
make quick-check   # Fast feedback (Ruff + MyPy)
make lint          # Full linting suite
make test          # Run tests
make test-cov      # Run tests with coverage
make test-parallel # Run tests in parallel (xdist)
```

### Maintenance
```bash
# Performance Report
poetry run python tools/performance_report.py --report-type summary
```

## 7. Agent Maintenance & Verification

### 7.1 Verification Loop
Before executing complex tasks or when initializing context, the agent SHALL:
1.  **Audit Source**: List `src/` subdirectories to confirm architecture alignment.
2.  **Verify Deps**: Read `pyproject.toml` to check for new libraries or version shifts.
3.  **Command Check**: Read `Makefile` to see if development entry points have evolved.
4.  **Confirm Assumptions**: If a documented command fails, use `grep` or `find` to discover the correct entry point rather than assuming the docs are perfect.

### 7.2 Self-Update Instructions
This `GEMINI.md` is a **living document**. The agent SHOULD update it when:
*   **Structural Changes**: New top-level directories or modules are added to `src/`.
*   **New Workflow**: A new standard way of running the pipeline is established (e.g., a new `make` target).
*   **New Tech**: Significant new dependencies are added to `pyproject.toml`.
*   **Deviations**: If the codebase consistently deviates from the "Coding Standards" section, update the standards to reflect actual project reality.

**Process**: To update, read the full file, plan the delta, and rewrite the file with preserved hierarchy.

