# ContentEngineAI Project Memory

## Project Overview

ContentEngineAI is an AI-powered video production pipeline for e-commerce platforms.

## Essential Commands

```bash
# Core workflow
poetry run python -m src.scraper.amazon.scraper --keywords <ASIN> --debug --clean
poetry run python -m src.video.producer outputs/<ASIN>/data.json slideshow_images --debug

# Batch processing
poetry run python -m src.video.producer --batch --batch-profile slideshow_images --debug

# Advanced scraper usage
poetry run python -m src.scraper.amazon.scraper --keywords "product" --min-price 15 --max-price 100 --min-rating 4 --debug --clean

# Performance monitoring
poetry run python tools/performance_report.py --report-type summary
```

## Code Standards

- **Naming**: snake_case functions, PascalCase classes, UPPER_CASE constants
- **Type Annotations**: Use modern Python typing (`dict[str, Any]`, `| None`)
- **Error Handling**: Specific exceptions, structured logging
- **Configuration**: Centralized in `src/video/video_config.py`

## Development Guidelines

- Use Poetry for dependency management
- Use imperative commit messages (e.g., "Add subtitle generation")
- **Never mention Claude Code or AI tools in git commit messages**
- Save project status to STATUS.md (not this file)
- Create implementation plans for features/fixes before coding
- Test scraper with: `poetry run python -m src.scraper.amazon.scraper --debug --clean`
