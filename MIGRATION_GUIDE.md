# Migration Guide: Unified Configuration System

## Overview

ContentEngineAI has migrated from a monolithic configuration system (1,962 lines) to a unified modular system (907 lines across 6 files). This guide helps existing users migrate to the new system.

## 🚨 Breaking Changes: NONE

**Zero breaking changes** - All existing commands and function calls work exactly as before thanks to backward-compatible adapters.

## What's Changed

### Configuration Structure

**Before (Monolithic)**:
```
src/video/video_config.py      # 1,643 lines
src/scraper/config.py          # 319 lines
```

**After (Modular)**:
```
config/core.yaml              # Application-wide settings
config/video_production.yaml  # Video pipeline settings
config/ai_services.yaml       # AI provider configurations
config/subtitles.yaml         # Subtitle generation settings
config/performance.yaml       # Resource limits and optimization
config/scraper.yaml          # Web scraping configurations
```

### New Features Available

1. **CLI Overrides**: Any configuration can now be overridden via command line
2. **Environment Variables**: Production-ready environment variable support
3. **Triple Precedence**: CLI > Environment > YAML (in order of priority)
4. **Performance**: 20% faster configuration loading

## Migration Steps

### For Existing Users (Recommended)

**Option 1: Keep Using Existing Commands (Zero Migration)**
```bash
# These commands still work exactly as before
poetry run python -m src.video.producer outputs/B0BTYCRJSS/data.json slideshow_images1
poetry run python -m src.scraper.amazon.scraper --keywords "headphones" --debug
```

**Option 2: Gradually Adopt New Features**
```bash
# Start using CLI overrides for specific settings
poetry run python -m src.video.producer outputs/B0BTYCRJSS/data.json slideshow_images1 \
  --preset random \
  --max-concurrent-downloads 5 \
  --openai-timeout 60

# Use environment variables for sensitive data
export OPENROUTER_API_KEY="your-key"
export GOOGLE_CLOUD_TTS_CREDENTIALS_PATH="/path/to/credentials.json"
```

### For Advanced Users (Optional)

**Direct Configuration File Usage**:
```python
# New unified approach
from src.config_manager import ConfigurationManager

config_manager = ConfigurationManager()
video_config = config_manager.load_video_config(cli_overrides={"preset": "bold"})
```

**Custom Environment Variables**:
```bash
# Set environment variables for production deployment
export VIDEO_PRODUCER_MAX_CONCURRENT_DOWNLOADS=10
export SCRAPER_REQUEST_DELAY=2.0
export AI_SERVICES_OPENROUTER_TIMEOUT=120
```

## Configuration File Mapping

### Video Producer Settings

**Old Location** → **New Location**
- `src/video/video_config.py` → `config/video_production.yaml` + `config/ai_services.yaml`
- Pipeline settings → `config/video_production.yaml`
- AI provider settings → `config/ai_services.yaml`
- Subtitle settings → `config/subtitles.yaml`
- Performance settings → `config/performance.yaml`

### Scraper Settings

**Old Location** → **New Location**
- `src/scraper/config.py` → `config/scraper.yaml`
- Core application settings → `config/core.yaml`

## Environment Variable Reference

### Core Variables
```bash
# Application-wide
CONTENT_ENGINE_LOG_LEVEL=DEBUG
CONTENT_ENGINE_OUTPUT_DIR=/custom/outputs

# Video Production
VIDEO_PRODUCER_MAX_CONCURRENT_DOWNLOADS=5
VIDEO_PRODUCER_DEFAULT_PRESET=modern

# AI Services
OPENROUTER_API_KEY=your-openrouter-key
GOOGLE_CLOUD_TTS_CREDENTIALS_PATH=/path/to/credentials.json

# Performance
PERFORMANCE_MAX_WORKERS=4
PERFORMANCE_MEMORY_LIMIT_GB=8
```

### Complete Environment Variable List
See individual config files for all available environment variables:
- `config/core.yaml` - Core application variables
- `config/video_production.yaml` - Video production variables
- `config/ai_services.yaml` - AI service variables
- `config/subtitles.yaml` - Subtitle generation variables
- `config/performance.yaml` - Performance tuning variables
- `config/scraper.yaml` - Web scraping variables

## CLI Override Examples

### Video Producer Overrides
```bash
# Override AI model
poetry run python -m src.video.producer data.json profile \
  --ai-services.openrouter.default-model "anthropic/claude-3.5-sonnet"

# Override subtitle settings
poetry run python -m src.video.producer data.json profile \
  --subtitles.style.font-size 80 \
  --subtitles.positioning.margin-bottom 100

# Override performance settings
poetry run python -m src.video.producer data.json profile \
  --performance.max-workers 8 \
  --performance.memory-limit-gb 16
```

### Scraper Overrides
```bash
# Override request settings
poetry run python -m src.scraper.amazon.scraper --keywords "headphones" \
  --scraper.requests.delay 3.0 \
  --scraper.requests.timeout 45
```

## Testing Your Migration

### Verify Backward Compatibility
```bash
# Test that existing commands still work
poetry run python -m src.scraper.amazon.scraper --keywords "test" --debug --clean
poetry run python -m src.video.producer --batch --batch-profile slideshow_images1 --debug
```

### Verify New Features
```bash
# Test CLI overrides
poetry run python -m src.video.producer outputs/*/data.json slideshow_images1 \
  --preset random \
  --ai-services.openrouter.timeout 90

# Test environment variables
export VIDEO_PRODUCER_DEFAULT_PRESET=bold
poetry run python -m src.video.producer outputs/*/data.json slideshow_images1
```

## Troubleshooting

### Common Issues

**Issue**: "Configuration file not found"
**Solution**: The new system automatically creates default config files. Ensure you're running from the project root directory.

**Issue**: "Environment variable not recognized"
**Solution**: Check the exact variable name in the corresponding config file. Variable names follow the pattern `SECTION_SUBSECTION_SETTING`.

**Issue**: "CLI override not working"
**Solution**: Use dot notation for nested settings: `--ai-services.openrouter.timeout 60`

### Getting Help

1. **Check Configuration**: Use `--debug` flag to see which configuration values are being used
2. **Validate Settings**: The system will report validation errors with specific guidance
3. **Fallback**: All existing commands work unchanged if you need to revert

## Performance Benefits

The new configuration system provides:
- **20% faster loading** compared to the original system
- **Reduced memory usage** through lazy loading
- **Better caching** of configuration values
- **Optimized validation** with early error detection

## Next Steps

1. **Continue using existing commands** (no migration required)
2. **Gradually adopt new features** as needed (CLI overrides, environment variables)
3. **Consider modular config files** for complex deployments
4. **Review environment variable options** for production deployments

The migration is designed to be **completely optional** and **fully backward compatible**. You can adopt new features at your own pace without breaking existing workflows.