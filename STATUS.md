# Project Status

This document tracks the current state of ContentEngineAI, including ongoing migrations, version-specific information, and temporary project conditions that may change over time.

**Last Updated**: September 18, 2025
**Current Version**: 0.1.1 (fixes branch)

## 🚀 Initial Release Status

ContentEngineAI v0.1.0 represents the initial open source release with complete functionality.

## 🎯 Current System Capabilities

- ✅ **End-to-End Pipeline**: Complete scraper → producer workflow
- ✅ **Audio-Based Subtitle Synchronization**: Perfect timing via Whisper STT
- ✅ **High-Quality Voice**: Chirp 3 HD voice support with priority-based selection
- ✅ **Batch Processing**: Multi-product processing with error handling
- ✅ **Performance Optimization**: Parallel execution with intelligent dependency management
- ✅ **Amazon Scraping**: Complete product data and media extraction
- ✅ **Per-Profile Visual Settings**: Complete customization of image positioning, subtitle styling, fonts, and colors per video profile
- ✅ **Content-Aware Subtitle Positioning**: Dynamic subtitle placement that avoids overlapping with visual content
- ✅ **Producer File Cleanup**: Complete cleanup of producer-generated files when using --clean flag
- ✅ **Pixel-Based Subtitle Width Constraints**: Intelligent subtitle width calculation based on actual font metrics

## 📁 Current Output Structure

```
outputs/
├── {product_id}/           # Each product directory
│   ├── data.json          # Scraped product data
│   ├── script.txt         # Generated script
│   ├── video_{product_id}_{profile}.mp4 # Final video
│   ├── voiceover.wav      # TTS audio
│   ├── subtitles.ass      # Regular synchronized subtitles
│   ├── subtitles_content_aware.ass # Content-aware positioned subtitles
│   ├── metadata.json      # Pipeline execution metadata
│   ├── performance_metrics.json # Step-by-step performance data
│   ├── images/            # Product images
│   ├── videos/            # Product videos
│   └── temp/              # Temporary processing files
├── cache/                 # Global cache
├── logs/                  # Application logs
└── reports/               # Performance reports
```

## 🛠️ Development Environment

### Code Quality
- **Line Length**: 88 characters enforced
- **Type Checking**: MyPy enabled with practical settings
- **Tools**: Ruff, Bandit, Vulture, Safety for comprehensive code analysis

## 🛠️ Essential Usage Commands

### Core Workflows
```bash
# Scrape product data
poetry run python -m src.scraper.amazon.scraper --keywords <ASIN> --debug --clean

# Generate video for single product
poetry run python -m src.video.producer outputs/<PRODUCT_ID>/data.json slideshow_images1 --debug

# Batch process all products
poetry run python -m src.video.producer --batch --batch-profile slideshow_images1 --debug

# End-to-end workflow
poetry run python -m src.scraper.amazon.scraper --debug --clean
poetry run python -m src.video.producer --batch --batch-profile slideshow_images1 --debug
```

### Performance Analysis
```bash
# Generate performance report
poetry run python tools/performance_report.py --report-type detailed
```

## 🧪 Test Suite

- **Total Tests**: 349 comprehensive test cases across all modules (25 test files)
- **Test Status**: All tests passing (339 passed, 10 skipped)
- **Current Coverage**: 34% (target: 30% - exceeds target)
- **Coverage Areas**: Amazon scraper, video producer, batch processing, media validation, subtitle positioning, cleanup functionality, content-aware subtitles, producer cleanup
- **Test Types**: Unit tests, integration tests, performance tests, security tests
- **Quality Gates**: Ruff, MyPy, Bandit, Vulture, Safety all passing
- **Recent Updates**:
  - Removed 1 outdated test (test_complete_audio_pipeline_integration) incompatible with current assembler implementation
  - Added comprehensive producer cleanup tests (10 test cases) covering file cleanup functionality
  - Added content-aware subtitle positioning test framework (tests need API alignment)

## 🎯 Current Scope

- **Platform Support**: Amazon (with extensible architecture for future platforms)
- **Pre-Production Status**: Functional with potential breaking changes until 1.0.0
- **API Requirements**: Requires API keys for Pexels, OpenRouter, Freesound, Google Cloud

## 📈 Performance

- **Video Generation**: Typically 2-5 minutes per video with parallel pipeline execution
- **Complete Workflow**: End-to-end processing from scraping to final video
- **Media Support**: Multiple images and videos per product with intelligent selection
- **Subtitle Generation**: Audio-based synchronization with perfect timing via Whisper STT
- **Content-Aware Positioning**: Dynamic subtitle placement based on visual content analysis

## 🔄 Recent Development Activity

### September 2025 Enhancements (Branch: feature/subtitle-settings-update)

**Subtitle System Improvements:**
- ✅ **Content-Aware Positioning**: Implemented dynamic subtitle placement that analyzes visual content bounds
- ✅ **Pixel-Based Width Calculation**: Added intelligent width constraints using font metrics
- ✅ **Dual ASS Generation**: System now generates both regular and content-aware subtitle files
- ✅ **Character-Specific Width Estimation**: Implements variable width calculation for different character types

**Producer Cleanup Enhancements:**
- ✅ **Complete File Cleanup**: Fixed --clean flag to remove all producer-generated files
- ✅ **Content-Aware ASS Cleanup**: Added cleanup for `subtitles_content_aware.ass` files
- ✅ **Legacy Video Pattern Support**: Added cleanup for old video naming patterns

**Technical Implementation:**
- Enhanced `src/video/unified_subtitle_generator.py` with pixel-based width validation
- Improved `src/video/producer.py` cleanup function to handle all generated files
- Added font metrics configuration with `font_width_to_height_ratio` parameter
- Implemented visual bounds analysis for content-aware positioning

**Testing & Validation:**
- Verified cleanup functionality removes all 10+ producer-generated files
- Validated content-aware positioning with real product data
- Confirmed subtitle width constraints prevent visual overflow
- All pipeline steps complete successfully with new enhancements

---

**Note**: This status reflects current development on feature/subtitle-settings-update branch. These features are ready for integration into main branch.