# Project Status

This document tracks the current state of ContentEngineAI, including ongoing migrations, version-specific information, and temporary project conditions that may change over time.

**Last Updated**: September 21, 2025
**Current Version**: 0.3.0 (Font/color randomization system and comprehensive code quality improvements)

## 🚀 Current Release Status

ContentEngineAI v0.3.0 introduces a comprehensive font and color randomization system with deterministic seeding, new font management capabilities, and significant code quality improvements across the subtitle generation pipeline.

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
- ✅ **AI Video Descriptions**: Generate social media descriptions with hashtags and compliance features
- ✅ **Font and Color Randomization**: Deterministic randomization system with product-specific fonts and colors

## 📁 Current Output Structure

```
outputs/
├── {product_id}/           # Each product directory
│   ├── data.json          # Scraped product data
│   ├── script.txt         # Generated script
│   ├── description.txt    # AI-generated social media description
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

- **Total Tests**: 423 comprehensive test cases across all modules (27 test files)
- **Test Status**: Currently 39.03% coverage (exceeds 30% target)
- **Quality Gates**: Ruff, MyPy, Bandit, Vulture, Safety all passing
- **Test Status**: All tests passing (413 passed, 10 skipped)
- **Recent Updates**: Updated test documentation, verified all tests against current codebase

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

## 🔄 Recent Changes

### Latest Updates - Font/Color Randomization and Test Suite Updates
- **FIXED**: Font and color randomization system now working correctly across all products
- **UPDATED**: Test suite documentation and verification against current codebase
- **IMPROVED**: Test coverage from 20.16% to 39.03% (exceeds 30% target)
- **VERIFIED**: All 423 tests passing with proper randomization functionality
- **ADDED**: Missing test files to documentation (test_result_types.py, test_unified_subtitle_generator.py, test_subtitle_positioning.py)
- **CONFIRMED**: No outdated tests found - all tests valid for current codebase

### v0.2.1 Release - Critical Path Fixes
- Added AI-generated video descriptions for social media platforms
- New `description.txt` output file with optimized content for TikTok, YouTube, Instagram
- Automatic #ad hashtag compliance for advertising disclosure
- **FIXED**: Description generator path resolution issue that was causing pipeline failures
- **FIXED**: Producer now works correctly from any working directory
- Full backward compatibility with existing configurations