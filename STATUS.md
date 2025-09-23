# Project Status

**Version**: 0.3.1

## 🚀 Current Release

Optimized style preset system with 4 presets and font/color randomization.

## ✅ Core Features

- End-to-End Pipeline (scraper → producer)
- Style Presets (minimal, modern, bold, random)
- Content-Aware Subtitles with audio sync
- Batch Processing with error handling
- Amazon Scraping with complete product data
- AI Video Descriptions for social media

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

## 🛠️ Quick Commands

```bash
# Scrape + generate video
poetry run python -m src.scraper.amazon.scraper --keywords <ASIN> --debug --clean
poetry run python -m src.video.producer outputs/<PRODUCT_ID>/data.json slideshow_images1 --debug

# Batch processing
poetry run python -m src.video.producer --batch --batch-profile slideshow_images1 --debug
```

## 📊 Current Status

- **Tests**: 424 cases (39.19% coverage) - All passing
- **Quality**: Ruff, MyPy, Bandit, Vulture, Safety all passing
- **Platform**: Amazon (extensible architecture)
- **Performance**: 2-5 minutes per video
- **APIs**: Requires Pexels, OpenRouter, Freesound, Google Cloud keys

## 🔄 v0.3.1 Changes

- Reduced presets from 5 to 4 (minimal, modern, bold, random)
- Added RANDOM preset with deterministic font/color randomization
- Limited effects to 1 per preset to prevent clutter
- Added CLI --preset argument support
- Fixed ASS effects application and random effect selection
- Updated comprehensive test suite (424 tests passing)
