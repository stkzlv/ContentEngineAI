# ContentEngineAI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-312/)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Development Status](https://img.shields.io/badge/status-pre--production-orange.svg)](VERSIONING.md)

**Version**: 0.14.0 | **License**: MIT | **Status**: Pre-Production

> **🚀 Latest Update (v0.14.0)**: Async I/O architecture, Pydantic configuration models, and type-safe scraper settings with improved performance.

**ContentEngineAI** is an AI-powered pipeline for generating short, vertical (9:16) promotional videos for e-commerce products. It automates the complete workflow from scraping product data to delivering final videos with AI-generated scripts, voiceovers, and content-aware subtitles.

## ✨ Key Features

- **🤖 End-to-End Automation**: Complete video production from scraping to final output
- **📱 Social Media Ready**: Vertical 9:16 format optimized for TikTok, Instagram, YouTube Shorts
- **🎥 Advanced Video Extraction**: M3U8/HLS support with strict quality filtering and deduplication
- **🎬 Product Video Modes**: Sequential assembly, gallery transitions, and image-only slideshows
- **🎯 Content-Aware Subtitles**: Dynamic positioning with CTA-synchronized dual-line support
- **🎤 Premium Audio**: Google Chirp 3 HD voices with Whisper STT synchronization
- **🎵 Stock Music**: Freesound.org integration with OAuth2 auth and local fallback
- **⚙️ Production-Ready Config**: Modular YAML + CLI overrides + environment variables

## 🚀 Quick Start

```bash
# 1. Setup (requires Python 3.12+, FFmpeg, Poetry)
git clone https://github.com/stkzlv/ContentEngineAI.git && cd ContentEngineAI
poetry install && poetry run playwright install
cp .env.example .env  # Configure API keys

# 2. Generate a video (30 seconds to first result)
poetry run python -m src.scraper.amazon.scraper --keywords "B0BTYCRJSS" --debug
poetry run python -m src.video.producer outputs/B0BTYCRJSS/data.json slideshow_images1 --preset random --debug

# 3. Batch process multiple products
poetry run python -m src.video.producer --batch --batch-profile slideshow_images1
```

**📖 Complete Guide**: [INSTALL.md](INSTALL.md) • **⚙️ Configuration**: [CONFIGURATION.md](CONFIGURATION.md)

## 🔄 Batch Processing

Process multiple products efficiently with batch mode support for both scraper and producer.

### Scraper Batch Mode

**Product ID Lists** - Scrape specific products by ASIN:
```bash
# CLI: Multiple product IDs
poetry run python -m src.scraper.amazon.scraper --product-ids B0BTYCRJSS B0D6GZF3T4 B0CTTZJRL6 --debug

# YAML Configuration (config/scraper.yaml)
batch:
  product_ids:
    - B0BTYCRJSS
    - B0D6GZF3T4
    - B0CTTZJRL6
```

**Keyword Search** - Find products by search terms with filters:
```bash
# CLI: Multiple keywords with filters
poetry run python -m src.scraper.amazon.scraper \
  --keywords "wireless earbuds" "bluetooth headphones" \
  --min-price 20 --max-price 100 --min-rating 4.0 --prime-only \
  --debug

# YAML Configuration
batch:
  keywords:
    - "wireless earbuds"
    - "bluetooth headphones"
scrapers:
  amazon:
    search_filters:
      min_price: 20.0
      max_price: 100.0
      min_rating: 4.0
      prime_only: true
```

**Mixed Mode** - Combine product IDs and keywords:
```bash
# Both sources in one batch
poetry run python -m src.scraper.amazon.scraper \
  --product-ids B0BTYCRJSS \
  --keywords "wireless earbuds" \
  --debug
```

**Configuration Precedence**: CLI arguments override YAML configuration, which overrides defaults.

**Error Handling**:
- `--fail-fast`: Stop on first error (default: continue processing)
- Invalid ASINs are skipped with warnings
- Duplicate products (by ASIN) are automatically removed

**Batch Summary** - View statistics after completion:
```
BATCH SCRAPING SUMMARY
Total Attempted: 3
  - Product IDs: 2
  - Keywords: 1
Successful: 3
Failed: 0
Media Collection Statistics:
  - total_images: 42
  - total_videos: 6
Duration: 45.32 seconds
```

### Producer Batch Mode

Process all scraped products automatically:
```bash
poetry run python -m src.video.producer --batch --batch-profile slideshow_images1 --debug
```

## 🏗️ Architecture

<details>
<summary>Pipeline Overview</summary>

ContentEngineAI follows a **7-step modular pipeline** with parallel execution:

```mermaid
graph TD
    A[Gather Visuals] --> B[Generate Script]
    B --> C[Generate Description]
    C --> D[Create Voiceover]
    D --> E[Generate Subtitles]
    D --> F[Download Music]
    E --> G[Assemble Video]
    F --> G
```

**📖 Detailed architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
</details>

## 📚 Documentation

| Guide | Description |
|-------|-------------|
| **[🛠️ INSTALL.md](INSTALL.md)** | Complete installation and setup guide |
| **[⚙️ CONFIGURATION.md](CONFIGURATION.md)** | Complete configuration reference and options |
| **[🏗️ ARCHITECTURE.md](ARCHITECTURE.md)** | Technical architecture and system design |
| **[🔧 TROUBLESHOOTING.md](TROUBLESHOOTING.md)** | Solutions for common issues and debugging |

<details>
<summary><strong>Developer Documentation</strong></summary>

| Document | Purpose |
|----------|---------|
| **[👨‍💻 DEVELOPMENT.md](DEVELOPMENT.md)** | Development setup and contribution workflow |
| **[🧪 TESTING.md](TESTING.md)** | Testing framework and quality assurance |
| **[✨ LINTING.md](LINTING.md)** | Code quality standards and tools |
| **[🤝 CONTRIBUTING.md](CONTRIBUTING.md)** | How to contribute code and documentation |

</details>

## 🛠️ Development & Contributing

<details>
<summary><strong>Developer Quick Start</strong></summary>

```bash
# Setup development environment
git clone https://github.com/stkzlv/ContentEngineAI.git && cd ContentEngineAI
poetry install --with dev && make install-dev

# Quality assurance commands
make lint      # Complete quality check (7 tools)
make test      # Full test suite with coverage
make security  # Security vulnerability scan
```

**📖 Complete guide**: [DEVELOPMENT.md](DEVELOPMENT.md) • [CONTRIBUTING.md](CONTRIBUTING.md)

</details>

---

## 📄 License

**MIT License** - see [LICENSE](LICENSE) for details

<div align="center">

**[🛠️ Installation](INSTALL.md)** • **[⚙️ Configuration](CONFIGURATION.md)** • **[🤝 Contributing](CONTRIBUTING.md)** • **[🐛 Issues](https://github.com/stkzlv/ContentEngineAI/issues)**

</div>
