# ContentEngineAI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-312/)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Development Status](https://img.shields.io/badge/status-pre--production-orange.svg)](docs/versioning.md)

**License**: MIT | **Status**: Pre-Production

**ContentEngineAI** is an AI-powered pipeline for generating short, vertical (9:16) promotional videos for e-commerce products. It automates the complete workflow from scraping product data to delivering final videos with AI-generated scripts, voiceovers, and content-aware subtitles.

## Key Features

- **End-to-End Automation**: Complete video production from scraping to publishing
- **Social Media Publishing**: Auto-scheduling with slot detection + cleanup after publish
- **Link-in-Bio Integration**: Auto-add affiliate links to bio page after publishing (Lnk.Bio)
- **Published Products Registry**: Track all published products in JSON/CSV format
- **Batch Processing**: Process hundreds of products with unified scrape + produce + publish pipeline
- **Platform-Specific Metadata**: AI-generated titles, captions, hashtags for YouTube, TikTok, Instagram
- **Content-Aware Subtitles**: Dynamic positioning with CTA-synchronized dual-line support
- **Premium Audio**: Google Chirp 3 HD voices with Whisper STT synchronization

## Quick Start

```bash
# 1. Setup (requires Python 3.12+, FFmpeg, Poetry)
git clone https://github.com/stkzlv/ContentEngineAI.git && cd ContentEngineAI
poetry install && poetry run playwright install
cp .env.example .env  # Configure API keys

# 2. Generate a video
poetry run python -m src.scraper.amazon.scraper --product-ids B0BTYCRJSS --debug
poetry run python -m src.video.producer outputs/B0BTYCRJSS/data.json slideshow_images1 --debug

# 3. Batch pipeline (scrape + produce + publish)
poetry run python -m src.pipeline.global_batch \
  --keywords "wireless earbuds" \
  --profile slideshow_images1 \
  --platforms youtube tiktok \
  --debug
```

See [Installation](docs/installation.md) for complete setup instructions.

## Documentation

| Guide | Description |
|-------|-------------|
| [Installation](docs/installation.md) | Setup guide with prerequisites and API keys |
| [Configuration](docs/configuration.md) | YAML config reference and CLI overrides |
| [Scraper](docs/scraper.md) | Product data extraction from Amazon |
| [Video Producer](docs/video-producer.md) | Video production CLI and profiles |
| [TTS Voice Profiles](docs/tts-voice-profiles.md) | Voice presets, providers, and style direction |
| [Pycaps Subtitles](docs/pycaps-subtitles.md) | Optional animated caption engine (TikTok/Reels style) |
| [Platform Safe Zones](docs/platform-safe-zones.md) | Subtitle safe zones for TikTok, YouTube Shorts, Reels |
| [Batch Processing](docs/batch-processing.md) | Multi-product pipelines and automation |
| [Publisher](docs/publisher.md) | Social media publishing via Late.dev |
| [Architecture](docs/architecture.md) | System design and module overview |
| [Troubleshooting](docs/troubleshooting.md) | Common issues and debugging tips |

<details>
<summary><strong>Developer Documentation</strong></summary>

| Document | Description |
|----------|-------------|
| [Development](docs/development.md) | Dev setup and contribution workflow |
| [Testing](docs/testing.md) | Test framework and coverage |
| [Linting](docs/linting.md) | Code quality tools (Ruff, MyPy, Bandit) |
| [Requirements](docs/requirements.md) | Project requirements and specs |
| [Subtitle Best Practices](docs/subtitle-best-practices.md) | Caption design research for TikTok/Shorts/Reels |
| [Versioning](docs/versioning.md) | Semantic versioning and releases |
| [Contributing](CONTRIBUTING.md) | How to contribute |

</details>

## License

**MIT License** - see [LICENSE](LICENSE) for details
