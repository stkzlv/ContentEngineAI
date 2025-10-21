# Product Overview

## Product Purpose

ContentEngineAI is an AI-powered video production pipeline that automates the creation of short, vertical (9:16) promotional videos for e-commerce products. It solves the time-intensive and technically complex problem of producing professional-quality social media content at scale, transforming product data into engaging videos ready for TikTok, Instagram Reels, and YouTube Shorts.

## Target Users

### Primary Users
**E-commerce Marketers & Content Creators** who need to produce high-volume social media content for product promotion.

**User Needs:**
- Rapid video production without video editing expertise
- Consistent, professional-quality output across multiple products
- Social media optimization (9:16 vertical format, engaging scripts, subtitles)
- Cost-effective alternative to hiring video editors or agencies

**Pain Points:**
- Manual video creation is slow and requires specialized skills
- Scaling content production for multiple products is labor-intensive
- Maintaining consistent quality and style across campaigns
- Integrating product data, voiceovers, subtitles, and music manually

## Key Features

1. **End-to-End Automation**: Complete pipeline from product scraping to final video delivery with minimal human intervention
2. **AI-Generated Scripts**: LLM-powered promotional scripts optimized for social media engagement
3. **Content-Aware Subtitles**: Dynamic subtitle positioning that intelligently avoids visual content overlap using pixel-based analysis
4. **Premium Audio Stack**: Google Chirp 3 HD voices with Whisper STT synchronization for perfect timing
5. **Style Presets**: 5 production-ready visual styles (minimal, modern, bold, animated, random) with configurable effects
6. **Multi-Provider Fallbacks**: OpenRouter, Google Cloud, and local models ensure reliability and uptime
7. **Batch Processing**: Process multiple products in sequence with configurable delays
8. **Production-Ready Config**: Modular YAML configuration with CLI overrides and environment variable support

## Business Objectives

- **Scale Content Production**: Enable users to generate 10-100x more video content with the same resources
- **Reduce Production Costs**: Eliminate need for video editors, voiceover artists, and manual subtitle creation
- **Accelerate Time-to-Market**: Generate first video in ~30 seconds, complete batches in minutes
- **Democratize Video Marketing**: Make professional video production accessible without technical expertise
- **Maintain Quality Standards**: Deliver consistent, platform-optimized content that meets social media best practices

## Success Metrics

- **Production Speed**: Generate first complete video in <60 seconds from product data
- **Quality Score**: >90% of generated videos require no manual editing
- **User Efficiency**: 10x reduction in time spent per video vs manual production
- **System Reliability**: >95% success rate on end-to-end pipeline execution
- **Cost Effectiveness**: <$0.50 per video in API costs (LLM, TTS, STT, stock media)

## Product Principles

1. **Automation Over Manual Intervention**: Every step should be fully automated with intelligent defaults; manual edits should be optional refinements, not requirements
2. **Quality Through Intelligence**: Use AI and content-aware algorithms to make smart decisions (subtitle positioning, script generation, timing) rather than relying on fixed templates
3. **Performance at Scale**: Design for batch processing and parallel execution; optimize for users generating hundreds of videos, not just one
4. **Modular Flexibility**: Each component (scraping, script generation, TTS, assembly) should be independently configurable and replaceable
5. **Fail Gracefully**: Implement multi-level fallbacks (provider switching, retry logic, graceful degradation) to ensure pipeline completion even with partial failures

## Monitoring & Visibility

- **Dashboard Type**: Web-based dashboard (localhost:3000) for workflow management and approval processes
- **Real-time Updates**: WebSocket-based real-time status updates for pipeline progress
- **Key Metrics Displayed**:
  - Pipeline execution progress (current step, completion percentage)
  - Performance metrics (step timings, resource usage, bottlenecks)
  - Error rates and fallback triggers
  - Historical trends and cross-session analysis
- **Sharing Capabilities**:
  - Performance reports (summary, detailed, trends)
  - Log files with structured debugging output
  - Attribution files for stock media licensing

**Monitoring Tools:**
```bash
make perf-report                                    # Quick performance summary
poetry run python tools/performance_report.py      # Detailed metrics
```

## Future Vision

### Potential Enhancements

- **Remote Access**:
  - Cloud deployment for multi-user teams
  - Tunnel features for sharing dashboards with stakeholders
  - Remote queue management for distributed processing

- **Analytics**:
  - A/B testing different script styles and video profiles
  - Historical performance tracking across campaigns
  - ROI metrics linking videos to conversion rates

- **Collaboration**:
  - Multi-user approval workflows for brand compliance
  - Commenting and feedback on generated videos
  - Team-based batch processing and queue management

- **Platform Expansion**:
  - Support for additional e-commerce platforms (eBay, Walmart, Shopify)
  - Multi-platform scraping in single pipeline run
  - Cross-platform product comparison videos

- **Advanced Customization**:
  - User-trained voice models for brand consistency
  - Custom visual effects and transition libraries
  - Industry-specific script templates (fashion, electronics, home goods)
  - Brand voice and tone customization for scripts
