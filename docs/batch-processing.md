# Batch Processing Guide

Complete guide to batch processing modes in ContentEngineAI for efficient multi-product workflows.

## Overview

ContentEngineAI supports three batch processing modes:

1. **[Scraper Batch Mode](#scraper-batch-mode)** - Scrape multiple products
2. **[Producer Batch Mode](#producer-batch-mode)** - Generate videos for all scraped products
3. **[Global Batch Pipeline](#global-batch-pipeline)** - End-to-end automation (scrape + produce + publish)

---

## Scraper Batch Mode

Scrape multiple products efficiently using product IDs, keyword search, or both.

### Product ID Lists

Scrape specific products by ASIN:

```bash
# CLI: Multiple product IDs
poetry run python -m src.scraper.amazon.scraper \
  --product-ids B0BTYCRJSS B0D6GZF3T4 B0CTTZJRL6 \
  --debug
```

**YAML Configuration** (`config/scraper.yaml`):
```yaml
batch:
  product_ids:
    - B0BTYCRJSS
    - B0D6GZF3T4
    - B0CTTZJRL6
```

### Keyword Search

Find products by search terms with optional filters:

```bash
# CLI: Multiple keywords with filters
poetry run python -m src.scraper.amazon.scraper \
  --keywords "wireless earbuds" "bluetooth headphones" \
  --min-price 20 --max-price 100 \
  --min-rating 4.0 \
  --prime-only \
  --debug
```

**YAML Configuration**:
```yaml
batch:
  keywords:
    - "wireless earbuds"
    - "bluetooth headphones"
  max_products: 10          # Global cap across all keywords
  products_per_keyword: 2   # Limit per individual keyword

scrapers:
  amazon:
    search_filters:
      min_price: 20.0
      max_price: 100.0
      min_rating: 4.0
      prime_only: true
```

### Mixed Mode

Combine product IDs and keywords in a single batch:

```bash
poetry run python -m src.scraper.amazon.scraper \
  --product-ids B0BTYCRJSS \
  --keywords "wireless earbuds" \
  --debug
```

### Configuration Precedence

Settings are applied in this order (highest to lowest priority):
1. **CLI arguments** - only when **explicitly provided**
2. **YAML configuration** (config files)
3. **Default values** (built-in defaults)

**Note**: CLI arguments only override YAML values when explicitly provided. Omitting a flag uses the YAML value.

### Error Handling

**Fail-Fast Mode**:
```bash
# Stop on first error
poetry run python -m src.scraper.amazon.scraper \
  --product-ids B0TEST1 B0TEST2 \
  --fail-fast \
  --debug
```

**Graceful Continuation** (default):
- Invalid ASINs are skipped with warnings
- Scraping errors are logged and reported in summary
- Duplicate products (by ASIN) are automatically removed

### Batch Summary

After completion, view detailed statistics:

```
================================================================================
BATCH SCRAPING SUMMARY
================================================================================
Total Attempted: 3
  - Product IDs: 2
  - Keywords: 1
Successful: 3
Failed: 0

Media Collection Statistics:
  - total_images: 42
  - total_videos: 6
  - avg_images_per_product: 14.0
  - avg_videos_per_product: 2.0

Duration: 45.32 seconds
================================================================================
```

---

## Producer Batch Mode

Generate videos for all scraped products with flexible profile selection.

### Fixed Profile Mode

Use the same video profile for all products:

```bash
poetry run python -m src.video.producer \
  --batch \
  --batch-profile slideshow_images1 \
  --debug
```

**YAML Configuration** (`config/video_production.yaml`):
```yaml
batch:
  profile: slideshow_images1
```

### Random Profile Selection

Assign profiles deterministically per product for variety:

```bash
# Random selection from all available profiles
poetry run python -m src.video.producer \
  --batch \
  --random-profile \
  --debug

# Random selection from specific profile pool
poetry run python -m src.video.producer \
  --batch \
  --random-profile \
  --profile-pool slideshow_images1 video_sequential mixed_media \
  --debug
```

**YAML Configuration**:
```yaml
batch:
  random_profile: true
  profile_pool:
    - slideshow_images1
    - video_sequential
    - mixed_media
    # Empty list defaults to all available profiles
```

### Profile Randomization Features

- **Deterministic Assignment**: Same product ID always receives the same profile (reproducible builds)
- **Configuration Precedence**: CLI `--profile-pool` > YAML `profile_pool` > All available profiles
- **Mutual Exclusivity**: Cannot use both `--batch-profile` and `--random-profile` simultaneously
- **Distribution Tracking**: Summary displays profile usage statistics

### Batch Summary

Example output after batch completion:

```
================================================================================
BATCH PRODUCTION SUMMARY
================================================================================
Products Processed: 15
  - Successful: 14
  - Skipped: 1 (insufficient media)
  - Failed: 0

Profile Distribution:
  - slideshow_images1: 5 (35.7%)
  - video_sequential: 4 (28.6%)
  - mixed_media: 3 (21.4%)
  - slideshow_images2: 2 (14.3%)

Total Duration: 42.5 seconds
================================================================================
```

---

## Global Batch Pipeline

End-to-end automation combining scraping, video production, and publishing in a single unified command.

### Pipeline Architecture

The global batch pipeline orchestrates four phases:

1. **Scraping Phase** - Acquire product data from specified sources (product IDs, keywords)
2. **Handoff Phase** - Discover scraped products and filter by media availability
3. **Production Phase** - Generate videos using configured profile settings
4. **Publishing Phase** - Upload and publish videos to social media platforms (optional)

### Usage Examples

#### Product IDs Only

```bash
poetry run python -m src.pipeline.global_batch \
  --product-ids B0BTYCRJSS B0D6GZF3T4 \
  --profile slideshow_images1 \
  --debug
```

#### Keywords Only

```bash
poetry run python -m src.pipeline.global_batch \
  --keywords "wireless earbuds" "bluetooth speaker" \
  --max-products 5 \
  --profile video_sequential \
  --debug
```

#### Mixed Input with Filters

```bash
poetry run python -m src.pipeline.global_batch \
  --product-ids B0BTYCRJSS \
  --keywords "smart watch" \
  --min-price 20 --max-price 100 \
  --min-rating 4.0 \
  --profile product_video_hybrid \
  --debug
```

#### Random Profile Mode

```bash
poetry run python -m src.pipeline.global_batch \
  --keywords "wireless earbuds" \
  --random-profile \
  --profile-pool slideshow_images1 video_sequential mixed_media \
  --debug
```

#### Process All Existing Products

Process all products already in the outputs directory (skip scraping phase):

```bash
poetry run python -m src.pipeline.global_batch \
  --process-all-products \
  --profile slideshow_images1 \
  --debug
```

#### With Social Media Publishing

```bash
# Publish to YouTube and TikTok with auto-scheduling (default)
poetry run python -m src.pipeline.global_batch \
  --keywords "wireless earbuds" \
  --profile slideshow_images1 \
  --platforms youtube tiktok \
  --debug

# Skip publishing (video production only)
poetry run python -m src.pipeline.global_batch \
  --keywords "wireless earbuds" \
  --profile slideshow_images1 \
  --skip-publish \
  --debug

# Explicit schedule time
poetry run python -m src.pipeline.global_batch \
  --product-ids B0BTYCRJSS \
  --profile slideshow_images1 \
  --platforms youtube tiktok instagram \
  --schedule-time "2025-01-20T10:00:00+00:00" \
  --debug
```

**Publishing Behavior**:
- **Auto-Scheduling** (default): Finds first available unoccupied slot in `config/publisher.yaml` recurring schedule
- **Explicit Scheduling**: Use `--schedule-time` with ISO 8601 format to override auto-scheduling
- **Cleanup**: Removes product directories after successful multi-platform publish (configurable in `config/publisher.yaml`)

### YAML Configuration

Define persistent pipeline settings in `config/pipeline.yaml`:

```yaml
global_batch:
  # Input Configuration
  product_ids:
    - B0BTYCRJSS
    - B0D6GZF3T4
  keywords:
    - "wireless earbuds"

  # Product Limits (Two-Tier System)
  max_products: 10        # Global cap across all keywords
  products_per_keyword: 2 # Limit per individual keyword

  # Scraper Filters
  scraper_filters:
    min_price: 20.0
    max_price: 100.0
    min_rating: 4.0
    prime_only: true

  # Video Production Settings
  profile: slideshow_images1  # Fixed profile mode
  # OR
  random_profile: true        # Random profile mode
  profile_pool:
    - slideshow_images1
    - video_sequential

  # Error Handling
  fail_fast: false  # Continue on errors (default)

  # Common Settings
  outputs_dir: outputs
  debug: false
```

**Note**: Publishing options (`--skip-publish`, `--platforms`, `--schedule-time`, `--fail-fast-publish`) are CLI-only and not supported in YAML configuration.

**Publishing Configuration**: Publishing behavior is controlled by `config/publisher.yaml` (see [Publisher](publisher.md) for details):
- `immediate_publish: false` enables auto-scheduling
- `recurring_schedule.slots` defines available time slots
- `cleanup.enabled: true` removes product directories after successful publish

### Configuration Precedence

Settings are merged with this priority order:
1. **CLI arguments** (highest priority) - only when **explicitly provided**
2. **YAML configuration** (`config/pipeline.yaml`)
3. **Default values** (lowest priority)

**Important**: CLI arguments only override YAML values when explicitly provided by the user. Omitting a CLI flag uses the YAML value, not a hardcoded default.

### Product Limits

The pipeline uses a **two-tier limit system** for keyword searches:

- **`max_products`** (default: 10): Global cap on total products collected
- **`products_per_keyword`** (default: 2): Maximum products per individual keyword

**Behavior**:
- Processing iterates through keywords, collecting up to `products_per_keyword` from each
- Stops immediately when `max_products` is reached, even if keywords remain
- Product IDs count toward `max_products` but are not limited by `products_per_keyword`

**Example**: With `max_products: 10` and `products_per_keyword: 2` across 6 keywords:
- Keywords 1-5: 2 products each = 10 total (global cap reached)
- Keyword 6: Skipped (global cap already reached)

### Error Handling

**Fail-Fast Mode**:
```bash
# Stop immediately on any failure (scraping OR production)
poetry run python -m src.pipeline.global_batch \
  --keywords "product" \
  --profile slideshow_images1 \
  --fail-fast \
  --debug
```

**Graceful Continuation** (default):
- Pipeline continues processing remaining products after failures
- All failures are tracked and reported in final summary
- Partial success scenarios are clearly identified

### Pipeline Summary

Comprehensive end-to-end statistics:

```
================================================================================
GLOBAL PIPELINE SUMMARY
================================================================================

SCRAPING PHASE:
  Total Attempted: 3
  Successful: 3
  Failed: 0

  Media Statistics:
    - Total Images: 42
    - Total Videos: 6
  Duration: 25.4s

VIDEO PRODUCTION PHASE:
  Total Attempted: 3
  Successful: 3
  Failed: 0
  Skipped: 0

  Profile Distribution:
    - slideshow_images1: 2 (66.7%)
    - video_sequential: 1 (33.3%)
  Duration: 87.6s

PUBLISHING PHASE:
  Total Attempted: 3
  Successful: 3
  Failed: 0
  Skipped: 0

  Platform Results:
    - youtube: 3 successful, 0 failed
    - tiktok: 3 successful, 0 failed
    - instagram: 3 successful, 0 failed
  Duration: 45.2s

END-TO-END RESULTS:
  Complete Success (scraped + produced + published): 3
  Partial Success (scraped/produced only): 0
  Total Failures: 0

Total Pipeline Duration: 158.2s
================================================================================
```

### Key Features

- **Unified Workflow**: Single command for complete scrape-to-publish pipeline
- **Four-Phase Orchestration**: Automatic coordination across scraping, handoff, production, and publishing
- **Flexible Input Sources**: Support for product IDs, keywords, or both simultaneously
- **Smart Filtering**: Handoff phase validates products have sufficient media before production
- **Profile Management**: Fixed profile or deterministic random selection per product
- **Resume Capability**: Continue interrupted pipelines from last checkpoint with `--resume` flag
- **Parallel Publishing**: Concurrent uploads to multiple platforms per video for faster publishing
- **Dry-Run Mode**: Preview pipeline plan without executing with `--dry-run` flag
- **JSON Output**: Machine-readable summaries with `--output-format json` for automation
- **Webhook Notifications**: Non-blocking POST to configured URL on phase/pipeline events
- **Auto-Scheduling**: Finds first available unoccupied slot in recurring schedule by querying Late.co API
- **Smart Cleanup**: Removes product directories after successful multi-platform publish
- **Comprehensive Reporting**: Detailed phase-by-phase statistics with end-to-end metrics
- **Error Resilience**: Graceful failure handling with optional fail-fast mode per phase

---

## Best Practices

### Production Workflows

1. **Preview with Dry-Run**
   ```bash
   # Validate configuration and see planned actions
   poetry run python -m src.pipeline.global_batch \
     --keywords "wireless earbuds" \
     --profile slideshow_images1 \
     --platforms youtube tiktok \
     --dry-run
   ```

2. **Test with Small Batches**
   ```bash
   # Test with 2-3 products first
   poetry run python -m src.pipeline.global_batch \
     --keywords "test product" \
     --max-products 2 \
     --profile slideshow_images1 \
     --debug
   ```

3. **Use YAML for Reproducible Runs**
   - Store configuration in `config/pipeline.yaml`
   - Version control your configuration files
   - Override specific settings with CLI arguments

4. **Monitor Resource Usage**
   - Video production is CPU/memory intensive
   - Consider processing large batches in stages
   - Use `--debug` flag for detailed logging

5. **Profile Selection Strategy**
   - **Fixed profile**: Consistent branding across all products
   - **Random profiles**: Content variety for social media feeds
   - **Profile pool**: Balance between variety and brand consistency

### Error Recovery

If a batch fails midway:

1. **Use `--resume` flag** - Continue from the last successful phase:
   ```bash
   poetry run python -m src.pipeline.global_batch --resume
   ```
2. **Check outputs directory** - Successfully processed products are saved
3. **Review error logs** - Identify specific failures in `outputs/logs/global_pipeline.log`
4. **Use fail-fast for debugging** - Isolate issues quickly

The pipeline automatically saves state to `outputs/.pipeline_state.json` after each phase. On successful completion, the state file is cleared.

### Performance Optimization

- **Parallel processing**: Scraper uses async I/O for multiple products
- **Caching**: Media downloads are cached to avoid re-fetching
- **Resource limits**: Set `max_products` to control batch size

---

## Troubleshooting

### Common Issues

**Issue**: "No products found for batch processing"
- **Cause**: Handoff phase filters products without sufficient media
- **Solution**: Check `outputs/` directory for `data.json` files and verify media availability

**Issue**: "Profile not found"
- **Cause**: Invalid profile name specified
- **Solution**: Check available profiles in `config/video_production.yaml` under `video_profiles:` section

**Issue**: "Insufficient media for production"
- **Cause**: Product has no images or videos after scraping
- **Solution**: Verify product ASIN is valid and media is available on Amazon

**Issue**: "Configuration file not found"
- **Cause**: YAML configuration path doesn't exist
- **Solution**: Create configuration file or verify path in command

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
poetry run python -m src.pipeline.global_batch \
  --keywords "product" \
  --profile slideshow_images1 \
  --debug  # Detailed logs for each phase
```

---

## Related Documentation

- **[Configuration](configuration.md)** - Complete configuration reference
- **[Architecture](architecture.md)** - Technical architecture details
- **[Troubleshooting](troubleshooting.md)** - Additional debugging guidance
