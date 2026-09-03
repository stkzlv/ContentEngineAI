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
    value:
      - "wireless earbuds"
      - "bluetooth headphones"
    novelty:
      - "smart ring"
  # max_products is read from scrapers.amazon, not from this block
  products_per_keyword: 1   # Limit per individual keyword

scrapers:
  amazon:
    default_search_parameters:
      min_price: 20.0
      max_price: 100.0
      min_rating: 4.0
      prime_only: true
```

### File-Based Input

Read product IDs or URLs from a file (one per line):

```bash
# Batch from file with custom output directory
poetry run python -m src.scraper.amazon.scraper \
  --input-file products.txt \
  --output-dir tmp \
  --batch-size 10 \
  --debug
```

The `--input-file` entries are merged with any `--product-ids` provided on the command line, and duplicates are removed automatically.

### URL Scraping

Scrape from full or shortened Amazon URLs (e.g. tr.ee, amzn.to):

```bash
poetry run python -m src.scraper.amazon.scraper \
  --product-ids "https://tr.ee/mUk1eH" "https://www.amazon.com/dp/B0CZ6TVK4Y" \
  --output-dir tmp \
  --debug
```

URLs are navigated directly in the browser and the ASIN is extracted from the redirected URL.

### Custom Output Directory

Override the default `outputs/` directory:

```bash
poetry run python -m src.scraper.amazon.scraper \
  --product-ids B0BTYCRJSS \
  --output-dir tmp \
  --debug
```

### Batch Sizing

Process products in sequential batches of N:

```bash
poetry run python -m src.scraper.amazon.scraper \
  --input-file products.txt \
  --batch-size 10 \
  --debug
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
# There is no `profile` key here. The producer requires --batch-profile or
# --random-profile on the command line; only `profile_pool` is read from YAML.
batch:
  profile_pool:
    - slideshow_images1
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
  --profile-pool slideshow_images1 product_video_sequential product_video_mixed \
  --debug
```

**YAML Configuration**:
```yaml
# `random_profile` is a CLI flag, not a YAML key.
batch:
  profile_pool:
    - slideshow_images1
    - product_video_sequential
    - product_video_mixed
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
--- PRODUCER SUMMARY ---
Products: 15 attempted, 14 successful, 0 failed, 1 skipped
Profiles: slideshow_images1: 5, product_video_sequential: 4, product_video_mixed: 3, slideshow_images2: 2
---
```

---

## Global Batch Pipeline

End-to-end automation combining scraping, video production, and publishing in a single unified command.

### Pipeline Architecture

The global batch pipeline orchestrates four phases:

1. **Scraping Phase** - Acquire product data from specified sources (product IDs, keywords). A topic run has no listing behind it, so this phase prepares the topic records instead of scraping.
2. **Handoff Phase** - Discover scraped products, filter by media availability, and drop any already recorded as published on every target platform (`--force` renders them anyway)
3. **Production Phase** - Generate videos using configured profile settings
4. **Publishing Phase** - Upload and publish videos to social media platforms (optional)

### Where a run's keywords come from

`global_batch.keywords` in `config/pipeline.yaml` is empty by default, so the
batch draws `batch.keywords` from `config/scraper.yaml`. One pool, one place to
edit. Setting the batch key *replaces* that pool for batch runs rather than
adding to it, so a single entry there narrows every batch run to one keyword.

A run does not search the whole pool. It searches `keywords_per_run`, which
defaults to what the run will actually consume (`max_products` divided by
`products_per_keyword`), taken in rotation by date. Consecutive days are
disjoint while that number is at most half the pool -- which the bundled 10
of 54 is -- so a daily cadence works through the pool instead of re-serving
the head of the list. Past half the pool consecutive days must overlap.
Keywords passed with `--keywords` are used exactly as given
and are not rotated.

### Usage Examples

#### Product IDs Only

```bash
poetry run python -m src.pipeline.global_batch \
  --product-ids B0BTYCRJSS B0D6GZF3T4 \
  --profile slideshow_images1 \
  --debug
```

#### Topics

A topic is rendered and published without a scraper run. The flags match the
producer's, so a command that works on one entry point works on the other.

```bash
poetry run python -m src.pipeline.global_batch \
  --topic "Why your wifi keeps dropping" \
  --topic-description "Router placement, channel congestion, 2.4 vs 5GHz." \
  --topic-keywords "wifi router, home network" \
  --profile slideshow_stock \
  --debug

# Several at once
poetry run python -m src.pipeline.global_batch \
  --topics-file topics.yaml --profile slideshow_stock
```

A topic needs a stock-sourced profile. A product profile gathers nothing and
the run **fails** with `No visual inputs were found or gathered for this
profile` -- it does not degrade to a skip, because visual gathering raises
before the media check that would report one. The batch refuses the
combination up front, so a misconfiguration is reported as one rather than as
a per-product render failure, and so
omitting `--profile` is safe: a topics run draws from the profiles that can
render one, replacing any pool configured in `pipeline.yaml` for product runs.
A pool named on the command line is refused rather than replaced, and so is a
`profile` set in `pipeline.yaml` -- that one carries no record of whether it
was meant for this run, so it is reported rather than overridden.

`--topic-keywords` is comma-separated so a phrase stays one search term.

Output lands in `outputs/topic-<slug>-<digest>/`, which `--clean` removes along
with product directories.

A topics-only run refuses a fixed `--profile` that draws no stock media, and
replaces a pool inherited from `pipeline.yaml`. A pool named on the command
line is refused rather than replaced. On a run that also carries products the
two pools coexist instead, and the CLI pool governs the products; the topics
draw from the stock-sourced profiles either way.

#### Topics in the daily cadence

A topic named on the command line is a one-off. To make the tutorial arm part
of the repeatable run, put the topics in `config/pipeline.yaml`:

```yaml
global_batch:
  # Left empty, so the batch draws `batch.keywords` from config/scraper.yaml.
  # Set it here only to override that pool for batch runs.
  keywords: {}
  topics:
    - title: "Why your wifi keeps dropping"
      description: "Router placement, channel congestion, 2.4 vs 5GHz bands."
      keywords: ["wifi router", "home network"]
    - title: "Your phone battery is not dying as fast as you think"
      description: "Background refresh, screen brightness, battery health."
  topics_per_run: 1
```

A run with no input flags then produces both formats: the configured keywords
are scraped and rendered as before, and `topics_per_run` topics are rendered
alongside them. `topics_per_run: 0` returns to products only.

Which topics a run takes rotates with the date, so a daily run works through
the list instead of re-rendering the first entry every morning. Interleaving
matters beyond variety: comparing the two content formats fairly needs them
mixed through the week rather than run in blocks, since a block comparison
cannot separate the format from whatever else changed that week. `registry
--summary` segments by `content_format` for the same reason.

Each record draws from its own profile pool on a mixed run -- topics from the
stock-sourced profiles, products from the rest -- so no profile has to serve
both. A fixed `--profile` cannot: one that draws no stock media is refused
rather than applied to the products and quietly swapped for the topics.

CLI inputs still replace the configured set entirely. `--keywords earbuds`
renders products only, and `--topic "..."` renders that topic only; neither
picks up the other arm from the config file.

#### Keywords Only

```bash
poetry run python -m src.pipeline.global_batch \
  --keywords "wireless earbuds" "bluetooth speaker" \
  --max-products 5 \
  --profile product_video_sequential \
  --debug
```

#### Mixed Input with Filters

```bash
poetry run python -m src.pipeline.global_batch \
  --product-ids B0BTYCRJSS \
  --keywords "smart watch" \
  --min-price 20 --max-price 100 \
  --min-rating 4.0 \
  --profile product_video_primary \
  --debug
```

#### Random Profile Mode

```bash
poetry run python -m src.pipeline.global_batch \
  --keywords "wireless earbuds" \
  --random-profile \
  --profile-pool slideshow_images1 product_video_sequential product_video_mixed \
  --debug
```

#### Pillar-Scoped Run

```bash
poetry run python -m src.pipeline.global_batch \
  --keywords "smart door lock" \
  --pillar utility \
  --profile slideshow_images1 \
  --debug
```

`--pillar <name>` filters the script template pool to templates configured under that pillar in `config/ai_services.yaml::script_templates.pillars`, prepends the per-pillar preamble to the LLM prompt, and substitutes `{AUDIENCE}` with the pillar's audience hint. Defaults: `value` (mass-appeal staples), `novelty` (lesser-known finds), `utility` (problem/solution framing). Without the flag, the product record's own pillar applies when it has one — the scraper attaches the source keyword's group. With neither, all templates are eligible and the global `target_audience` applies. See [Requirements](requirements.md) "Content Pillars" for the full system.

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
  # Empty draws the pool from config/scraper.yaml::batch.keywords. A
  # non-empty value REPLACES that pool for batch runs, so one entry here
  # narrows every run to one keyword.
  keywords: {}
  # keywords_per_run: 10  # default: max_products / products_per_keyword

  # Product Limits (Two-Tier System)
  max_products: 10        # Global cap across all keywords
  products_per_keyword: 1 # Limit per individual keyword

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
    - product_video_sequential

  # Error Handling
  fail_fast: false  # Continue on errors (default)

  # Common Settings
  outputs_dir: outputs
  debug: false
```

#### Clean Before Run

Remove stale product directories before starting a fresh run:

```bash
# Clean specific products
poetry run python -m src.pipeline.global_batch \
  --product-ids B0BTYCRJSS B0D6GZF3T4 \
  --profile slideshow_images1 \
  --clean \
  --debug

# Clean every run directory (ASIN-shaped and topic-*)
poetry run python -m src.pipeline.global_batch \
  --keywords "wireless earbuds" \
  --profile slideshow_images1 \
  --clean \
  --debug
```

A run that names its inputs removes only those, and a run carrying both kinds removes both: `--product-ids B0X --topic "Y"` removes `B0X` and the topic's directory. Keywords name nothing -- which products they produce is not known until the search runs -- so a run carrying any keyword, including a no-flag run reading them from the config, removes every run directory under outputs/, ASIN-shaped and `topic-*` alike. Directories that are not run outputs (logs/, coverage/) are preserved.

**Note**: Publishing options (`--skip-publish`, `--force`, `--platforms`, `--schedule-time`, `--fail-fast-publish`, `--clean`) are CLI-only and not supported in YAML configuration.

**Publishing Configuration**: Publishing behavior is controlled by `config/publisher.yaml` (see [Publisher](publisher.md) for details):
- `immediate_publish: false` enables auto-scheduling
- `recurring_schedule.slots` defines available time slots
- `cleanup.enabled: true` removes product directories after successful publish
- `link_in_bio.enabled: true` adds affiliate link to bio page after each publish

### Configuration Precedence

Settings are merged with this priority order:
1. **CLI arguments** (highest priority) - only when **explicitly provided**
2. **YAML configuration** (`config/pipeline.yaml`)
3. **Default values** (lowest priority)

**Important**: CLI arguments only override YAML values when explicitly provided by the user. Omitting a CLI flag uses the YAML value, not a hardcoded default.

### Product Limits

The pipeline uses a **two-tier limit system** for keyword searches:

- **`max_products`** (default: 10): Global cap on total products collected
- **`products_per_keyword`** (default: 1): Maximum products per individual keyword

**Behavior**:
- Processing iterates through keywords, collecting up to `products_per_keyword` from each
- Stops immediately when `max_products` is reached, even if keywords remain
- Product IDs count toward `max_products` but are not limited by `products_per_keyword`

**Example**: With `max_products: 10` and `products_per_keyword: 1` across 12 keywords:
- Keywords 1-10: 1 product each = 10 total (global cap reached)
- Keywords 11-12: Skipped (global cap already reached)

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
    - product_video_sequential: 1 (33.3%)
  Duration: 87.6s

PUBLISHING PHASE:
  Total Attempted: 3
  Successful: 3
  Failed: 0
  Skipped: 0

  Platform Results:
    - Youtube: 3/3 (100.0%)
    - Tiktok: 3/3 (100.0%)
    - Instagram: 3/3 (100.0%)
  Duration: 45.2s

END-TO-END RESULTS:
  Complete Success (scraped + produced): 3
  Partial Success (scraped only): 0
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
- **Auto-Scheduling**: Finds first available unoccupied slot in recurring schedule by querying the Zernio API
- **Smart Cleanup**: Removes product directories after successful multi-platform publish
- **Comprehensive Reporting**: Detailed phase-by-phase statistics with end-to-end metrics
- **Error Resilience**: Graceful failure handling with optional fail-fast mode per phase
- **Low-Priority Mode**: `make batch-lowpri` runs with reduced CPU, I/O, and memory priority
- **Pre-Run Cleanup**: `--clean` removes stale product directories before starting

### Low-Priority Batch Mode

For long-running batch jobs on shared or resource-constrained machines, use low-priority Makefile targets. They wrap commands with `nice`, `ionice`, and `systemd-run` memory limits:

```bash
# Full pipeline (scrape + produce + publish)
make batch-lowpri ARGS="--keywords 'wireless earbuds' --profile slideshow_images1 --debug"

# Scraping only
make scrape-lowpri ARGS="--keywords 'wireless earbuds' --debug"

# Video production only
make produce-lowpri ARGS="--batch --batch-profile slideshow_images1 --debug"

# Override resource limits (defaults: MEM_LIMIT=6G, NICE_LEVEL=15)
make batch-lowpri ARGS="--product-ids B0ASIN1 --debug" MEM_LIMIT=4G NICE_LEVEL=19
```

Requires `ionice` (from `util-linux`). Falls back to `nice` + `ionice` without memory cap if `systemd-run` is unavailable.

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

### Exit Codes

| Outcome | Exit code | Log line |
|---|---|---|
| Every product completed end-to-end | 0 | `PIPELINE COMPLETED SUCCESSFULLY` |
| Some products completed, some were lost | 0 (1 with `--strict`) | `PIPELINE COMPLETED WITH LOSSES` |
| No product completed end-to-end | 1 | `PIPELINE FAILED` |
| Every product was already published, nothing else lost | 0 (0 with `--strict`) | `PIPELINE COMPLETED SUCCESSFULLY` |

The last row is not a loss: nothing was asked for that does not exist, so
`--strict` leaves it at 0 too. Its condition is narrower than row three's,
which it would otherwise contradict: anything genuinely lost alongside the
already-published products -- a keyword that returned nothing, a product
rejected for insufficient media -- puts the run back on row three at exit 1. It
is a normal outcome once the keyword
rotation has walked the pool, since the same keywords then return the same
already-published top results.

"Lost" covers both a product whose step failed and one reported skipped for
insufficient media.

A partial loss exits 0 by default: a run that loses one product of twenty
has done most of what was asked, and failing the whole run would stop a
schedule over a single bad listing. Pass `--strict` when a lost product
matters more than the interruption — an unattended run whose output is
posted on a cadence, for instance.

`--strict` counts skips as well as failures. The two are reported separately
because they have different causes, but for the exit code they are the same
thing: a video that was asked for and does not exist. A profile misconfigured
so that every product is rejected for insufficient media loses the whole run
while reporting no failures at all, and that is the silence the flag exists
to break.

The standalone scraper takes the same flag, for what it can see: a product id that yielded nothing, and a keyword whose search returned nothing or raised. It has no skip outcome — a product dropped during media validation is not counted there at all.

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
