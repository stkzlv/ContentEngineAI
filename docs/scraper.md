# ContentEngineAI Scraper Module

The scraper extracts product data from Amazon for video production. It handles media downloading, validation, and batch processing with anti-detection (Botasaurus).

## Quick Start

```bash
# Single product by ASIN
poetry run python -m src.scraper.amazon.scraper --keywords "B0BTYCRJSS" --debug

# Keyword search
poetry run python -m src.scraper.amazon.scraper --keywords "wireless earbuds" --min-rating 4.5 --debug

# Multiple products
poetry run python -m src.scraper.amazon.scraper --product-ids B0BTYCRJSS B08DTZM7LM B07ZPC9QD4

# From shortened/full URLs
poetry run python -m src.scraper.amazon.scraper --product-ids "https://tr.ee/mUk1eH" --output-dir tmp --debug

# From file, chunked
poetry run python -m src.scraper.amazon.scraper --input-file products.txt --batch-size 10 --debug

# Override product counts
poetry run python -m src.scraper.amazon.scraper --keywords "earbuds" "headphones" --products-per-keyword 3 --max-products 5 --debug
```

## CLI Reference

Both the standalone scraper and the global batch pipeline accept these flags.

### Core Arguments
| Argument | Description | Example |
|----------|-------------|---------|
| `--keywords` | Keywords or ASINs to search | `--keywords "headphones" "B0..."` |
| `--product-ids` | ASINs or URLs for direct scraping | `--product-ids B0123... B0456...` |
| `--max-products` | Global cap on total products | `--max-products 10` |
| `--products-per-keyword` | Max products per keyword | `--products-per-keyword 2` |
| `--fail-fast` | Stop batch on first error | `--fail-fast` |
| `--clean` | Delete output dir before scraping | `--clean` |
| `--input-file` | Read product IDs/URLs from file | `--input-file products.txt` |
| `--batch-size` | Process products in chunks of N | `--batch-size 10` |
| `--output-dir` | Override output directory | `--output-dir tmp` |
| `--profile` | Video profile for media validation | `--profile slideshow_images1` |

### Filtering
| Argument | Description | Example |
|----------|-------------|---------|
| `--min-price` | Minimum price | `--min-price 25.00` |
| `--max-price` | Maximum price | `--max-price 100.00` |
| `--min-rating` | Minimum star rating (1-5) | `--min-rating 4.0` |
| `--prime-only` | Prime-eligible only | `--prime-only` |
| `--free-shipping` | Free shipping only | `--free-shipping` |
| `--brands` | Filter by brand | `--brands Sony Bose` |
| `--sort` | Sort order | `--sort rating` |

### Debugging
| Argument | Description |
|----------|-------------|
| `--debug` | Detailed logging, visible browser |
| `--verbose` | Even more logging |
| `--save-screenshots` | Screenshots at key steps |
| `--save-page-source` | Save HTML source |
| `--analyze-images` | Deep image analysis |
| `--pause-on-error` | Pause on errors |

## Product Count Logic

Two settings control how many products get scraped:

| Setting | Config path | Default | CLI override |
|---------|------------|---------|--------------|
| `products_per_keyword` | `batch.products_per_keyword` | `1` | `--products-per-keyword N` |
| `max_products` | `scrapers.amazon.max_products` | `50` | `--max-products N` |

### How they interact

Each keyword is scraped independently with `products_per_keyword` as the per-keyword limit. After collecting results from a keyword, the total is checked against `max_products`. If the total reaches `max_products`, remaining keywords are skipped.

With the defaults (1 per keyword, 50 max), every keyword in the batch gets processed and contributes 1 product each. The `max_products: 50` just acts as a safety cap.

### Product IDs vs keywords

Product IDs and keywords are handled differently:

- **Product IDs** (`--product-ids`): each ID is scraped individually, always returns 1 product per ID. All IDs are processed regardless of `max_products`.
- **Keywords** (`--keywords` or `batch.keywords` in config): each keyword returns up to `products_per_keyword` results. The keyword loop stops when total >= `max_products`.

### No CLI args (config-only mode)

When you run the scraper without `--keywords` or `--product-ids`, it falls back to `config/scraper.yaml`:

1. Check `batch.product_ids` and `batch.keywords`
2. If both empty, fall back to `scrapers.amazon.keywords` (single-product mode)
3. If that's also empty, error out

### Media validation filtering

When `count_products_with_media: true` (default), only products with enough images/videos count toward the limit. Products that fail media validation are discarded and don't count.

To compensate for rejected products, the scraper fetches more raw results than the target:

| Setting | Config path | Default | Purpose |
|---------|------------|---------|---------|
| `prefetch_multiplier` | `global_settings.batch_processing.prefetch_multiplier` | `3` | Fetch 3x the target per page |
| `max_batch_size` | `global_settings.batch_processing.max_batch_size` | `15` | Cap on products fetched per page |
| `max_pages` | `global_settings.batch_processing.max_pages` | `7` | Max search result pages to scan |
| `max_scrape_attempts` | `global_settings.batch_processing.max_scrape_attempts` | `50` | Hard stop on raw products examined |

For example, if `products_per_keyword: 3`, the scraper fetches ~9 products per page (3x multiplier, capped at 15), validates each, and loops through pages until 3 valid products are found or the safety limits are hit.

### Config precedence

CLI > YAML > code defaults. If you pass `--products-per-keyword 5` on the command line, it overrides `batch.products_per_keyword` in the YAML. If neither is set, code defaults apply (2 for products_per_keyword, 10 for max_products).

### Examples

**Default config (no CLI args)**: 11 keywords configured, `products_per_keyword: 1`, `max_products: 50`
- Result: 11 products (1 per keyword, all keywords processed)

**3 keywords, 2 per keyword, no cap**: `--keywords "a" "b" "c" --products-per-keyword 2`
- Result: up to 6 products (2 per keyword x 3 keywords)

**3 keywords, 2 per keyword, cap at 4**: `--keywords "a" "b" "c" --products-per-keyword 2 --max-products 4`
- Result: 4 products (2 from "a", 2 from "b", "c" skipped because cap reached)

**5 product IDs**: `--product-ids B0A B0B B0C B0D B0E`
- Result: 5 products (all IDs processed, `max_products` doesn't apply to product IDs)

## Configuration

Config file: `config/scraper.yaml`. CLI arguments override YAML settings when provided.

```yaml
scrapers:
  amazon:
    enabled: true
    headless: true
    max_products: 50  # global cap across all keywords

batch:
  keywords: ["smart ring", "mini projector"]  # default keywords when no CLI args
  product_ids: []
  products_per_keyword: 1  # how many products per keyword
  fail_fast: false

global_settings:
  count_products_with_media: true  # only count products passing media validation
  batch_processing:
    max_scrape_attempts: 50
    prefetch_multiplier: 3
    max_batch_size: 15
```

## Troubleshooting

### Rate Limiting / CAPTCHA
1. Run with `--debug` to see the browser.
2. Increase `rate_limiting` delays in `config/scraper.yaml`.
3. Try a VPN if your IP is blocked.

### Missing Media
1. Check connection speed.
2. Verify `download_config` timeouts in config.
3. Run with `--debug` to check if URLs are extracted but fail validation.

### "No products scraped"
1. Check your filters (strict price/rating can exclude everything).
2. Verify the keyword returns results on Amazon.com.
3. Check logs for selector errors (Amazon UI changes break extraction).
