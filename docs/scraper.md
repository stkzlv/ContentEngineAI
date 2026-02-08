# ContentEngineAI Scraper Module

The Scraper Module is a powerful, multi-platform data extraction engine designed for e-commerce video production. It currently supports Amazon with advanced anti-detection capabilities (Botasaurus), media extraction, and automated validation.

## 🚀 Quick Start

### Single Product Scraping
Scrape a specific product by ASIN (Amazon Standard Identification Number):
```bash
poetry run python -m src.scraper.amazon.scraper --keywords "B0BTYCRJSS" --debug
```

### Keyword Search
Search for products and scrape the top results:
```bash
poetry run python -m src.scraper.amazon.scraper --keywords "wireless earbuds" --min-rating 4.5 --prime-only --debug
```

### Batch Processing
Scrape multiple specific products:
```bash
poetry run python -m src.scraper.amazon.scraper --product-ids B0BTYCRJSS B08DTZM7LM B07ZPC9QD4 --fail-fast
```

### URL Scraping
Scrape from full or shortened Amazon URLs (e.g. tr.ee, amzn.to):
```bash
poetry run python -m src.scraper.amazon.scraper --product-ids "https://tr.ee/mUk1eH" --output-dir tmp --debug
```

### Batch from File
Read product IDs or URLs from a file, process in batches:
```bash
poetry run python -m src.scraper.amazon.scraper --input-file products.txt --output-dir tmp --batch-size 10 --debug
```

## 📖 CLI Reference

The scraper is run via `src.scraper.amazon.scraper`.

### Core Arguments
| Argument | Description | Example |
|----------|-------------|---------|
| `--keywords` | List of keywords or ASINs to search/scrape | `--keywords "headphones" "B0..."` |
| `--product-ids` | Explicit list of ASINs for direct scraping | `--product-ids B0123... B0456...` |
| `--max-products` | Global cap on total products to collect | `--max-products 10` |
| `--products-per-keyword` | Maximum products per individual keyword | `--products-per-keyword 2` |
| `--fail-fast` | Stop batch processing on first error | `--fail-fast` |
| `--clean` | Delete output directory before scraping | `--clean` |
| `--input-file` | Read product IDs/URLs from file (one per line) | `--input-file products.txt` |
| `--batch-size` | Process products in batches of N | `--batch-size 10` |
| `--output-dir` | Override output directory (default: `outputs`) | `--output-dir tmp` |

### Filtering & Search
| Argument | Description | Example |
|----------|-------------|---------|
| `--min-price` | Minimum price filter | `--min-price 25.00` |
| `--max-price` | Maximum price filter | `--max-price 100.00` |
| `--min-rating` | Minimum star rating (1-5) | `--min-rating 4.0` |
| `--prime-only` | Filter for Prime-eligible items | `--prime-only` |
| `--free-shipping` | Filter for Free Shipping items | `--free-shipping` |
| `--brands` | Filter by specific brands | `--brands Sony Bose` |
| `--sort` | Sort order (`relevance`, `price-low`, `price-high`, `rating`, `newest`) | `--sort rating` |

### Debugging & Development
| Argument | Description |
|----------|-------------|
| `--debug` | Enable detailed logging and show browser (headless=False) |
| `--verbose` | Enable even more detailed logging |
| `--save-screenshots` | Save screenshots at key steps (debug mode only) |
| `--save-page-source` | Save HTML source for analysis (debug mode only) |
| `--analyze-images` | Deep analysis of all page images (debug mode only) |
| `--pause-on-error` | Pause execution when errors occur (debug mode only) |

## ⚙️ Configuration

The scraper is configured via `config/scraper.yaml`. CLI arguments override YAML settings **only when explicitly provided**.

```yaml
scrapers:
  amazon:
    enabled: true
    headless: true  # Set false to see browser
    max_products: 5 # Default limit per keyword
    
    # Browser settings
    browser:
      window_size: [1920, 1080]
      user_agent_rotate: true
      
    # Timing
    timeouts:
      page_load: 30000
      element_wait: 5000

batch:
  # Default batch to run if no CLI args provided
  keywords: []
  product_ids: []
```

## 🔧 Troubleshooting

### Rate Limiting / CAPTCHA
The scraper uses Botasaurus to evade detection. If you encounter CAPTCHAs:
1.  Run with `--debug` to see the browser.
2.  Increase `rate_limiting` delays in `config/scraper.yaml`.
3.  Ensure your IP is not blacklisted (try a VPN/proxy if configured).

### Missing Media
If videos/images aren't downloading:
1.  Check connection speed.
2.  Verify `download_config` timeouts in `config/scraper.yaml`.
3.  Run with `--debug` to see if URLs are being extracted but failing validation.

### "No products scraped"
1.  Check your filters (e.g., strict price/rating).
2.  Verify the keyword returns results on Amazon.com manually.
3.  Check logs for selector errors (Amazon UI changes).
