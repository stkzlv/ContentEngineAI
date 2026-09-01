# ContentEngineAI Scraper Module

The scraper extracts product data from Amazon for video production. It handles media downloading, validation, and batch processing with anti-detection (Botasaurus).

## Quick Start

```bash
# Single product by ASIN
poetry run python -m src.scraper.amazon.scraper --product-ids B0BTYCRJSS --debug

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
| `--fail-fast` / `--no-fail-fast` | Stop batch on first error. Omitted, `batch.fail_fast` in the config decides | `--fail-fast` |
| `--strict` | Exit non-zero when any product or keyword produced nothing | `--strict` |
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
| `--debug` | Detailed logging; visible browser on X11 (on Wayland use `make scrape-watch`) |
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
| `max_products` | `scrapers.amazon.max_products` | `5` code default (bundled `config/scraper.yaml` sets `1`) | `--max-products N` |
| `max_products_per_search` | `scrapers.amazon.max_products_per_search` | `10` (Pydantic default `5`) | none |

`max_products` is the total validated-product cap for the run. `max_products_per_search` is the distinct per-page extraction cap (how many products are read from a single search results page).

### How they interact

Each keyword is scraped independently with `products_per_keyword` as the per-keyword limit. After collecting results from a keyword, the total is checked against `max_products`. If the total reaches `max_products`, remaining keywords are skipped.

The bundled `config/scraper.yaml` sets `max_products: 1`, so a run stops after the first validated product. Raise `scrapers.amazon.max_products` (or pass `--max-products N`) to collect more across keywords.

### Product IDs vs keywords

Product IDs and keywords are handled differently:

- **Product IDs** (`--product-ids`): each ID is scraped individually, always returns 1 product per ID. All IDs are processed regardless of `max_products`.
- **Keywords** (`--keywords` or `batch.keywords` in config): each keyword returns up to `products_per_keyword` results. The keyword loop stops when total >= `max_products`. When `batch.keywords` is a dict keyed by pillar (the default shape), each scraped product carries the pillar from its source keyword through to the producer. A flat list is still accepted for backward compatibility (no pillar attached).

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

This applies to keyword searches only. A URL or an ASIN names one product, so every later page would re-resolve the same listing; those inputs are scraped in a single pass and reported as failed if they do not pass validation.

### Config precedence

CLI > YAML > code defaults. If you pass `--products-per-keyword 5` on the command line, it overrides `batch.products_per_keyword` in the YAML. If neither is set, the YAML defaults apply (1 for `products_per_keyword`; `scrapers.amazon.max_products` is 1 in the bundled `config/scraper.yaml`, with a code default of 5 when unset).

### Examples

**Default config (no CLI args)**: keywords from config, `products_per_keyword: 1`, bundled `max_products: 1`
- Result: 1 product (the run stops once the cap is reached). Raise `--max-products` to collect one per keyword across the pool.

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
    max_products: 1  # total validated-product cap for the run
    max_products_per_search: 10  # per-page extraction cap

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

## Affiliate URLs

The scraper writes the affiliate URL for each scraped product into `data.json::affiliate_link` using `build_affiliate_url`, which canonicalises every URL to `https://www.amazon.com/dp/<ASIN>?tag=<AMAZON_ASSOCIATE_TAG>`. The tag is read from the `AMAZON_ASSOCIATE_TAG` environment variable (or, if unset, from `scrapers.amazon.associate_tag` in `config/scraper.yaml`).

The standalone scraper CLI loads `.env` at startup, so a tag present in `.env` is visible to the URL builder without any shell-side `export`. If the tag resolves empty (no env var, no config value), `build_affiliate_url` logs a WARNING and returns the input URL unchanged: this signals lost affiliate attribution on every scrape in the session and is grep-able in `outputs/logs/scraper.log`.

### Running without an affiliate program

The warning above assumes a missing tag is a mistake. When there is no program to attribute to, say so explicitly:

```yaml
scrapers:
  amazon:
    affiliate_links:
      enabled: false
```

`build_affiliate_url` then strips tracking parameters instead of preserving them, canonicalising to a bare `https://www.amazon.com/dp/<ASIN>`, and drops the log line to DEBUG. Use this when the account is closed or was never opened, rather than leaving a dead account's tag in the config: a tag belonging to a terminated account is still sent to Amazon on every link.

`AMAZON_AFFILIATE_LINKS_ENABLED` overrides the YAML, so an install can declare "no program" from `.env` without editing a tracked config file. This mirrors how the tag itself prefers the environment.

A typo inside the block (`enabld: false`) is rejected at config load rather than silently ignored. A typo in the block name itself still falls back to the default, since the surrounding model ignores unknown keys.

The flag only governs the missing-tag path. A tag from `AMAZON_ASSOCIATE_TAG` or `associate_tag` is still applied while the flag is off, so it cannot silently discard a working program.

## URL shortener

Configured in `config/url_shortener.yaml`. Two providers ship:

| Provider | Default | Requires | Behaviour |
|---|---|---|---|
| `bare` | yes | nothing | Returns the canonical affiliate URL unchanged. No third-party dependency. |
| `picsee` | opt-in | `PICSEE_API_KEY` in `.env`, Picsee account | Mints a `stte.psee.io` short code that 302s to the affiliate URL. |

The canonical Amazon URL is ~50 characters and fits every supported platform's caption budget after the first-line disclosure, description, and hashtags land. The character savings from a shortener don't justify the vendor dependency for most setups. The bare provider is the recommended default.

The Picsee path remains for setups already invested in `stte.psee.io` shorts (e.g. live published captions reference existing short codes). One known caveat: Picsee captures the input URL verbatim rather than re-canonicalising, so a shortened code minted while the upstream affiliate URL was bad (no tag, or stale account tag) will keep redirecting to that bad URL until the redirect target is updated server-side via Picsee's API or dashboard. The bare provider sidesteps this class of issue by design.

A third theoretical option, Amazon's own `amzn.to` shortener, isn't available programmatically: SiteStripe mints the short codes in-browser and there's no public API. If you want short URLs in captions and have access to SiteStripe, mint them manually and bypass this layer.

## Anti-bot detection and browser mode

The scraper drives a real Chrome through Botasaurus (a fork of `nodriver`, itself descended
from `undetected-chromedriver`). Two facts drive the whole design: Botasaurus is unreliable in
headless mode, and Amazon runs a serious anti-bot stack. The scraper answers both by running a
real, visible-capable browser with human-like navigation.

### Why the scraper never runs true headless

Every browser config path hardcodes `headless: False` (see `config.py` and
`browser_functions.py::_build_browser_config`). Headless is avoided for two separate reasons,
both observed in this project and confirmed upstream:

1. Detection. In headless mode Botasaurus/nodriver does not fully patch the browser
   fingerprint. The user agent still reports `HeadlessChrome/<version>` and other headless
   signals leak (missing plugins, GPU/renderer mismatches, `navigator.webdriver` traces).
   Anti-bot services flag this immediately. Botasaurus's own docs state that headless mode
   "will surely result in identification by services like Cloudflare and DataDome" and
   recommend it only for sites with no bot protection. Amazon is not such a site.
2. Stability. nodriver's headless startup path has crash bugs. This project hit a
   `StopIteration` during the headless connection setup (hence the `# Disabled - causes
   StopIteration in headless mode` comments), and upstream tracks a related
   `TypeError: cannot unpack non-iterable NoneType` in the headless connection-prep code.
   Headful avoids both.

The cost of headful is that Chrome needs a display. On X11 (Ubuntu 22 and earlier) the session
exported `DISPLAY`, so this was invisible. On Wayland (Ubuntu 26) `DISPLAY` is empty, so headful
Chrome has nowhere to draw and `google_get` hangs until the 60s document-ready timeout. That
failure looks like an anti-bot block in the logs but is purely a missing display.

### Virtual display (Xvfb) for non-debug runs

Botasaurus has built-in support for a headful-but-invisible browser via a virtual framebuffer.
When `headless=False` and `enable_xvfb_virtual_display=True`, it starts a `pyvirtualdisplay`
Xvfb session itself (`botasaurus_driver/core/config.py`). This is the right mode for unattended
scraping: a real, non-headless browser with no visible window, so the fingerprint stays clean
and nothing pops up on screen.

Two gotchas:

- Botasaurus only auto-starts Xvfb when `is_vmish` is true (Docker, a VM with `VM=true`,
  Gitpod, or Kubernetes). A normal Linux desktop is not `is_vmish`, so the virtual display must
  be requested explicitly with `enable_xvfb_virtual_display=True`.
- The Xvfb binary comes from the `xvfb` apt package (`sudo apt-get install -y xvfb`). The
  `pyvirtualdisplay` Python package is already installed but only wraps the binary. If the
  binary is missing, Botasaurus prints a one-line notice and silently falls back to
  `--headless=new`, putting you right back in the detectable/unstable headless mode. Grep
  `outputs/logs/scraper.log` for `install Xvfb` to catch this.

Debug mode (`--debug`) behaviour depends on the session:

- On a real X11 desktop it uses the live session display (`enable_xvfb_virtual_display=False`),
  so the browser window is visible.
- On a live Wayland session it runs on a virtual Xvfb display (no visible window): a headful
  window on Wayland freezes Chromium's CDP (DevTools won't connect, then per-navigation
  `Response not received` hangs). To watch a debug scrape on Wayland, run `make scrape-watch` —
  it starts a dedicated Xvfb plus `x11vnc` and the browser is viewable at `localhost:5900`.

Running the module directly (`poetry run python -m src.scraper.amazon.scraper ... --debug`)
works the same way; only `make scrape-watch` adds the VNC view.

### Amazon's anti-bot stack (AWS WAF, not Cloudflare)

Amazon does not use Cloudflare. Amazon.com is fronted by CloudFront (CDN) and protected by
AWS WAF plus Amazon's own "Robot Check" page. The layers a scraper actually meets:

| Layer | What it does |
|---|---|
| AWS WAF silent Challenge | Background JavaScript interrogation and lightweight proof-of-work that issues a token before content loads. No user interaction; a real browser passes, a thin HTTP client or leaky headless browser fails. |
| Robot Check CAPTCHA | Amazon's image-text CAPTCHA page, shown when the silent challenge is not satisfied or risk is high. |
| Fingerprinting | TLS/JA3, HTTP/2 frame order, header consistency, and JS browser-environment checks (the headless tells above). |
| IP reputation and rate limits | Datacenter-IP blocks, per-IP request-rate thresholds, repeated-pattern detection. |
| Behavioral / ML analysis | AWS WAF targeted protections score traffic statistics (timing, navigation patterns, prior URL) for anomalies indicative of coordinated bots. |

This is why the scraper navigates with `driver.google_get(url, bypass_cloudflare=True)` and
human-like cursor motion rather than fetching the URL directly: an organic referer and real
browser behavior are what satisfy the silent AWS WAF challenge. Direct `requests`-style fetches
get the Robot Check immediately.

### Cloudflare Turnstile and the bypass flag

The `bypass_cloudflare=True` argument on `google_get` is generic Botasaurus machinery, not
Amazon-specific. Cloudflare Turnstile runs non-interactive JS challenges (proof-of-work and
browser-environment signals); most legitimate visitors never see a visible widget. When a
visible Turnstile checkbox does appear, `solve_cloudflare_captcha.py` walks the iframe and shadow
DOM and clicks it with a restored human cursor. Since Amazon does not run Cloudflare, this code
path is mostly dormant for Amazon scraping. It still matters because `google_get` routes through
the same flow, and the `wait_till_document_is_ready` step that times out on a missing display
lives in that module. Treat a 60s document-ready timeout as a display or navigation failure
first, not as a Cloudflare block.

### Practical implications

- Run non-debug scrapes with a virtual display; install `xvfb` so Botasaurus does not silently
  fall back to headless.
- A sudden run of 0 products with 60s timeouts after an OS or session change is almost always a
  display problem, not Amazon blocking you. Check `echo $DISPLAY` and whether the session is
  Wayland or X11 before assuming a ban.
- If Amazon genuinely starts challenging (Robot Check in `--save-page-source` output, captcha
  text on the page), slow down: raise `rate_limiting` delays, reduce per-page volume, and
  consider a residential IP. Headless is never the answer here; it makes detection worse.

Sources: [Botasaurus](https://github.com/omkarcloud/botasaurus),
[nodriver headless bot detection (undetected-chromedriver #2003)](https://github.com/ultrafunkamsterdam/undetected-chromedriver/issues/2003),
[nodriver headless exception (#2120)](https://github.com/ultrafunkamsterdam/undetected-chromedriver/issues/2120),
[AWS WAF Bot Control](https://docs.aws.amazon.com/waf/latest/developerguide/aws-managed-rule-groups-bot.html),
[Amazon CAPTCHA / AWS WAF overview](https://2captcha.com/p/amazon-captcha-bypass),
[Cloudflare Turnstile](https://developers.cloudflare.com/turnstile/).

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
