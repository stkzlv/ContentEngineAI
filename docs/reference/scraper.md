# Scraper reference

The command line of the Amazon scraper, the keys in `config/scraper.yaml`, and the rules that decide how many products a run collects. For the steps of a scrape, see [the scraping guide](../guides/scraping.md); for why the scraper behaves as it does, see [scraping explained](../explanation/scraping.md). The requirements are in [the scraper requirements](../requirements/scraper.md), and the defects behind the code in [the scraper module notes](../notes/scraper.md).

## Command line

```bash
poetry run python -m src.scraper.amazon.scraper [options]
```

The parser is `build_argument_parser` in `src/scraper/amazon/cli.py`. A run with `--product-ids`, or with more than one keyword, goes through the batch controller; a run with one keyword goes through the single-keyword path. Both print a summary at the end.

### Inputs and run control

| Flag | Value | Default | Effect |
|---|---|---|---|
| `--keywords` | one or more strings | none | Keywords or ASINs to search. Overrides the configured keywords. |
| `--product-ids` | one or more ASINs or URLs | none | Products to scrape directly. Full and shortened URLs are followed to the product page (`REQ-SCR-002`). |
| `--input-file` | `FILE` | none | Reads product ids or URLs, one per line, and merges them with `--product-ids`, dropping duplicates. A relative path resolves against the repository root. |
| `--max-products` | `N` | `scrapers.amazon.max_products` | Cap on the total products collected across all keywords. |
| `--products-per-keyword` | `N` | `batch.products_per_keyword` | Cap on the products collected for one keyword. |
| `--batch-size` | `N` | all at once | Processes the product ids in chunks of N. Keywords are searched once, with the first chunk. |
| `--fail-fast` / `--no-fail-fast` | switch | `batch.fail_fast` | Stops the batch at the first failed input, or continues past it. |
| `--strict` | switch | off | Exits non-zero when any product id or keyword produced nothing, not only when the run scraped nothing. |
| `--clean` | switch | off | Before scraping, deletes the product directories (named by ASIN), the `cache`, `temp` and `screenshots` directories, and loose `.json`, `.csv`, `.xlsx` and `.html` files under the configured output base. Logs and files whose names start with `report` stay. |
| `--output-dir` | `DIR` | `global_settings.output_config.base_directory` | Writes the scraped products under this directory. |
| `--no-profile-uses-videos` | switch | off | Scrapes for an image-only profile: no video is extracted or downloaded, and images must meet the minimums on their own. |
| `--profile-uses-videos` | switch | off | Keeps the configured media requirements, which count videos. The same as passing neither flag. |

`--profile-uses-videos` and `--no-profile-uses-videos` are mutually exclusive.

### Search filters

The filters apply to keyword searches. Each one overrides the matching field of `scrapers.amazon.default_search_parameters` (`REQ-SCR-031`).

| Flag | Value | Effect |
|---|---|---|
| `--min-price` | `PRICE` (float) | Minimum price. `0` is a real bound. |
| `--max-price` | `PRICE` (float) | Maximum price. |
| `--min-rating` | `RATING` (float, 1 to 5) | Minimum star rating. |
| `--prime-only` | switch | Prime-eligible items only. |
| `--free-shipping` | switch | Items with free shipping only. |
| `--brands` | one or more `BRAND` | Brand names to filter by. |
| `--sort` | `relevance`, `price-low`, `price-high`, `rating`, `newest` or `featured` | Sort order of the results. The default `relevance` leaves the configured `sort_order` in force. |

`--sort` values map to Amazon's sort tokens: `relevance` -> `relevanceblender`, `price-low` -> `price-asc-rank`, `price-high` -> `price-desc-rank`, `rating` -> `review-rank`, `newest` -> `date-desc-rank`, `featured` -> `featured-rank`. The config key `sort_order` takes the token, not the CLI value. Invalid combinations, such as a minimum price above the maximum, stop the run before the browser starts.

### Debugging

| Flag | Effect |
|---|---|
| `--debug` | Detailed logging and debug mode. On X11 the browser window is visible; on Wayland the browser runs on a virtual display, and `make scrape-watch` shows it over VNC. |
| `--verbose` | Debug mode with more detailed logging than `--debug`. |
| `--save-screenshots` | Saves screenshots at key steps. |
| `--save-page-source` | Saves the HTML source of the pages visited. |
| `--analyze-images` | Analyses every image found on the page. |
| `--dump-image-urls` | Saves every discovered image URL to a file. |

The last five take effect only together with `--debug`. Setting `global_settings.debug_mode: true` turns on debug mode without the flag.

### Exit status

A run that scrapes no product exits 1, including one that stops before scraping because the input file is missing, no inputs are configured or the search filters are invalid (`REQ-SCR-049`). A run that loses some inputs exits 0 unless `--strict` is passed (`REQ-SCR-050`).

### Flags the global batch does not take

`python -m src.pipeline` accepts the common flags. It does not take `--free-shipping`, `--brands`, `--sort`, `--input-file`, `--batch-size`, `--output-dir`, `--verbose`, the four debug switches (`--save-screenshots`, `--save-page-source`, `--analyze-images`, `--dump-image-urls`), or the `--profile-uses-videos` pair, since the batch works out video use from the profile itself. Its output flag is `--outputs-dir`. See [batch processing](../guides/batch-processing.md).

## Product count

Three settings control how many products a run scrapes:

| Setting | Config path | Bundled value | Model default | CLI override |
|---|---|---|---|---|
| `products_per_keyword` | `batch.products_per_keyword` | `1` | `2` | `--products-per-keyword N` |
| `max_products` | `scrapers.amazon.max_products` | `1` | `2` | `--max-products N` |
| `max_products_per_search` | `global_settings.browser_config.max_products_per_search` | `10` | `5` | none |

`max_products` is the cap on validated products for the run. `max_products_per_search` is a separate cap on how many products are read from one search results page.

### Keywords and the total cap

Each keyword is scraped on its own, with `products_per_keyword` as its limit. After each keyword, the total is checked against `max_products`; once the total reaches it, the remaining keywords are skipped (`REQ-SCR-034` to `REQ-SCR-038`).

The bundled `config/scraper.yaml` sets `max_products: 1`, so a run stops after the first validated product. Raise `scrapers.amazon.max_products`, or pass `--max-products N`, to collect more across keywords.

### Product ids and keywords

- **Product ids** (`--product-ids`): each id is scraped on its own and yields one product. Every id is processed, whatever `max_products` says.
- **Keywords** (`--keywords`, or `batch.keywords` in the config): each keyword yields up to `products_per_keyword` products, and the keyword loop stops when the total reaches `max_products`. When `batch.keywords` is a mapping keyed by pillar (the bundled shape), each scraped product carries the pillar of its source keyword through to the producer. A flat list is also accepted, and attaches no pillar.

### Inputs from the config

A run with neither `--keywords` nor `--product-ids` (nor `--input-file`) reads its inputs from `config/scraper.yaml`:

1. `batch.product_ids` and `batch.keywords`.
2. If both are empty, `scrapers.amazon.keywords`.
3. If that is empty too, the run stops with an error.

A configured keyword pool is rotated by date before the search: every keyword stays in the list, and the starting point advances each day by what one run consumes, `max_products` divided by `products_per_keyword`, rounded up (`REQ-SCR-006`). Keywords passed with `--keywords` are searched in the order given (`REQ-SCR-007`). [Scraping explained](../explanation/scraping.md#keyword-pool-rotation) has the reason.

### Media validation and page scanning

With `global_settings.count_products_with_media: true` (the default), only products whose media pass validation count toward the limits; the rest are discarded. To make up for them, a keyword search fetches more raw results than its target and scans further result pages (`REQ-SCR-033`):

| Setting | Config path | Default | Effect |
|---|---|---|---|
| `prefetch_multiplier` | `global_settings.batch_processing.prefetch_multiplier` | `3` | Fetches this many times the remaining target from each page. |
| `max_batch_size` | `global_settings.batch_processing.max_batch_size` | `15` | Caps the products fetched from one page. |
| `max_pages` | `global_settings.batch_processing.max_pages` | `7` | The last result page the standalone loop scans. Not in the bundled YAML. |
| `max_scrape_attempts` | `global_settings.batch_processing.max_scrape_attempts` | `50` | Stops the loop once this many validated products are collected, checked before each page. It binds only when set below the target, a single page can overshoot it, and it stays at 0 while nothing validates, where `max_pages` is the only stop. |
| `max_retry_pages` | `global_settings.batch_processing.max_retry_pages` | `5` | The highest result page the global batch retries to. It runs pages 2 to N, so the default scans four extra pages, and any value below 2 disables the retry. Global batch only. |

For example, with `products_per_keyword: 3` the scraper fetches about 9 products per page (3 x 3, under the cap of 15), validates each, and moves through pages until 3 products pass or `max_pages` is passed.

This applies to keyword searches only. A URL or an ASIN names one product, so a later page would resolve the same listing again; those inputs are scraped in one pass and reported as failed if they don't pass validation.

### Precedence

A flag on the command line overrides the YAML, and the YAML overrides the model default ([decision 0003](../decisions/0003-config-precedence.md)). `--products-per-keyword 5` overrides `batch.products_per_keyword`. Without the flag, the bundled YAML gives `1` for `products_per_keyword` and `1` for `max_products`; without the YAML keys, the model defaults are `2` and `2`.

### Examples

| Command | Result |
|---|---|
| No inputs on the command line, bundled config | 1 product: the run stops at the `max_products: 1` cap. Raise `--max-products` to take one per keyword across the pool. |
| `--keywords "a" "b" "c" --products-per-keyword 2 --max-products 10` | Up to 6 products, 2 per keyword. |
| `--keywords "a" "b" "c" --products-per-keyword 2 --max-products 4` | 4 products: 2 from "a", 2 from "b", and "c" skipped because the cap is reached. |
| `--product-ids B0A B0B B0C B0D B0E` | 5 products: every id is processed, and `max_products` doesn't apply to ids. |

## Configuration keys

`config/scraper.yaml` has three top-level blocks: `global_settings`, `batch` and `scrapers`. The Pydantic models are in `src/scraper/config_models.py`; every block refuses keys it doesn't declare, so a misspelled key fails at load instead of reading as a default. Where the bundled YAML and the model default differ, the table shows both as `YAML (model)`. The general configuration layout is in [the configuration reference](configuration.md#6-scraper-configuration-configscraperyaml).

### `global_settings`

| Key | Type | Default | Effect |
|---|---|---|---|
| `debug_mode` | bool | `false` | Debug mode without `--debug`. |
| `count_products_with_media` | bool | `true` | Counts only products that pass media validation. |
| `retries` | int | `3` | Browser task retries. Not in the bundled YAML. |
| `proxy` | string or null | `null` | Browser proxy. Not in the bundled YAML. |
| `output_config.base_directory` | string | `outputs` | Base directory for scraper outputs. |
| `output_config.file_patterns` | mapping | `product_file: "{keyword}_products.json"`, `image_file: "{asin}_image_{index}.{ext}"`, `video_file: "{asin}_video_{index}.{ext}"` | File name patterns. |

#### `retry_config`

Retries of failed network operations, with a delay of `min(base_delay * backoff_factor ^ attempt, max_delay)`.

| Key | Type | Default | Effect |
|---|---|---|---|
| `default_max_retries` | int | `3` | Retries after the first attempt. |
| `base_delay` | float | `1.0` | First retry delay, seconds. |
| `max_delay` | float | `60.0` | Delay ceiling, seconds. |
| `backoff_factor` | float | `2.0` | Multiplier per retry. |
| `use_jitter` | bool | `true` | Randomises each delay. |
| `jitter_factor` | float, 0 to 1 | `0.5` | Spread of the jitter. |

#### `rate_limiting`

| Key | Type | Default | Effect |
|---|---|---|---|
| `video_validation_delay` | [min, max] float | `[0.5, 1.5]` | Random pause between video HEAD requests, seconds. |
| `debug_pause_duration` | int | `5` | Pause in debug operations, seconds. |
| `inter_input_delay_sec` | [min, max] float | `[2.0, 5.0]` | Random pause before each input after the first, on successful runs too. |
| `throttle_backoff_base_sec` | float | `60` | First wait after a throttled input, seconds. Doubles per attempt. |
| `throttle_backoff_max_sec` | float | `600` | Ceiling on one wait, seconds. |
| `throttle_max_attempts` | int | `6` | Retries of a throttled input. The waits are 60, 120, 240, 480, then 960 capped to the ceiling, so 6 is the first count that reaches a 600 s ceiling. |
| `throttle_max_total_wait_sec` | float | `3600` | Total throttle wait for the whole run, compared against the cost of the next retry. A success doesn't reset it. |
| `dead_query_after` | int | `3` | Error pages for one input, in a run where another input got through, before that input is named a dead query and skipped. |

[Scraping explained](../explanation/scraping.md#throttling-and-dead-queries) describes how the run tells the two cases apart.

#### `image_config`

| Key | Type | Default | Effect |
|---|---|---|---|
| `min_high_res_dimension` | int | `1500` | Smallest width or height counted as high resolution, pixels. |
| `min_high_res_file_size` | int | `10000` | Smallest image file kept, bytes (`REQ-SCR-018`). |
| `very_high_res_dimension` | int | `2000` | Images above this size are trusted without a HEAD request. |
| `max_images_per_product` | int | `15` (`10`) | Images extracted per product. |

#### `video_config`

| Key | Type | Default | Effect |
|---|---|---|---|
| `min_dimension` | int | `640` | Smallest width or height of a kept video, pixels. |
| `min_duration` | float | `1.0` | Shortest kept video, seconds. |
| `max_videos_per_product` | int | `10` | Videos extracted per product. |
| `enable_m3u8_monitoring` | bool | `false` | Watches network traffic for HLS stream URLs. |
| `m3u8_download_timeout` | int | `120` | Time limit for converting an HLS stream to MP4, seconds. |
| `network_capture_timeout` | int | `20` | How long network traffic is watched for HLS URLs, seconds. |

#### `download_config`

| Key | Type | Default | Effect |
|---|---|---|---|
| `download_timeout` | int | `30` | Download time limit, seconds. |
| `video_download_timeout` | int | `300` | Video download time limit, seconds. |
| `download_chunk_size` | int | `8192` | Bytes read per streaming iteration. |
| `validation_timeout` | int or null | `null` | Media validation request time limit, seconds. Not in the bundled YAML. |
| `min_image_file_size` | int | `10000` | Smallest downloaded image, bytes. Not in the bundled YAML. |
| `validation_range_bytes` | string | `0-1023` | Byte range requested to check that a video exists. |
| `concurrent_image_downloads` | int | `5` | Parallel image downloads. |
| `concurrent_video_downloads` | int | `3` | Parallel video downloads. |

#### `validation_config`

| Key | Type | Default | Effect |
|---|---|---|---|
| `essential_fields` | list | `[]` | Fields a product must have (`title`, `price`, `description`, `asin`, `rating`; `REQ-SCR-014`). |
| `min_total_media` | int | `3` | Deprecated and ignored: the scraper reads `video_settings.min_total_media` from `config/video_production.yaml`. |
| `min_images_if_no_video` | int | `5` | Deprecated and ignored: the scraper reads `video_settings.min_images_if_no_video` from `config/video_production.yaml`. |
| `min_images_with_video` | int | `2` | Deprecated and ignored: the scraper reads `video_settings.min_images_with_video` from `config/video_production.yaml`. |
| `media_validation_timeout` | int | `30` | Time limit for one media check, seconds. |
| `validation_report_top_issues` | int | `10` | Issues listed in a validation report. |

The three media minimums must match `video_settings` in `config/video_production.yaml`, so that the producer accepts what the scraper keeps (`REQ-SCR-021`).

#### `browser_config`

| Key | Type | Default | Effect |
|---|---|---|---|
| `debug_window_width`, `debug_window_height` | int | `1920`, `1200` | Window size in debug mode. |
| `fallback_window_position` | [x, y, width, height] | `[0, 0, 1920, 1080]` | Window geometry when monitor detection fails. |
| `search_result_timeout` | int | `10` | Wait for search results, seconds. |
| `max_products_per_search` | int | `10` (`5`) | Products read from one results page. |
| `page_load_timeout_ms` | int | `60000` | Page load limit, milliseconds. |
| `script_execution_timeout_ms` | int | `30000` | Injected JavaScript limit, milliseconds. |
| `element_selection_timeout` | int | `10` | Default element wait, seconds. |
| `max_title_selector_attempts` | int | `10` | Title selectors tried before giving up. |
| `search_result_wait_reduced` | int | `5` | Shorter wait used to detect an empty results page, seconds. |

#### Other `global_settings` blocks

| Block | Keys (default) |
|---|---|
| `debug_settings` | `create_media_validation_reports` (`true`), `save_screenshots` (`false`), `save_error_screenshots` (`true`) |
| `system_timeouts` | `system_command_timeout` (`5`), `head_request_timeout` (`10`), `system_profiler_timeout` (`10`), seconds |
| `media_config` | `default_image_extension` (`.jpg`), `amazon_media_domains` (`images-amazon.com`, `m.media-amazon.com`, `media-amazon.com`), `amazon_high_res_suffix` (`._AC_SL2000_.jpg`), `high_res_upgrade_dimension` (`2000`), `js_context_chars` (`500`), `valid_http_status_codes` (`[200, 206]`), `min_file_size_absolute` (`1000` bytes), `ffprobe_timeout_sec` (`30`) |
| `debug_config` | `title_preview_length` (`50`), `url_preview_length` (`100`), `result_preview_length` (`100`), characters shown in logs |
| `css_selectors` | `product_title_selectors` (`#productTitle`, `h1.a-size-large`, `.product-title`, `h1[data-automation-id='product-title']`), `search_result_card` (`div[data-component-type='s-search-result']`) |
| `asin_patterns` | `modern_asin_pattern` (`^B0[A-Z0-9]{8}$`), `legacy_asin_pattern` (`^[A-Z0-9]{10}$`), `url_asin_pattern` (`/dp/([A-Z0-9]{10})`) |
| `batch_processing` | See [media validation and page scanning](#media-validation-and-page-scanning). |

There is no `headless` key: the scraper always runs a headful browser, on a virtual display when no window is wanted. [Scraping explained](../explanation/scraping.md#why-the-scraper-never-runs-headless) has the reason.

### `batch`

| Key | Type | Default | Effect |
|---|---|---|---|
| `product_ids` | list | `[]` | Product ids scraped when the command line names no input. |
| `keywords` | mapping of pillar to list, or list | keywords under `value`, `novelty` and `utility` (`[]`) | Keywords searched when the command line names no input. |
| `products_per_keyword` | int | `1` (`2`) | Products per keyword. |
| `fail_fast` | bool | `false` | Stops at the first failed input. |
| `logging.duration_decimal_places` | int | `2` | Decimals of durations in the summary. |
| `logging.media_stats_decimal_places` | int | `2` | Decimals of media statistics in the summary. |

### `scrapers.amazon`

| Key | Type | Default | Effect |
|---|---|---|---|
| `enabled` | bool | `true` | Whether the scraper is available. |
| `base_url` | string | `https://www.amazon.com` | Site the scraper visits. |
| `keywords` | list | `["keyboard"]` | Fallback keywords when `batch` names none. |
| `max_products` | int | `1` (`2`) | Cap on validated products for the run. |
| `associate_tag` | string | `""` | Associate tag, used when `AMAZON_ASSOCIATE_TAG` is unset. |
| `affiliate_links.enabled` | bool | `true` | Whether the install takes part in an affiliate program. See [affiliate URLs](#affiliate-urls). |
| `default_search_parameters` | mapping | see below | Search filters for every keyword search, in the standalone scraper and the batch alike (`REQ-SCR-030`). |
| `filter_parameters` | mapping | `price_to_cents_multiplier: 100`, `rating_codes` for 4, 3, 2 and 1 stars, `prime_filter_code: p_85:2470955011`, `free_shipping_filter_code: p_76:419122011` | Amazon's codes for the filters. |
| `http_headers` | mapping | `video_validation`, `media_download` and `standard` header sets | Headers sent with media requests; see below. |

`default_search_parameters`:

| Key | Type | Default | Effect |
|---|---|---|---|
| `min_price` | float or null | `20` (`null`) | Minimum price. |
| `max_price` | float or null | `250` (`null`) | Maximum price. |
| `min_rating` | float or null, 1 to 5 | `4` (`null`) | Minimum rating. |
| `prime_only` | bool | `false` | Prime-eligible only. |
| `free_shipping` | bool | `false` | Free shipping only. |
| `brands` | list | `[]` | Brand names. |
| `sort_order` | string | `relevanceblender` | Amazon sort token (see [search filters](#search-filters)). |

`http_headers`, each a mapping of header name to value:

| Set | Headers | Sent with |
|---|---|---|
| `video_validation` | `User-Agent`, `Accept`, `Accept-Language`, `Accept-Encoding` (`identity`), `Referer` (`https://www.amazon.com/`) | Requests that check a video URL before download. |
| `media_download` | `User-Agent`, `Accept`, `Accept-Language`, `Referer` (`https://www.amazon.com/`) | Image and video downloads. |
| `standard` | `User-Agent` | The fallback User-Agent when a set has none. |

## Affiliate URLs

The scraper writes each product's affiliate URL into `data.json` as `affiliate_link`, and the page URL as visited into `url` (`REQ-SCR-053`). `build_affiliate_url` in `src/scraper/amazon/utils.py` canonicalises every URL to `<host>/dp/<ASIN>?tag=<tag>`, taking the ASIN from a `/dp/`, `/gp/product/`, `/gp/aw/d/` or `/product/` path, or from the product when the URL carries none; the host is the URL's own Amazon marketplace, `www.amazon.com` otherwise (`REQ-SCR-051`).

| Source | Precedence |
|---|---|
| Associate tag | `AMAZON_ASSOCIATE_TAG`, then `scrapers.amazon.associate_tag` (`REQ-SCR-052`). |
| Affiliate program flag | `AMAZON_AFFILIATE_LINKS_ENABLED` (`0`, `false`, `no` or `off` turn it off), then `scrapers.amazon.affiliate_links.enabled` (`REQ-SCR-059`). |

The standalone scraper CLI loads `.env` at startup, so a tag set only in `.env` reaches the URL builder without an `export` (`REQ-SCR-055`).

| Tag resolves | `affiliate_links.enabled` | Result |
|---|---|---|
| yes | either | Canonical URL with the tag. The flag can't discard a working tag (`REQ-SCR-058`). |
| no | `true` | Input URL unchanged, and a WARNING that affiliate attribution is lost, in `outputs/logs/scraper-<date>.log` (`REQ-SCR-056`). |
| no | `false` | Bare `https://www.amazon.com/dp/<ASIN>` with tracking parameters stripped, and a DEBUG line instead of the warning (`REQ-SCR-057`). |

A misspelled key inside `affiliate_links` (`enabld: false`), or a misspelled block name, is refused at config load.

## URL shortener

Configured in `config/url_shortener.yaml` with `url_shortener.provider`; the full key list is in [the configuration reference](configuration.md#7-url-shortener-configuration-configurl_shorteneryaml).

| Provider | Default | Requires | Behaviour |
|---|---|---|---|
| `bare` | yes | nothing | Returns the canonical affiliate URL unchanged, with no network call (`REQ-SCR-064`). |
| `picsee` | opt-in | `PICSEE_API_KEY` in `.env`, a Picsee account | Mints a `stte.psee.io` short code that redirects (302) to the affiliate URL. |

[Scraping explained](../explanation/scraping.md#url-shortening) covers why `bare` is the default.
