# Scraper requirements

Ids use the prefix `REQ-SCR`. The format and the statuses are described in [the requirements index](README.md).

## Inputs

- **REQ-SCR-001** `shipped` The scraper accepts product ids, keyword searches, or both in one run.
- **REQ-SCR-002** `shipped` The scraper accepts a full or shortened product URL as a product input and follows its redirects to the product page.
- **REQ-SCR-003** `shipped` When `--input-file` is passed, the scraper reads product ids or URLs from the file, one per line, and merges them with `--product-ids`.
- **REQ-SCR-004** `shipped` When `--batch-size N` is passed, the scraper processes its inputs in chunks of N.
- **REQ-SCR-005** `shipped` If a product id doesn't match the platform's id format, the scraper rejects it.
- **REQ-SCR-006** `shipped` When no keywords are given on the command line, the scraper rotates its configured keyword pool by date, so consecutive daily runs start at different keywords.
- **REQ-SCR-007** `shipped` Keywords passed with `--keywords` are searched in the order given.
- **REQ-SCR-008** `shipped` The scraper processes inputs one at a time, with a delay between inputs.
- **REQ-SCR-009** `shipped` The scraper keeps one copy of a product found through more than one input.
- **REQ-SCR-065** `shipped` When `--clean` is passed, the scraper removes every product directory, `cache/`, `temp/`, `screenshots/` and its loose `.json`, `.csv`, `.xlsx` and `.html` files from the outputs directory before running, and keeps report files and logs.
- **REQ-SCR-066** `shipped` When no inputs are given on the command line, the scraper takes `batch.product_ids` and `batch.keywords` from `scraper.yaml`, then `scrapers.amazon.keywords`, and refuses to run if none is configured.

## Product data

- **REQ-SCR-010** `shipped` The scraper records each product's title, price, description, product id, rating and reviews.
- **REQ-SCR-011** `shipped` The scraper records the price as a plain decimal number, parsed from both US (comma grouping, dot decimal) and European (dot grouping, comma decimal) price formats.
- **REQ-SCR-012** `shipped` When the product page yields no rating, the scraper records the rating shown in the search results.
- **REQ-SCR-013** `shipped` By default, the scraper rejects a product without a title.
- **REQ-SCR-014** `shipped` Where `global_settings.validation_config.essential_fields` lists fields (`title`, `price`, `description`, `asin`, `rating`), the scraper rejects a product missing any of them.

## Media

- **REQ-SCR-015** `shipped` The scraper downloads each product's high-resolution images.
- **REQ-SCR-016** `shipped` The scraper downloads product videos only when the target profile uses them.
- **REQ-SCR-017** `shipped` Where the target profile uses no video, the scraper skips video extraction and download entirely.
- **REQ-SCR-018** `shipped` The scraper discards a downloaded image below the configured minimum file size (`image_config.min_high_res_file_size`, default 10 KB).
- **REQ-SCR-019** `shipped` If a downloaded file fails validation, the scraper removes it and logs the rejection with its reason.
- **REQ-SCR-020** `shipped` The scraper stores each product's media in a directory named after its product id.
- **REQ-SCR-021** `shipped` The scraper keeps a product only when its validated media meet the configured minimums: at least 3 files in total, at least 5 images without a video, and at least 2 images with a video.
  - Why: the minimums mirror the producer's, so a kept product is one the producer can render.
- **REQ-SCR-022** `shipped` Where the target profile uses no video, the scraper ignores downloaded videos when it checks the media minimums.
- **REQ-SCR-023** `shipped` If a product fails the media minimums, the scraper removes its output directory and logs the reason.
- **REQ-SCR-024** `shipped` The media counts in the scraper's summaries are validated files on disk per product, not URLs found on the page.
- **REQ-SCR-067** `shipped` The scraper extracts at most `image_config.max_images_per_product` images (15 in the bundled config) and `video_config.max_videos_per_product` videos (default 10) per product.
- **REQ-SCR-068** `shipped` Where `video_config.enable_m3u8_monitoring` is on (off by default), the scraper also captures product videos from the HLS streams the page loads when its video player starts.

## Search and filtering

- **REQ-SCR-025** `shipped` Keyword searches accept filters for price range, minimum rating, free shipping, Prime eligibility and brands.
- **REQ-SCR-026** `shipped` When `--sort` is passed, the scraper requests search results in that order.
- **REQ-SCR-027** `shipped` When `--debug` is passed and the site redirects to a different regional site, the scraper logs the redirect.
- **REQ-SCR-028** `shipped` The scraper skips sponsored, placeholder and non-product search-result cards without waiting on a per-element timeout.
- **REQ-SCR-029** `shipped` The time the scraper spends classifying one search-result card is bounded.
- **REQ-SCR-030** `shipped` The configured `scrapers.amazon.default_search_parameters` (price, rating, Prime) apply to the standalone scraper and the batch alike.
- **REQ-SCR-031** `shipped` When a filter flag is passed on the command line, it overrides that field of the default search parameters.
- **REQ-SCR-032** `shipped` The scraper's search log line states the filters in force.
- **REQ-SCR-033** `shipped` When a keyword search yields fewer validated products than its per-keyword target, the scraper searches further result pages, up to a configured page limit.
- **REQ-SCR-069** `shipped` If the resolved search parameters are invalid, such as a price or rating out of range, the scraper logs each error and refuses to run.

## Product limits

- **REQ-SCR-034** `shipped` `max_products` caps the total number of products collected across all keywords.
- **REQ-SCR-035** `shipped` `products_per_keyword` caps the number of products collected for one keyword.
- **REQ-SCR-036** `shipped` `max_products` and `products_per_keyword` have the same meaning in the scraper and the batch, and the same flags, `--max-products` and `--products-per-keyword`.
  - Check: other flags are not shared by name, for example the scraper's output directory flag is `--output-dir` and the batch's is `--outputs-dir`.
- **REQ-SCR-037** `shipped` While `max_products` isn't reached, the scraper moves on to the next keyword.
- **REQ-SCR-038** `shipped` When `max_products` is reached, the scraper stops, even if keywords remain.

## Detection and failures

- **REQ-SCR-039** `shipped` The scraper uses a randomised browser user agent.
- **REQ-SCR-040** `shipped` The scraper pauses for a short random interval between page actions.
- **REQ-SCR-041** `shipped` The scraper varies the browser window size but keeps it desktop-width (at least 1280 pixels wide).
  - Why: a narrow window gets the mobile layout, which has no extractable product cards.
- **REQ-SCR-042** `shipped` If one input fails, the scraper continues with the next input.
- **REQ-SCR-043** `shipped` When `--fail-fast` is passed, the scraper stops at the first failed input.
- **REQ-SCR-070** `shipped` `batch.fail_fast` in `scraper.yaml` sets whether the scraper stops at the first failed input, and `--fail-fast` or `--no-fail-fast` overrides it for one run.
- **REQ-SCR-044** `shipped` If the site's error page blocks every input in the run, the scraper treats it as throttling and retries with growing waits (default 60 s, doubling, capped at 600 s, at most 6 attempts).
- **REQ-SCR-045** `shipped` The scraper spends at most a configured total on throttle waits in one run (default 3600 s).
- **REQ-SCR-046** `shipped` If an input keeps returning the error page while other inputs in the run succeed, the scraper calls it a dead query after a configured number of attempts (default 3) and moves on.
- **REQ-SCR-047** `shipped` The scraper's run summary reports success and failure counts.
- **REQ-SCR-048** `shipped` The scraper's run summary lists dead queries and throttled inputs separately.
- **REQ-SCR-049** `partial` If a run scrapes no product, the scraper exits non-zero.
  - Gap: a run that stops before scraping because the input file is missing, no inputs are configured or the search filters are invalid exits 0 (#587).
- **REQ-SCR-050** `shipped` When `--strict` is passed, the scraper also exits non-zero if any product id or keyword produced nothing.
- **REQ-SCR-071** `shipped` When `--debug` or `--verbose` is passed, or `global_settings.debug_mode` is true, the scraper runs a visible browser and logs at debug level.
- **REQ-SCR-072** `shipped` While debug mode is on, when the site shows its error page or a search page yields no product cards, the scraper saves a screenshot, unless `global_settings.debug_settings.save_error_screenshots` is false.

## Debugging

- **REQ-SCR-073** `shipped` When `--debug` is passed, `--save-page-source`, `--save-screenshots`, `--analyze-images` and `--dump-image-urls` each write that artifact for every product page to the `debug/image_analysis` folder of the temp directory.
- **REQ-SCR-074** `shipped` Without `--debug`, `--save-page-source`, `--save-screenshots`, `--analyze-images` and `--dump-image-urls` have no effect.
- **REQ-SCR-075** `shipped` `make scrape-watch` runs a debug scrape in a dedicated virtual display that can be watched over VNC on `localhost:5900`, stops the display afterwards and exits with the scraper's exit code.

## Affiliate URLs

- **REQ-SCR-051** `partial` The scraper canonicalises every scraped product's affiliate URL to `https://www.amazon.com/dp/<ASIN>?tag=<associate_tag>` before writing `data.json`.
  - Gap: a product URL without a `/dp/<ASIN>` path keeps its original form with only the tag appended (#584).
- **REQ-SCR-052** `shipped` The scraper reads the associate tag from the `AMAZON_ASSOCIATE_TAG` environment variable, falling back to `scrapers.amazon.associate_tag` in the YAML.
- **REQ-SCR-053** `shipped` `data.json` carries the canonical link in `affiliate_link` and the page URL as visited in `url`.
- **REQ-SCR-054** `shipped` The publisher's link-in-bio reads `affiliate_link` before `url`.
- **REQ-SCR-055** `shipped` The standalone scraper loads `.env` at startup, so a tag set only in `.env` is applied.
- **REQ-SCR-056** `shipped` If no associate tag resolves, the scraper keeps the input URL unchanged and logs a warning that affiliate attribution is lost.
- **REQ-SCR-057** `shipped` Where `scrapers.amazon.affiliate_links.enabled` is `false` and no tag resolves, the scraper strips the URL to `https://www.amazon.com/dp/<ASIN>` and logs at debug level instead of warning.
- **REQ-SCR-058** `shipped` When an associate tag resolves, the scraper applies it even where `scrapers.amazon.affiliate_links.enabled` is `false`.
- **REQ-SCR-059** `shipped` The `AMAZON_AFFILIATE_LINKS_ENABLED` environment variable overrides `scrapers.amazon.affiliate_links.enabled`.

## URL shortener

- **REQ-SCR-060** `shipped` The URL shortener provider is chosen with `url_shortener.provider`.
- **REQ-SCR-061** `shipped` The bundled providers are `bare` and `picsee`.
- **REQ-SCR-062** `shipped` The `picsee` provider is opt-in and requires an API key.
- **REQ-SCR-063** `shipped` The default provider is `bare`, so a fresh install needs no shortener API key.
- **REQ-SCR-064** `shipped` The `bare` provider makes no network calls and returns the canonical affiliate URL unchanged.
- **REQ-SCR-076** `shipped` Where `url_shortener.enabled` and `url_shortener.integration.shorten_on_scrape` are on, the scraper writes a `shortened_affiliate_link` for each product to `data.json`.
- **REQ-SCR-077** `shipped` If shortening a product's link fails and `url_shortener.integration.fallback_to_original` is on, the scraper writes the canonical affiliate link as its `shortened_affiliate_link`.
- **REQ-SCR-078** `shipped` If the active provider's API key environment variable is unset, the scraper skips shortening and writes no `shortened_affiliate_link`.
- **REQ-SCR-079** `shipped` Where `url_shortener.picsee.custom_domain` is set, the `picsee` provider mints short links on that domain.
