# Scrape products from Amazon

The scraper collects product data and media from Amazon for video production, and writes each product to `outputs/<ASIN>/`. This guide covers the common runs. Every flag and key is listed in [the scraper reference](../reference/scraper.md); multi-product runs and the global pipeline are covered in [batch processing](batch-processing.md).

A full scrape drives Chromium and holds memory for the whole search. On a machine you are also working on, run it through `make scrape-lowpri ARGS="..."`, which caps its memory and lowers its priority; the commands below use the bare module form so that the flags are easy to read.

## Before you start

1. Install the project as described in [installation](installation.md).
2. Install the `xvfb` package (`sudo apt-get install -y xvfb`). The scraper runs a headful browser on a virtual display except in a debug run on X11; without Xvfb, it falls back to a detectable headless mode. [Scraping explained](../explanation/scraping.md#virtual-display) has the reason.
3. Set `AMAZON_ASSOCIATE_TAG` in `.env` if you take part in the Amazon affiliate program. The scraper loads `.env` at startup.

## Scrape one product

```bash
poetry run python -m src.scraper.amazon.scraper --product-ids B0BTYCRJSS --debug
```

Pass several ids to scrape them all in one run:

```bash
poetry run python -m src.scraper.amazon.scraper --product-ids B0BTYCRJSS B08DTZM7LM B07ZPC9QD4
```

Every id is processed, whatever `max_products` is set to.

## Scrape from URLs

`--product-ids` also takes full or shortened Amazon URLs. The scraper follows the redirect and reads the ASIN from the product page URL:

```bash
poetry run python -m src.scraper.amazon.scraper --product-ids "https://tr.ee/mUk1eH" --output-dir tmp --debug
```

## Search by keyword

```bash
poetry run python -m src.scraper.amazon.scraper --keywords "wireless earbuds" --min-rating 4.5 --debug
```

The filters from `scrapers.amazon.default_search_parameters` in `config/scraper.yaml` apply to every search; a filter flag overrides its field for the run. See [search filters](../reference/scraper.md#search-filters).

## Collect more than one product

The bundled config stops a run after the first validated product (`max_products: 1`). Set the caps on the command line:

```bash
poetry run python -m src.scraper.amazon.scraper --keywords "earbuds" "headphones" --products-per-keyword 3 --max-products 5 --debug
```

`--products-per-keyword` caps each keyword and `--max-products` caps the run. [Product count](../reference/scraper.md#product-count) has the rules and worked examples.

## Scrape from a file

1. Write the product ids or URLs to a file, one per line.
2. Run the scraper with `--input-file`, and `--batch-size` to process the entries in chunks:

   ```bash
   poetry run python -m src.scraper.amazon.scraper --input-file products.txt --batch-size 10 --debug
   ```

The file's entries are merged with any `--product-ids`, and duplicates are dropped.

## Scrape from the config

Run the scraper with no inputs to take them from `config/scraper.yaml`:

```bash
poetry run python -m src.scraper.amazon.scraper --debug
```

The run uses `batch.product_ids` and `batch.keywords`, then `scrapers.amazon.keywords` if both are empty. The keyword pool is rotated by date, so consecutive daily runs start at different keywords.

## Run without an affiliate program

If there is no program to attribute links to, declare it so that a missing tag isn't reported as a mistake:

```yaml
scrapers:
  amazon:
    affiliate_links:
      enabled: false
```

Product URLs are then cleaned to a bare `https://www.amazon.com/dp/<ASIN>`, and the missing-tag warning drops to DEBUG. To declare it without editing the tracked config, set `AMAZON_AFFILIATE_LINKS_ENABLED=false` in `.env`. A tag that does resolve is still applied. See [affiliate URLs](../reference/scraper.md#affiliate-urls).

## Watch a debug scrape

On an X11 desktop, `--debug` shows the browser window. On Wayland, the browser runs on a virtual display; to watch it:

1. Install the `xvfb` and `x11vnc` packages.
2. Start the scrape through the watch target:

   ```bash
   make scrape-watch ARGS="--keywords 'wireless earbuds' --max-products 1"
   ```

3. Connect a VNC viewer to `localhost:5900`.

[Troubleshooting](troubleshooting.md#watching-a-debug-scrape-on-wayland-make-scrape-watch) covers the details.

## When a scrape fails

The scraper exits non-zero when it scrapes nothing; add `--strict` to also fail when any input produced nothing. Look for the cause in `outputs/logs/scraper-<date>.log`, then see [the scraper section of troubleshooting](troubleshooting.md#scraper-issues).
