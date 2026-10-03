# Global batch reference

The command line of the global batch pipeline, which scrapes, renders and publishes in one run. For the steps of a batch run, see [the batch processing guide](../guides/batch-processing.md); its exit codes are in [Exit Codes](../guides/batch-processing.md#exit-codes). Its `global_batch` keys in `config/pipeline.yaml` are under [Configuration keys](#configuration-keys). The requirements are in [the batch requirements](../requirements/batch.md).

## Command line

```bash
poetry run python -m src.pipeline [options]
make batch-lowpri ARGS="[options]"
```

The parser is `create_argument_parser` in `src/pipeline/cli.py`. A full run goes through `make batch-lowpri`, which caps its memory. A flag passed on the command line overrides the matching `global_batch` key; a flag left out keeps the YAML value. Each switch backed by a key has a `--no-` form, so a YAML `true` can be turned off for one run, and `--profile` on the command line wins over a YAML `random_profile: true`.

### Input

| Flag | Value | Default | Effect |
|---|---|---|---|
| `--product-ids` | one or more `ASIN` | none | Products to scrape, render and publish. |
| `--keywords` | one or more `KEYWORD` | none | Keywords to search for products. |
| `--topic` | `TITLE` | none | Renders a video about a topic instead of a scraped product. Skips scraping; the record is built from the title. |
| `--topic-description` | `TEXT` | none | Source material the topic script is written from, with `--topic`. |
| `--topic-keywords` | `TERMS` | none | Comma-separated stock media search terms for the topic. |
| `--topics-file` | `FILE` | none | A YAML list of topics to render, each with a title and an optional description and keywords. |
| `--max-products` | `N` | `global_batch.max_products` | Cap on the products collected across all keywords. |
| `--products-per-keyword` | `N` | `global_batch.products_per_keyword` | Cap on the products scraped for one keyword. |

### Scraper filters

| Flag | Value | Effect |
|---|---|---|
| `--min-price` | `PRICE` | Minimum price. |
| `--max-price` | `PRICE` | Maximum price. |
| `--min-rating` | `RATING` (1 to 5) | Minimum star rating. |
| `--prime-only` / `--no-prime-only` | switch | Keeps Prime-eligible items only. |

### Video production

| Flag | Value | Effect |
|---|---|---|
| `--profile` | `NAME` | The profile for every product. Mutually exclusive with `--random-profile`. |
| `--random-profile` / `--no-random-profile` | switch | Picks a profile per product, deterministically from the product id, from `--profile-pool` or from every profile. |
| `--profile-pool` | one or more `PROFILE` | The profiles `--random-profile` picks from. |
| `--voice-profile` | `NAME` | Overrides the voice profile. |
| `--script-template` | `NAME` | Overrides the script template (the name without `.md`). |
| `--cta` | `LINE` | Overrides the closing call to action. It must be one of the configured options; otherwise selection runs as usual. |
| `--pillar` | `NAME` | The content pillar for the run. Prepends the pillar preamble to the prompt and picks the pillar audience; on a product render it also narrows the script templates to that pillar's. |
| `--subtitle-format` | `srt` or `ass` | The subtitle format for the FFmpeg engine. The pycaps engine ignores it, and the bundled YAML selects pycaps, so pair it with `--subtitle-engine ffmpeg`. |
| `--subtitle-engine` | `ffmpeg` or `pycaps` | The caption engine. `ffmpeg` burns SRT or ASS through libass; `pycaps` renders animated captions after assembly and needs the optional `pycaps` dependency group. |
| `--pycaps-template` | `NAME` | Forces one pycaps template for every product by clearing the template pool. |
| `--pycaps-template-pool` | one or more `NAME` | The pycaps templates picked per product, deterministically. |
| `--pycaps-renderer` | `css` or `pictex` | The pycaps renderer. `css` (Playwright and Chromium) is the production renderer; `pictex` is a preview only and renders words without gaps. |

### Run control

| Flag | Value | Default | Effect |
|---|---|---|---|
| `--fail-fast` / `--no-fail-fast` | switch | off | Stops the pipeline at the first failure, publishing included unless `--no-fail-fast-publish` is passed. |
| `--strict` | switch | off | Exits non-zero when any product was lost to a failure or a skip, not only when none succeeded. |
| `--process-all-products` / `--no-process-all-products` | switch | off | Renders every product in the outputs directory, not only those this run scraped. |
| `--outputs-dir` | `PATH` | `outputs` | Where the scraper writes and the producer reads. Its default shadows the YAML's `global_output_directory` even when the flag isn't passed (`REQ-OPS-002`). |
| `--debug` / `--no-debug` | switch | off | Debug logging. |
| `--resume` | switch | off | Resumes an interrupted run from its checkpoint, skipping completed products and phases. |
| `--dry-run` | switch | off | Validates the configuration and prints the planned products, profiles and platforms without running anything. |
| `--clean` | switch | off | Removes product directories from the outputs directory before the run; with `--product-ids`, only those products. |
| `--output-format` | `text` or `json` | `text` | The format of the end-of-run summary. `json` carries every statistic and timestamp. |

### Publishing

| Flag | Value | Default | Effect |
|---|---|---|---|
| `--skip-publish` / `--no-skip-publish` | switch | off | Skips the publishing phase. |
| `--force` | switch | off | Renders and publishes products already recorded as published. Without it the batch skips them before the render. |
| `--platforms` | one or more `PLATFORM` | `default_platforms` in `config/publisher.yaml` | The platforms to publish to. |
| `--schedule-time` | `ISO8601` | none | Schedules the posts for this time instead of the next free slot. |
| `--fail-fast-publish` / `--no-fail-fast-publish` | switch | off | Stops publishing at the first failure. |
| `--platform-specific` / `--no-platform-specific` | switch | off | Creates a separate post per platform with that platform's metadata, instead of one post for all platforms. |

## Configuration keys

The `global_batch` block of `config/pipeline.yaml`. A boolean set here holds until a flag says otherwise, in either direction.

### Inputs

| Key | Default | Effect |
|---|---|---|
| `product_ids` | `[]` | Products every run scrapes. |
| `keywords` | `{}` | Keywords grouped by content pillar, or a flat list with no pillar. Empty means the run draws from `batch.keywords` in `config/scraper.yaml`; a value here replaces that pool for batch runs. |
| `keywords_per_run` | unset | How many keywords one run searches, taken in rotation by date. Unset, it is what the run consumes, `max_products` divided by `products_per_keyword`. |
| `topics` | two sample topics | Topics rendered without scraping, each a `title` with an optional `description` and `keywords`. Same shape as `--topics-file`. |
| `topics_file` | `null` | A file whose topics replace `topics`. A relative path resolves against `config/`. `PIPELINE_TOPICS_FILE` overrides it; a missing file is an error. |
| `topics_per_run` | `1` | How many topics a run with no input flags includes, taken in rotation by date. `0` renders products only. Topics on the command line ignore it. |
| `alternate_formats` | `true` | A run with no input flags renders one side per day, topics or products, chosen by date parity. A side with nothing configured falls back to the other with a warning. |
| `max_products` | `10` | Cap on the products collected across all keywords. |
| `products_per_keyword` | `1` | Cap on the products scraped for one keyword. |
| `scraper_filters` | all `null` | `min_price`, `max_price`, `min_rating` and `prime_only`. A `null` takes `scrapers.amazon.default_search_parameters` from `config/scraper.yaml`; a flag overrides both. |

### Production and run control

| Key | Default | Effect |
|---|---|---|
| `profile` | `null` | The profile for every product. Mutually exclusive with `random_profile`. |
| `random_profile` | `false` | Picks a profile per product, deterministically from the product id. |
| `profile_pool` | `[]` | The profiles `random_profile` picks from; empty means every profile. |
| `fail_fast` | `false` | Stops the pipeline at the first failure. |
| `outputs_dir` | `outputs` | Where the scraper writes and the producer reads. |
| `debug` | `false` | Debug logging. |

### `webhook`

Posts phase and pipeline events to an HTTP endpoint. A failed call never stops the pipeline.

| Key | Default | Effect |
|---|---|---|
| `url` | `null` | The endpoint, `http://` or `https://`. `null` turns webhooks off. |
| `enabled` | `true` | Sends events when `url` is set. |
| `timeout_sec` | `5.0` | Timeout for one call. |
| `max_retries` | `3` | Retries for a failed call, with the delay doubling each time. |
| `retry_delay_sec` | `1.0` | The first retry delay. |
| `events` | all four | Which of `phase.complete`, `phase.failed`, `pipeline.complete` and `pipeline.failed` to send. |
