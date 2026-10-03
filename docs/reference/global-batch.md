# Global batch reference

The command line of the global batch pipeline, which scrapes, renders and publishes in one run. For the steps of a batch run, see [the batch processing guide](../guides/batch-processing.md); its exit codes are in [Exit Codes](../guides/batch-processing.md#exit-codes). The `global_batch` keys of `config/pipeline.yaml` are in [the configuration reference](configuration.md). The requirements are in [the batch requirements](../requirements/batch.md).

## Command line

```bash
poetry run python -m src.pipeline [options]
make batch-lowpri ARGS="[options]"
```

The parser is `create_argument_parser` in `src/pipeline/cli.py`. A full run goes through `make batch-lowpri`, which caps its memory. A flag passed on the command line overrides the matching `global_batch` key; a flag left out keeps the YAML value.

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
| `--prime-only` | switch | Keeps Prime-eligible items only. |

### Video production

| Flag | Value | Effect |
|---|---|---|
| `--profile` | `NAME` | The profile for every product. Mutually exclusive with `--random-profile`. |
| `--random-profile` | switch | Picks a profile per product, deterministically from the product id, from `--profile-pool` or from every profile. |
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
| `--fail-fast` | switch | off | Stops the pipeline at the first failure. |
| `--strict` | switch | off | Exits non-zero when any product was lost to a failure or a skip, not only when none succeeded. |
| `--process-all-products` | switch | off | Renders every product in the outputs directory, not only those this run scraped. |
| `--outputs-dir` | `PATH` | `outputs` | Where the scraper writes and the producer reads. Its default shadows the YAML's `global_output_directory` even when the flag isn't passed (`REQ-OPS-002`). |
| `--debug` | switch | off | Debug logging. |
| `--resume` | switch | off | Resumes an interrupted run from its checkpoint, skipping completed products and phases. |
| `--dry-run` | switch | off | Validates the configuration and prints the planned products, profiles and platforms without running anything. |
| `--clean` | switch | off | Removes product directories from the outputs directory before the run; with `--product-ids`, only those products. |
| `--output-format` | `text` or `json` | `text` | The format of the end-of-run summary. `json` carries every statistic and timestamp. |

### Publishing

| Flag | Value | Default | Effect |
|---|---|---|---|
| `--skip-publish` | switch | off | Skips the publishing phase. |
| `--force` | switch | off | Renders and publishes products already recorded as published. Without it the batch skips them before the render. |
| `--platforms` | one or more `PLATFORM` | `default_platforms` in `config/publisher.yaml` | The platforms to publish to. |
| `--schedule-time` | `ISO8601` | none | Schedules the posts for this time instead of the next free slot. |
| `--fail-fast-publish` | switch | off | Stops publishing at the first failure. |
| `--platform-specific` | switch | off | Creates a separate post per platform with that platform's metadata, instead of one post for all platforms. |
