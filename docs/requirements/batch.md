# Batch requirements

Ids use the prefix `REQ-BAT`. The format and the statuses are described in [the requirements index](README.md).

## Producer batch mode

- **REQ-BAT-001** `shipped` When `--batch` is passed, the producer renders every product directory in the outputs directory that holds a `data.json`.
- **REQ-BAT-002** `shipped` The producer's batch mode leaves topic directories out of discovery.
- **REQ-BAT-003** `shipped` When `--product-ids` is passed with `--batch`, the producer renders only those products.
- **REQ-BAT-004** `shipped` When `--batch-profile` is passed, the producer renders every product with that profile.
- **REQ-BAT-005** `shipped` When `--random-profile` is passed, the producer picks a profile for each product from the profile pool.
- **REQ-BAT-006** `shipped` If `--batch` is passed with both or neither of `--batch-profile` and `--random-profile`, the producer refuses to start.
- **REQ-BAT-007** `shipped` The producer pauses for a random delay between products (`video_settings.inter_product_delay_min_sec` to `inter_product_delay_max_sec`, default 1.5-4.0 s).
- **REQ-BAT-008** `shipped` When `--fail-fast` is passed, the producer's batch stops at the first failed product.
- **REQ-BAT-009** `shipped` The producer's batch summary reports attempted, succeeded, failed and skipped counts, the total and average duration, and, with `--random-profile`, the profile distribution.
- **REQ-BAT-010** `shipped` When `--output-format json` is passed, the producer prints its batch summary as JSON.
- **REQ-BAT-011** `shipped` If the producer's batch renders no video, the producer exits non-zero.
- **REQ-BAT-012** `shipped` When `--strict` is passed, the producer also exits non-zero if any product failed or was skipped.

## Global pipeline

- **REQ-BAT-013** `shipped` The batch runs scraping, video production and publishing end to end from a single command.
- **REQ-BAT-014** `shipped` If a product fails in one phase, the batch continues with its other products and later phases.
- **REQ-BAT-015** `shipped` The batch reads its settings from `config/pipeline.yaml`, and a CLI flag overrides the matching YAML setting.
- **REQ-BAT-016** `shipped` The batch accepts the producer's render override flags `--voice-profile`, `--script-template`, `--cta`, `--pillar`, `--subtitle-format`, `--subtitle-engine`, `--pycaps-template`, `--pycaps-template-pool` and `--pycaps-renderer`.
- **REQ-BAT-017** `planned #566` The batch accepts the same render override flags as the producer.

## Execution phases

- **REQ-BAT-018** `shipped` In the scraping phase, the batch collects product data, images and videos for its product inputs.
- **REQ-BAT-019** `shipped` For topic inputs, the batch writes a topic record in place of a scrape.
- **REQ-BAT-020** `shipped` In the handoff phase, the batch passes on only products that have a `data.json` and were scraped or prepared in the current run.
- **REQ-BAT-021** `shipped` When `--process-all-products` is passed, the handoff passes on every product in the outputs directory.
- **REQ-BAT-022** `shipped` The batch refuses `--process-all-products` only on a topics-only run.
- **REQ-BAT-023** `shipped` In the production phase, the batch renders each handed-off product with its selected profile.
- **REQ-BAT-024** `shipped` In the publishing phase, the batch uploads and schedules each rendered video to the target platforms.

## Already-published products

- **REQ-BAT-025** `shipped` The batch drops a product already recorded as published on every target platform before rendering it.
  - Why: this filter is the batch's only guard against a second post for a product it re-scraped.
- **REQ-BAT-026** `shipped` The batch keeps a product that is published on some target platforms but not all.
- **REQ-BAT-027** `shipped` The batch never drops a topic as already published.
- **REQ-BAT-028** `shipped` When `--force` is passed, the batch renders and publishes already-published products.
- **REQ-BAT-029** `shipped` When `--skip-publish` is passed, the batch renders already-published products.
- **REQ-BAT-030** `shipped` The batch summary lists the products dropped as already published.

## Inputs and selection

- **REQ-BAT-031** `shipped` Where `global_batch.alternate_formats` is on and no input flags are given, the batch runs either its topics or its products on a given day, chosen by date parity.
- **REQ-BAT-032** `shipped` If the side chosen for the day has nothing configured, the batch runs the other side and logs a warning.
- **REQ-BAT-033** `shipped` When inputs are passed on the command line, the batch runs them without alternation.
- **REQ-BAT-034** `shipped` When no input flags are given, the batch rotates its configured keyword pool by date, so consecutive runs search different keywords.
- **REQ-BAT-035** `shipped` The batch takes each search filter (`min_price`, `max_price`, `min_rating`, `prime_only`) from the CLI flag first, then `global_batch.scraper_filters`, then the scraper's default search parameters.
  - Check: a `null` in `scraper_filters` applies the scraper's default, not "no filter".
- **REQ-BAT-036** `shipped` When `--clean` is passed with `--product-ids`, the batch removes only those products' directories before running.

## Profile randomization

- **REQ-BAT-037** `shipped` With random profiles, the batch picks a profile for each product from the configured pool.
- **REQ-BAT-038** `shipped` When no pool is configured, the batch picks from all profiles except `base` and `slideshow_stock`.
- **REQ-BAT-039** `shipped` `base` and `slideshow_stock` stay usable as an explicit profile choice.
- **REQ-BAT-040** `partial` The batch picks the same random profile for the same product on every run.
  - Gap: the choice is stable within one run but differs from one run to the next.
- **REQ-BAT-041** `shipped` If a product lacks the media its profile needs, the batch skips it during rendering and counts it as skipped.
- **REQ-BAT-042** `shipped` If a topic run would draw a profile that sources no stock media, the batch refuses to start.

## Run control

- **REQ-BAT-043** `shipped` When `--resume` is passed, the batch continues an interrupted run from its last checkpoint without redoing completed phases or products.
- **REQ-BAT-044** `shipped` When `--dry-run` is passed, the batch validates its config and shows the planned products, profiles and platforms without running anything.
- **REQ-BAT-045** `shipped` When `--output-format json` is passed, the batch prints its summary as JSON.
- **REQ-BAT-046** `shipped` Where `global_batch.webhook.url` is set, the batch sends a webhook on each `phase.complete`, `phase.failed`, `pipeline.complete` and `pipeline.failed` event.
- **REQ-BAT-047** `shipped` If a webhook call fails, the batch logs it and carries on.
- **REQ-BAT-048** `shipped` If no product completes end to end, the batch exits non-zero.
- **REQ-BAT-049** `shipped` If no product completes because every product was already published and nothing else was lost, the batch exits 0.
- **REQ-BAT-050** `shipped` If some products complete and others fail or are skipped, the batch exits 0.
- **REQ-BAT-051** `shipped` When `--strict` is passed, the batch exits non-zero if any product failed or was skipped.

## Summary reporting

- **REQ-BAT-052** `shipped` The batch summary reports success, failure and skipped counts for each phase.
- **REQ-BAT-053** `shipped` The batch summary reports the end-to-end success count, the scraped-only count, total failures and total duration.
- **REQ-BAT-054** `shipped` The batch summary reports how many products each profile rendered.
- **REQ-BAT-055** `partial` The batch summary's media counts are validated files on disk per product, matching the scraper's own count.
  - Gap: the batch counts the files the download step reported, which differs from the files on disk when earlier files already existed.
