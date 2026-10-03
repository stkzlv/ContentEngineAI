# Operations requirements

Ids use the prefix `REQ-OPS`. The format and the statuses are described in [the requirements index](README.md).

## Configuration

- **REQ-OPS-001** `partial` The config resolves each setting from four tiers, highest first: CLI flags, the machine environment, the profile, the YAML files.
  - Gap: environment overrides are applied when the YAML loads and the profile merges afterwards, so for a key a profile also sets, the profile wins over the environment.
- **REQ-OPS-002** `partial` A CLI flag overrides a lower tier only when the user passes it.
  - Gap: `--outputs-dir` defaults to `outputs`, so it shadows `global_output_directory` from the YAML even when it isn't passed.
- **REQ-OPS-003** `planned (decision 0003)` The environment holds only secrets and machine-specific settings; behaviour settings live in the YAML files or a profile.
- **REQ-OPS-004** `shipped` The YAML files under `config/` hold the application settings, contain no secrets and are safe to commit.
- **REQ-OPS-005** `shipped` A CLI flag can override a nested setting, such as `--pycaps-template` overriding `subtitle_settings.pycaps.template_name`.
- **REQ-OPS-006** `shipped` If the config is invalid, the pipeline stops at startup with an error that names the setting and the problem.

## Secrets

- **REQ-OPS-007** `shipped` The pipeline reads API keys and other secrets from the `.env` file or the environment, and git ignores `.env`.
- **REQ-OPS-008** `partial` The repository ships `.env.example`, listing the environment variables the pipeline reads.
  - Gap: it also lists `SUBTITLE_FONT`, `SUBTITLE_FONT_COLOR`, `SUBTITLE_OUTLINE_COLOR` and `SUBTITLE_BACKGROUND_COLOR`, which nothing reads.
- **REQ-OPS-009** `planned (decision 0003)` `.env.example` lists only secrets and machine-specific settings.
- **REQ-OPS-010** `shipped` Log output masks API keys, tokens and other secret-shaped values before they reach the console or a log file.

## Test isolation

- **REQ-OPS-011** `shipped` If the developer's `.env` file changes during a test run, the test suite fails.
- **REQ-OPS-012** `shipped` A test run writes nothing to the real log files under `outputs/logs/`.
- **REQ-OPS-013** `partial` A test run writes nothing to the real `outputs/` tree and doesn't depend on its contents.
  - Gap: tests can create `TEST*` product directories there, which are removed after each test, and nothing fails when one is written.

## Error handling and resilience

- **REQ-OPS-014** `shipped` If one item in a batch fails, the run continues with the remaining items.
- **REQ-OPS-015** `shipped` The pipeline retries transient network failures (timeouts, rate limits) with exponential backoff.
- **REQ-OPS-016** `shipped` If a service fails repeatedly, the pipeline stops calling it for a cool-down period instead of failing every request against it.
- **REQ-OPS-017** `shipped` If required configuration is missing, the pipeline reports which setting is missing.
- **REQ-OPS-018** `shipped` When `--fail-fast` is passed, the run stops at the first failed item.

## Logging

- **REQ-OPS-019** `shipped` When `--debug` is passed, every component logs at debug level.
- **REQ-OPS-020** `shipped` Batch operations log their progress as `[N/total]`.
- **REQ-OPS-021** `shipped` At the end of its work, each module (scraper, producer, publisher, audio) logs a summary in a shared format with the key counts, the product ids and the duration.
- **REQ-OPS-022** `shipped` Module summaries contain no emojis.
- **REQ-OPS-023** `partial` A logged duration is measured on a monotonic clock.
  - Gap: the batch, its phases, the publisher batch and the scraper batch measure durations on the wall clock, so a clock change skews them.
- **REQ-OPS-024** `shipped` A logged count names what it counts, such as URLs found on a page against files validated on disk.
- **REQ-OPS-025** `shipped` A log message describes the run as executed, not the mode a flag asked for, so a debug run on a virtual display says so.
- **REQ-OPS-026** `shipped` Each log record holds one event.
- **REQ-OPS-027** `shipped` Every log record carries the run id and the product id it belongs to, with `-` when there is none.
- **REQ-OPS-028** `shipped` Logs are written to one file per component per day under `outputs/logs/` and kept for 45 days.
- **REQ-OPS-029** `shipped` Third-party library debug logs are suppressed in every mode, including `--debug`.

## Outputs directory

- **REQ-OPS-030** `shipped` The pipeline writes all its artifacts under one outputs directory, `outputs/` by default.
- **REQ-OPS-031** `shipped` When `--outputs-dir` is passed, the pipeline uses that directory instead.
- **REQ-OPS-032** `shipped` Each product gets `outputs/<product_id>/`, holding `data.json`, `metadata.json`, the `images/`, `videos/`, `music/` and `temp/` subdirectories, the media validation reports and the final `video_*.mp4`.
- **REQ-OPS-033** `shipped` A topic render writes to `outputs/topic-<slug>-<digest>/`, which has no `images/` or `videos/` subdirectory.
- **REQ-OPS-034** `shipped` A topic's directory name is stable for its title, so a re-run resumes its own directory.
- **REQ-OPS-035** `shipped` Two different topic titles never share a directory.
  - Why: a shared directory would let the second topic inherit the first one's completed state and return its video.
- **REQ-OPS-036** `shipped` Shared directories sit beside the product directories: `cache/`, `logs/`, `reports/`, `performance_history/`, `temp/` and `state/`.
- **REQ-OPS-037** `shipped` Records that must outlive product cleanup (publish tracking, the schedule, the published-products registry, analytics) live in `outputs/state/`, which cleanup never removes.

## Resource limits

- **REQ-OPS-038** `partial` The pipeline limits how many FFmpeg, I/O and network operations run at once.
  - Gap: only FFmpeg operations are limited.
- **REQ-OPS-039** `partial` The concurrency limit per operation type is set by `optimization_settings.async_ffmpeg_max_concurrent`, `async_io_max_concurrent` and `async_network_max_concurrent`.
  - Gap: nothing reads these keys; the limits are fixed in code.
- **REQ-OPS-040** `shipped` The scrape, produce, batch, publish and test runs each have a low-priority `make` target (`scrape-lowpri`, `produce-lowpri`, `batch-lowpri`, `publish-lowpri`, `test-lowpri`) that runs them at reduced CPU and I/O priority under a memory cap with swap disabled.
  - Check: a run that exceeds `MEM_LIMIT` (default 6G) is stopped without other applications on the machine being killed.

## Resource cleanup

- **REQ-OPS-041** `shipped` Connections, file handles and temporary resources are released when the operation that uses them ends, including on failure.
- **REQ-OPS-042** `shipped` If a step fails, it removes the partial files it wrote.
- **REQ-OPS-043** `shipped` HTTP clients reuse connections from a shared pool, which is closed when the run ends.

## Validation and caching

- **REQ-OPS-044** `shipped` The pipeline validates the config and scraped product data against typed schemas when it loads them.
- **REQ-OPS-045** `shipped` The scraper rejects a product id that isn't a well-formed ASIN.
- **REQ-OPS-046** `shipped` The producer caches a media file's probed duration for 24 hours, keyed on the file and its modification time, so an unchanged file isn't probed again.

## Performance measurement

- **REQ-OPS-047** `shipped` Each render step records its wall-clock duration, memory at start, peak and end, CPU percent and disk I/O.
- **REQ-OPS-048** `shipped` Peak memory is sampled while a step runs, every `optimization_settings.performance_monitoring_interval_sec` (default 0.1 s).
- **REQ-OPS-049** `shipped` A step's memory and CPU figures include its child processes (FFmpeg, the subtitle renderer, speech-to-text).
- **REQ-OPS-050** `shipped` A step that fails records its error next to its resource figures.

## Performance history

- **REQ-OPS-051** `shipped` Each render is recorded with its product id, profile name, outcome, total duration and aggregated resource usage.
- **REQ-OPS-052** `shipped` When a render finishes, its record is saved to the performance history without a separate command.
- **REQ-OPS-053** `shipped` History keeps render runs and single-step (`--step`) runs as separate kinds.
- **REQ-OPS-054** `shipped` On every save, history keeps the newest `optimization_settings.performance_history_max_runs` (default 100) runs of each kind and drops the rest.
- **REQ-OPS-055** `shipped` A skipped render (insufficient media) is recorded as skipped, and the success rate leaves it out instead of counting it as a failure.
- **REQ-OPS-056** `shipped` When history is loaded, a corrupt entry is skipped and the rest load.

## Threshold warnings

- **REQ-OPS-057** `partial` When a render finishes, the pipeline logs a warning for each step that ran longer than `debug_settings.operation_timing_threshold_sec` (default 180 s).
  - Gap: failed and skipped renders get no threshold warnings.
- **REQ-OPS-058** `partial` When a render finishes, the pipeline logs a warning for each step whose process-tree peak memory exceeded `debug_settings.memory_usage_warning_mb` (default 5000 MB).
  - Gap: failed and skipped renders get no threshold warnings.

## Performance reports

- **REQ-OPS-059** `shipped` The reports cover render runs only, unless `--include-steps` is passed.
- **REQ-OPS-060** `shipped` The summary report shows the success rate, duration statistics with p50, p95 and p99, memory and CPU averages, the product and profile distribution, and a per-step breakdown.
- **REQ-OPS-061** `shipped` The trends report shows daily aggregates with per-step daily averages over the last `--days` days (default 30), and `--product-id` narrows it to one product.
- **REQ-OPS-062** `shipped` The detailed report lists individual runs with a per-step breakdown, and `--format csv` exports it.
- **REQ-OPS-063** `shipped` The comparison report shows each profile's run count, success rate, duration percentiles and memory side by side.
- **REQ-OPS-064** `shipped` The regressions report compares the last `--window` runs (default 10) with the runs before them and flags each step that became more than twice as slow.
- **REQ-OPS-065** `shipped` `--limit` (default 50) caps the number of runs the summary, detailed and comparison reports read.
