# Operations requirements

Ids use the prefix `REQ-OPS`. The format and the statuses are described in [the requirements index](README.md).

## Configuration

- **REQ-OPS-001** `shipped` The config resolves each setting from four tiers, highest first: CLI flags, the machine environment, the profile, the YAML files.
- **REQ-OPS-002** `shipped` A CLI flag overrides a lower tier only when the user passes it.
- **REQ-OPS-003** `shipped` The environment holds only secrets, machine-specific settings and the operator's account values ([decision 0008](../decisions/0008-operator-account-values-in-the-environment.md)); other behaviour settings live in the YAML files or a profile.
- **REQ-OPS-004** `shipped` The YAML files under `config/` hold the application settings, contain no secrets and are safe to commit.
- **REQ-OPS-005** `shipped` A CLI flag can override a nested setting, such as `--pycaps-template` overriding `subtitle_settings.pycaps.template_name`.
- **REQ-OPS-006** `shipped` If the config is invalid, the pipeline stops at startup with an error that names the setting and the problem.

## Secrets

- **REQ-OPS-007** `shipped` The pipeline reads API keys and other secrets from the `.env` file or the environment, and git ignores `.env`.
- **REQ-OPS-008** `shipped` The repository ships `.env.example`, listing the environment variables the pipeline reads.
- **REQ-OPS-009** `shipped` `.env.example` lists only secrets, machine-specific settings, and the operator's account values (affiliate tag, link-in-bio address, topics file) that the public YAML can't carry.
- **REQ-OPS-010** `shipped` Log output masks API keys, tokens and other secret-shaped values before they reach the console or a log file.

## Test isolation

- **REQ-OPS-011** `shipped` If the developer's `.env` file changes during a test run, the test suite fails.
- **REQ-OPS-012** `shipped` A test run writes nothing to the real log files under `outputs/logs/`.
- **REQ-OPS-013** `shipped` A test run writes nothing to the real `outputs/` tree and doesn't depend on its contents.

## Error handling and resilience

- **REQ-OPS-014** `shipped` If one item in a batch fails, the run continues with the remaining items.
- **REQ-OPS-015** `shipped` The pipeline retries transient network failures (timeouts, rate limits) with exponential backoff.
- **REQ-OPS-016** `shipped` If a service fails repeatedly, the pipeline stops calling it for a cool-down period instead of failing every request against it.
- **REQ-OPS-017** `shipped` If required configuration is missing, the pipeline reports which setting is missing.
- **REQ-OPS-018** `shipped` When `--fail-fast` is passed, the run stops at the first failed item.
- **REQ-OPS-066** `shipped` Where `pipeline_timeout_sec` sets a render's total time budget, the speech-to-text time limit on every attempt, retries included, is capped at what remains of that budget.
  - Why: an uncapped step can spend the render's budget and let the timeout fire in a later step, which then takes the blame.
- **REQ-OPS-067** `shipped` If FFmpeg is found neither on `PATH` nor at `ffmpeg_settings.executable_path`, the producer exits 1 at startup, before rendering anything.

## Logging

- **REQ-OPS-019** `shipped` When `--debug` is passed, every component logs at debug level.
- **REQ-OPS-020** `shipped` Batch operations log their progress as `[N/total]`.
- **REQ-OPS-021** `shipped` At the end of its work, each module (scraper, producer, publisher, audio) logs a summary in a shared format with the key counts, the product ids and the duration.
- **REQ-OPS-022** `shipped` Module summaries contain no emojis.
- **REQ-OPS-023** `shipped` A logged duration is measured on a monotonic clock.
- **REQ-OPS-024** `shipped` A logged count names what it counts, such as URLs found on a page against files validated on disk.
- **REQ-OPS-025** `shipped` A log message describes the run as executed, not the mode a flag asked for, so a debug run on a virtual display says so.
- **REQ-OPS-026** `shipped` Each log record holds one event.
- **REQ-OPS-027** `shipped` Every log record carries the run id and the product id it belongs to, with `-` when there is none.
- **REQ-OPS-028** `shipped` Logs are written to one file per component per day under `outputs/logs/` and kept for 45 days.
- **REQ-OPS-029** `shipped` Third-party library debug logs are suppressed in every mode, including `--debug`.

## Outputs directory

- **REQ-OPS-030** `shipped` The pipeline writes all its artifacts under one outputs directory, `outputs/` by default.
- **REQ-OPS-031** `shipped` When `--outputs-dir` (producer, publisher, global batch) or `--output-dir` (scraper) is passed, the run uses that directory instead.
- **REQ-OPS-032** `shipped` Each product gets `outputs/<product_id>/`, holding `data.json`, `metadata.json`, the `images/`, `videos/`, `music/` and `temp/` subdirectories, the media validation reports and the final `video_*.mp4`.
- **REQ-OPS-033** `shipped` A topic render writes to `outputs/topic-<slug>-<digest>/`, which has no `images/` or `videos/` subdirectory.
- **REQ-OPS-034** `shipped` A topic's directory name is stable for its title, so a re-run resumes its own directory.
- **REQ-OPS-035** `shipped` Two different topic titles never share a directory.
  - Why: a shared directory would let the second topic inherit the first one's completed state and return its video.
- **REQ-OPS-036** `shipped` Shared directories sit beside the product directories: `cache/`, `logs/`, `reports/`, `performance_history/`, `temp/` and `state/`.
- **REQ-OPS-037** `shipped` Records that must outlive product cleanup (publish tracking, the schedule, the published-products registry, analytics) live in `outputs/state/`, which cleanup never removes.
- **REQ-OPS-068** `shipped` `make clean-outputs` lists the files and directories under the outputs directory that sit outside the expected layout and are older than `cleanup_settings.max_age_days` (default 7), and deletes nothing.
- **REQ-OPS-069** `shipped` When `CONFIRM=1` is passed, `make clean-outputs` deletes the items it lists, keeping anything that matches `cleanup_settings.preserve_patterns`, which by default include `state/` and the publish records.
- **REQ-OPS-070** `shipped` The outputs cleanup removes items matching `cleanup_settings.force_cleanup_patterns` (such as `*.tmp` and `.DS_Store`) whatever their age, unless a preserve pattern also matches them.
- **REQ-OPS-071** `shipped` If `cleanup_settings.enabled` is false, the outputs cleanup tool refuses to run and exits 1, unless `--force` is passed.
- **REQ-OPS-072** `shipped` After a cleanup that deletes, the outputs cleanup writes `cleanup_report.json` (`cleanup_settings.report_file`) to the outputs directory, listing every action, unless `cleanup_settings.create_report` is false.
- **REQ-OPS-073** `shipped` If removing any item fails, the outputs cleanup exits 1.
- **REQ-OPS-074** `shipped` When a render succeeds without `--debug`, the producer deletes the product's `temp/` directory, which holds the intermediate files.
- **REQ-OPS-075** `shipped` If a render fails or is skipped, or `--debug` is passed, the producer keeps the product's `temp/` directory, so the product can be resumed or inspected.
- **REQ-OPS-076** `shipped` When `--debug` is passed, a successful render writes `performance.json` with its step metrics to the product's `temp/` directory.
- **REQ-OPS-077** `shipped` The final assembly writes the FFmpeg command it ran to an `*_ffmpeg_command.log` file in the product's `temp/` directory.
- **REQ-OPS-078** `shipped` The producer records each render's step progress in `pipeline_state.json` in the product's `temp/` directory.
  - Why: a re-run resumes from this file; without it every re-run starts from the first step.

## Resource limits

- **REQ-OPS-038** `partial` The pipeline limits how many FFmpeg, I/O and network operations run at once.
  - Gap: the FFmpeg encodes take a limit (final assembly, caption burn and the stock-clip transcodes, which the visual builder starts all at once); probes, I/O and network calls are not limited.
- **REQ-OPS-039** `shipped` The limit on concurrent FFmpeg encodes (final assembly, caption burn and stock-clip transcodes) is set by `optimization_settings.async_ffmpeg_max_concurrent` (default 4).
- **REQ-OPS-040** `shipped` The scrape, produce, batch, publish and test runs each have a low-priority `make` target (`scrape-lowpri`, `produce-lowpri`, `batch-lowpri`, `publish-lowpri`, `test-lowpri`) that runs them at reduced CPU and I/O priority under a memory cap with swap disabled.
  - Check: a run that exceeds `MEM_LIMIT` (default 6G) is stopped without other applications on the machine being killed.
- **REQ-OPS-104** `shipped` If the machine runs out of memory during a low-priority run, the run is the process killed: its scope asks systemd-oomd to kill it under sustained memory pressure, and its OOM score is raised above other applications'.
- **REQ-OPS-105** `shipped` Before each render, in the producer, the global batch and the topics batch, the pipeline waits until the machine has `memory_guard.min_available_gb` available and, where there is swap, `min_available_gb + min_swap_free_gb` of available memory and free swap together; after `wait_sec` it fails that render and stops the batch instead of starting it.
- **REQ-OPS-106** `shipped` Inside a capped scope, after each render, the log records the scope's peak memory so far and its cap.

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

- **REQ-OPS-057** `shipped` When a render finishes, the pipeline logs a warning for each step that ran longer than `debug_settings.operation_timing_threshold_sec` (default 180 s).
- **REQ-OPS-058** `shipped` When a render finishes, the pipeline logs a warning for each step whose process-tree peak memory exceeded `debug_settings.memory_usage_warning_mb` (default 5000 MB).

## Performance reports

- **REQ-OPS-059** `shipped` The reports cover render runs only, unless `--include-steps` is passed.
- **REQ-OPS-060** `shipped` The summary report shows the success rate, duration statistics with p50, p95 and p99, memory and CPU averages, the product and profile distribution, and a per-step breakdown.
- **REQ-OPS-061** `shipped` The trends report shows daily aggregates with per-step daily averages over the last `--days` days (default 30), and `--product-id` narrows it to one product.
- **REQ-OPS-062** `shipped` The detailed report lists individual runs with a per-step breakdown, and `--format csv` exports it.
- **REQ-OPS-063** `shipped` The comparison report shows each profile's run count, success rate, duration percentiles and memory side by side.
- **REQ-OPS-064** `shipped` The regressions report compares the last `--window` runs (default 10) with the runs before them and flags each step that became more than twice as slow.
- **REQ-OPS-065** `shipped` `--limit` (default 50) caps the number of runs the summary, detailed and comparison reports read.
- **REQ-OPS-079** `shipped` When `--format json` is passed, the performance report prints the report as JSON.
- **REQ-OPS-080** `shipped` When `--output <file>` is passed, the performance report saves the report as JSON to that file instead of printing it, whatever `--format` says.
- **REQ-OPS-081** `shipped` The performance report reads history from `--history-dir`, which defaults to the repository's `outputs/performance_history` from any working directory.

## Content research

- **REQ-OPS-107** `shipped` `python -m src.research demand` measures Google Trends relative interest for every scraper keyword and every pool topic against one anchor term per side, in each configured country, over 12 months and 5 years, records each term's peak month, and lists Google autocomplete suggestions for the configured stems; a source that is rate-limited or returns nothing is reported as missing, not as zero.
- **REQ-OPS-108** `shipped` The demand report marks a scraper keyword as a drop candidate when its interest is below `drop_below` of its side's median in every country and has not risen over the last quarter, lists rising related searches from the configured product seeds that are not already keywords as add candidates, and lists topic autocomplete suggestions that no pool topic covers.
- **REQ-OPS-109** `shipped` The `sample` stage generates scripts text-only, with no scraping and no rendering, through the producer's own script step, for the first N pool topics under each configured variant (`shipped`, `step_lists`, `task_answer_first`) and the N most recently scraped products under `shipped`, into a scratch outputs root; a sample that fails or is dropped is recorded with its reason.
- **REQ-OPS-110** `shipped` Every sampled script is measured for word count against its band (the step-list band where a step list applies), the search phrase (a topic's `search`, else its title; a product's keyword) in the first sentence, openings repeated across the sample, the script lint, the CTA as its last sentence, fact-check flags and rewrites, and a task topic written with a symptom or mistake template, and the report compares the variants side by side.
- **REQ-OPS-111** `planned #686` Each sampled topic script's steps and claims are checked by a grounded model call that returns correct, wrong, outdated or unverified with a source URL per verdict; a verdict without a source counts as unverified.
- **REQ-OPS-112** `planned #686` The research report recommends a config change only when a variant beats the shipped config on the measured checks of the same sample, names the key and value, and never edits configuration.

## Release and CI

- **REQ-OPS-082** `shipped` On every pull request to `main`, CI fails unless the `pyproject.toml` version is the next patch, minor or major version after the base branch's.
- **REQ-OPS-083** `shipped` On every pull request and push to `main`, CI fails unless the first version heading in `CHANGELOG.md` matches the `pyproject.toml` version and carries a date that isn't in the future.
- **REQ-OPS-084** `shipped` On every pull request and push to `main`, CI fails if the `[Unreleased]` section of `CHANGELOG.md` holds entries.
- **REQ-OPS-085** `shipped` If the release's CHANGELOG section has a `**Breaking**` entry, the pull request check fails on a patch bump.
- **REQ-OPS-086** `shipped` `make release-check` runs the pull request's release check locally against `origin/main`.
- **REQ-OPS-087** `shipped` When a push to `main` passes the release, lint and test jobs and its version has no GitHub release, CI tags it `v<version>` and creates the release with that version's CHANGELOG section as its notes.
- **REQ-OPS-088** `shipped` If the version's tag already exists, the release job reuses it, so a re-run after a failed release step still creates the release.
- **REQ-OPS-089** `shipped` When a `v*` tag is pushed by hand, CI runs the tests and lint and creates a GitHub release from that version's CHANGELOG section, marked prerelease where the tag contains `alpha`, `beta` or `rc`.
- **REQ-OPS-090** `shipped` On every push and pull request to `main`, CI runs `ruff check`, `ruff format --check`, mypy and the test suite with coverage, except as REQ-OPS-103 allows.
- **REQ-OPS-103** `shipped` On a push to `main`, CI skips the test suite only when the merge commit came from a pull request whose head has the identical tree and whose test jobs all passed; anything else, including a failure to find out, runs the suite.
- **REQ-OPS-091** `shipped` The security workflow runs Bandit, Safety and Vulture on every push and pull request to `main` and weekly, and uploads the Bandit and Safety reports.
- **REQ-OPS-092** `shipped` The security workflow never fails the build, whatever its scans find.

## Development

- **REQ-OPS-093** `shipped` `make lint` runs ruff, the ruff format check, mypy, Bandit, Vulture and Safety, and fails if any of them fails.
- **REQ-OPS-094** `shipped` `make lint-fix` runs the same tools with ruff's automatic fixes and formatting applied.
- **REQ-OPS-095** `shipped` `make lint-tool TOOL=<name>` runs one lint tool, and fails when `TOOL` isn't set.
- **REQ-OPS-096** `shipped` `make lint-report` writes the lint results to `outputs/reports/lint-report.json`.
- **REQ-OPS-097** `shipped` `make test` runs the test suite, and `make test-cov` runs it with an HTML coverage report in `outputs/coverage/`.
- **REQ-OPS-098** `shipped` `make test-parallel` runs the test suite on `PYTEST_WORKERS` workers (default `auto`, one per core).
- **REQ-OPS-099** `shipped` `make quick-check` runs ruff and the type check, and `make full-check` runs the lint, security and coverage targets.
- **REQ-OPS-100** `shipped` `make install` and `make install-dev` fail before installing anything unless Python 3.12 and a working Poetry are present.
- **REQ-OPS-101** `shipped` `make clean` removes the tool caches, `__pycache__` directories, the coverage report and the build output.
- **REQ-OPS-102** `shipped` `make clean-all` also removes the project's virtualenvs.
