# ContentEngineAI Project Memory

## Project Overview

ContentEngineAI is an AI-powered video production pipeline for e-commerce platforms.

## Session Start

At the very beginning of every session, run this check:
```bash
pyenv version && python3 --version && which python3
```

Expected output:
- pyenv version: `ContentEngineAI` (set by `.python-version`)
- Python: `3.12.x`
- Path: pyenv shim (`~/.pyenv/shims/python3`)

The project uses a **pyenv virtualenv**. The `.python-version` file auto-activates it via pyenv shims. No manual activation, `PYENV_VIRTUAL_ENV` prefixes, or `PATH` overrides needed. Just use `python3`, `pytest`, `ruff` directly.

If pyenv version shows something else, fix with:
```bash
pyenv activate ContentEngineAI
```

**A foreign `VIRTUAL_ENV` breaks more than it looks like.** If another project's virtualenv is active in the shell (`echo $VIRTUAL_ENV` points outside this repo), it sets `VIRTUAL_ENV`, leads `PATH`, and is what Poetry reports, because `poetry.toml` sets `virtualenvs.create=false`. `poetry run <anything>` then fails with `Please change python executable via the "env use" command`, so the CI gates can't be run through `poetry run` at all. Deactivate it, or run the gates through the venv directly (`~/.pyenv/versions/ContentEngineAI/bin/ruff`, `.../mypy`, `.../pytest`). The same hijack is why the `*-lowpri` targets consult `.python-version` before any ambient source.

## Private Overlay Files

The repo supports private, contributor-specific overlays that stay out of git. Any file matching `*.private.md` or living under `.business/` is gitignored (see `.gitignore`). Use this for business motivation, account-specific context, personal planning, or decisions you don't want to publish.

**Naming convention**: a private overlay sits next to its public counterpart with the same basename plus `.private.md`. Example: `docs/roadmap.md` (public) and `docs/roadmap.private.md` (private).

**At session start**: list any `*.private.md` files relevant to the current task and read them. They usually explain *why* a public item exists when the public doc only describes *what*. Typical check:
```bash
find . -name '*.private.md' -not -path './.venv/*' -not -path './outputs/*'
```

**Keeping public and private in sync** when a pair exists:
- Items and structure stay aligned. When an item is added, removed, reordered, or changes horizon in one file, mirror it in the other.
- Public describes the capability. Private adds motivation, constraints, and decisions.
- `Done when` criteria may differ: public stays generic and testable; private can reference signals (metrics, sample sizes, thresholds specific to the contributor's use case).
- The same rule applies to any other paired docs (e.g., `docs/strategy.md` + `docs/strategy.private.md`).

**Never leaks into the public tree**:
- Content or direct quotes from any `*.private.md` file.
- Account handles, follower counts, financial numbers, real persona names, or any contributor-specific identifiers.
- References to the private file's existence or path in commit messages, PR descriptions, issues, or committed code.
- Config values keyed to the contributor's accounts. Public YAML ships generic defaults; real values live in `.env` or in a gitignored override file.

This pattern is generic. Any contributor can create `docs/<public-doc>.private.md` (or a `.business/` subtree) for their own overlay without project-side configuration changes.

## Logs

Pipeline logs are in `outputs/logs/`:
- `global_pipeline.log` — batch pipeline (scrape + produce + publish)
- `scraper.log` — standalone scraper runs
- `producer.log` — standalone video production
- `publisher.log` — standalone publishing

## Resource discipline (read before running anything below)

The scraper and the producer are the heavy commands. The producer pipeline peaks around 2-2.5 GB RSS per render (Whisper STT, FFmpeg encoding, pycaps Chromium) and runs for 3-6 minutes on a single 30-45s output. The scraper drives Botasaurus + Chromium and holds RAM for the duration of a search. Running either bare while the user is working on the same machine causes systemd-oomd to kill unrelated session apps (Chrome, VSCode) — see the 0.44.0 changelog for why we now ship `MemorySwapMax=0` in the lowpri cgroup.

**Rule: full scrape and full produce ALWAYS go through `make scrape-lowpri` / `make produce-lowpri` (or `make batch-lowpri` for the global pipeline).** These targets wrap the command in a `systemd-run --user --scope` cgroup with `MEM_LIMIT` (default 6G) + `nice` (default 15) + `MemorySwapMax=0`. Tune via `NICE_LEVEL=19` when thrashing; **do not lower `MEM_LIMIT` below the 6G default for a render**: at 4G the kernel memcg OOM-killed ffmpeg (2.3 GB anon) during an image-profile assembly with the Whisper model still resident, and the batch died before its publish phase (2026-09-07). The cap contains a blow-up; it does not make a render need less. `nice`/`ionice` are CPU/IO priority only and do nothing for OOM; `MemoryMax` + `MemorySwapMax=0` are what contain a memory blow-up to the pipeline cgroup instead of letting it (or systemd-oomd) kill unrelated session apps. The bare `poetry run python -m src.scraper.amazon.scraper` and `poetry run python -m src.video.producer` forms are reserved for:

- A targeted pytest run (a file or a directory, no full render). The **full suite** is not exempt: it holds the machine for several minutes, and `make test-parallel` is one uncapped worker per core, so use `make test-lowpri` (`ARGS=` takes pytest arguments) or bound the workers with `make test-parallel PYTEST_WORKERS=4`. **When the desktop is busy (a browser holding several GB), even the lowpri suite trips the host's low-memory guard and is killed before it prints anything; `make test-lowpri ARGS="-n 4"` bounds the xdist workers and ran the full suite in two minutes where the unbounded run was killed twice.** The cgroup cap contains the suite's own blow-up; it does nothing about the host running out around it.
- Single-step debug runs that pass `--step <name>` and skip the heavy steps.
- Dry runs (`--dry-run`).
- One-second invocations that just print help or load config.

If the command will scrape products, render audio/video, or run the full pipeline end-to-end, use lowpri. No exceptions for "I just need one quick test render" — quick test renders are exactly when the user is also using the machine.

**The `*-lowpri` recipes deliberately do NOT use `poetry run` — don't "simplify" them back to it.** `systemd-run --user --scope` starts the process through the user service manager, which doesn't carry the caller's virtualenv, so `poetry run python` inside the scope resolves an interpreter without the project's dependencies and the run dies on import (which module varies by entry point). The recipes resolve an interpreter that can actually import a project dependency and exec it directly with `PATH` forwarded. `poetry run python` is also unusable as the probe: with `virtualenvs.create=false` in `poetry.toml` it reports the base interpreter, not the project venv. Only the lowpri targets need this; the plain targets run outside the scope and `poetry run` is correct there.

When invoking lowpri, pass the args via `ARGS="..."`:

```bash
make produce-lowpri ARGS="outputs/<ASIN>/data.json slideshow_images1 --clean --debug"
make scrape-lowpri  ARGS="--product-ids B0XXXXXXXX --debug"
make batch-lowpri   ARGS="--product-ids B0XXXXXXXX --profile slideshow_images1 --debug"
```

For batch operations, `make batch-lowpri` is documented below as the default for global pipeline runs. Apply the same rule to single-product runs by reaching for `make produce-lowpri` first.

## Essential Commands

```bash
# Core workflow
poetry run python -m src.scraper.amazon.scraper --keywords <ASIN> --debug --clean
poetry run python -m src.video.producer outputs/<ASIN>/data.json slideshow_images1 --debug

# Topic render (no scraper run); output lands in outputs/topic-<slug>/
poetry run python -m src.video.producer slideshow_stock --topic "Why wifi drops" --topic-description "..." --topic-keywords "wifi router, home network"
poetry run python -m src.video.producer slideshow_stock --topics-file topics.yaml

# Batch scraping (product IDs)
poetry run python -m src.scraper.amazon.scraper --product-ids B0ASIN1 B0ASIN2 B0ASIN3 --debug

# Batch scraping (keywords with filters)
poetry run python -m src.scraper.amazon.scraper --keywords "wireless earbuds" "headphones" --min-price 20 --max-price 100 --min-rating 4.0 --debug

# Batch scraping (mixed mode with fail-fast)
poetry run python -m src.scraper.amazon.scraper --product-ids B0ASIN1 --keywords "product" --fail-fast --debug

# Scraping from URLs (shortened or full Amazon URLs)
poetry run python -m src.scraper.amazon.scraper --product-ids "https://tr.ee/mUk1eH" --output-dir tmp --debug

# Batch scraping from file with chunked processing
poetry run python -m src.scraper.amazon.scraper --input-file products.txt --output-dir tmp --batch-size 10 --debug

# Batch video production (fixed profile)
poetry run python -m src.video.producer --batch --batch-profile slideshow_images1 --debug

# Batch video production (random profile per product - deterministic)
poetry run python -m src.video.producer --batch --random-profile --debug

# Batch video production (random from specific pool)
poetry run python -m src.video.producer --batch --random-profile --profile-pool slideshow_images1 video_sequential --debug

# Batch video production (specific products only)
poetry run python -m src.video.producer --batch --random-profile --product-ids B0ASIN1 B0ASIN2 --debug
make produce-lowpri ARGS="--batch --random-profile --product-ids B0ASIN1 B0ASIN2 --debug" MEM_LIMIT=6G

# Global batch pipeline (always use make batch-lowpri for batch runs)
make batch-lowpri ARGS="--product-ids B0ASIN1 B0ASIN2 --profile slideshow_images1 --debug"

# Global batch pipeline (random profiles with filters)
make batch-lowpri ARGS="--keywords 'wireless earbuds' --max-products 10 --min-price 20 --min-rating 4.0 --random-profile --debug"

# Global batch pipeline (skip publishing)
make batch-lowpri ARGS="--keywords 'smart watch' --skip-publish --debug"

# Global batch pipeline (clean stale outputs before run)
make batch-lowpri ARGS="--product-ids B0ASIN1 --clean --debug"

# Scraping only (low priority)
make scrape-lowpri ARGS="--keywords 'wireless earbuds' --debug"

# Video production only (low priority)
make produce-lowpri ARGS="--batch --batch-profile slideshow_images1 --debug"

# Tune resource limits if needed (defaults: MEM_LIMIT=6G, NICE_LEVEL=15)
make batch-lowpri ARGS="--product-ids B0ASIN1 --debug" MEM_LIMIT=4G NICE_LEVEL=19

# Publish single product (auto-schedules to next slot)
poetry run python -m src.publisher.late single B0ASIN1 --debug
make publish ARGS="single B0ASIN1 --debug"

# Schedule all unpublished products
make publish ARGS="schedule --debug"
make publish-lowpri ARGS="schedule --debug" MEM_LIMIT=6G NICE_LEVEL=15

# Publish to specific platforms
poetry run python -m src.publisher.late single B0ASIN1 --platform youtube --platform tiktok --debug

# Published products registry
poetry run python -m src.publisher.late registry --rebuild --outputs-dir outputs
poetry run python -m src.publisher.late registry --rebuild --scan-dir tmp --outputs-dir outputs

# Performance monitoring
poetry run python tools/performance_report.py --report-type summary
poetry run python tools/performance_report.py --report-type trends --days 30
poetry run python tools/performance_report.py --report-type detailed --limit 10
poetry run python tools/performance_report.py --report-type comparison
poetry run python tools/performance_report.py --report-type regressions --window 10
poetry run python tools/performance_report.py --report-type detailed --format csv --limit 5
```

## End-to-End Pipeline Test Cases

After any change that alters runtime behavior, run the real path it touches and inspect the real artifact (file, video, log line, published post), not just a green test run. Match the check to the change: scraper change -> a scrape; producer/subtitle/audio change -> a produce + `ffprobe`/frame check; publisher change -> the publish-option runbook; config-model change -> the one path that consumes the field; cross-phase change -> a full batch. When the changed logic exists in both a standalone module CLI and `global_batch` (Module/Batch Alignment Rule), verify BOTH paths. Trust the artifact over the exit code (the global batch exits non-zero when nothing completes end-to-end, with one exception -- a run whose products were all already published and which lost nothing else exits 0, logging `PIPELINE COMPLETED SUCCESSFULLY` with an `Already Published (not re-rendered):` line -- and a partial failure exits 0 unless `--strict` is passed — grep the phase-summary log lines to confirm). Full change-type -> check table and worked runbooks in `docs/testing.md` ("Verifying a change end-to-end").

The three worked full-pipeline cases, their verification steps and the
caveats baked in from real runs are in
[docs/notes/end-to-end-checks.md](docs/notes/end-to-end-checks.md).

## Code Standards

- **Naming**: snake_case functions, PascalCase classes, UPPER_CASE constants
- **Type Annotations**: Use modern Python typing (`dict[str, Any]`, `| None`)
- **Error Handling**: Specific exceptions (never bare `except Exception`), structured logging
- **Logging**: Use lazy format (`logger.debug("msg: %s", val)`) not f-strings -- ruff's `G`/`LOG` groups enforce this, so a new f-string call fails `ruff check` rather than being caught in review. A literal `%` has to be written `%%` in any message that takes arguments (`logging` applies `%` formatting only when there are some, so escaping one in an argument-less message prints the `%%` verbatim), and `tests/utils/test_lazy_logging.py` counts placeholders against arguments, which no linter does: a miscount raises inside `logging`, which prints it to stderr and logs nothing. No emojis in log messages (existing emoji-laden lines are pre-existing tech debt to clean up over time; new code emits plain text).
- **Configuration**: Centralized in `src/video/config/` (Pydantic models)
- **Secrets wiring**: the render pipeline's secrets dict is built once, by `collect_producer_secrets` in `src/video/producer/utils.py`, for both entry points (producer CLI and global batch). Adding an env var to the config model is the whole change; there are no per-entry-point copies to keep aligned any more. The audio provider `audio_providers[].settings` env vars are read dynamically; other modules use hardcoded lists.

## Session Continuity

After every context compaction (session continuation), run `/github-workflow` to check CI status and catch any issues from the previous session. This is non-optional.

## Development Guidelines

- Use Poetry for dependency management
- Use imperative commit messages (e.g., "Add subtitle generation")
- **NEVER mention Claude Code, AI tools, or assistants in commits/PRs — no `Co-Authored-By`, no AI references anywhere**
- Document project status in relevant documentation files
- Create implementation plans for features/fixes before coding

**Important Documentation**:
- **CONTRIBUTING.md**: GitHub Flow workflow, branch naming, code style, testing requirements
- **docs/development.md**: Architecture, performance optimization, component development, debugging
- **docs/versioning.md**: Semantic versioning rules, release process, version support policy

*These files are automatically read by the github-workflow skill during iteration start and releases.*

### Git Commit & PR Guidelines

**Commit Messages**:
- Use imperative mood (e.g., "Add feature", not "Added feature" or "Adds feature")
- Keep first line under 50 characters
- **CRITICAL: NEVER include `Co-Authored-By` trailers, author attributions, or any mention of Claude Code / AI tools / assistants**
- Keep messages short and simple
- Explain what and why, not how
- Track follow-up work as GitHub Issues with the `follow-up` label, not as `docs/*-followups.md` files. Issues survive renames, link cleanly from PRs, and don't bit-rot when section numbers shift.

**Pull Request Descriptions**:
- **CRITICAL: NEVER mention authors, Claude Code, AI tools, or assistants in PR titles or descriptions**
- Keep descriptions short and simple
- Use PR template if available in `.github/`
- Focus on what changed, why it changed, and how to test
- Don't reference internal follow-up/todo tracker docs in PR descriptions. Describe the change on its own terms.

## Development Workflow (GitHub Flow)

ContentEngineAI follows **GitHub Flow** - a branch-based workflow for features and bug fixes.

### Branch Management

1. **Create Branch from Main**:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/your-feature-name
   ```

2. **Branch Naming Conventions**:
   - `feature/` - New features
   - `bugfix/` - Bug fixes
   - `hotfix/` - Critical fixes
   - `docs/` - Documentation updates

### Quality Gates (Required Before PR)

```bash
make lint          # Ruff, MyPy, Bandit, Vulture, Safety
make test          # Pytest with coverage
make security      # Security scans
```

**Code Standards Enforced**:
- 88-character line limit
- Modern Python typing (`dict[str, Any]`, `| None`)
- Unit tests >90% coverage, Integration >80% coverage
- Security scanning with Bandit and Safety

### Development Commands

```bash
make dev-setup     # Complete development environment setup
make quick-check   # Essential checks (ruff + type-check)
make full-check    # All checks (lint + security + test-cov)
make lint-fix      # Run linting with automatic fixes
make test-cov      # Run tests with coverage report
```

### Pull Request Process

1. **Push and Create PR**:
   ```bash
   git add .
   git commit -m "Add subtitle generation"  # Imperative messages
   git push origin feature/your-feature-name
   ```

2. **PR Requirements**:
   - Target `main` branch
   - Use conventional commit format (`feat:`, `fix:`, `docs:`)
   - Complete PR template checklist
   - All CI checks must pass
   - Include tests for new functionality

3. **CI/CD Pipeline**:
   - **CI Workflow**: Runs on push/PR to main (lint, test, coverage)
   - **Security Workflow**: Weekly scans + PR checks
   - **Release Workflow**: Triggered by version tags

### Merge Process

- Squash merge for clean history
- Address all review feedback
- Ensure all CI checks pass

### Release Process

**Version Bumping**:
- Follow semantic versioning: `MAJOR.MINOR.PATCH`
- Determine version bump based on changes:
  - **Major** (e.g., 1.0.0 → 2.0.0): Breaking API changes
  - **Minor** (e.g., 0.17.0 → 0.18.0): New features (backward compatible)
  - **Patch** (e.g., 0.17.0 → 0.17.1): Bug fixes only
- Update version in `pyproject.toml`

**Releases are automated via CI/CD**:
1. Review changes and determine version bump
2. Update version in `pyproject.toml` and code files
3. Update `CHANGELOG.md` with release notes following [Keep a Changelog](https://keepachangelog.com/) format
4. Commit version bump: `git commit -m "Bump version to 0.18.0"`
5. Merge PR and switch to main branch
6. Create and push version tag: `git tag -a v0.18.0 -m "Release v0.18.0"`
7. CI workflow automatically creates GitHub release with:
   - Release notes extracted from CHANGELOG.md
   - Build artifacts (wheel and source distribution)
   - Tests and linting verification

**Note**: Do not manually create GitHub releases - CI handles this when tags are pushed

### Dependency Updates (Dependabot)

Dependabot PRs are batched into patch releases per `docs/versioning.md`:

1. Dependabot PRs stay open until the next patch release cycle
2. At release time: `gh pr checkout <PR>`, rebase onto main, install deps, run full test suite
3. Bump version in `pyproject.toml`, add a "Dependencies" section in CHANGELOG
4. Commit version bump on the Dependabot branch, force-push (rebase changed history), squash-merge
5. Tag and push from main as usual

Security-critical updates can trigger an immediate patch release without waiting.

## Duplicates Are Acceptable

**A duplicate this project's normal operation produces is not a defect here.** A `--force` republish that creates a second live Zernio post, a knowing re-render that goes out again, and an ASIN carrying more than one lnk.bio link are all expected outcomes. Take the action that was asked for, note the duplicate afterwards if it is informative, and don't stop to ask whether it is wanted or offer to clean it up.

This is a tolerance, not a licence to remove the guards. `_drop_already_published` stays on unless `--force` or `--skip-publish` says otherwise (the latter publishes nothing, so there is no duplicate to prevent), `is_already_published` unless `--force`, and the link-in-bio check is unconditional; and a guard that silently stops working is still a defect worth raising: the handoff filter is the batch's only guard against a second post for a product that run re-scraped. It is not the whole publish history that is at stake, because the handoff covers the current run's scrape unless `process_all_products` is set, and cleanup has removed the directories of most products that already went out.

The bio's multi-link ASINs are that distinction in practice. `LinkInBioManager.update` does check, by scanning `list_links()` for the product id, but `/lnk/list` returns one un-paginated page of 50 against a bio several times that size, so a link older than the window is invisible and gets added again. The check is doing its job within the window it can see; the duplicates past it are accepted rather than evidence the check is wrong.

## Module notes

One file per module in [docs/notes/](docs/notes/), each an entry per defect:
what broke, why it was invisible, and what catches it now. **Read the file for
a module before changing it** -- most of those entries exist because the
obvious change had already been tried.

| Module | Notes |
|---|---|
| Render pipeline (producer, assembler, prompts, timeouts, state) | [video.md](docs/notes/video.md) |
| Subtitles (the pycaps engine, its fallbacks and templates) | [subtitles.md](docs/notes/subtitles.md) |
| Publishing (provider SDK, disclosures, scheduling, tracking, analytics) | [publisher.md](docs/notes/publisher.md) |
| Scraping (config, throttling, extraction, the browser stack) | [scraper.md](docs/notes/scraper.md) |
| Audio providers | [audio.md](docs/notes/audio.md) |
| Link-in-bio | [link-in-bio.md](docs/notes/link-in-bio.md) |
| CI, gates and dependency pinning | [ci-and-dependencies.md](docs/notes/ci-and-dependencies.md) |
| Where the batch and the standalone modules have drifted | [batch-alignment.md](docs/notes/batch-alignment.md) |
| The worked end-to-end cases | [end-to-end-checks.md](docs/notes/end-to-end-checks.md) |

## Module/Batch Alignment Rule

**CRITICAL**: Standalone module CLIs (publisher, scraper, producer) and `global_batch.py` often have parallel implementations of the same logic (scheduling, validation, retry, cleanup). When fixing or adding behavior in one path, **proactively check the other path** for the same issue or missing feature. Don't wait for it to break separately. The batch pipeline re-implements logic from standalone modules rather than calling them, so drift is common and silent.

Where the two have actually drifted, and what each drift cost, is in
[docs/notes/batch-alignment.md](docs/notes/batch-alignment.md).

## Available MCP Servers

The project has access to these MCP servers for enhanced development capabilities:

### Context7 Server
- **Purpose**: Library documentation and code examples
- **Usage**: Get up-to-date documentation for any library
- **Tools**: `resolve-library-id`, `query-docs`
- **Example**: Get Next.js documentation, React hooks examples, Python library docs

### GitHub Server
- **Purpose**: GitHub repository management and automation
- **Capabilities**:
  - Repository operations (create, fork, search)
  - Issue management (create, update, comment, sub-issues)
  - Pull request workflow (create, review, merge, status)
  - Workflow automation (run, cancel, retry)
  - Code search and file operations
- **Integration**: Use for automating PR creation, issue tracking, code reviews
