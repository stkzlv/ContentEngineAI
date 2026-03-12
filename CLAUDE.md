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

## Essential Commands

```bash
# Core workflow
poetry run python -m src.scraper.amazon.scraper --keywords <ASIN> --debug --clean
poetry run python -m src.video.producer outputs/<ASIN>/data.json slideshow_images1 --debug

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

# Tune resource limits if needed (defaults: MEM_LIMIT=8G, NICE_LEVEL=10)
make batch-lowpri ARGS="--product-ids B0ASIN1 --debug" MEM_LIMIT=6G NICE_LEVEL=15

# Publish single product (auto-schedules to next slot)
poetry run python -m src.publisher.late single B0ASIN1 --debug

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

## Code Standards

- **Naming**: snake_case functions, PascalCase classes, UPPER_CASE constants
- **Type Annotations**: Use modern Python typing (`dict[str, Any]`, `| None`)
- **Error Handling**: Specific exceptions (never bare `except Exception`), structured logging
- **Logging**: Use lazy format (`logger.debug("msg: %s", val)`) not f-strings
- **Configuration**: Centralized in `src/video/config/` (Pydantic models)

### Video Module Notes

- **Profile settings**: `get_profile_merged_settings()` returns typed `MergedProfileSettings` (not dicts). Access via `.video_settings.field` and `.subtitle_settings.field`. Use `.model_dump()` when downstream functions need dicts.
- **Video centering**: Use `video_vertical_align: center` in profile config. Setting `video_top_position_percent: 0.0` puts video at top, not center. The `center` value triggers FFmpeg's `(oh-ih)/2` pad expression.
- **Subtitle positioning**: For centered images with mixed aspect ratios (landscape + portrait), use AVERAGE `video_top` across all images to balance positioning
- **Visual bounds**: Calculated from actual image dimensions, not frame dimensions
- **Two-part subtitles**: Upper (static URL) + Lower (voiceover-synced) handled in `two_part_subtitles.py`. Upper subtitle is dynamically repositioned per visual segment in `subtitle_builder._create_content_aware_upper_ass_file()` using actual geometry from the assembler
- **Upper subtitle positioning gotcha**: The pre-assembly `calculate_visual_bounds()` can produce wrong bounds (e.g., `ctx.scraped_videos` empty). The real fix is in the assembler where per-segment `VisualGeometry` data is available
- **Scraper-producer alignment**: Pass `profile_uses_videos` to scraper so media validation counts only what the profile actually uses
- **TTS voice profiles**: Configured in `config/subtitles.yaml` under `tts_config.voice_profiles`. Profiles specify provider (`google_cloud` or `gemini`), style prompt, voice criteria, and markup rules
- **Gemini TTS**: Uses same `google.cloud.texttospeech` SDK but with `SynthesisInput(text=..., prompt=...)`. Requires `Vertex AI User` IAM role on the service account. Falls back to Google Cloud TTS on failure
- **TTS hash slice**: Voice profiles use md5 hex `[16:24]` (fonts use `[0:8]`, colors `[8:16]`, voice within profile `[24:32]`)
- **TTS metadata**: Profile name and voice name saved in `pipeline_state.json` under `create_voiceover.tts_metadata`
- **Script templates**: 15 prompt templates in `src/ai/prompts/scripts/` with different styles (curiosity hook, problem-solution, storytelling, etc.). Configured in `config/ai_services.yaml` under `llm_settings.script_templates`. Selection is deterministic per product using salted md5 hash (`md5(product_id + ":script_template")`). Override with `--script-template NAME` CLI arg. Template name saved in `pipeline_state.json` under `generate_script.script_template`
- **LLM provider fallback**: Gemini is primary, OpenRouter is automatic fallback. Configured via `llm_settings.fallback_provider` in `ai_services.yaml` (self-referencing `LLMSettings`). Both `global_batch.py` and `cli.py` must include fallback provider's API key env var in the secrets dict. Shared dispatch in `src/ai/llm_client.py`
- **LLM config fields**: `model_blocklist`, `min_context_length`, `retry_attempts`/`retry_min_wait_sec`/`retry_max_wait_sec`, and `script_validation` (min_chars, min_words) all live on `LLMSettings`. Don't hardcode these in generator code
- **`.env` safety**: `update_env_file()` in `freesound_client.py` only updates existing keys, never adds new lines

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

**Pull Request Descriptions**:
- **CRITICAL: NEVER mention authors, Claude Code, AI tools, or assistants in PR titles or descriptions**
- Keep descriptions short and simple
- Use PR template if available in `.github/`
- Focus on what changed, why it changed, and how to test

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

## Publisher Module Notes

- **TikTok content disclosure**: Posts **must** include `commercial_content_type: "brand_organic"` and `is_brand_organic_post: true` in `tiktokSettings`. Without these, TikTok rejects with "Commercial content disclosure is enabled but no option selected". The fix is in `src/publisher/late/client.py` — settings are sent both per-platform (`platformSpecificData.tiktokSettings`) and at top-level (`tiktok_settings`).
- **Fixing failed TikTok posts**: Use Late SDK `posts.aupdate()` to set correct `tiktokSettings` per-platform, then the platform status resets from `failed` → `pending` and auto-publishes. No need to call `retry()` — the update triggers re-publish automatically. Calling `retry()` after that gives 409 "Post is currently publishing".
- **Late SDK post methods**: `create`, `get`, `update`, `delete`, `retry`, `list` (+ async variants `acreate`, etc.)
- **Publishing modes**: Unified (default) = 1 post to all platforms; platform-specific (`--platform-specific` or `use_platform_specific_content: true`) = 1 post per platform with optimized metadata. Shared helper: `src/publisher/publish_modes.py`.
- **Config loading gotcha**: `publisher.yaml` may contain keys not in `PublisherConfig` dataclass (e.g. deprecated `backoff_multiplier`). The config loader in `src/publisher/config.py` strips unknown keys before constructing `PublisherConfig(**config_dict)`.
- **`.env` file**: Auto-loaded by the CLI via `load_dotenv()`. No manual sourcing needed.
- **CLI format**: `poetry run python -m src.publisher.late single <PRODUCT_ID> --debug`. Takes product ID (not file paths). Video is auto-discovered from `outputs/<product_id>/`. Don't pass `--immediate` or `--platform-specific` by default.

## Link-in-Bio Module Notes

- **CLI flags**: `--link-in-bio` and `--no-link-in-bio` override `link_in_bio.enabled` config for single publish
- **Affiliate URL fallback**: Uses `affiliate_link` field first, falls back to `url` if unavailable
- **Image fallback**: Uses `images[0]` URL first, falls back to `downloaded_images[0]` local file upload
- **Lnk.Bio auth**: Requires HTTP Basic Auth (not form-encoded), plus `User-Agent: ContentEngineAI/1.0` header to bypass Cloudflare
- **Lnk.Bio API endpoints**: Auth: `POST /oauth/token`, Add: `POST /oauth/v1/lnk/add`, List: `GET /oauth/v1/lnk/list`, Delete: `POST /oauth/v1/lnk/delete`
- **Non-blocking**: Failures never block video publishing; logged as warnings

## Scraper Module Notes

- **URL support**: Scraper accepts full URLs (including shortened URLs like tr.ee) via `--product-ids` or `--input-file`. URLs are detected by `startswith("http")`, navigated directly in the browser, and ASIN is extracted from the redirected URL via regex `/dp/([A-Z0-9]{10})`.
- **CLI args**: `--input-file FILE` (one URL/ASIN per line), `--batch-size N` (process in chunks), `--output-dir DIR` (override output directory).
- **Botasaurus output dir override**: Botasaurus framework callbacks don't accept custom parameters. Use module-level `set_output_dir()` in `botasaurus_output.py` before running the scraper. The `_effective_dir()` helper resolves: explicit param > module override > None (config default).
- **Variable initialization in browser_functions.py**: Variables used after `if is_url / elif is_asin / else` branching (like `count_products_with_media`, `products_with_media_count`, `max_products`) must be initialized **before** the branch, not inside one branch.

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
