# ContentEngineAI Project Memory

## Project Overview

ContentEngineAI is an AI-powered video production pipeline for e-commerce platforms.

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

# Global batch pipeline (fixed profile)
poetry run python -m src.pipeline.global_batch --product-ids B0ASIN1 B0ASIN2 --profile slideshow_images1 --debug

# Global batch pipeline (random profiles with filters)
poetry run python -m src.pipeline.global_batch --keywords "wireless earbuds" --max-products 10 --min-price 20 --min-rating 4.0 --random-profile --profile-pool slideshow_images1 video_sequential --debug

# Global batch pipeline (mixed mode with fail-fast)
poetry run python -m src.pipeline.global_batch --product-ids B0ASIN1 --keywords "smart watch" --profile slideshow_images1 --fail-fast --debug

# Published products registry
poetry run python -m src.publisher.late registry --rebuild --outputs-dir outputs
poetry run python -m src.publisher.late registry --rebuild --scan-dir tmp --outputs-dir outputs

# Performance monitoring
poetry run python tools/performance_report.py --report-type summary
```

## Code Standards

- **Naming**: snake_case functions, PascalCase classes, UPPER_CASE constants
- **Type Annotations**: Use modern Python typing (`dict[str, Any]`, `| None`)
- **Error Handling**: Specific exceptions (never bare `except Exception`), structured logging
- **Logging**: Use lazy format (`logger.debug("msg: %s", val)`) not f-strings
- **Configuration**: Centralized in `src/video/config/` (Pydantic models)

### Video Module Notes

- **Subtitle positioning**: For centered images with mixed aspect ratios (landscape + portrait), use AVERAGE `video_top` across all images to balance positioning
- **Visual bounds**: Calculated from actual image dimensions, not frame dimensions
- **Two-part subtitles**: Upper (static URL) + Lower (voiceover-synced) handled in `two_part_subtitles.py`

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
- **Config loading gotcha**: `publisher.yaml` may contain keys not in `PublisherConfig` dataclass (e.g. deprecated `backoff_multiplier`, `use_platform_specific_content`). The config loader in `src/publisher/config.py` strips unknown keys before constructing `PublisherConfig(**config_dict)`.
- **`.env` file**: Must be sourced before running publisher CLI (`set -a && source .env && set +a`), or use `poetry run` which auto-loads `.env` if `python-dotenv` is installed.

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
