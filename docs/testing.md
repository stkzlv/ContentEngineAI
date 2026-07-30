# Testing Guide

ContentEngineAI uses a comprehensive test suite across unit, integration, and end-to-end categories. Run `make test` (or `poetry run pytest --collect-only -q | tail -1`) for the current count.

## Quick Start

```bash
# Run all tests
make test

# Run with coverage report
make test-cov

# Run specific test file
poetry run pytest tests/test_video_config.py -v

# Run by category
poetry run pytest -m unit          # Unit tests only
poetry run pytest -m integration   # Integration tests only
poetry run pytest -m e2e           # End-to-end tests only
```

## Test Categories

### Unit Tests (`@pytest.mark.unit`)
- **Purpose**: Test individual functions in isolation
- **Coverage**: Core utilities, configuration validation
- **Speed**: Fast (< 1s per test)

### Integration Tests (`@pytest.mark.integration`)
- **Purpose**: Test component interactions
- **Coverage**: Pipeline steps, data flow
- **Speed**: Medium (1-10s per test)

### End-to-End Tests (`@pytest.mark.e2e`)
- **Purpose**: Test complete workflows
- **Coverage**: Full pipeline execution
- **Speed**: Slow (10+ seconds per test)

### Video Verification Tests
- **Purpose**: Validate final video output quality
- **Coverage**: Audio levels, subtitle positioning, image sizing
- **Method**: Screenshot analysis, FFprobe audio analysis
- **Dependencies**: Requires pipeline-generated videos

<details>
<summary><strong>Running Video Verification Tests</strong></summary>

```bash
# Run all verification tests
poetry run pytest tests/test_slideshow_images1_verification.py -v

# Run specific verification
poetry run pytest tests/test_slideshow_images1_verification.py::TestSlideshowImagesVerification::test_slideshow_images1_background_music_verification -v
```

**Verification artifacts** stored in: `outputs/<ASIN>/temp/verification/`

</details>

## Test Structure

<details>
<summary><strong>Test Directory Structure</strong></summary>

```
tests/
├── conftest.py                      # Pytest config & shared fixtures
├── run_tests.py                     # Test runner script
│
├── # AI/Platform Metadata Tests
├── ai/
│   ├── test_instagram_generator.py  # Instagram metadata generation
│   ├── test_tiktok_generator.py     # TikTok metadata generation
│   ├── test_youtube_generator.py    # YouTube metadata generation
│   ├── test_text_formatter.py       # Text formatting utilities
│   ├── test_platform_metadata_models.py  # Pydantic models
│   ├── test_metadata_cache.py       # Metadata caching (25 tests)
│   ├── test_ab_testing.py           # A/B testing for prompts (25 tests)
│   ├── test_batch_generation.py     # Batch metadata generation (25 tests)
│   ├── test_metadata_export.py      # Multi-format export (31 tests)
│   └── test_trend_aware_hashtags.py # Trend-aware hashtags (13 tests)
│
├── # End-to-End Tests
├── e2e/
│   ├── test_publisher_workflow.py   # Publisher workflow tests
│   └── test_publisher_schedule_cleanup.py  # Schedule cleanup tests
│
├── # Integration Tests
├── integration/
│   ├── test_freesound_integration.py  # Freesound API integration
│   ├── test_late_publisher.py       # Zernio publisher integration
│   └── test_platform_metadata_integration.py  # Metadata integration
│
├── # Pipeline Tests
├── pipeline/
│   ├── test_global_batch_integration.py  # Global batch pipeline
│   ├── test_global_batch_orchestrator.py  # Orchestrator tests
│   └── test_global_batch_publishing.py  # Publishing integration
│
├── # Publisher Tests
├── publisher/
│   ├── late/
│   │   └── test_client.py           # Zernio API client
│   ├── test_accounts.py             # Account management
│   ├── test_base.py                 # Base publisher interface
│   ├── test_batch.py                # Batch publishing
│   ├── test_cleanup.py              # Cleanup functionality
│   ├── test_conflict_resolution.py  # Schedule conflict resolution
│   ├── test_metadata.py             # Platform metadata loading
│   ├── test_models.py               # Publisher models
│   ├── test_product_registry.py     # Published products registry
│   ├── test_publish_modes.py        # Unified/platform-specific modes
│   ├── test_registry.py             # Provider registry
│   ├── test_schedule*.py            # Scheduling tests (4 files)
│   ├── test_tracking_extended.py    # Tracking atomic writes/retry queue
│   └── test_webhooks.py             # Webhook event handling
│
├── # Scraper Tests
├── scraper/
│   ├── test_config_models.py        # Pydantic config models
│   ├── test_video_integration.py    # Video pipeline integration
│   ├── test_m3u8_video_extraction.py # M3U8/HLS video support
│   ├── test_batch_controller.py     # Batch mode unit tests
│   └── test_batch_integration.py    # Batch mode integration tests
│
├── # Video Producer Tests
├── video/producer/
│   ├── test_profile_selection.py    # Profile selection utilities
│   └── test_batch_profile_integration.py  # Batch profile randomization
│
└── # Root-level Tests (core components)
    ├── test_video_config.py         # Video configuration
    ├── test_ai_script_generator.py  # AI script generation
    ├── test_subtitle_*.py           # Subtitle system
    ├── test_audio.py                # Audio processing
    ├── test_tts.py                  # Text-to-speech
    └── test_*.py                    # Other component tests
```

</details>

## Writing Tests

### Test Template

```python
import pytest
from unittest.mock import Mock, patch

@pytest.mark.unit  # or integration, e2e
class TestYourComponent:
    """Test suite for your component."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock()
        config.setting = "value"
        return config

    def test_basic_functionality(self, mock_config):
        """Test basic functionality."""
        result = your_function(mock_config)
        assert result == expected_value

    @patch('module.external_service')
    def test_with_mocked_service(self, mock_service):
        """Test with mocked external service."""
        mock_service.return_value = "mocked_data"
        result = your_function()
        assert result == "expected_result"
```

### Best Practices

1. **Use descriptive test names** that explain the scenario
2. **Mock external dependencies** (APIs, file systems)
3. **Test both success and failure** scenarios
4. **Include edge cases** and boundary conditions
5. **Use fixtures** for shared setup logic

<details>
<summary><strong>Coverage Requirements</strong></summary>

- **Unit tests**: >90% coverage target
- **Integration tests**: >80% coverage target
- **Overall minimum**: enforced by the CI `--cov-fail-under` gate; run `make test-cov` for the current number

**Generate coverage report:**
```bash
make test-cov
# Opens HTML report in browser: outputs/coverage/index.html
```

</details>

## Test Configuration

### Pytest Configuration (`pyproject.toml`)

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "e2e: End-to-end tests",
    "slow: Slow running tests",
    "external: Tests requiring external services",
    "mock: Tests using mocks",
    "asyncio: Async tests",
]
```

### Shared Fixtures (`tests/conftest.py`)

Common fixtures available in all tests:
- `mock_config` - Mock video configuration
- `temp_dir` - Temporary directory (auto-cleanup)
- `sample_product_data` - Sample product data for testing

## Advanced Testing

<details>
<summary><strong>Parallel Test Execution</strong></summary>

```bash
# Run tests in parallel (faster)
poetry run pytest -n auto

# Run with specific number of workers
poetry run pytest -n 4
```

</details>

<details>
<summary><strong>Test Filtering</strong></summary>

```bash
# Run tests by name pattern
poetry run pytest -k "config"          # All tests with 'config' in name
poetry run pytest -k "not slow"        # Exclude slow tests

# Run failed tests from last run
poetry run pytest --lf

# Run tests that failed, then all others
poetry run pytest --ff
```

</details>

<details>
<summary><strong>Debugging Tests</strong></summary>

```bash
# Show print statements
poetry run pytest -s

# Stop at first failure
poetry run pytest -x

# Drop into debugger on failure
poetry run pytest --pdb

# Verbose output with full tracebacks
poetry run pytest -vv --tb=long
```

</details>

## Verifying a change end-to-end

The automated suite catches regressions in code paths it covers; it does not prove your change produces a correct real artifact. After any change that alters runtime behavior, run the real path it touches and inspect the real output (a file, a video, a log line, a published post), not just a green test run. Two principles drive everything below:

- **Run the narrowest path that exercises the change, then widen.** A scraper change needs a scrape, not a full batch. Only run the whole pipeline when the change spans phases or you're verifying a release.
- **Verify both implementations when the logic is duplicated.** The standalone module CLIs (scraper, producer, publisher) and `global_batch` re-implement the same logic (scheduling, validation, retry, cleanup) rather than calling each other. A change to shared behavior has to be checked on both paths, because they drift silently.

Match the change to the check:

| Change touches | Run | Inspect |
|---|---|---|
| Scraper (`src/scraper/`) | `make scrape-lowpri` on a keyword, plus an ASIN and a URL | `data.json` fields, image count vs the media-validation floor, `outputs/logs/scraper.log` warnings |
| Producer / assembler (`src/video/`) | `xvfb-run -a make produce-lowpri` on an existing ASIN | `ffprobe` the video (codec/resolution/duration), eyeball a frame, `producer.log` step summaries and threshold warnings |
| Subtitles / pycaps (`src/video/pycaps_engine/`, `subtitle_*`) | produce with the affected `--subtitle-engine` / `--pycaps-renderer` | captions on a sampled frame; no `overlay skipped` or pycaps burn-fail warnings |
| Audio (`src/audio/`, `audio_builder`) | produce | `ffprobe`/loudness on the output audio; no provider-chain errors |
| AI / prompts (`src/ai/`) | produce, or `--step generate_script` | the rendered prompt at `outputs/<ASIN>/temp/script_prompt.txt` and the script output |
| Publisher (`src/publisher/`) | the publish-option runbook below | Zernio post status, `publish_history.json` / `published_products.json`, disclosures, first comments |
| Config models / YAML (`config/`, `src/*/config/`) | load the config, then run the one path that consumes the field | the override actually reaches the runtime (profile-level gating and the secrets dict are the usual silent-drop traps) |
| Pipeline / batch orchestration (`src/pipeline/`) | full `make batch-lowpri` | phase-summary log lines and exit behavior on both the standalone and batch paths |
| Pure refactor (no behavior change) | `make test` + `make lint`, plus one targeted smoke on the touched path | output unchanged vs before |

Resource and environment rules apply to every row: heavy scrape/produce/batch runs go through the `*-lowpri` make targets (cgroup memory cap), and on a Wayland box any producer run is wrapped in `xvfb-run -a` (the bundled pycaps CSS renderer hangs without an X display). One heavy job at a time.

Trust the artifact over the exit code. The global batch now exits non-zero when no product completes end-to-end, but partial failures still exit 0. Confirm success by grepping the phase-summary log lines (`Scraping phase complete:`, `Production phase complete:`, `Publishing phase complete:`), not `$?`.

The two runbooks below are worked instances of this: the smoke test exercises the full scrape -> produce -> publish chain, and the publish-option runbook exercises every publishing path.

## Step-by-step pipeline smoke test

Before a release, or after a refactor that touched the scraper, producer, or publisher surfaces, it's worth running the full pipeline end-to-end against a single product and stopping between phases to eyeball the outputs. This is slower than `make batch-lowpri` in one shot, but catches regressions the automated suite doesn't: Whisper transcript drift, subtitle positioning on a new product, publish-side SDK changes.

The three `*-lowpri` make targets map one-to-one to the batch phases. Pick a keyword at random from `global_batch.keywords` in `config/pipeline.yaml`. Different product categories stress different code paths (image count, video availability, description length). Then:

```bash
# 1. Scrape
make scrape-lowpri ARGS="--keywords 'mini projector' --debug" \
  MEM_LIMIT=4G NICE_LEVEL=19

# 2. Produce (replace ASIN with what scrape found)
make produce-lowpri ARGS="--batch --random-profile --product-ids B0ASIN123 --clean --debug" \
  MEM_LIMIT=4G NICE_LEVEL=19

# 3. Publish
make publish-lowpri ARGS="single B0ASIN123 --debug" \
  MEM_LIMIT=4G NICE_LEVEL=19
```

`MEM_LIMIT=4G` is tighter than the `make batch-lowpri` default (`6G`) on purpose. Whisper STT peaks near 2.3 GB on a 40-word transcript, so 4 GB leaves room but not much. If a memory regression pushes Whisper over the cap, it fails loudly instead of hiding.

### What to check between phases

After **scrape**: open `outputs/<ASIN>/data.json` and confirm the product has a title, a price, and an affiliate link (the full `?tag=` Amazon URL; the bundled `config/url_shortener.yaml` ships `provider: bare`, so there's no shortened link unless Picsee is enabled). Count the images under `outputs/<ASIN>/images/`; fewer than 3 and the producer will reject the product (media validation floor). If the scrape returns a product short on images, stop there. Running produce on insufficient media wastes about five minutes.

After **produce**: `ffprobe` the output video.

```bash
ffprobe -v error \
  -show_entries format=duration,size,bit_rate \
  -show_entries stream=width,height,codec_name \
  -of default=nw=1 outputs/<ASIN>/video_<ASIN>_<profile>.mp4
```

Codec should be `h264 + aac`, resolution `1080×1920`, duration within a second of the voiceover length. The producer log calls out any warnings worth a closer look: `Duration mismatch`, `Subtitle content similarity to script is low`, threshold breaches for a particular step.

After **publish**: the Late SDK returns a post ID and a per-platform status map. All three platforms (TikTok, YouTube, Instagram) should show `scheduled`. The publisher cleanup step removes the product directory once Zernio confirms the scheduled state, so `outputs/<ASIN>/` disappears when this phase finishes cleanly.

### Why not just `make batch-lowpri`

The one-shot target runs the same phases back-to-back without pausing. When everything works it's fine. When something goes wrong in produce, you've already lost the scrape state (the directory will still exist, but you've burned the scraper's Amazon session and might get throttled re-running). Going step by step means a failed phase leaves the earlier artifacts intact for inspection.

## End-to-end publish-option verification

The smoke test above publishes once. This runbook exercises *every* publishing option and verifies the result on Zernio, which is the right check before a publisher refactor or release. It creates real posts: immediate ones go live, scheduled ones publish at the next free slot. Schedule to a future slot and delete with `python -m src.publisher.late delete <POST_ID>` if you don't want the content to ship.

### The shape

Render one product per option combination with publishing skipped, then publish that product with one option set, then verify. The two publish code paths re-implement the same logic, so cover both (Module/Batch Alignment Rule):

- `single <ASIN>` -- unified (one post, all platforms) by default, or `--platform-specific` (one post per platform); `--immediate` (live now) or default (next free slot); `--link-in-bio` / `--no-link-in-bio`; `--force` (republish).
- `schedule auto` -- config-driven mode (`use_platform_specific_content`), `--dry-run`, `--auto-resolve`. Runs the `record_publish` + `add_to_registry` writes before cleanup.

Render step (per product), publishing skipped:

```bash
xvfb-run -a make batch-lowpri \
  ARGS="--keywords '<keyword>' --max-products 1 --products-per-keyword 1 --random-profile --skip-publish --debug"
```

A three-product matrix covers the surface: (1) `single --immediate` unified live; (2) `single --platform-specific` scheduled; (3) `schedule auto --auto-resolve`. Across the three you also touch dry-run, the dedup guard (`single <ASIN>` again with no `--force` skips already-published platforms), and conflict resolution.

### Verification surfaces

Local state lives at the outputs root and survives product-dir cleanup: `outputs/publish_history.json` (one row per `<ASIN>:<platform>`), `outputs/published_products.json`, `outputs/schedule.json`. Diff their counts before and after.

Live state comes from Zernio. The authoritative read is the post status, not the local files (`publish_history.json` records queue time, not live time). `client.posts.get(<id>)` returns the payload under `.post`; dump with `model_dump(by_alias=True, mode="json")["post"]` and read top `status` plus `platforms[*].status` / `scheduledFor`. Check the disclosure flags while you're there: each YouTube leg carries `containsSyntheticMedia: true`, and each TikTok leg carries `platformSpecificData.tiktokSettings.commercial_content_type: brand_organic` + `is_brand_organic_post: true` (without these TikTok rejects the post at publish time). First-comment delivery is only visible in the platform inbox, not in `posts.get` -- use `verify-comments`, or `LatePublisher.get_post_comments(<platformPostId>, <accountId>)` and look for the comment with `from.isOwner == true`.

### Behaviors and gotchas

- `link-in-bio` runs only on the `single` path, after a successful publish, and reads `outputs/<ASIN>/data.json`. Config defaults it on, so pass `--no-link-in-bio` to keep it off. If the platforms are all already published, `single` returns before the link-in-bio step, so `--link-in-bio` alone is a no-op there. After the product dir is cleaned the `data.json` is gone; to set the link later, reconstruct a `data.json` from `published_products.json` in a temp dir and call `LinkInBioManager.update(<ASIN>, <tmp_outputs>)`. The bio thumbnail comes from `images[0]` (remote URL) or `downloaded_images[0]` (local file) in `data.json`, neither of which the registry stores, so a title-plus-`affiliate_link` reconstruction produces a text-only link with no thumbnail. Re-scrape the ASIN (regenerates `images/` + `data.json`) if the thumbnail matters.
- `schedule auto` needs `--auto-resolve` to take an alternative slot when the preferred slot is occupied (2h `min_post_spacing` by default). Without it, a conflict counts the product as failed and suggests slots. The publisher schedule path exits non-zero on failure; the global batch exits non-zero when nothing completes end-to-end, but partial failures still exit 0. Verify the batch by grepping the phase-summary log lines, not `$?`.
- Cleanup is conservative: it skips while a leg is still `publishing` (immediate runs, the dir stays) and runs once a post is `scheduled` (the dir is removed after the tracking/registry writes).
- A published TikTok leg often returns `platformPostUrl: ""`, which the SDK's strict URL model rejects. `list_posts` / `get_status` tolerate this (they coerce the empty URL to null before validating), so `verify-comments`, slot detection, and blob retention keep working through them. A **direct** `client.posts.list` / `posts.get` call still raises on such a post, so for a one-off script read status via the raw REST API (`GET /api/v1/posts/<id>` or `?page=N&limit=50` with `Authorization: Bearer $LATE_API_KEY`) and check comments via the inbox rather than the SDK.

## Continuous Integration

Tests run automatically on:
- **Pull requests** to main branch
- **Commits** to main branch
- **Scheduled** nightly builds

CI pipeline includes:
- ✅ All test categories (unit, integration, e2e)
- ✅ Code coverage reporting
- ✅ Linting and type checking
- ✅ Security scanning

## Troubleshooting

### Common Issues

**Import errors:**
```bash
# Ensure package is installed in editable mode
poetry install
```

**Fixture not found:**
```bash
# Check conftest.py is in correct location
# Verify fixture name matches exactly
```

**Tests pass locally but fail in CI:**
```bash
# Check for environment-specific dependencies
# Verify all test dependencies are in pyproject.toml
```

**Slow test execution:**
```bash
# Run only unit tests for faster feedback
poetry run pytest -m unit

# Use parallel execution
poetry run pytest -n auto
```

### Test Status

**Current Statistics:**
- **Total Tests**: run `poetry run pytest --collect-only -q | tail -1` for the current count
- **Coverage**: enforced by the CI `--cov-fail-under` gate; see the CI config for the current threshold

## Quick Reference

| Command | Description |
|---------|-------------|
| `make test` | Run all tests |
| `make test-cov` | Run tests with coverage |
| `make test-parallel` | Run tests in parallel |
| `poetry run pytest -v` | Verbose test output |
| `poetry run pytest -x` | Stop on first failure |
| `poetry run pytest -k "pattern"` | Run tests matching pattern |
| `poetry run pytest --lf` | Run last failed tests |
| `poetry run pytest -n auto` | Parallel execution |

## Resources

- **Pytest Documentation**: https://docs.pytest.org
- **Coverage Documentation**: https://coverage.readthedocs.io
- **Mocking Guide**: https://docs.python.org/3/library/unittest.mock.html
- **Contributing Guide**: [CONTRIBUTING.md](../CONTRIBUTING.md)

---

**💡 Tip**: Use `poetry run pytest -m unit` for fast feedback during development, then run full suite before committing.
