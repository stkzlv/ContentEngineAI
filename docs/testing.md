# Testing Guide

ContentEngineAI uses a comprehensive test suite with **1888 tests** across unit, integration, and end-to-end categories.

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

**Verification artifacts** stored in: `outputs/videos/{product_id}/{profile_name}/temp/verification/`

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
│   ├── test_late_publisher.py       # Late.dev publisher integration
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
│   │   └── test_client.py           # Late.dev API client
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
- **Overall minimum**: 50% (currently at 56%)

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
- **Total Tests**: 1741 passing
- **Coverage**: 56% (target 50% minimum)

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
