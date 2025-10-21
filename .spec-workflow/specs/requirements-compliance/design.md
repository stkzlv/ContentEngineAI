# Design Document

## Overview

**What This Spec Does**: Creates comprehensive tests to verify that ContentEngineAI follows all rules documented in REQUIREMENTS.md.

**Why It Matters**: The system already implements most features, but lacks automated tests to verify compliance with requirements. This spec adds test coverage to ensure features work correctly and don't break in the future.

**What Gets Built**: Test files in `tests/compliance/` that validate configuration precedence, scraper behavior, and video production features against documented requirements.

## Steering Document Alignment

### Technical Standards (tech.md)

- **Uses existing test framework**: Pytest with async support (already in pyproject.toml)
- **Follows existing code style**: Ruff linting, type hints, 88-char lines
- **Integrates with CI/CD**: Tests run in existing GitHub Actions workflows

### Project Structure (structure.md)

**New files added**:
```
tests/
└── compliance/                       # NEW folder for requirements tests
    ├── test_config_compliance.py    # Tests for configuration system
    ├── test_scraper_compliance.py   # Tests for scraper requirements
    └── test_video_compliance.py     # Tests for video production
```

**Follows existing patterns**: Test files use `test_*.py` naming, functions use `test_req_*` naming to reference requirement numbers.

## Code Reuse Analysis

### What Already Exists (and we'll test)

**Configuration System** (`src/config_manager.py`):
- `UnifiedConfigManager` class handles CLI > ENV > YAML precedence
- `apply_precedence_rules()` method applies overrides in correct order
- `_apply_env_overrides()` reads environment variables
- `_apply_cli_overrides()` applies command-line arguments
- **We'll test**: That precedence actually works correctly

**Video Configuration** (`src/video/config_adapter.py`):
- `ModularConfigAdapter` loads YAML files from `config/` directory
- Merges 5 YAML files into single configuration dictionary
- **We'll test**: That merged config contains all required fields

**Scraper System** (`src/scraper/`):
- `BaseScraper` abstract class defines interface for all platforms
- `AmazonScraper` implements Amazon-specific scraping
- **We'll test**: That scrapers follow multi-platform architecture rules

### What We'll Create (new test code)

**Test Fixtures** (shared test data):
```python
@pytest.fixture
def mock_yaml_config() -> dict:
    """Provides sample YAML configuration for testing."""
    return {
        "debug_mode": False,
        "subtitle_settings": {"anchor": "bottom", "margin": 0.1}
    }

@pytest.fixture
def mock_env_vars() -> dict:
    """Provides sample environment variables."""
    return {"DEBUG_MODE": "true", "SUBTITLE_ANCHOR": "top"}
```

**Validation Helpers** (assertion functions):
```python
def assert_cli_wins(final_value, cli_value, env_value, yaml_value):
    """Verify CLI value overrides others."""
    assert final_value == cli_value

def assert_no_secrets_in_yaml(yaml_file_path):
    """Scan YAML for API keys/secrets."""
    content = yaml_file_path.read_text()
    secret_patterns = ["api_key:", "token:", "password:"]
    for pattern in secret_patterns:
        assert pattern not in content.lower()
```

## Architecture

**How Tests are Organized**:

```
Requirements Document (12 requirements)
    ↓
Configuration Tests (Req 1, 11, 12)
    → test_cli_precedence()
    → test_env_precedence()
    → test_yaml_fallback()
    → test_secret_isolation()
    → test_validation_errors()
    ↓
Scraper Tests (Req 2, 3)
    → test_base_scraper_interface()
    → test_product_data_extraction()
    → test_media_download()
    → test_directory_structure()
    ↓
Video Production Tests (Req 4-10)
    → test_duration_matching()
    → test_subtitle_positioning()
    → test_two_part_subtitles()
    → test_profile_overrides()
    → test_style_presets()
    → test_ass_effects()
```

**Test Strategy**:
1. **Unit tests**: Test individual functions in isolation (e.g., precedence logic)
2. **Integration tests**: Test multiple components together (e.g., load config → validate → use in pipeline)
3. **Compliance tests**: Verify specific requirements are met (this spec)

## Components and Interfaces

### Component 1: Configuration Compliance Tests

**File**: `tests/compliance/test_config_compliance.py`

**What it tests**:
- Requirement 1: Three-tier configuration precedence (CLI > ENV > YAML)
- Requirement 11: Global debug mode
- Requirement 12: Configuration validation

**Example test**:
```python
def test_req_1_cli_overrides_everything():
    """REQ-1: CLI arguments must override ENV and YAML."""
    # Setup
    manager = UnifiedConfigManager()
    yaml_config = {"debug_mode": False}

    # Simulate: YAML=false, ENV=maybe, CLI=true
    with mock.patch.dict(os.environ, {"DEBUG_MODE": "false"}):
        cli_overrides = {"debug": True}
        result = manager.apply_precedence_rules(yaml_config, cli_overrides)

    # Verify: CLI wins
    assert result["debug_mode"] is True
```

**Tests to add**: ~10 tests covering different precedence scenarios

### Component 2: Scraper Compliance Tests

**File**: `tests/compliance/test_scraper_compliance.py`

**What it tests**:
- Requirement 2: Multi-platform scraper architecture
- Requirement 3: Product media discovery and storage

**Example test**:
```python
async def test_req_2_basescraper_interface():
    """REQ-2: Platform-specific scrapers extend BaseScraper."""
    from src.scraper.base import BaseScraper
    from src.scraper.amazon.scraper import AmazonScraper

    # Verify: AmazonScraper inherits from BaseScraper
    assert issubclass(AmazonScraper, BaseScraper)

    # Verify: Implements required abstract methods
    assert hasattr(AmazonScraper, 'scrape_products')
    assert hasattr(AmazonScraper, 'validate_product_id')
```

**Tests to add**: ~8 tests covering scraper requirements

### Component 3: Video Production Compliance Tests

**File**: `tests/compliance/test_video_compliance.py`

**What it tests**:
- Requirement 4: Dynamic video assembly
- Requirement 5: Unified subtitle positioning
- Requirement 6: Two-part subtitle system
- Requirement 7: Profile-specific settings
- Requirement 8: Style presets
- Requirement 9: ASS effects system
- Requirement 10: AI service integration

**Example test**:
```python
async def test_req_5_content_aware_positioning():
    """REQ-5: Content-aware mode adjusts subtitle position."""
    # Setup
    config = VideoConfig(subtitle_settings={
        "anchor": "below_content",
        "content_aware": True,
        "margin": 0.05
    })

    # Mock image with detected content boundaries
    mock_image_bounds = {"top": 100, "bottom": 800}

    # Execute
    subtitle_position = calculate_subtitle_position(
        config, mock_image_bounds, frame_height=1080
    )

    # Verify: Position is below content (> 800 pixels)
    assert subtitle_position > 800
```

**Tests to add**: ~25 tests covering video production requirements

## Data Models

**Test Case Data Structure**:
```python
# Simple dictionary for test parameters
test_params = {
    "requirement_id": "REQ-1.1",
    "yaml_value": False,
    "env_value": "true",
    "cli_value": True,
    "expected_result": True  # CLI wins
}
```

No complex data models needed - tests use simple dictionaries and fixtures.

## Error Handling

**What we test for**:

1. **Missing configuration** → Test expects clear error message
2. **Invalid types** → Test expects validation error with field name
3. **API keys in YAML** → Test fails with security warning
4. **Multiple ASS effects** → Test expects exactly 1 effect per video

**Example**:
```python
def test_req_12_validation_error_messages():
    """REQ-12: Validation errors must be clear."""
    invalid_config = {"subtitle_settings": {"margin": 2.0}}  # Invalid: >0.5

    with pytest.raises(ValidationError) as exc:
        VideoConfig(**invalid_config)

    error_msg = str(exc.value)
    assert "margin" in error_msg
    assert "0.0" in error_msg and "0.5" in error_msg
```

## Testing Strategy

### How to Run Tests

```bash
# Run all compliance tests
pytest tests/compliance/ -v

# Run only configuration tests
pytest tests/compliance/test_config_compliance.py -v

# Run specific requirement
pytest tests/compliance/ -k "test_req_1" -v

# Run with coverage
pytest tests/compliance/ --cov=src --cov-report=term-missing
```

### What Gets Tested

**Unit Tests** (~15 tests):
- Individual functions: precedence logic, validation, positioning calculations
- Fast execution (<1 second total)
- No external dependencies

**Integration Tests** (~10 tests):
- Multi-component flows: load config → validate → use in pipeline
- Medium execution time (~5 seconds total)
- Mock external APIs (LLM, TTS, scraping)

**End-to-End Tests** (~5 tests):
- Complete workflows: scrape → generate script → create video
- Slower execution (~30 seconds total)
- Full pipeline validation

**Total**: ~30 new tests added

### Success Criteria

- All 12 requirements have at least one corresponding test
- Tests pass consistently (not flaky)
- Coverage of requirements-related code reaches >90%
- Tests run in CI/CD pipeline without failures
