# ContentEngineAI Test Suite

This directory contains comprehensive tests for the ContentEngineAI project, including unit tests, integration tests, and end-to-end tests.

## Test Structure

<details>
<summary><strong>Complete Test File Structure</strong></summary>

```
tests/
├── __init__.py                      # Test package initialization
├── conftest.py                      # Pytest configuration and shared fixtures
├── run_tests.py                     # Test runner script
│
├── # Core Component Tests
├── test_utils.py                    # Unit tests for utility functions
├── test_video_config.py             # Unit tests for video configuration
├── test_ai_script_generator.py      # Unit tests for AI script generation
├── test_ai_description_generator.py # Unit tests for AI description generation
├── test_audio.py                    # Unit tests for audio processing (Freesound)
├── test_tts.py                      # Unit tests for text-to-speech
├── test_assembler.py                # Unit tests for video assembly (FFmpeg)
├── test_assembler_decomposed.py     # Unit tests for decomposed assembler components
├── test_subtitle_utils.py           # Unit tests for subtitle utilities
├── test_subtitle_validation.py      # Unit tests for subtitle validation
├── test_subtitle_positioning.py     # Unit tests for subtitle positioning system
├── test_unified_subtitle_positioning.py # Unit tests for unified subtitle positioning
├── test_unified_subtitle_generator.py   # Unit tests for unified subtitle generator
├── test_result_types.py             # Unit tests for pipeline result types
├── test_stock_media.py              # Unit tests for stock media fetching
├── test_circuit_breaker.py          # Unit tests for circuit breaker functionality
│
├── # Configuration & Validation Tests
├── test_config_validator.py         # Unit tests for configuration validation
├── test_config_extensions.py        # Tests for configuration extensions
├── test_scraper_config_enhanced.py  # Tests for enhanced scraper configuration
├── test_media_validator.py          # Unit tests for media validation
├── test_media_validation.py         # Integration tests for media validation
│
├── # Pipeline & Structure Tests
├── test_pipeline_graph.py           # Tests for pipeline dependency graph
├── test_producer_cleanup.py         # Tests for producer cleanup functionality
├── test_outputs_structure.py        # Tests for output directory structure
│
├── # Integration & Performance Tests
├── test_optimization_integration.py # Integration tests for optimizations
├── test_performance.py             # General performance tests
├── test_slideshow_images1_verification.py  # Video output verification tests
│
└── TESTING.md                       # This file
```

</details>

The test suite contains comprehensive test coverage across multiple categories.

## Video Verification Tests

The test suite includes comprehensive video verification tests that validate the final pipeline output:

### What is Verified
- **Audio Quality**: Background music presence, volume levels, duration matching
- **Visual Layout**: Image positioning, sizing, subtitle placement 
- **Configuration Compliance**: Resolution, frame rate, codec specifications
- **Profile Requirements**: slideshow_images1 profile uses only scraped images
- **Content Synchronization**: Subtitle-voiceover timing alignment

### How Verification Works
1. **Screenshot Extraction**: Captures frames every 2 seconds from real videos
2. **Audio Analysis**: Uses FFprobe to analyze audio streams and levels
3. **Visual Analysis**: Measures positioning and sizing against configuration
4. **Artifact Storage**: Saves analysis results in pipeline output directory

### Running Video Verification
```bash
# Run all video verification tests
poetry run pytest tests/test_slideshow_images1_verification.py -v

# Run background music verification only
poetry run pytest tests/test_slideshow_images1_verification.py::TestSlideshowImagesVerification::test_slideshow_images1_background_music_verification -v

# Run with real video (requires existing pipeline output)
poetry run pytest tests/test_slideshow_images1_verification.py::TestSlideshowImagesVerification::test_slideshow_images1_profile_video_verification -v
```

### Verification Artifacts
Test artifacts are stored alongside pipeline outputs:
```
outputs/videos/{product_id}/{profile_name}/temp/verification/
├── screenshots/                    # Extracted video frames
├── audio_analysis_results.json     # Audio analysis data
├── visual_analysis_results.json    # Visual analysis data
└── verification_report.json        # Overall test results
```

**Note**: Video verification tests require actual pipeline-generated videos. Run the pipeline first to generate test videos.

## Test Categories

### Unit Tests (`@pytest.mark.unit`)
- **Purpose**: Test individual functions and classes in isolation
- **Coverage**: Core utilities, configuration validation, helper functions
- **Speed**: Fast execution (< 1 second per test)
- **Dependencies**: Minimal external dependencies, heavy use of mocking

### Integration Tests (`@pytest.mark.integration`)
- **Purpose**: Test interaction between multiple components
- **Coverage**: Pipeline steps, component communication, data flow
- **Speed**: Medium execution (1-10 seconds per test)
- **Dependencies**: Mocked external services, real file system operations

### End-to-End Tests (`@pytest.mark.e2e`)
- **Purpose**: Test complete workflows from start to finish
- **Coverage**: Full pipeline execution, real API interactions
- **Speed**: Slow execution (10+ seconds per test)
- **Dependencies**: Real external services, full system setup

### Video Verification Tests
- **Purpose**: Verify final video output quality and compliance
- **Coverage**: Audio levels, subtitle positioning, image sizing, configuration compliance
- **Method**: Screenshot analysis, FFprobe audio analysis, real video validation
- **Speed**: Slow execution (60+ seconds per test)
- **Dependencies**: Real pipeline-generated videos, FFmpeg tools

## Running Tests

### Prerequisites

1. **Install dependencies**:
   ```bash
   poetry install
   ```

2. **Install test dependencies**:
   ```bash
   poetry install --with dev
   ```

3. **Verify installation**:
   ```bash
   poetry run pytest --version
   ```

### Basic Test Commands

#### Run All Tests
```bash
poetry run pytest
```

#### Run Specific Test Categories
```bash
# Unit tests only
poetry run pytest -m unit

# Integration tests only
poetry run pytest -m integration

# End-to-end tests only
poetry run pytest -m e2e
```

#### Run Specific Test Files
```bash
# Run utility tests
poetry run pytest tests/test_utils.py

# Run configuration tests
poetry run pytest tests/test_video_config.py

# Run AI script generator tests
poetry run pytest tests/test_ai_script_generator.py

# Run AI description generator tests
poetry run pytest tests/test_ai_description_generator.py

# Run video verification tests (requires real pipeline output)
poetry run pytest tests/test_slideshow_images1_verification.py
```

#### Run Specific Test Functions
```bash
# Run specific test function
poetry run pytest tests/test_utils.py::TestFormatTimestamp::test_format_timestamp_seconds

# Run all tests in a class
poetry run pytest tests/test_utils.py::TestDownloadFile
```

### Using the Test Runner Script

The `run_tests.py` script provides convenient ways to run different types of tests:

```bash
# Run all tests with coverage
python tests/run_tests.py --type all

# Run only unit tests
python tests/run_tests.py --type unit

# Run integration tests with verbose output
python tests/run_tests.py --type integration --verbose

# Run tests in parallel
python tests/run_tests.py --type all --parallel 4

# Run tests with specific markers
python tests/run_tests.py --markers "unit and not slow"

# Generate HTML coverage report
python tests/run_tests.py --type coverage --output html
```

### Test Runner Options

- `--type`: Type of tests to run (`unit`, `integration`, `all`, `coverage`, `lint`, `type-check`)
- `--verbose, -v`: Verbose output
- `--parallel, -n`: Number of parallel processes
- `--markers`: Run only tests with specific markers
- `--output`: Test output format (`text`, `html`, `xml`)

## Test Configuration

### Pytest Configuration (`pytest.ini`)

The project uses a comprehensive pytest configuration:

- **Test discovery**: Automatically finds test files and functions
- **Coverage reporting**: Generates coverage reports in multiple formats
- **Timeout handling**: Prevents tests from hanging indefinitely
- **Markers**: Defines test categories and metadata
- **Async support**: Automatic async/await handling

### Shared Fixtures (`conftest.py`)

The test suite provides many shared fixtures:

#### Core Fixtures
- `temp_dir`: Temporary directory for test files
- `sample_product_data`: Sample product data for testing
- `sample_video_profile`: Sample video profile for testing
- `mock_config`: Mock video configuration
- `mock_aiohttp_session`: Mock HTTP session
- `mock_aioresponses`: Mock HTTP responses

#### Test Data Fixtures
- `sample_script`: Sample generated script
- `sample_srt_content`: Sample SRT subtitle content
- `mock_ffmpeg_path`: Mock FFmpeg executable
- `mock_google_cloud_credentials`: Mock Google Cloud credentials
- `mock_env_vars`: Mock environment variables

#### Configuration Fixtures

**Real Config Loading Pattern** - Recommended for complex Pydantic models:

```python
@pytest.fixture
def test_config(temp_outputs_dir):
    """Create test config using modular config loading with overrides."""
    from src.video.config_adapter import load_video_config_modular

    # Load modular config system
    config = load_video_config_modular()

    # Override specific settings for testing
    config.global_output_directory = str(temp_outputs_dir)
    if hasattr(config, 'cleanup_settings'):
        config.cleanup_settings.enabled = True
        config.cleanup_settings.dry_run = False

    return config
```

**Benefits of Real Config Pattern**:
- ✅ Eliminates Pydantic validation errors
- ✅ Tests against actual configuration schema  
- ✅ Maintains realistic test environment
- ✅ Easy to override specific settings

## Test Coverage

### Coverage Targets

- **Overall Project**: >30% line coverage (current target)
- **Configuration**: >50% line coverage
- **Utilities**: >40% line coverage
- **Core Components**: >35% line coverage
- **Integration Paths**: >30% coverage
- **Error Handling**: >40% coverage for critical paths
- **Amazon Scraper**: >45% line coverage with comprehensive feature testing

### Coverage Reports

Generate coverage reports:

```bash
# Terminal coverage report
poetry run pytest --cov=src --cov-report=term-missing

# HTML coverage report
poetry run pytest --cov=src --cov-report=html:htmlcov

# XML coverage report (for CI/CD)
poetry run pytest --cov=src --cov-report=xml
```

View HTML coverage report:
```bash
# Open in browser
open htmlcov/index.html
```

## Writing Tests

### Test Naming Conventions

- **Test files**: `test_<module_name>.py`
- **Test classes**: `Test<ClassName>`
- **Test functions**: `test_<function_name>_<scenario>`

### Test Structure

```python
import pytest
from unittest.mock import AsyncMock, patch

class TestExampleClass:
    """Test class for ExampleClass."""

    def test_example_function_success(self, temp_dir: Path):
        """Test successful execution of example function."""
        # Arrange
        input_data = "test input"
        expected_output = "expected result"

        # Act
        result = example_function(input_data)

        # Assert
        assert result == expected_output

    @pytest.mark.asyncio
    async def test_async_function_error(self, mock_aiohttp_session: AsyncMock):
        """Test async function error handling."""
        # Arrange
        mock_aiohttp_session.get.side_effect = Exception("Network error")

        # Act & Assert
        with pytest.raises(Exception, match="Network error"):
            await async_function(mock_aiohttp_session)
```

### Using Fixtures

```python
def test_with_fixtures(sample_product_data: ProductData, temp_dir: Path):
    """Test using shared fixtures."""
    # Use sample_product_data fixture
    assert sample_product_data.title == "Test Product - Wireless Bluetooth Headphones"

    # Use temp_dir fixture for file operations
    test_file = temp_dir / "test.txt"
    test_file.write_text("test content")
    assert test_file.exists()
```

### Mocking External Dependencies

```python
@patch("src.module.external_api_call")
def test_with_mocking(mock_api_call, temp_dir: Path):
    """Test with mocked external API."""
    # Mock API response
    mock_api_call.return_value = {"status": "success", "data": "test"}

    # Test function that uses external API
    result = function_that_calls_api()

    # Verify API was called
    mock_api_call.assert_called_once()
    assert result == "test"
```

### Async Testing

```python
@pytest.mark.asyncio
async def test_async_function(mock_aiohttp_session: AsyncMock):
    """Test async function."""
    # Mock async response
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = {"result": "success"}
    mock_aiohttp_session.get.return_value.__aenter__.return_value = mock_response

    # Test async function
    result = await async_function(mock_aiohttp_session)

    assert result == "success"
```

## Continuous Integration

### Pre-commit Hooks

The project includes pre-commit hooks that run tests automatically:

```bash
# Install pre-commit hooks
poetry run pre-commit install

# Run pre-commit hooks manually
poetry run pre-commit run --all-files
```

### CI/CD Pipeline

The test suite is designed to work with CI/CD pipelines:

1. **Code Quality**: Linting and type checking
2. **Unit Tests**: Fast unit tests with high coverage
3. **Integration Tests**: Component interaction tests
4. **Coverage Reporting**: Generate coverage reports
5. **Test Reports**: Generate test result reports

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure you're in the correct directory
cd ContentEngineAI

# Install dependencies
poetry install --with dev

# Check Python path
poetry run python -c "import src; print('Import successful')"
```

#### Test Failures
```bash
# Run tests with verbose output
poetry run pytest -v

# Run specific failing test
poetry run pytest tests/test_specific.py::test_failing_function -v

# Run tests with debug output
poetry run pytest --tb=long
```

#### Coverage Issues
```bash
# Check coverage configuration
poetry run pytest --cov=src --cov-report=term-missing

# Generate detailed coverage report
poetry run pytest --cov=src --cov-report=html:htmlcov --cov-report=term-missing
```

### Common Test Issues

#### Pydantic Validation Errors
**Problem**: `AttributeError: Mock object has no attribute 'resolution'`  
**Solution**: Use real config loading instead of complex mocks

```python
# ❌ Problematic pattern (causes validation errors)
mock_video_settings = Mock(spec=VideoSettings)
config = VideoConfig(video_settings=mock_video_settings)

# ✅ Recommended pattern (works reliably)
config = load_video_config(config_path)
config.global_output_directory = str(temp_dir)  # Override as needed
```

#### Async Test Issues
**Problem**: `TypeError: 'coroutine' object does not support the asynchronous context manager protocol`  
**Solution**: Proper async mock setup

```python
# ✅ Correct async mock pattern
@pytest.mark.asyncio
async def test_async_function():
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_session.get.return_value.__aenter__.return_value = mock_response
    
    result = await async_function(mock_session)
```

#### Complex Integration Test Management
**Problem**: Producer pipeline tests failing due to missing dependencies  
**Solution**: Skip complex integration tests during development

```python
@pytest.mark.skip(reason="Complex integration - requires extensive mocking")
def test_complex_integration():
    pass
```

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Set debug environment variable
export DEBUG=1

# Run tests with debug output
poetry run pytest -v --log-cli-level=DEBUG
```

## Best Practices

### Test Design

1. **Arrange-Act-Assert**: Structure tests with clear sections
2. **Single Responsibility**: Each test should test one thing
3. **Descriptive Names**: Use clear, descriptive test names
4. **Minimal Dependencies**: Mock external dependencies
5. **Fast Execution**: Keep tests fast for quick feedback

### Configuration Testing

1. **Use Real Config Files**: Load actual configuration and override specific values
2. **Avoid Complex Mocks**: Don't mock Pydantic models - use real instances
3. **Override Minimally**: Only override what's needed for the specific test
4. **Test Validation**: Ensure Pydantic validation works correctly

```python
# ✅ Recommended pattern for configuration tests
@pytest.fixture
def test_config(temp_dir):
    from src.video.config_adapter import load_video_config_modular
    config = load_video_config_modular()
    config.global_output_directory = str(temp_dir)
    return config

def test_with_config(test_config):
    # Test uses real, validated configuration
    assert test_config.global_output_directory != "outputs"  # Overridden
    assert test_config.video_settings.frame_rate == 30  # Real default
```

### Async Testing

1. **Use AsyncMock**: For async functions and context managers
2. **Proper Context Managers**: Set up async context managers correctly  
3. **Mock External Services**: Don't make real network calls in tests
4. **Test Error Handling**: Test async error scenarios

```python
# ✅ Recommended async test pattern
@pytest.mark.asyncio
async def test_async_function():
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = {"result": "success"}
    mock_session.get.return_value.__aenter__.return_value = mock_response
    
    result = await async_function(mock_session)
    assert result == "success"
```

### Integration Test Strategy

1. **Start Simple**: Begin with unit tests, add integration gradually
2. **Mock Boundaries**: Mock at service boundaries, not internal functions
3. **Test Key Paths**: Focus on critical business logic flows
4. **Use Real Data Structures**: Use actual data models, not mocks

### Test Data

1. **Use Fixtures**: Leverage shared fixtures for common data
2. **Minimal Test Data**: Use only necessary data for each test  
3. **Realistic Data**: Use realistic but minimal test data
4. **Cleanup**: Ensure proper cleanup after tests

### Error Testing

1. **Test Error Paths**: Ensure error conditions are tested
2. **Exception Testing**: Test that appropriate exceptions are raised
3. **Edge Cases**: Test boundary conditions and edge cases
4. **Graceful Degradation**: Test how the system handles failures

## Contributing

### Adding New Tests

1. **Follow Naming Conventions**: Use established naming patterns
2. **Add Appropriate Markers**: Mark tests with appropriate categories
3. **Update Documentation**: Update this README if adding new test types
4. **Maintain Coverage**: Ensure new code has adequate test coverage

### Test Maintenance

1. **Keep Tests Updated**: Update tests when code changes
2. **Remove Obsolete Tests**: Remove tests for removed functionality
3. **Refactor Tests**: Refactor tests to improve maintainability
4. **Monitor Performance**: Ensure tests remain fast and efficient

## Quick Reference

### Essential Commands
```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src --cov-report=html

# Run specific category
poetry run pytest -m unit
poetry run pytest -m integration

# Run single test file
poetry run pytest tests/test_utils.py

# Run with verbose output
poetry run pytest -v --tb=short

# Check test count
poetry run pytest --collect-only -q | tail -3
```

### Test Markers
- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests (slower, more dependencies)  
- `@pytest.mark.e2e` - End-to-end tests (slowest, full system)
- `@pytest.mark.asyncio` - Async tests
- `@pytest.mark.skip(reason="...")` - Skip test with reason

### AI Component Testing
The AI components include comprehensive test coverage for:

#### AI Description Generator (`TestAIDescriptionGenerator`)
- Tests prompt template loading and formatting
- Validates description completeness and format requirements
- Ensures hashtag validation and #ad compliance
- Tests error handling for missing templates and API keys

#### AI Script Generator (`TestAIScriptGenerator`)
- Tests script generation logic and validation
- Validates prompt formatting and template handling
- Ensures proper error handling and fallback mechanisms

```bash
# Run AI description generator tests
poetry run pytest tests/test_ai_description_generator.py -v

# Run AI script generator tests
poetry run pytest tests/test_ai_script_generator.py -v
```

### Additional Component Testing

#### Circuit Breaker (`TestCircuitBreaker`)
- Tests circuit breaker functionality for external service resilience
- Validates failure threshold handling and automatic recovery
- Ensures proper state transitions and error reporting

#### Media Validation (`TestMediaValidation`)
- Tests media file validation and quality checks
- Validates image and video format requirements
- Ensures proper error handling for corrupted media

```bash
# Run circuit breaker tests
poetry run pytest tests/test_circuit_breaker.py -v

# Run media validation tests
poetry run pytest tests/test_media_validation.py -v
poetry run pytest tests/test_media_validator.py -v
```

### Test Status
**Current Test Statistics (as of latest update):**
- **Total Tests**: 485+ tests collected (28 new tests added for enhanced scraper configuration)
- **Test Coverage**: 20.91% (working towards 30% minimum target)
- **Test Status**: 429+ passed, 7 skipped, remainder in progress
- **Recent Updates**:
  - **NEW**: Added comprehensive scraper configuration tests (28 tests in `test_scraper_config_enhanced.py`)
    - Browser timeout configuration validation
    - Media processing settings verification
    - Configuration documentation completeness checks
    - Value range and type validation tests
  - Added comprehensive tests for unified configuration system (19 tests)
  - Added enhanced fallback logic tests (15 tests)
  - Updated tests to use new modular config files (`video_production.yaml`)
  - Fixed deprecated config file references
  - Improved configuration system coverage: config_manager.py from 85% to 77%, scraper config from 32% to 61%
  - Enhanced error handling and fallback testing across all config systems

Run `poetry run pytest --collect-only -q | tail -3` to see current test counts and status.

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [Coverage.py](https://coverage.readthedocs.io/)
- [Python Mocking](https://docs.python.org/3/library/unittest.mock.html)
