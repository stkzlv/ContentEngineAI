# Changelog

All notable changes to ContentEngineAI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Centralized Logging Utility**: New `src/utils/logging_setup.py` module provides standardized logging configuration
- **LLM Validation Settings**: Added description validation thresholds to configuration (`min_description_chars`, `min_description_words`, `description_retry_attempts`)
- **Cleanup Configuration**: New `cleanup_on_failure` setting to control temp file retention on pipeline failures
- **Configuration Audit**: Comprehensive `docs/CONFIG_AUDIT.md` documenting hardcoded values, unused settings, and improvement roadmap
- **Debug Documentation**: Expanded TROUBLESHOOTING.md with debug files reference table and configuration guidance

### Changed
- **Debug Mode Logging**: Eliminated 60+ lines of duplicated logging setup code between producer and scraper
- **FFmpeg Logging Logic**: Simplified `_should_create_ffmpeg_logs()` method with clearer fallback behavior and explicit error logging
- **Debug Settings**: Enhanced `DebugSettings` model with separate `cleanup_on_success` and `cleanup_on_failure` controls
- **Configuration Comments**: Improved inline documentation in `config/performance.yaml` for cleanup behavior

### Fixed
- **Logging Configuration**: Producer and scraper now use shared `setup_debug_logging()` function for consistency
- **Debug Mode Hierarchy**: Clarified CLI `--debug` flag precedence over config file settings in documentation

### Technical
- **Code Deduplication**: Removed redundant logging setup code (producer.py:71-99, scraper.py:955-1001)
- **Configuration Analysis**: Identified 25+ hardcoded values, 7 unused settings, and 5 duplicate configuration conflicts
- **Test Coverage**: All tests passing (469 passed, 10 skipped) with 41% coverage maintained

## [0.4.0] - 2025-10-01

### Added
- **Unified Configuration System**: Modular YAML architecture with 6 specialized config files (core, video_production, ai_services, subtitles, performance, scraper)
- **Triple Precedence Configuration**: CLI arguments override environment variables override YAML defaults
- **CLI Configuration Overrides**: Command-line parameters can override any YAML configuration value
- **Environment Variable Support**: All configuration settings can be set via environment variables
- **Configuration Validation**: Enhanced validation with Pydantic models and clear error messages
- **Backward Compatibility Layer**: Adapter classes maintain 100% compatibility with existing code

### Changed
- **Configuration Architecture**: Split monolithic `config/video_producer.yaml` into 6 modular files
- **Complexity Reduction**: 54% reduction in configuration complexity (1,962 → 1,047 lines)
- **Performance Improvement**: 20% faster configuration loading through lazy loading and better caching
- **Documentation Overhaul**: Completely rewritten CONFIGURATION.md with modular system guide
- **Architecture Documentation**: Updated ARCHITECTURE.md with configuration system overview
- **Project Documentation**: Streamlined README.md and consolidated STATUS.md content

### Technical
- **Modular Loading**: Independent loading of configuration modules with dependency resolution
- **Memory Optimization**: Reduced memory footprint through lazy configuration loading
- **Configuration Caching**: Improved caching of parsed configuration values
- **Test Coverage**: Enhanced test suite with configuration validation tests (424 tests maintained)
- **Zero Breaking Changes**: All existing function signatures preserved through adapter pattern

## [0.3.1] - 2025-09-23

### Added
- **RANDOM Preset**: New style preset with deterministic randomization using product-specific seeding for fonts, colors, and single animation effects
- **CLI Style Override**: Added `--preset` command-line argument for easy video styling control (minimal, modern, bold, random)
- **Enhanced Randomization**: Improved font and color randomization system with better effect selection

### Changed
- **Optimized Preset System**: Reduced preset count from 5 to 4 (removed `animated` and `classic`, kept `minimal`, `modern`, `bold`)
- **Effect Limitation**: Limited effects to 1 per preset to prevent visual clutter and rendering issues
- **Improved Documentation**: Updated README.md for simplicity with collapsible sections

### Fixed
- **ASS Effects Application**: Fixed ASS effects not applying by changing condition from >1 to >0 effects
- **Random Effect Selection**: Enabled randomize_effects for RANDOM preset to activate effect system properly
- **Configuration Alignment**: Updated all documentation to match actual 4-preset codebase implementation

### Technical
- **Deterministic Randomization**: RANDOM preset uses product ID-based seeding for consistent per-video styling
- **CLI Integration**: Producer now accepts preset override parameter for flexible styling
- **Test Coverage**: Updated comprehensive test suite to reflect new preset system (424 tests across 27 files)
- **Code Quality**: All quality gates pass with optimized preset system implementation

## [0.3.0] - 2025-09-21

### Added
- **Font and Color Randomization System**: New comprehensive deterministic randomization system for subtitle fonts and colors
- **New Font Manager**: Added `font_color_manager.py` module for centralized font and color management
- **Product-Specific Seeding**: Deterministic font/color selection based on product ID for consistent results
- **Enhanced Subtitle Configuration**: New subtitle settings with font/color randomization options
- **Comprehensive Test Coverage**: Added new test suites for subtitle validation and unified subtitle generation

### Changed
- **Code Quality Improvements**: Fixed 18 linting issues across 6 core files for better maintainability
- **Type Annotations**: Enhanced type checking with proper annotations and MyPy compliance
- **Security Compliance**: Added proper security warning suppressions for non-cryptographic randomization
- **Configuration Enhancement**: Updated video producer configuration with new subtitle randomization options
- **Documentation Updates**: Updated architecture and testing documentation

### Fixed
- **Line Length Issues**: Fixed E501 violations by splitting long debug messages across multiple lines
- **Import Sorting**: Resolved I001 violations with proper import organization
- **Docstring Issues**: Fixed missing parameter descriptions and formatting issues
- **Type Checking**: Resolved MyPy errors with proper SubtitleSettings object usage
- **Constructor Parameters**: Added missing optional parameters to UnifiedSubtitleConfig

### Technical
- **Subtitle Pipeline**: Enhanced subtitle generation with randomization capabilities
- **Performance Monitoring**: Maintained consistent pipeline performance (232-283 seconds)
- **Testing Framework**: All 413 tests pass with improved coverage
- **Code Standards**: Achieved compliance with Ruff, MyPy, Bandit, Vulture, and Safety tools

## [0.2.1] - 2025-09-20

### Fixed
- **Missing Pipeline Step**: Added missing `generate_description` step to pipeline execution - description generation was completely skipped despite having all the code
- **Critical Path Resolution**: Fixed description generator failing due to relative path issues when run from different working directories
- **Producer Cleanup**: Fixed missing `description.txt` and erroneous directories (`~`, `outputs`) in cleanup process with `--clean` flag
- **Whisper Model Caching**: Fixed literal `~` directory creation by properly expanding home directory path with `os.path.expanduser()`
- **Pipeline Reliability**: Ensured producer works correctly regardless of current working directory

### Changed
- Enhanced producer cleanup to remove all temporary and generated files consistently
- Improved path handling throughout the pipeline for better portability
- Updated test documentation to reflect current structure (365 tests across 23 files)
- Updated project status documentation with current capabilities and fixes

### Technical
- Added `generate_description` step to pipeline graph with proper dependency on `generate_script` step
- Made description generator use absolute paths for template loading
- Added proper home directory expansion in Whisper model configuration
- Enhanced producer file cleanup logic with comprehensive file removal
- Improved error handling and path resolution across multiple modules

## [0.2.0] - 2025-09-20

### Added
- **AI-Generated Video Descriptions**: New feature for generating social media descriptions using LLM providers
- New `description_generator.py` module with template-based prompt formatting and hashtag validation
- `DescriptionSettings` configuration class with platform targeting and validation options
- Social media compliance with required #ad hashtag for advertising disclosure
- Integration with video producer pipeline as new `STEP_GENERATE_DESCRIPTION` step
- Comprehensive test suite for description generation functionality

### Changed
- Extended video producer pipeline to include description generation step
- Updated configuration schema to include `description_settings` section
- Enhanced product files structure to include `description.txt` output
- Updated all test fixtures to support new configuration requirements

### Technical
- Added circuit breaker pattern for API resilience in description generation
- Implemented async/await patterns following existing LLM integration standards
- Added Pydantic validation for description settings and content quality
- Extended configuration loading to validate new description settings

## [0.1.2] - 2025-09-18

### Fixed
- Fixed CI test failures by adding FFmpeg to release workflow
- Resolved FFmpeg dependency validation issues in test environment
- Fixed media validator test error message expectations
- Improved test reliability in CI environments

### Changed
- Enhanced subtitle positioning system with improved style presets
- Renamed DYNAMIC subtitle preset to RELATIVE for better clarity
- Added font_width_to_height_ratio configuration to all subtitle style presets
- Updated video producer configuration with enhanced subtitle settings

### Technical
- Added FFmpeg installation to GitHub Actions release workflow
- Improved CI/CD pipeline reliability and test coverage
- Enhanced configuration validation for production environments

## [0.1.1] - 2025-09-17

### Fixed
- Resolved all CI linting and type checking issues
- Fixed MyPy type annotation errors in media validator and assembler modules
- Updated test expectations to match implementation changes
- Fixed hardcoded path issues in test files for better portability
- Improved code style compliance with 88-character line limit

### Changed
- Enhanced debug logging and error handling in assembler module
- Improved test reliability with proper mock configurations

### Technical
- All quality gates now pass: Ruff, MyPy, Bandit, Vulture, Safety, pytest
- GitHub Actions CI pipeline fully functional
- Enhanced type safety and code maintainability

## [0.1.0] - Initial Release

### Added
- Initial open source release
- Complete AI video production pipeline for e-commerce products
- Amazon product scraper with configurable search parameters
- Multi-provider AI service support (OpenRouter, Google Cloud, OpenAI)
- Professional video assembly with FFmpeg
- Audio-synchronized subtitle generation
- Background music integration
- Batch processing capabilities
- Performance monitoring and optimization framework
- Comprehensive test suite with 280+ test cases
- Modular, extensible architecture supporting future platforms

### Technical Features
- **Pipeline Processing**: 6-step modular pipeline with parallel execution
- **Multi-Provider Support**: Fallback mechanisms for reliability
- **Configuration Management**: 100+ customizable parameters via YAML
- **Output Management**: Clean, product-centric directory structure
- **Code Quality**: Comprehensive linting, type checking, and security scanning