# Requirements Document

## Introduction

This specification ensures ContentEngineAI adheres to all requirements documented in REQUIREMENTS.md. The requirements cover three critical areas: **configuration system architecture**, **scraper capabilities**, and **video production features**. This compliance specification will validate, test, and document that all stated requirements are properly implemented and maintained across the codebase.

**Purpose**: Establish a comprehensive testing and validation framework that ensures the system meets all documented requirements for configuration management, multi-platform scraping, and advanced video production features.

**Value to Users**: Guarantees system reliability, prevents regression, and ensures all advertised features work as specified, reducing bugs and improving user confidence in the platform.

## Alignment with Product Vision

This specification directly supports multiple product principles and objectives from product.md:

- **Automation Over Manual Intervention**: Validates that the three-tier configuration system (CLI > env > YAML) provides intelligent defaults while allowing runtime customization
- **Quality Through Intelligence**: Ensures content-aware subtitle positioning, ASS effects system, and AI service integrations work correctly
- **Modular Flexibility**: Verifies that each component (configuration, scraping, video production) is independently configurable as documented
- **Fail Gracefully**: Tests multi-provider fallbacks, error handling, and graceful degradation scenarios
- **System Reliability >95%**: Comprehensive requirements testing contributes to the >95% success rate target for end-to-end pipeline execution

## Requirements

### Requirement 1: Three-Tier Configuration System

**User Story:** As a developer or power user, I want a predictable configuration precedence system (CLI > env > YAML), so that I can override settings at runtime without modifying files and keep secrets secure in environment variables.

#### Acceptance Criteria

1. WHEN a CLI argument is provided THEN the system SHALL use that value regardless of environment variables or YAML configuration
2. WHEN an environment variable is set AND no CLI argument provided THEN the system SHALL use the environment variable value over YAML defaults
3. WHEN neither CLI argument nor environment variable exists THEN the system SHALL fall back to YAML configuration values
4. IF API keys or credentials are present in configuration THEN they SHALL only be stored in `.env` file (never in YAML files)
5. WHEN `.env` file exists THEN it SHALL be loaded at runtime and values SHALL be injected into configuration system
6. IF `.env.example` template exists THEN it SHALL provide examples for all required environment variables
7. WHEN YAML files reference secrets THEN they SHALL use `api_key_env_var` field pattern to reference environment variables

### Requirement 2: Multi-Platform Scraper Architecture

**User Story:** As a content creator, I want to scrape product data from multiple e-commerce platforms, so that I can generate videos for products from Amazon, eBay, Walmart, and other sources.

#### Acceptance Criteria

1. WHEN platform-specific scraper is implemented THEN it SHALL extend `BaseScraper` abstract class
2. IF scraper supports direct product ID lookup THEN it SHALL validate product ID format against platform standards
3. WHEN scraper extracts product data THEN it SHALL include: title, price, description, ID, ratings, review count at minimum
4. IF product lacks essential data THEN scraper SHALL skip that product and continue processing
5. WHEN multiple ASINs (or product IDs) are provided THEN scraper SHALL handle each individually (not in single search query)
6. IF keyword search is used THEN scraper SHALL support filters for: price range, rating, shipping options, brand names
7. WHEN scraper downloads media THEN it SHALL filter out low-quality images and invalid file types
8. IF stealth techniques are required THEN scraper SHALL implement human-like interactions to evade detection
9. WHEN scraper encounters failure THEN it SHALL handle gracefully without halting entire processing

### Requirement 3: Product Media Discovery and Storage

**User Story:** As a video producer, I want high-resolution product images and videos automatically downloaded and organized, so that the video pipeline has quality visual assets without manual intervention.

#### Acceptance Criteria

1. WHEN product scraping completes THEN all media SHALL be stored in `outputs/<product_id>/` directory structure
2. IF product has images THEN scraper SHALL download high-resolution versions preferentially
3. WHEN product has videos THEN scraper SHALL include video URLs in product data and download if configured
4. IF image quality validation fails THEN that image SHALL be excluded from downloaded assets
5. WHEN directory structure is configurable THEN users SHALL be able to customize output path patterns
6. IF cleanup function is invoked THEN it SHALL remove unexpected files/directories not matching expected structure

### Requirement 4: Dynamic Video Assembly

**User Story:** As a marketer creating promotional videos, I want video duration to automatically match voiceover length with properly timed image transitions, so that videos are professional and synchronized without manual editing.

#### Acceptance Criteria

1. WHEN voiceover audio duration is determined THEN video duration SHALL match voiceover length exactly
2. IF voiceover is 30 seconds THEN image display duration SHALL be 2-3 seconds each (configurable)
3. WHEN calculating image count THEN system SHALL determine count based on: `total_duration / per_image_duration`
4. IF insufficient images exist for duration THEN system SHALL reuse images to match voiceover length
5. WHEN transitions are applied THEN they SHALL create smooth visual flow between images

### Requirement 5: Unified Subtitle Positioning System

**User Story:** As a video creator, I want flexible subtitle positioning with content-aware intelligence, so that subtitles never overlap product images and maintain consistent spacing.

#### Acceptance Criteria

1. WHEN subtitle positioning is configured THEN system SHALL support anchor options: `top`, `center`, `bottom`, `above_content`, `below_content`
2. IF content_aware mode is enabled (`content_aware=true`) THEN subtitles SHALL dynamically adjust position based on visual content boundaries
3. WHEN content_aware mode is disabled (`content_aware=false`) THEN subtitles SHALL use fixed positioning with anchor + margin
4. IF margin is configured THEN it SHALL be specified as fraction of frame height (0.0-0.5 range)
5. WHEN subtitle text is rendered THEN width SHALL NOT exceed image width (text constraints enforced)
6. IF content-aware positioning detects overlap THEN subtitle SHALL be repositioned to avoid visual content
7. WHEN spacing is applied THEN system SHALL maintain consistent spacing between content and subtitles

### Requirement 6: Two-Part Subtitle System

**User Story:** As a marketer promoting products, I want to display a persistent product link at the top and timed voiceover subtitles at the bottom, so that viewers always see the purchase link while following the narration.

#### Acceptance Criteria

1. WHEN two-part mode is enabled in profile THEN system SHALL display two independent subtitle lines simultaneously
2. IF upper line is configured THEN it SHALL display shortened product URL from `data.json` by default
3. WHEN upper line data source is customized THEN system SHALL support configurable field selection (e.g., `product_url`, `product_link`, custom field)
4. IF upper line uses `above_content` anchor THEN it SHALL be positioned above image with content-aware positioning
5. WHEN upper line is rendered THEN it SHALL remain visible throughout entire video (static, not timed to voiceover)
6. IF lower line is enabled THEN it SHALL display standard timed subtitles synchronized to voiceover audio
7. WHEN lower line uses `below_content` anchor THEN it SHALL be positioned below image with content-aware positioning
8. IF lower line uses STT-based timing THEN it SHALL synchronize perfectly with voiceover word-level timestamps
9. WHEN both lines are styled THEN system SHALL support independent styling for upper and lower lines
10. IF profile configuration specifies margins THEN each line SHALL have separate margin/positioning control
11. WHEN content-aware positioning is active THEN both lines SHALL adjust position based on visual content boundaries
12. IF two-part mode is disabled THEN system SHALL fall back to single-line subtitle mode (backward compatibility)

### Requirement 7: Profile-Specific Visual Settings

**User Story:** As a video producer managing multiple campaigns, I want to configure visual settings per video profile, so that I can create different styles for different product categories without global configuration changes.

#### Acceptance Criteria

1. WHEN video profile is defined THEN all visual settings SHALL be configurable per profile
2. IF profile specifies image positioning THEN it SHALL override global defaults: width, position, aspect ratio
3. WHEN profile defines subtitle settings THEN it SHALL override global: positioning, styling, fonts, colors, effects
4. IF profile merging occurs THEN profile settings SHALL take precedence over global configuration
5. WHEN legacy code uses global configuration THEN system SHALL maintain backward compatibility
6. IF unified subtitle positioning is used THEN profiles SHALL support anchor-based layout configuration

### Requirement 8: Style Preset System with Font & Color Management

**User Story:** As a video creator, I want pre-defined visual presets including a random option, so that I can quickly apply professional styles or get creative variety without manual design work.

#### Acceptance Criteria

1. WHEN style preset is selected THEN system SHALL support 5 presets: `minimal`, `modern`, `bold`, `animated`, `random`
2. IF `minimal` preset is chosen THEN video SHALL use clean, simple styling with no effects
3. WHEN `modern` preset is applied THEN video SHALL have contemporary look with subtle karaoke effects only
4. IF `bold` preset is selected THEN video SHALL use high contrast, bold styling with fade effects only
5. WHEN `animated` preset is chosen THEN video SHALL include movement effects only (no other effects)
6. IF `random` preset is selected THEN system SHALL randomize: font selection, color pairs, and exactly one animation effect
7. WHEN font randomization occurs THEN selection SHALL be from curated collection with deterministic seeding per video
8. IF color randomization occurs THEN coordinated text/outline color combinations SHALL ensure proper contrast
9. WHEN preset is applied THEN system SHALL be fully compatible with ASS, SRT, and FFmpeg rendering

### Requirement 9: ASS Effects System

**User Story:** As a video editor, I want consistent, properly-formatted ASS subtitle effects per video, so that animations enhance engagement without visual clutter or rendering errors.

#### Acceptance Criteria

1. WHEN effect is selected for video THEN exactly 1 effect SHALL be applied per video (not per subtitle segment)
2. IF ASS override codes are used THEN all codes SHALL be enclosed in curly braces `{}` to prevent literal text display
3. WHEN effect variety is needed THEN system SHALL support: scale_pulse, rotation_bounce, glow, typewriter, karaoke, fade, movement effects
4. IF `random` preset is active THEN system SHALL select exactly 1 effect from all available using product ID seeding
5. WHEN `minimal` preset is used THEN no effects SHALL be applied
6. IF `modern` preset is chosen THEN karaoke effect only SHALL be applied
7. WHEN `bold` preset is selected THEN fade effect only SHALL be applied
8. IF `animated` preset is chosen THEN movement effect only SHALL be applied
9. WHEN karaoke timing is implemented THEN word-by-word highlighting SHALL use proper `\k` tag formatting in centiseconds
10. IF ASS effects are rendered THEN they SHALL render correctly through FFmpeg's libass library
11. WHEN visual consistency is required THEN animation style SHALL be coherent throughout individual videos

### Requirement 10: AI Service Integration

**User Story:** As a pipeline operator, I want automatic AI model selection with fallbacks, so that the system continues operating even when preferred services are unavailable.

#### Acceptance Criteria

1. WHEN OpenRouter API is used THEN system SHALL auto-select models from available options
2. IF free models are available THEN system SHALL prioritize them with configuration fallback
3. WHEN TTS provider is selected THEN system SHALL prioritize Google Cloud Chirp 3 HD voices
4. IF voice selection occurs THEN skipped voices SHALL be hidden in logs even in debug mode
5. WHEN primary AI service fails THEN system SHALL fall back to secondary provider automatically

### Requirement 11: Global Debug Mode and Error Handling

**User Story:** As a developer debugging issues, I want comprehensive debug mode across all components with clear error messages, so that I can quickly identify and resolve problems.

#### Acceptance Criteria

1. WHEN `--debug` flag is provided THEN all components SHALL enable verbose logging
2. IF configuration is invalid at startup THEN system SHALL validate and provide clear error messages
3. WHEN environment variable is missing THEN error message SHALL specify which variable and where to configure it
4. IF individual component fails THEN system SHALL continue processing other components (graceful degradation)
5. WHEN service is unavailable THEN system SHALL attempt fallback providers before failing
6. IF unexpected error occurs THEN system SHALL log full stack trace in debug mode

### Requirement 12: Configuration Validation at Startup

**User Story:** As a system operator, I want configuration validated at startup with clear error messages, so that I know immediately if settings are incorrect before pipeline execution.

#### Acceptance Criteria

1. WHEN system starts THEN configuration SHALL be validated using Pydantic models
2. IF required field is missing THEN startup SHALL fail with error specifying missing field name and expected type
3. WHEN value type is incorrect THEN validation SHALL fail with clear type mismatch error
4. IF environment variable reference is invalid THEN startup SHALL fail indicating which env var is missing
5. WHEN YAML syntax is malformed THEN parser SHALL provide line number and syntax error details
6. IF CLI argument conflicts with valid values THEN system SHALL show allowed options and exit

## Non-Functional Requirements

### Code Architecture and Modularity
- **Single Responsibility Principle**: Configuration loading, scraper implementations, and video assembly SHALL be in separate modules
- **Modular Design**: Each requirement category (config, scraper, video) SHALL have isolated, reusable components
- **Dependency Management**: Configuration system SHALL NOT depend on scraper or video modules; scraper SHALL NOT depend on video production
- **Clear Interfaces**: `BaseScraper`, configuration loaders, and video profile systems SHALL define clean contracts between components

### Performance
- **Configuration Loading**: YAML parsing and environment variable injection SHALL complete in <100ms
- **Validation Speed**: Pydantic validation at startup SHALL complete in <500ms even with complex configurations
- **Test Execution**: Full requirements compliance test suite SHALL execute in <5 minutes

### Security
- **Secret Isolation**: API keys and credentials SHALL only exist in `.env` file (enforced by security tests)
- **Environment Variable Access**: System SHALL never log or print API key values, even in debug mode
- **File Permissions**: `.env` file SHALL have restricted read permissions (owner-only on Unix systems)

### Reliability
- **Test Coverage**: All requirements SHALL have corresponding unit and integration tests (targeting >90% coverage)
- **Backward Compatibility**: Configuration system changes SHALL not break existing YAML configurations or CLI arguments
- **Graceful Degradation**: Missing optional features SHALL not prevent pipeline execution (e.g., missing style presets fall back to minimal)

### Usability
- **Clear Documentation**: Each requirement SHALL be documented in user-facing CONFIGURATION.md
- **Error Messages**: Validation failures SHALL include actionable guidance for fixing configuration issues
- **Examples**: `.env.example` and YAML comments SHALL provide clear examples for all configurable values
