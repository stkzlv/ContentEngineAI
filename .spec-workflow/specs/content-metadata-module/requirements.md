# Requirements Document: Content Metadata Module

## Introduction

This spec defines the complete requirements for the ContentEngineAI Content Metadata Module, which generates platform-optimized metadata (titles, descriptions, hashtags, keywords) for YouTube, TikTok, and Instagram. The module supports both unified metadata (single set for all platforms) and optimized metadata (platform-tailored content with SEO optimization).

## Alignment with Product Vision

The Content Metadata Module directly supports the product principles defined in product.md:

- **Automation Over Manual Intervention**: LLM-powered metadata generation requires no manual copywriting
- **Modular Flexibility**: Factory pattern enables adding new platforms without changing core logic
- **Fail Gracefully**: Validation catches issues before publishing, fallback to unified mode if optimization fails
- **Performance at Scale**: Async LLM calls with caching handle batch metadata generation efficiently

## Requirements

### Section 1: Platform-Specific Metadata Generation

#### Requirement 1: Multi-Platform Metadata Architecture

**User Story:** As a content creator, I want platform-specific metadata generated automatically, so that my videos perform optimally on each social media platform.

##### Acceptance Criteria

1. WHEN generating metadata THEN system SHALL support YouTube, TikTok, and Instagram platforms
2. WHEN platform is YouTube THEN system SHALL generate: title (≤100 chars), description (≤5000 chars), 3-5 hashtags, SEO keywords
3. WHEN platform is TikTok THEN system SHALL generate: caption (100-300 chars optimal, ≤2200 max), 3-5 niche hashtags
4. WHEN platform is Instagram THEN system SHALL generate: caption (short 3-5 words OR SEO 100-200 chars), 15-30 hashtags
5. WHEN generating for any platform THEN system SHALL include `#ad` hashtag for FTC compliance

#### Requirement 2: YouTube Shorts Optimization

**User Story:** As a YouTube content creator, I want Shorts-optimized metadata, so that my videos rank well in search and attract viewers.

##### Acceptance Criteria

1. WHEN generating YouTube metadata THEN system SHALL create titles between 50-60 characters for SEO optimization
2. IF video is vertical (9:16) THEN system SHALL include `#Shorts` hashtag automatically
3. WHEN generating description THEN system SHALL include SEO keywords within first 200 characters
4. WHEN generating hashtags THEN system SHALL produce 3-5 relevant tags including product category

#### Requirement 3: TikTok SEO Optimization

**User Story:** As a TikTok content creator, I want SEO-focused captions, so that my videos appear in search results.

##### Acceptance Criteria

1. WHEN generating TikTok caption THEN system SHALL optimize for 100-300 characters (engagement sweet spot)
2. WHEN selecting hashtags THEN system SHALL prefer niche-specific tags over generic ones
3. WHEN generating hashtags THEN system SHALL avoid generic tags like `#fyp`, `#foryoupage`, `#viral`
4. WHEN generating content THEN system SHALL use exact search phrases users might search for

#### Requirement 4: Instagram Reels Optimization

**User Story:** As an Instagram content creator, I want Reels-optimized captions with extensive hashtag usage.

##### Acceptance Criteria

1. WHEN generating Instagram caption THEN system SHALL support two styles: short (3-5 words) or SEO (100-200 chars)
2. WHEN generating hashtags THEN system SHALL produce 15-30 relevant hashtags
3. WHEN emoji_enabled is true THEN system SHALL include relevant emojis in caption
4. WHEN generating content THEN system SHALL optimize for Instagram Reels discoverability

#### Requirement 5: Character Limit Enforcement

**User Story:** As a content creator, I want automatic character limit enforcement, so that my content never gets rejected by platforms.

##### Acceptance Criteria

1. WHEN metadata exceeds platform limits THEN system SHALL truncate with ellipsis and log warning
2. WHEN validating YouTube THEN system SHALL enforce: title ≤100 chars, description ≤5000 chars
3. WHEN validating TikTok THEN system SHALL enforce: caption ≤2200 chars
4. WHEN validating Instagram THEN system SHALL enforce: caption ≤2200 chars, hashtags ≤30
5. WHEN character counts are tracked THEN system SHALL report counts in metadata output

#### Requirement 6: LLM-Powered Generation

**User Story:** As a developer, I want LLM-based metadata generation, so that content is unique and contextually relevant.

##### Acceptance Criteria

1. WHEN generating metadata THEN system SHALL use OpenRouter API with configurable model selection
2. WHEN calling LLM THEN system SHALL use platform-specific prompt templates
3. WHEN LLM call fails THEN system SHALL retry with exponential backoff (max 3 retries)
4. WHEN generating content THEN system SHALL inject product data (title, description, URL) into prompts
5. WHEN API key is missing THEN system SHALL fail with clear error message

#### Requirement 7: Validation and Quality Assurance

**User Story:** As a content creator, I want metadata validated before use, so that I know it meets platform requirements.

##### Acceptance Criteria

1. WHEN metadata is generated THEN system SHALL validate against platform-specific rules
2. IF validation fails THEN system SHALL return validation status and detailed messages
3. WHEN validating THEN system SHALL check: character limits, hashtag counts, required hashtags (#ad)
4. WHEN validation status is "error" THEN system SHALL include specific violation details
5. WHEN validation succeeds THEN system SHALL set status to "valid" or "warning" if minor issues exist

### Section 2: Unified vs Optimized Mode

#### Requirement 8: Dual Metadata Modes

**User Story:** As a content creator, I want to choose between unified and optimized metadata modes, so that I can balance consistency vs platform optimization.

##### Acceptance Criteria

1. WHEN unified mode is enabled THEN system SHALL generate single metadata set for all platforms
2. WHEN optimized mode is enabled THEN system SHALL generate platform-specific metadata for each target
3. WHEN no mode is specified THEN system SHALL default to unified mode
4. WHEN switching modes THEN system SHALL preserve product data and regenerate metadata accordingly

### Section 3: Configuration and CLI

#### Requirement 9: Platform Configuration

**User Story:** As a developer, I want configurable platform settings, so that I can tune metadata generation per platform.

##### Acceptance Criteria

1. WHEN loading configuration THEN system SHALL read from config/video.yaml under `platform_metadata` section
2. WHEN platform is disabled THEN system SHALL skip metadata generation for that platform
3. WHEN target_platform is "multi" THEN system SHALL generate metadata for all enabled platforms
4. WHEN target_platform is specific THEN system SHALL generate only for that platform
5. WHEN configuration is invalid THEN system SHALL fail with validation errors listing issues

#### Requirement 10: CLI Integration

**User Story:** As a user, I want CLI control over metadata generation, so that I can customize behavior per run.

##### Acceptance Criteria

1. WHEN `--platform-metadata` flag is set THEN system SHALL enable optimized metadata generation
2. WHEN `--platforms` is specified THEN system SHALL generate metadata for listed platforms only
3. WHEN `--unified-metadata` flag is set THEN system SHALL use unified mode
4. WHEN running video producer THEN system SHALL integrate metadata generation into pipeline

## Non-Functional Requirements

### Code Architecture

- **Strategy Pattern**: BasePlatformMetadataGenerator with platform-specific implementations
- **Factory Pattern**: PlatformMetadataFactory for creating generators by platform
- **Single Responsibility**: Separate modules for models, generators, utilities, prompts
- **Dependency Injection**: Generators receive configuration, not global state

### Performance

- **Async LLM Calls**: All API calls use aiohttp for non-blocking I/O
- **Batch Generation**: Support generating metadata for multiple products efficiently
- **Template Caching**: Load prompt templates once and reuse across products
- **Graceful Degradation**: Continue with available platforms if one fails

### Security

- **API Key Protection**: Never log or expose API keys in output
- **Credential Validation**: Verify API keys present before making calls
- **Input Sanitization**: Sanitize product data before injecting into prompts

### Reliability

- **Retry Logic**: Exponential backoff for transient LLM API failures
- **Validation**: Catch and report issues before metadata is used
- **Fallback**: Use unified mode if platform-specific generation fails
- **Error Messages**: Clear, actionable error messages for troubleshooting

### Usability

- **Progress Logging**: Log metadata generation progress with platform names
- **Validation Feedback**: Provide detailed validation messages
- **Debug Mode**: Enhanced logging when --debug flag is set
- **Output Format**: JSON-serializable metadata for storage and publishing
