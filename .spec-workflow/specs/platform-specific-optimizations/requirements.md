# Requirements Document

## Introduction

This feature enables ContentEngineAI to generate platform-optimized metadata (titles, descriptions, captions, hashtags) for YouTube, TikTok, and Instagram. Currently, the system generates unified descriptions suitable for all platforms, but research shows each platform has distinct optimization strategies that significantly impact discoverability and engagement.

**Purpose:** Maximize video reach and engagement by applying platform-specific SEO best practices to metadata generation.

**Value to Users:** Increase video views, improve search rankings, and drive higher engagement rates by delivering metadata optimized for each platform's unique algorithm and audience expectations.

## Alignment with Product Vision

This feature directly supports ContentEngineAI's business objective to "Maintain Quality Standards: Deliver consistent, platform-optimized content that meets social media best practices" (product.md).

**Key Alignments:**
- **Scale Content Production**: Enable users to target multiple platforms with optimized metadata without manual customization
- **Accelerate Time-to-Market**: Automatically generate platform-specific metadata during video production pipeline
- **Democratize Video Marketing**: Abstract platform-specific complexity, making optimization accessible without marketing expertise
- **Maintain Quality Standards**: Apply current 2025 SEO best practices per platform automatically

## Requirements

### Requirement 1: Multi-Platform Metadata Generation

**User Story:** As an e-commerce marketer, I want to generate platform-optimized titles, descriptions, and hashtags for YouTube, TikTok, and Instagram, so that my videos rank higher in search and reach more potential customers on each platform.

#### Acceptance Criteria

1. WHEN user specifies target platform(s) THEN system SHALL generate metadata optimized for specified platform(s)
2. WHEN user selects YouTube THEN system SHALL generate title (50-60 chars), description (up to 5000 chars), and 3-5 hashtags with SEO keyword placement
3. WHEN user selects TikTok THEN system SHALL generate SEO-focused caption (100-300 chars optimal) and 3-5 niche-specific hashtags
4. WHEN user selects Instagram THEN system SHALL generate caption (3-5 words OR 100-200 chars SEO style) and 15-30 relevant hashtags
5. WHEN user selects multi-platform mode THEN system SHALL generate optimized metadata for all three platforms simultaneously
6. WHEN metadata generation completes THEN system SHALL save platform-specific files (metadata_youtube.json, metadata_tiktok.json, metadata_instagram.json)

### Requirement 2: Platform-Specific SEO Optimization

**User Story:** As a content creator, I want my video metadata to follow platform-specific SEO best practices, so that my content appears in relevant searches and recommendations on each platform.

#### Acceptance Criteria

1. WHEN generating YouTube metadata THEN system SHALL place primary keywords at beginning of title and first 150 characters of description
2. WHEN generating YouTube metadata THEN system SHALL include `#Shorts` hashtag for vertical videos under 60 seconds
3. WHEN generating TikTok captions THEN system SHALL use exact search phrases (e.g., "morning skincare routine for sensitive skin" vs "This saved my skin")
4. WHEN generating TikTok hashtags THEN system SHALL avoid generic tags (#foryoupage, #fyp, #viral) and prioritize niche community hashtags
5. WHEN generating Instagram Reels metadata THEN system SHALL include 15-30 hashtags in caption (not comments) with mix of trending, niche, and evergreen tags
6. WHEN generating Instagram captions THEN system SHALL support both ultra-short (3-5 words) and SEO-descriptive (100-200 chars) styles
7. WHEN validation detects missing required elements (e.g., #ad disclosure) THEN system SHALL warn user and add required elements

### Requirement 3: Character Limit Enforcement

**User Story:** As a video producer, I want the system to respect platform-specific character limits, so that my metadata displays correctly without truncation on each platform.

#### Acceptance Criteria

1. WHEN generating YouTube title THEN system SHALL enforce 50-60 character optimal length
2. WHEN generating YouTube description THEN system SHALL support up to 5000 characters with first 150 characters prioritized
3. WHEN generating TikTok caption THEN system SHALL optimize for 100-300 characters while supporting up to 2200 characters
4. WHEN generating Instagram caption THEN system SHALL support 3-5 words (short style) or 100-200 characters (SEO style)
5. WHEN metadata exceeds platform limits THEN system SHALL truncate gracefully and log warning
6. WHEN metadata generation completes THEN system SHALL include character_counts field in output for validation

### Requirement 4: Configurable Platform Targeting

**User Story:** As a batch processing user, I want to configure target platforms per video profile or globally, so that I can automate platform-specific metadata generation across hundreds of products.

#### Acceptance Criteria

1. WHEN user sets target_platform in YAML config THEN system SHALL use that platform for all videos unless overridden
2. WHEN user provides --target-platform CLI argument THEN system SHALL override YAML configuration
3. WHEN user configures platform per video profile THEN system SHALL apply profile-specific platform targeting
4. IF target_platform is "multi" THEN system SHALL generate metadata for all three platforms
5. WHEN batch processing with random profiles THEN system SHALL apply consistent platform targeting per product
6. IF target platform not specified THEN system SHALL default to multi-platform mode (backward compatibility)

### Requirement 5: LLM-Powered Generation with Platform-Specific Prompts

**User Story:** As a system administrator, I want platform-specific metadata generated using optimized LLM prompts, so that output quality matches platform best practices and audience expectations.

#### Acceptance Criteria

1. WHEN generating YouTube metadata THEN system SHALL use youtube_metadata.md prompt template
2. WHEN generating TikTok metadata THEN system SHALL use tiktok_caption.md prompt template
3. WHEN generating Instagram metadata THEN system SHALL use instagram_caption.md prompt template
4. WHEN prompt template loads THEN system SHALL inject product data (title, description, URL)
5. WHEN LLM generates metadata THEN system SHALL validate completeness (hashtag count, character limits, required elements)
6. IF validation fails THEN system SHALL retry with fallback model up to 2 attempts
7. WHEN all models fail THEN system SHALL log error and fall back to unified metadata generator (backward compatibility)

### Requirement 6: Unified Voiceover Script

**User Story:** As a content creator producing videos for multiple platforms, I want a single reusable voiceover script, so that I can upload the same video file to YouTube, TikTok, and Instagram without re-recording audio.

#### Acceptance Criteria

1. WHEN generating voiceover script THEN system SHALL create platform-agnostic content (no platform-specific jargon)
2. WHEN script includes call-to-action THEN system SHALL use generic phrases ("link in bio," "follow for more") suitable for all platforms
3. WHEN video production completes THEN system SHALL reuse same voiceover audio for all platform exports
4. IF platform-specific requirements exist THEN system SHALL handle them via metadata only, not voiceover modifications

### Requirement 7: Validation and Quality Assurance

**User Story:** As a quality-conscious marketer, I want the system to validate generated metadata against platform requirements, so that I catch errors before publishing.

#### Acceptance Criteria

1. WHEN metadata generation completes THEN system SHALL validate character limits per platform
2. WHEN hashtag count exceeds platform maximum THEN system SHALL truncate to recommended range and log warning
3. WHEN required hashtags missing (e.g., #ad for sponsored content) THEN system SHALL add required tags
4. WHEN validation detects issues THEN system SHALL log detailed warnings with specific platform guidelines
5. WHEN metadata passes validation THEN system SHALL include validation_status field in output JSON
6. IF critical validation failures occur THEN system SHALL fallback to unified metadata generator

## Non-Functional Requirements

### Code Architecture and Modularity
- **Single Responsibility Principle**: Separate metadata generator per platform (YouTubeMetadataGenerator, TikTokMetadataGenerator, InstagramMetadataGenerator)
- **Modular Design**: Abstract BasePlatformMetadataGenerator interface for extensibility
- **Dependency Management**: Metadata generation should not depend on video assembly logic
- **Clear Interfaces**: Unified PlatformMetadata dataclass for all platforms

### Performance
- **Generation Speed**: Metadata generation SHALL complete in <5 seconds per platform
- **Parallel Processing**: Multi-platform mode SHALL generate all platforms concurrently using asyncio.gather()
- **Caching**: LLM responses SHALL be cached to avoid regenerating identical metadata
- **Batch Efficiency**: Batch processing SHALL reuse HTTP sessions and connection pools across products

### Security
- **API Key Protection**: Platform-specific API keys (if future platforms require) SHALL be stored in .env only
- **No Hardcoded Secrets**: LLM prompts and configuration SHALL not contain sensitive information
- **Input Sanitization**: Product data injected into prompts SHALL be sanitized to prevent prompt injection attacks

### Reliability
- **Fallback Strategy**: IF platform-specific generation fails THEN system SHALL fall back to unified metadata generator
- **Retry Logic**: LLM requests SHALL retry up to 2 times with exponential backoff
- **Error Handling**: Metadata generation failures SHALL not block video production pipeline
- **Validation**: All generated metadata SHALL be validated before storage

### Usability
- **CLI Simplicity**: Single `--target-platform youtube` argument SHALL override all configuration
- **Clear Defaults**: System SHALL default to multi-platform mode for backward compatibility
- **Documentation**: Each platform's character limits and best practices SHALL be documented in YAML comments
- **Error Messages**: Validation failures SHALL include specific guidance (e.g., "YouTube title exceeds 60 chars, recommended max is 60")
