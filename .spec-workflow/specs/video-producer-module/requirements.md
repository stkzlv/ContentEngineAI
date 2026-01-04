# Requirements Document: Video Producer Module

## Introduction

This spec defines the complete requirements for the ContentEngineAI Video Producer Module, which transforms scraped product data into polished videos with voiceovers, subtitles, background music, and visual effects. The module supports single product processing, batch operations with profile randomization, and multiple video assembly strategies.

## Alignment with Product Vision

The Video Producer Module directly supports the product principles defined in product.md:

- **Automation Over Manual Intervention**: Batch processing with automatic product discovery eliminates manual file selection
- **Modular Flexibility**: Profile system enables per-product customization without code changes
- **Fail Gracefully**: Graceful degradation ensures batch completion despite individual failures
- **Performance at Scale**: Efficient batch processing with profile randomization handles diverse product catalogs

## Requirements

### Section 1: Video Assembly

#### Requirement 1: Video Assembly Modes

**User Story:** As a video producer, I want multiple strategies for combining product videos, so that I can optimize video content based on available media.

##### Acceptance Criteria

1. WHEN assembly mode is "sequential" THEN the system SHALL concatenate all product videos in order with configurable crossfade transitions
2. IF assembly mode is "single_best" THEN the system SHALL select and use only the longest product video
3. WHEN assembly mode is "mixed_media" THEN the system SHALL interleave videos and images for visual variety
4. IF assembly mode is "video_first_fallback" THEN the system SHALL prioritize videos when available and gracefully fall back to images when needed
5. WHEN a single video exists THEN the system SHALL loop or trim it to match target duration
6. IF no product videos exist THEN the system SHALL fall back to image-only assembly

#### Requirement 2: Aspect Ratio Handling

**User Story:** As a video producer, I want consistent aspect ratio handling across mixed media, so that output videos have uniform dimensions.

##### Acceptance Criteria

1. WHEN aspect mode is "letterbox" THEN the system SHALL add black bars to preserve original aspect ratio
2. IF aspect mode is "crop_to_fit" THEN the system SHALL center-crop to fill target dimensions
3. WHEN aspect mode is "smart_scale" THEN the system SHALL use intelligent scaling with minimal cropping
4. IF source aspect differs from target THEN the system SHALL apply the configured mode consistently
5. WHEN processing mixed media THEN the system SHALL apply the same aspect handling to all clips

#### Requirement 3: Audio Normalization

**User Story:** As a video producer, I want control over original video audio, so that I can either remove it or mix it with background music.

##### Acceptance Criteria

1. WHEN audio handling is "remove" THEN the system SHALL strip all audio from product videos before assembly
2. IF audio handling is "mixed" THEN the system SHALL reduce original audio volume to configured level (default -30dB)
3. WHEN mixing audio THEN the system SHALL combine original audio with background music track
4. IF original audio is missing THEN the system SHALL proceed with background music only
5. WHEN volume is configured THEN the system SHALL accept values in dB range (-60 to 0)

#### Requirement 4: Format Normalization

**User Story:** As a video producer, I want all media normalized to consistent format, so that FFmpeg concatenation works without re-encoding issues.

##### Acceptance Criteria

1. WHEN normalizing video format THEN the system SHALL convert to H.264 codec
2. IF frame rate differs THEN the system SHALL normalize to 30 fps
3. WHEN pixel format varies THEN the system SHALL convert to yuv420p
4. IF format normalization is disabled THEN the system SHALL skip re-encoding (may cause concat issues)
5. WHEN processing images THEN the system SHALL create video clips at target frame rate

#### Requirement 5: Duration Matching Algorithm

**User Story:** As a video producer, I want video duration to match voiceover length, so that audio and visual content align properly.

##### Acceptance Criteria

1. WHEN assembling video THEN the system SHALL target voiceover duration plus padding
2. IF video content is shorter THEN the system SHALL loop or extend with additional media
3. WHEN video content is longer THEN the system SHALL trim to target duration
4. IF duration difference is within tolerance (±1 second) THEN the system SHALL accept without adjustment
5. WHEN no voiceover exists THEN the system SHALL use configured default duration

#### Requirement 6: Video Transition System

**User Story:** As a video producer, I want smooth transitions between video clips, so that assembled videos look professional.

##### Acceptance Criteria

1. WHEN transition type is "crossfade" THEN the system SHALL blend adjacent clips with configurable duration
2. IF transition duration exceeds clip length THEN the system SHALL use shorter transition
3. WHEN multiple clips exist THEN the system SHALL apply transitions between each pair
4. IF transition is disabled THEN the system SHALL hard-cut between clips

### Section 2: Subtitle System

#### Requirement 7: Unified Anchor-Based Positioning

**User Story:** As a video producer, I want consistent subtitle positioning using anchor points, so that subtitles appear correctly across different video formats.

##### Acceptance Criteria

1. WHEN anchor is "top" THEN the system SHALL position subtitles at top of frame with configured margin
2. IF anchor is "center" THEN the system SHALL center subtitles vertically
3. WHEN anchor is "bottom" THEN the system SHALL position at bottom with margin
4. IF anchor is "above_content" THEN the system SHALL position above visual content area
5. WHEN anchor is "below_content" THEN the system SHALL position below visual content area
6. IF content-aware mode is enabled THEN the system SHALL avoid overlapping with product visuals

#### Requirement 8: Two-Part Subtitle System

**User Story:** As a video producer, I want separate styling for URL and voiceover text, so that each serves its purpose effectively.

##### Acceptance Criteria

1. WHEN two-part subtitles enabled THEN the system SHALL display upper line (URL) and lower line (voiceover)
2. IF upper line source is "affiliate_url" THEN the system SHALL display shortened affiliate link
3. WHEN prefix_replace is configured THEN the system SHALL replace URL prefix for cleaner display
4. IF full_duration mode is enabled THEN the system SHALL display upper line throughout video
5. WHEN styling differs THEN the system SHALL apply separate font, size, and color to each part

#### Requirement 9: ASS Effect System

**User Story:** As a video producer, I want animated subtitle effects, so that videos have dynamic visual appeal.

##### Acceptance Criteria

1. WHEN karaoke effect enabled THEN the system SHALL animate word-by-word reveal with timing
2. IF fade effect enabled THEN the system SHALL fade in/out subtitles with configurable duration
3. WHEN typewriter effect enabled THEN the system SHALL reveal characters progressively
4. IF glow effect enabled THEN the system SHALL add animated glow around text
5. WHEN bounce effect enabled THEN the system SHALL apply subtle rotation animation
6. IF pulse effect enabled THEN the system SHALL scale text rhythmically

#### Requirement 10: Style Presets

**User Story:** As a video producer, I want predefined style presets, so that I can quickly apply consistent subtitle styling.

##### Acceptance Criteria

1. WHEN preset is "minimal" THEN the system SHALL apply clean, simple styling
2. IF preset is "modern" THEN the system SHALL apply contemporary styling with subtle effects
3. WHEN preset is "bold" THEN the system SHALL apply high-contrast, impactful styling
4. IF preset is "animated" THEN the system SHALL enable full animation effects
5. WHEN preset is "random" THEN the system SHALL randomly select effects for variety
6. IF profile overrides preset THEN the system SHALL use profile-specific styling

### Section 3: Background Music (Freesound Integration)

#### Requirement 11: Async Music Search with Duration Matching

**User Story:** As a video producer, I want background music that matches video duration, so that audio transitions are seamless.

##### Acceptance Criteria

1. WHEN searching for music THEN the system SHALL use async HTTP client for concurrent API calls
2. IF target duration is specified THEN the system SHALL filter results to within ±30% of target
3. WHEN multiple results match THEN the system SHALL prefer closest duration match
4. IF no duration match found THEN the system SHALL return best available result
5. WHEN search completes THEN the system SHALL return sorted list by relevance score

#### Requirement 12: OAuth2 Authentication with Token Refresh

**User Story:** As a system administrator, I want automatic token refresh, so that long-running batch operations don't fail due to expired credentials.

##### Acceptance Criteria

1. WHEN OAuth2 token expires THEN the system SHALL automatically request new token using refresh token
2. IF refresh token is valid THEN the system SHALL update stored access token without user intervention
3. WHEN new tokens received THEN the system SHALL persist to .env file
4. IF refresh fails with 401/403 THEN the system SHALL fail fast without retry
5. WHEN token refresh succeeds THEN the system SHALL retry original request automatically

#### Requirement 13: High-Quality OAuth2 Downloads with API Key Fallback

**User Story:** As a video producer, I want highest quality audio downloads, so that background music sounds professional.

##### Acceptance Criteria

1. WHEN downloading music THEN the system SHALL attempt OAuth2 HQ download first
2. IF OAuth2 download fails THEN the system SHALL fall back to API key preview download
3. WHEN using API key fallback THEN the system SHALL prefer HQ-MP3 preview over LQ
4. IF both methods fail THEN the system SHALL log error and continue without music
5. WHEN download succeeds THEN the system SHALL validate audio file integrity

#### Requirement 14: Circuit Breaker for API Resilience

**User Story:** As a system operator, I want protection against API outages, so that batch operations continue despite Freesound unavailability.

##### Acceptance Criteria

1. WHEN consecutive API failures reach threshold (3) THEN the system SHALL open circuit breaker
2. IF circuit is open THEN the system SHALL fast-fail without making API calls
3. WHEN timeout period expires (30s) THEN the system SHALL enter half-open state
4. IF half-open request succeeds THEN the system SHALL close circuit and resume normal operation
5. WHEN circuit opens THEN the system SHALL log state transition for monitoring

#### Requirement 15: Local Fallback with Memory-Mapped I/O

**User Story:** As a video producer, I want local music fallback, so that videos can be produced even when Freesound is unavailable.

##### Acceptance Criteria

1. WHEN API search fails THEN the system SHALL fall back to local music library
2. IF local library exists THEN the system SHALL search by tags and duration
3. WHEN loading local files THEN the system SHALL use memory-mapped I/O for efficiency
4. IF no local match found THEN the system SHALL use any available local track
5. WHEN local fallback used THEN the system SHALL log source for attribution tracking

#### Requirement 16: Attribution Metadata Tracking

**User Story:** As a content creator, I want Creative Commons attribution tracked, so that I can comply with licensing requirements.

##### Acceptance Criteria

1. WHEN music is selected THEN the system SHALL capture: title, author, license, source URL
2. IF license requires attribution THEN the system SHALL include in video metadata
3. WHEN batch processing THEN the system SHALL aggregate attribution for all tracks
4. IF attribution data missing THEN the system SHALL log warning and use available info

### Section 4: Batch Processing

#### Requirement 17: Automatic Product Discovery

**User Story:** As a batch operator, I want automatic discovery of products to process, so that I don't need to manually specify each product.

##### Acceptance Criteria

1. WHEN --batch flag is set THEN the system SHALL scan outputs directory for product subdirectories
2. IF directory contains valid data.json THEN the system SHALL include in batch queue
3. WHEN scanning THEN the system SHALL skip global directories (cache, logs, reports)
4. IF data.json is invalid THEN the system SHALL skip with warning and continue
5. WHEN discovery completes THEN the system SHALL report count of valid products found

#### Requirement 18: Batch Execution with Profile Consistency

**User Story:** As a batch operator, I want consistent profile application, so that batch outputs follow predictable patterns.

##### Acceptance Criteria

1. WHEN --batch-profile is specified THEN the system SHALL use same profile for all products
2. IF profile not found THEN the system SHALL fail immediately with clear error
3. WHEN processing batch THEN the system SHALL validate profile exists before starting
4. IF profile has errors THEN the system SHALL report issues at batch start

#### Requirement 19: Profile Randomization

**User Story:** As a content creator, I want varied video styles across products, so that content library has visual diversity.

##### Acceptance Criteria

1. WHEN --random-profile is enabled THEN the system SHALL select profile randomly per product
2. IF --profile-pool is specified THEN the system SHALL limit selection to specified profiles
3. WHEN selecting profile THEN the system SHALL use product ID hash for deterministic selection
4. IF same product processed twice THEN the system SHALL select same profile (deterministic)
5. WHEN batch completes THEN the system SHALL report profile usage distribution

#### Requirement 20: Progress Tracking and Logging

**User Story:** As a batch operator, I want clear progress visibility, so that I know batch status at all times.

##### Acceptance Criteria

1. WHEN processing products THEN the system SHALL log in `[N/total]` format
2. IF a product succeeds THEN the system SHALL log: `[N/total] SUCCESS: product_id`
3. WHEN a product fails THEN the system SHALL log: `[N/total] FAILED: product_id - error`
4. IF a product is skipped THEN the system SHALL log: `[N/total] SKIPPED: product_id - reason`
5. WHEN batch completes THEN the system SHALL output summary with counts

#### Requirement 21: Error Handling and Resilience

**User Story:** As a batch operator, I want individual failures isolated, so that one bad product doesn't stop the entire batch.

##### Acceptance Criteria

1. WHEN a product fails THEN the system SHALL log error and continue with next product
2. IF --fail-fast is enabled THEN the system SHALL stop on first failure
3. WHEN fail-fast stops THEN the system SHALL report failed item and pending count
4. IF network error occurs THEN the system SHALL retry with exponential backoff
5. WHEN batch completes with failures THEN the system SHALL list all failed products

#### Requirement 22: Summary Reporting

**User Story:** As a batch operator, I want a comprehensive summary, so that I can assess overall batch results.

##### Acceptance Criteria

1. WHEN batch completes THEN the system SHALL output:
   - Total products attempted
   - Successful: count and IDs
   - Failed: count, IDs, and error messages
   - Skipped: count, IDs, and reasons
   - Duration: total time and per-product average
2. IF profile randomization used THEN the summary SHALL include profile distribution
3. WHEN --output-format=json THEN the system SHALL output machine-readable summary

### Section 5: Profile System

#### Requirement 23: Per-Profile Configuration Override

**User Story:** As a video producer, I want per-profile media selection, so that different video styles use appropriate content sources.

##### Acceptance Criteria

1. WHEN profile specifies use_scraped_images THEN the system SHALL use product images accordingly
2. IF profile specifies stock_image_count THEN the system SHALL fetch that many stock images
3. WHEN profile enables use_scraped_videos THEN the system SHALL include product videos
4. IF profile overrides video_assembly_mode THEN the system SHALL use profile-specific mode
5. WHEN profile overrides subtitle settings THEN the system SHALL apply profile-specific styling

#### Requirement 24: Profile Validation

**User Story:** As a developer, I want profile validation at load time, so that configuration errors are caught early.

##### Acceptance Criteria

1. WHEN loading profile THEN the system SHALL validate all required fields present
2. IF video_assembly_mode is invalid THEN the system SHALL fail with clear error
3. WHEN profile references fonts THEN the system SHALL verify font availability
4. IF stock counts are negative THEN the system SHALL reject with validation error
5. WHEN profile is valid THEN the system SHALL cache parsed configuration

### Section 6: AI Service Integration

#### Requirement 25: LLM Script Generation

**User Story:** As a video producer, I want AI-generated voiceover scripts, so that videos have engaging narration.

##### Acceptance Criteria

1. WHEN generating script THEN the system SHALL use product data (title, description, features)
2. IF target duration specified THEN the system SHALL generate script of appropriate length
3. WHEN script fails THEN the system SHALL retry with exponential backoff
4. IF all retries fail THEN the system SHALL use fallback template-based script

#### Requirement 26: Text-to-Speech Voiceover

**User Story:** As a video producer, I want natural-sounding voiceovers, so that videos have professional narration.

##### Acceptance Criteria

1. WHEN generating voiceover THEN the system SHALL use configured TTS provider
2. IF TTS fails THEN the system SHALL attempt fallback provider
3. WHEN voiceover succeeds THEN the system SHALL save audio and duration metadata
4. IF circuit breaker opens THEN the system SHALL skip voiceover gracefully

## Non-Functional Requirements

### Code Architecture

- **Single Responsibility**: Each component (strategy, builder, client) has one purpose
- **Strategy Pattern**: Video assembly modes implemented as interchangeable strategies
- **Circuit Breaker**: External API calls protected with circuit breaker pattern
- **Dependency Injection**: Components receive configuration, not global state

### Performance

- Batch processing SHALL handle 100+ products in a single run
- Individual video production SHALL complete in <2 minutes average
- Media downloads SHALL use async I/O for parallelization
- Profile selection SHALL be O(1) using hash-based lookup

### Security

- API credentials SHALL be stored in .env file only
- OAuth tokens SHALL be refreshed automatically before expiry
- Downloaded content SHALL be validated for file type
- Credentials SHALL not appear in logs (masked output)

### Reliability

- Network failures SHALL trigger retry with exponential backoff (max 3 retries)
- Partial failures SHALL not corrupt previously produced videos
- Circuit breaker SHALL prevent cascade failures
- Interrupted batches SHALL be resumable via step parameter

### Usability

- Error messages SHALL include actionable guidance
- Progress output SHALL use consistent `[N/total]` format
- Debug mode SHALL preserve intermediate artifacts for troubleshooting
- CLI SHALL provide 60+ configurable parameters for fine-tuning
