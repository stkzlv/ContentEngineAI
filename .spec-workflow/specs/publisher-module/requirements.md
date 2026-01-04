# Requirements Document: Publisher Module

## Introduction

This spec defines the complete requirements for the ContentEngineAI Publisher Module, which handles multi-platform video publishing through the Late.dev service, scheduling with recurring time slots, and post-publication cleanup. The module supports single video publishing, batch operations, automatic scheduling, and safe cleanup of published content.

## Alignment with Product Vision

The Publisher Module directly supports the product principles defined in product.md:

- **Automation Over Manual Intervention**: Auto-scheduling fills optimal time slots without manual selection
- **Modular Flexibility**: Provider-based architecture allows adding new publishing backends
- **Fail Gracefully**: Circuit breaker protection and graceful degradation ensure batch completion
- **Performance at Scale**: Batch publishing with concurrent uploads handles large video libraries

## Requirements

### Section 1: Late.dev Integration

#### Requirement 1: Late SDK Integration

**User Story:** As a developer, I want to use the official Late Python SDK for API interactions, so that I have reliable and maintained API access.

##### Acceptance Criteria

1. WHEN initializing the publisher THEN the system SHALL use the `late-sdk` Python package
2. IF API credentials are missing THEN the system SHALL fail with a clear error message
3. WHEN making API calls THEN the system SHALL handle rate limiting with exponential backoff
4. IF the SDK version is incompatible THEN the system SHALL log a warning with upgrade instructions

#### Requirement 2: Account Discovery

**User Story:** As a user, I want to automatically discover my connected social media accounts, so that I don't need to manually configure platform IDs.

##### Acceptance Criteria

1. WHEN the publisher initializes THEN the system SHALL fetch all connected accounts via Late API
2. IF accounts are cached THEN the system SHALL use cached data unless refresh is requested
3. WHEN an account is disconnected THEN the system SHALL detect and report the status
4. IF no accounts are connected THEN the system SHALL provide instructions for connecting accounts

#### Requirement 3: Media Upload Management

**User Story:** As a video producer, I want efficient media uploads that handle both small and large files appropriately.

##### Acceptance Criteria

1. WHEN video file size is ≤4MB THEN the system SHALL use direct base64 upload
2. IF video file size is >4MB THEN the system SHALL use Vercel Blob token upload
3. WHEN uploading large files THEN the system SHALL show progress indication
4. IF upload fails THEN the system SHALL retry with exponential backoff (max 3 retries)
5. WHEN file exceeds 500MB THEN the system SHALL reject with clear error message

#### Requirement 4: Platform-Specific Metadata

**User Story:** As a content creator, I want platform-optimized metadata automatically applied, so that my videos perform well on each platform.

##### Acceptance Criteria

1. WHEN publishing to YouTube THEN the system SHALL include: title (≤100 chars), description (≤5000 chars), tags, category, privacy
2. IF publishing to TikTok THEN the system SHALL include: title (≤150 chars), privacy, allow_comments, allow_duet, allow_stitch
3. WHEN publishing to Instagram THEN the system SHALL include: caption (≤2200 chars), share_to_feed
4. IF metadata exceeds platform limits THEN the system SHALL truncate with ellipsis and log warning
5. WHEN product data exists THEN the system SHALL auto-generate metadata from product title, description, and features

### Section 2: Publishing Operations

#### Requirement 5: Single Video Publishing

**User Story:** As a content creator, I want to publish a single video to one or more platforms with a simple command.

##### Acceptance Criteria

1. WHEN publishing a video THEN the system SHALL accept: video path, platforms list, optional schedule time
2. IF schedule time is provided THEN the system SHALL validate it is in the future
3. WHEN publishing succeeds THEN the system SHALL return: post ID, platform, status, URL (if available)
4. IF publishing fails THEN the system SHALL provide actionable error message

#### Requirement 6: Batch Publishing

**User Story:** As a content creator with many videos, I want to publish multiple videos efficiently with progress tracking.

##### Acceptance Criteria

1. WHEN batch publishing THEN the system SHALL process videos concurrently (configurable limit)
2. IF a video fails THEN the system SHALL log error and continue with remaining videos
3. WHEN --fail-fast is enabled THEN the system SHALL stop on first failure
4. IF progress tracking is enabled THEN the system SHALL log in `[N/total]` format
5. WHEN batch completes THEN the system SHALL output summary with success/failure counts

#### Requirement 7: Configuration Management

**User Story:** As a developer, I want flexible configuration with environment, file, and CLI options.

##### Acceptance Criteria

1. WHEN loading configuration THEN the system SHALL use precedence: CLI > ENV > YAML > defaults
2. IF config file exists at config/publisher.yaml THEN the system SHALL load platform-specific settings
3. WHEN credentials are in environment THEN the system SHALL read: LATE_API_KEY, LATE_USER_ID
4. IF configuration is invalid THEN the system SHALL fail with validation errors listing issues

### Section 3: Status and Error Handling

#### Requirement 8: Status Tracking

**User Story:** As a user, I want to track publishing status for scheduled and completed posts.

##### Acceptance Criteria

1. WHEN a video is scheduled THEN the system SHALL store: post_id, platform, scheduled_time, status
2. IF checking status THEN the system SHALL query Late API for current state
3. WHEN status changes THEN the system SHALL update local tracking file
4. IF a scheduled post fails THEN the system SHALL mark as failed with error details

#### Requirement 9: Error Handling and Resilience

**User Story:** As a user, I want robust error handling that doesn't lose my work.

##### Acceptance Criteria

1. WHEN API errors occur THEN the system SHALL categorize: retryable (429, 503) vs permanent (400, 401)
2. IF retryable error occurs THEN the system SHALL retry with exponential backoff
3. WHEN permanent error occurs THEN the system SHALL fail immediately with clear message
4. IF network timeout occurs THEN the system SHALL retry up to 3 times
5. WHEN circuit breaker opens THEN the system SHALL fast-fail without making API calls

#### Requirement 10: CLI Interface

**User Story:** As a user, I want a comprehensive CLI for all publishing operations.

##### Acceptance Criteria

1. WHEN running `python -m src.publisher.late` THEN the system SHALL provide subcommands: publish, status, accounts
2. IF --platforms is specified THEN the system SHALL publish to listed platforms only
3. WHEN --schedule is provided THEN the system SHALL schedule for future time (ISO 8601 format)
4. IF --dry-run is enabled THEN the system SHALL validate without actually publishing

### Section 4: Scheduling System

#### Requirement 11: Calendar View

**User Story:** As a content creator, I want to see my publishing schedule in a calendar format, so that I can plan content distribution.

##### Acceptance Criteria

1. WHEN requesting calendar view THEN the system SHALL display scheduled posts by date/time
2. IF --week flag is set THEN the system SHALL show 7-day view from today
3. WHEN --month flag is set THEN the system SHALL show 30-day view
4. IF posts exist for a slot THEN the system SHALL show: time, platform, product_id, status

#### Requirement 12: Recurring Schedule Configuration

**User Story:** As a content creator, I want to define recurring time slots, so that I maintain consistent posting schedule.

##### Acceptance Criteria

1. WHEN defining schedule THEN the system SHALL accept: days of week, times, platforms per slot
2. IF slot conflicts with existing THEN the system SHALL warn and require confirmation
3. WHEN schedule is saved THEN the system SHALL persist to config/schedule.json
4. IF timezone is specified THEN the system SHALL convert all times accordingly

#### Requirement 13: Schedule Validation

**User Story:** As a content creator, I want validation of my schedule, so that I don't accidentally double-post or violate platform limits.

##### Acceptance Criteria

1. WHEN scheduling a post THEN the system SHALL check for duplicate slots (same platform, same time)
2. IF minimum spacing is configured THEN the system SHALL enforce gap between posts (default 2 hours)
3. WHEN daily limit is set THEN the system SHALL reject posts exceeding limit per platform
4. IF validation fails THEN the system SHALL list all violations with suggested fixes

#### Requirement 14: Batch Scheduling

**User Story:** As a content creator with many videos, I want to auto-schedule multiple videos to available slots.

##### Acceptance Criteria

1. WHEN auto-scheduling THEN the system SHALL fill next available slots from recurring schedule
2. IF --start-date is provided THEN the system SHALL begin scheduling from that date
3. WHEN slots are full THEN the system SHALL report how many videos couldn't be scheduled
4. IF --preview flag is set THEN the system SHALL show proposed schedule without committing

### Section 5: Post-Publication Cleanup

#### Requirement 15: Automatic Cleanup

**User Story:** As a user, I want published product directories automatically cleaned up, so that disk space is managed efficiently.

##### Acceptance Criteria

1. WHEN a video is confirmed published on all platforms THEN the system SHALL mark directory for cleanup
2. IF --auto-cleanup is enabled THEN the system SHALL delete directory after configurable delay (default 24h)
3. WHEN cleaning up THEN the system SHALL verify publication status before deletion
4. IF verification fails THEN the system SHALL skip cleanup and log warning

#### Requirement 16: Manual Cleanup

**User Story:** As a user, I want to manually trigger cleanup for specific products or date ranges.

##### Acceptance Criteria

1. WHEN running cleanup command THEN the system SHALL accept: product IDs, date range, or --all flag
2. IF --dry-run is enabled THEN the system SHALL list directories that would be deleted
3. WHEN deleting THEN the system SHALL move to trash/archive before permanent deletion (configurable)
4. IF cleanup succeeds THEN the system SHALL log: product_id, platforms confirmed, space freed

#### Requirement 17: CLI Override for Cleanup

**User Story:** As a user, I want CLI control over cleanup behavior per operation.

##### Acceptance Criteria

1. WHEN --skip-cleanup is provided THEN the system SHALL not cleanup regardless of config
2. IF --force-cleanup is provided THEN the system SHALL cleanup immediately after publish
3. WHEN --archive-before-delete is set THEN the system SHALL copy to archive directory first
4. IF --cleanup-delay is specified THEN the system SHALL wait N hours before cleanup

#### Requirement 18: Safety Features

**User Story:** As a user, I want safety checks to prevent accidental data loss.

##### Acceptance Criteria

1. WHEN cleanup is triggered THEN the system SHALL verify video exists on target platforms
2. IF verification fails for any platform THEN the system SHALL abort cleanup with warning
3. WHEN --require-all-platforms is set THEN the system SHALL require confirmation from ALL platforms
4. IF unpublished videos exist THEN the system SHALL exclude from cleanup and report

## Non-Functional Requirements

### Code Architecture

- **Provider Pattern**: BasePublisher interface with LatePublisher implementation
- **Single Responsibility**: Separate modules for client, metadata, scheduling, cleanup
- **Circuit Breaker**: API calls protected with circuit breaker pattern
- **Dependency Injection**: Components receive configuration, not global state

### Performance

- Batch publishing SHALL handle 50+ videos in a single run
- Media uploads SHALL use streaming for files >4MB
- Schedule validation SHALL be O(n) for n scheduled posts
- Cleanup verification SHALL use concurrent API calls

### Security

- API credentials SHALL be stored in environment variables only
- OAuth tokens SHALL not be logged or stored in output files
- Cleanup operations SHALL require verification before deletion
- Audit log SHALL track all cleanup operations

### Reliability

- Network failures SHALL trigger retry with exponential backoff (max 3 retries)
- Partial failures SHALL not affect already-published videos
- Interrupted batches SHALL be resumable via status tracking
- Circuit breaker SHALL prevent cascade failures

### Usability

- Error messages SHALL include actionable guidance
- Progress output SHALL use consistent `[N/total]` format
- Dry-run mode SHALL be available for all destructive operations
- Calendar view SHALL be human-readable and machine-parseable
