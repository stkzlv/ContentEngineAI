# Requirements Document

## Introduction

This feature adds advanced scheduling and automated cleanup capabilities to the ContentEngineAI Publisher module. It enables users to:

1. **Publishing Schedule & Calendar**: View, manage, and organize scheduled posts across multiple platforms with recurring time slots and validation rules
2. **Post-Publication Cleanup**: Automatically remove successfully published product directories from the outputs folder to manage disk space and maintain organization

**Value Proposition**: Streamlines content publishing workflows by providing calendar-based scheduling with recurring slots (eliminating manual scheduling), and automatic cleanup of published content (preventing disk space issues and maintaining organized outputs directory).

## Alignment with Product Vision

This feature aligns with ContentEngineAI's vision of automated video production pipelines by:

- **Automation**: Reduces manual intervention through recurring schedule slots and automatic cleanup
- **Scalability**: Enables high-volume publishing with batch scheduling and disk space management
- **Reliability**: Prevents scheduling conflicts and ensures publication success before cleanup
- **User Experience**: Provides clear visibility into scheduled posts and publication status

## Requirements

### Requirement 1: Calendar View

**User Story:** As a content publisher, I want to view all my scheduled posts in a calendar format with filtering options, so that I can manage my content schedule across multiple platforms.

#### Acceptance Criteria

1. WHEN user runs `calendar list` command THEN system SHALL display all scheduled posts with product ID, scheduled time, platforms, post IDs, and status
2. WHEN user provides `--platform` filter THEN system SHALL display only posts scheduled for that platform
3. WHEN user provides `--date-from` and `--date-to` filters THEN system SHALL display only posts within that date range
4. WHEN user provides `--status` filter THEN system SHALL display only posts matching that status (scheduled, published, failed, partial)
5. WHEN displaying scheduled times THEN system SHALL show both UTC time and local timezone conversion

### Requirement 2: Recurring Schedule Configuration

**User Story:** As a content publisher, I want to define recurring time slots for publishing, so that I can automate the scheduling of multiple videos without manual date/time entry.

#### Acceptance Criteria

1. WHEN user configures recurring schedule in YAML THEN system SHALL support day-of-week (monday-sunday) and time (HH:MM:SS) format
2. WHEN user enables recurring schedule THEN system SHALL use configured timezone for all slot times
3. WHEN user runs `schedule auto` command THEN system SHALL assign unpublished videos to next available recurring slots
4. IF `--dry-run` flag is provided THEN system SHALL preview schedule without creating posts
5. WHEN scheduling to slots THEN system SHALL validate minimum spacing between posts
6. IF no slots are available THEN system SHALL report error with next available slot time

### Requirement 3: Schedule Validation

**User Story:** As a content publisher, I want the system to prevent scheduling conflicts and enforce spacing rules, so that my posts don't violate platform rate limits or duplicate existing schedules.

#### Acceptance Criteria

1. WHEN user attempts to schedule post THEN system SHALL check for duplicate (same product + platform + time) and reject if exists
2. WHEN user schedules post THEN system SHALL enforce minimum spacing (configurable hours) between posts on same platform
3. IF scheduled time is in the past AND `allow_past_schedules: false` THEN system SHALL reject the schedule
4. WHEN user schedules post THEN system SHALL validate datetime includes timezone information
5. IF daily post limit is configured THEN system SHALL reject scheduling when limit reached for that day
6. WHEN validation fails THEN system SHALL provide clear error message with reason and suggestion

### Requirement 4: Batch Scheduling

**User Story:** As a content publisher, I want to schedule multiple videos from my outputs directory in a single command, so that I can efficiently publish large batches of content.

#### Acceptance Criteria

1. WHEN user runs `schedule auto` with `--outputs-dir` THEN system SHALL scan directory for unpublished videos
2. WHEN assigning videos to slots THEN system SHALL use sequential order by directory scan
3. IF video already published to specified platforms THEN system SHALL skip and log as already published
4. WHEN batch scheduling completes THEN system SHALL report summary with scheduled count, skipped count, and failed count
5. IF `--start-slot` is provided THEN system SHALL begin from that slot number (skip earlier slots)

### Requirement 5: Automatic Cleanup

**User Story:** As a content publisher, I want published product directories automatically removed from outputs, so that I don't accumulate unnecessary files and run out of disk space.

#### Acceptance Criteria

1. WHEN post is successfully published AND `cleanup.enabled: true` THEN system SHALL verify publication success via API status check
2. IF `cleanup.require_all_platforms: true` THEN system SHALL only cleanup when published to ALL configured platforms
3. WHEN cleanup is triggered THEN system SHALL log deleted directory with product ID, platforms, and post URLs to audit log
4. IF `cleanup.verify_before_delete: true` THEN system SHALL query API to confirm publication status before deletion
5. WHEN cleanup executes THEN system SHALL remove entire product directory (`outputs/<product_id>/`)
6. IF any platform has `auto_cleanup: false` THEN system SHALL skip cleanup for that product

### Requirement 6: Manual Cleanup

**User Story:** As a content publisher, I want to manually trigger cleanup for specific products or all published products, so that I can manage disk space on demand.

#### Acceptance Criteria

1. WHEN user runs `cleanup` with `--product-id` THEN system SHALL cleanup only that product
2. WHEN user runs `cleanup` with `--all` THEN system SHALL cleanup all successfully published products
3. IF `--dry-run` flag is provided THEN system SHALL preview cleanup without deletion
4. WHEN cleanup completes THEN system SHALL report summary with cleanup count, skipped count, and disk space freed
5. IF product not published to all platforms THEN system SHALL skip and log reason

### Requirement 7: CLI Override for Cleanup

**User Story:** As a content publisher, I want to disable cleanup on specific publish commands, so that I can preserve outputs when needed for debugging or manual review.

#### Acceptance Criteria

1. WHEN user provides `--no-cleanup` flag on `single` command THEN system SHALL skip cleanup regardless of configuration
2. WHEN user provides `--no-cleanup` flag on `batch` command THEN system SHALL skip cleanup for entire batch
3. WHEN `--no-cleanup` is used THEN system SHALL log that cleanup was disabled via CLI override

### Requirement 8: Cleanup Safety Features

**User Story:** As a content publisher, I want safety checks before cleanup execution, so that I don't accidentally delete content that hasn't been successfully published.

#### Acceptance Criteria

1. IF `cleanup.archive_before_delete: true` THEN system SHALL create ZIP archive in `archive_dir` before deletion
2. IF `cleanup.keep_published_days > 0` THEN system SHALL only cleanup products older than specified days
3. IF `cleanup.preserve_metadata: true` THEN system SHALL keep metadata JSON files when cleaning directory
4. IF `cleanup.preserve_logs: true` THEN system SHALL keep log files in `outputs/logs/` when cleaning directories
5. WHEN API status check fails THEN system SHALL skip cleanup and log error

## Non-Functional Requirements

### Code Architecture and Modularity

- **Single Responsibility Principle**: Separate modules for calendar operations, schedule validation, cleanup logic, and CLI commands
- **Modular Design**: Reusable components for date/time handling, platform filtering, and status checks
- **Dependency Management**: Minimize coupling between scheduling and cleanup modules
- **Clear Interfaces**: Define clean APIs between publisher client, schedule manager, and cleanup manager

### Performance

- **Calendar Queries**: List operations SHALL complete within 2 seconds for up to 1000 scheduled posts
- **Batch Scheduling**: Schedule 100 videos SHALL complete within 5 minutes (excluding upload time)
- **Cleanup Operations**: Delete product directory SHALL complete within 1 second per product
- **API Efficiency**: Use batch API endpoints where available to reduce API calls

### Security

- **Verification**: Always verify publication success via API before cleanup (prevent data loss)
- **Audit Trail**: Log all cleanup operations with timestamps, product IDs, and post URLs
- **Credentials**: Use existing three-tier configuration (CLI → .env → YAML) for API credentials
- **Error Handling**: Never delete on failed status checks or API errors

### Reliability

- **Idempotent Operations**: Cleanup operations SHALL be safely retryable
- **Graceful Degradation**: If API unavailable, skip cleanup rather than fail entire publish operation
- **Status Verification**: Double-check publication status before permanent deletion
- **Transaction Safety**: Use atomic file operations where possible

### Usability

- **Clear Feedback**: Provide progress indicators for batch scheduling and cleanup operations
- **Error Messages**: Display actionable error messages with suggestions for resolution
- **Dry Run Mode**: Support `--dry-run` for previewing operations without execution
- **Summary Reports**: Display comprehensive summaries after batch operations
- **Timezone Support**: Display times in both UTC and user's local timezone
