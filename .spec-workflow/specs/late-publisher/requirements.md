# Requirements Document

## Introduction

The Late Publisher module enables automated video publishing to social media platforms (YouTube, TikTok, Instagram) using the Late.dev scheduling service. This feature completes the end-to-end content pipeline: scrape → generate → publish, allowing users to automatically schedule and post generated videos across multiple platforms without manual intervention.

**Value Proposition**: Eliminates the final manual bottleneck in content production workflows. Users can configure publish schedules once and let ContentEngineAI handle complete content lifecycle from product data to live social media posts.

## Alignment with Product Vision

This feature directly supports the core product principles:

1. **Automation Over Manual Intervention**: Removes manual video uploading and scheduling across platforms
2. **Performance at Scale**: Batch publishing enables posting hundreds of videos with single command
3. **Modular Flexibility**: Late.dev integration is isolated and replaceable with other scheduling services
4. **Fail Gracefully**: Multi-level fallbacks ensure partial publishing success even with platform failures
5. **Quality Through Intelligence**: Leverages existing platform-specific metadata optimization (v0.17.0) for optimal engagement

**Business Impact**: Reduces time-to-publish from minutes (manual) to seconds (automated), enables scheduled campaigns, and provides centralized multi-platform management.

## Requirements

### Requirement 1: Late SDK Integration

**User Story:** As a developer, I want a Late client wrapper integrated into ContentEngineAI, so that the system can publish videos through the Late.dev service.

#### Acceptance Criteria

1. WHEN the system initializes the Late client THEN it SHALL load API credentials from environment variables (`.env`)
2. WHEN the Late client makes API calls THEN it SHALL use the `late-sdk` Python package (pip install late-sdk)
3. WHEN authentication fails THEN the system SHALL raise clear error messages indicating missing or invalid credentials
4. WHEN the Late client encounters rate limits THEN it SHALL implement exponential backoff with configurable retry attempts
5. IF the Late API is unavailable THEN the system SHALL log detailed error information and gracefully degrade

### Requirement 2: Account Discovery and Validation

**User Story:** As a content creator, I want to see which social media accounts are connected to my Late profile, so that I can verify the correct accounts before publishing.

#### Acceptance Criteria

1. WHEN the user requests account listing THEN the system SHALL fetch all connected accounts via `client.accounts.list()`
2. WHEN displaying accounts THEN the system SHALL show platform type, account username/handle, and account ID
3. WHEN no accounts are connected THEN the system SHALL display a clear message with instructions to connect accounts
4. IF account fetching fails THEN the system SHALL retry up to 3 times before reporting failure
5. WHEN validating accounts before publish THEN the system SHALL verify account IDs exist and are accessible

### Requirement 3: Media Upload Management

**User Story:** As a content creator, I want the system to automatically upload video files to Late, so that I don't need to manually handle file transfers.

#### Acceptance Criteria

1. WHEN uploading videos ≤4 MB THEN the system SHALL use `client.media.upload()` with API key authentication
2. WHEN uploading videos >4 MB and ≤500 MB THEN the system SHALL use `client.media.upload_large()` with Vercel token
3. WHEN video size >500 MB THEN the system SHALL raise an error indicating the file exceeds Late's limit
4. WHEN upload fails THEN the system SHALL retry up to 3 times with exponential backoff (2s, 4s, 8s)
5. WHEN upload succeeds THEN the system SHALL return and store the Late media ID for post creation
6. IF video file is missing or unreadable THEN the system SHALL validate file existence before upload attempt
7. WHEN uploading large files THEN the system SHALL display progress updates every 10% completion

### Requirement 4: Platform-Specific Metadata Integration

**User Story:** As a content creator, I want the system to use my platform-optimized metadata (titles, descriptions, hashtags) when publishing, so that posts are optimized for each platform's requirements.

#### Acceptance Criteria

1. WHEN publishing to YouTube THEN the system SHALL load metadata from `outputs/{product_id}/text/metadata_youtube.json`
2. WHEN publishing to TikTok THEN the system SHALL load metadata from `outputs/{product_id}/text/metadata_tiktok.json`
3. WHEN publishing to Instagram THEN the system SHALL load metadata from `outputs/{product_id}/text/metadata_instagram.json`
4. WHEN metadata files are missing THEN the system SHALL fall back to `UPLOAD_INSTRUCTIONS.txt` content extraction
5. IF no metadata is available THEN the system SHALL raise an error indicating missing metadata for target platform
6. WHEN constructing post content THEN the system SHALL apply platform-specific character limits and formatting rules

### Requirement 5: Single Video Publishing

**User Story:** As a content creator, I want to publish a single video to one or more platforms immediately or at a scheduled time, so that I can control individual post timing.

#### Acceptance Criteria

1. WHEN publishing immediately THEN the system SHALL set `publish_now=True` in `client.posts.create()`
2. WHEN scheduling for future THEN the system SHALL accept datetime string and convert to Late-compatible format
3. WHEN targeting multiple platforms THEN the system SHALL create separate platform objects with correct account IDs
4. WHEN post creation succeeds THEN the system SHALL return and log the Late post ID and scheduled time
5. IF post creation fails THEN the system SHALL log the specific platform error response
6. WHEN a scheduled time is in the past THEN the system SHALL raise validation error before API call
7. WHEN timezone is specified THEN the system SHALL convert schedule to UTC for Late API

### Requirement 6: Batch Publishing Workflow

**User Story:** As a content creator, I want to publish multiple videos from batch processing runs in a single command, so that I can automate posting entire campaigns.

#### Acceptance Criteria

1. WHEN batch publishing is enabled THEN the system SHALL scan outputs directory for completed videos
2. WHEN iterating videos THEN the system SHALL map each video to its platform-specific metadata files
3. WHEN publishing multiple videos THEN the system SHALL apply staggered delays between posts (configurable, default 30-60 seconds)
4. WHEN one video fails to publish THEN the system SHALL continue publishing remaining videos unless `--fail-fast` is specified
5. WHEN batch completes THEN the system SHALL generate summary report with successful/failed posts and published URLs
6. IF no videos with metadata are found THEN the system SHALL display message indicating no publishable content
7. WHEN rate limit is hit THEN the system SHALL wait for retry-after period before continuing batch

### Requirement 7: Publishing Configuration

**User Story:** As a content creator, I want to configure default publishing settings via YAML and override them with CLI arguments, so that I can maintain consistent workflows while allowing flexibility.

#### Acceptance Criteria

1. WHEN loading configuration THEN the system SHALL follow precedence: CLI args > environment variables > YAML config
2. WHEN YAML config specifies default platforms THEN the system SHALL use these unless CLI override provided
3. WHEN privacy settings are configured THEN the system SHALL apply platform-specific privacy levels (YouTube: public/unlisted/private, Instagram: everyone/followers)
4. WHEN schedule mode is set to "immediate" THEN the system SHALL publish as soon as upload completes
5. IF Late API key is missing from environment THEN the system SHALL raise clear error on initialization
6. WHEN retry configuration is specified THEN the system SHALL honor max retry attempts and backoff multiplier
7. WHEN upload timeout is configured THEN the system SHALL cancel upload after specified duration

### Requirement 8: Status Tracking and Reporting

**User Story:** As a content creator, I want to monitor upload progress and see published post URLs, so that I can verify successful publishing and share content links.

#### Acceptance Criteria

1. WHEN uploading video THEN the system SHALL display real-time progress percentage for large files
2. WHEN post is created THEN the system SHALL fetch and log the Late post ID
3. WHEN post is scheduled THEN the system SHALL display scheduled publish time in user's timezone
4. WHEN batch publishing THEN the system SHALL display progress with `[N/total]` format
5. WHEN publishing succeeds THEN the system SHALL log the published post URL (if immediately published)
6. WHEN generating summary THEN the system SHALL include: total attempted, successful, failed, skipped posts per platform
7. IF post status check fails THEN the system SHALL log error but continue with remaining posts

### Requirement 9: Error Handling and Resilience

**User Story:** As a content creator, I want the system to handle publishing failures gracefully, so that one failed post doesn't prevent other posts from being published.

#### Acceptance Criteria

1. WHEN network timeout occurs THEN the system SHALL retry upload up to configured max attempts
2. WHEN API returns 429 rate limit THEN the system SHALL extract retry-after header and wait before retry
3. WHEN authentication fails (401) THEN the system SHALL raise clear error about invalid/expired credentials
4. WHEN file validation fails THEN the system SHALL skip upload and log specific validation error
5. IF one platform fails in multi-platform post THEN the system SHALL continue attempting other platforms
6. WHEN Late API returns validation error THEN the system SHALL log the specific field and error message
7. WHEN upload fails after all retries THEN the system SHALL log detailed error and continue to next video in batch

### Requirement 10: CLI Interface

**User Story:** As a content creator, I want clear CLI commands to publish videos, so that I can integrate publishing into my workflows and automation scripts.

#### Acceptance Criteria

1. WHEN running `poetry run python -m src.publisher.late --video <path> --platform youtube` THEN the system SHALL publish single video to YouTube
2. WHEN specifying multiple platforms with `--platform youtube tiktok instagram` THEN the system SHALL post to all specified platforms
3. WHEN using `--schedule "2025-12-20 14:00:00"` THEN the system SHALL schedule post for specified datetime
4. WHEN using `--batch` THEN the system SHALL discover and publish all videos from outputs directory
5. WHEN using `--immediate` THEN the system SHALL override scheduled publishing and post immediately
6. WHEN using `--debug` THEN the system SHALL display detailed API request/response information
7. IF required arguments are missing THEN the system SHALL display usage help with examples

## Non-Functional Requirements

### Code Architecture and Modularity

- **Single Responsibility**: Separate modules for Late client wrapper, media upload, metadata loading, and batch orchestration
- **Modular Design**: Late client isolated in `src/publisher/late/` allowing future replacement with other services
- **Provider Pattern**: Abstract publisher interface allowing multiple scheduling service implementations
- **Clear Interfaces**: Define `BasePublisher` abstract class with methods: `authenticate()`, `upload_media()`, `create_post()`, `get_status()`
- **Configuration Loading**: Centralized config module handling three-tier precedence (CLI > env > YAML)

### Performance

- **Upload Optimization**: Stream large files in chunks to minimize memory usage (32KB chunks)
- **Batch Processing**: Process videos sequentially with configurable delays to respect rate limits
- **Async Operations**: Use `late-sdk` async methods (`acreate`, `aupload`) for concurrent operations where safe
- **Connection Pooling**: Reuse HTTP sessions across batch to reduce connection overhead
- **Progress Tracking**: Update progress every 10% for large file uploads to provide user feedback

### Security

- **Credential Storage**: Late API key and Vercel token stored in `.env` file (never in code or YAML)
- **Environment Validation**: Verify credentials exist before starting publish workflow
- **API Key Rotation**: Support updating credentials without code changes (read from environment)
- **Error Messages**: Never log full API keys in error messages (show first 4 characters only)
- **File Permissions**: Validate output videos are readable before attempting upload

### Reliability

- **Retry Logic**: Exponential backoff for transient failures (network timeouts, 5xx errors)
- **Rate Limit Handling**: Detect 429 responses and wait for retry-after period before retry
- **Graceful Degradation**: Continue batch processing even when individual posts fail
- **Validation**: Pre-validate video files, metadata files, account IDs before upload attempts
- **Error Recovery**: Clear error messages with actionable remediation steps

### Usability

- **Clear CLI Output**: Progress indicators, success confirmations, and error messages
- **Example Commands**: Documentation includes common use cases with copy-paste examples
- **Default Behavior**: Sensible defaults for immediate publishing without complex configuration
- **Batch Workflows**: Single command to publish entire batch without per-video interaction
- **Debug Mode**: Detailed logging for troubleshooting without cluttering normal output
