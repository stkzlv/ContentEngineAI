# Requirements Document

## Introduction

The Freesound Client is a modular, async-first Python client for integrating with Freesound.org's API to search, download, and manage background music for video production. It provides a production-ready implementation with OAuth2 authentication, circuit breaker resilience, and graceful fallback strategies, serving as the foundation for ContentEngineAI's stock music system.

**Purpose**: Enable automated discovery and download of Creative Commons licensed music tracks that match video duration requirements, with robust error handling and fallback to local stock files.

**Value**: Eliminates manual music selection, ensures proper licensing/attribution, and maintains video production quality through intelligent track selection and resilient API integration.

## Alignment with Product Vision

This feature directly supports ContentEngineAI's **Product Principles**:

1. **Automation Over Manual Intervention**: Fully automated music selection based on voiceover duration, no manual track selection required
2. **Quality Through Intelligence**: Smart track selection using ratings, duration matching, and quality filtering
3. **Performance at Scale**: Async operations with connection pooling for batch video production
4. **Modular Flexibility**: Extensible architecture supporting future music platforms (AudioJungle, Epidemic Sound)
5. **Fail Gracefully**: Multi-level fallbacks (OAuth2 → API key → local files) ensure pipeline completion

**Business Impact**:
- **Reduce Production Costs**: Eliminate music licensing fees through Creative Commons integration
- **Accelerate Time-to-Market**: Automated music selection reduces video generation time
- **Maintain Quality Standards**: Curated search queries and rating filters ensure professional audio quality

## Requirements

### Requirement 1: Async Music Search with Duration Matching

**User Story:** As a video producer, I want the system to automatically find background music tracks matching my voiceover duration, so that I don't have to manually search for and select music files.

#### Acceptance Criteria

1. WHEN voiceover duration is known THEN the system SHALL search Freesound with duration filter `[voiceover_duration TO max_duration]`
2. IF no tracks match duration requirements THEN the system SHALL fallback to broader search using configured filter string
3. WHEN search completes THEN the system SHALL return sorted tracks (by duration ascending) with metadata (id, name, duration, previews, license, username, url)
4. WHEN search times out after configured timeout (default 30s) THEN the system SHALL return empty list and log warning
5. IF circuit breaker is open THEN the system SHALL skip API call and return empty list immediately

### Requirement 2: OAuth2 Authentication with Token Refresh

**User Story:** As a system administrator, I want OAuth2 authentication to automatically refresh access tokens and persist refresh tokens, so that high-quality music downloads continue working without manual intervention.

#### Acceptance Criteria

1. WHEN OAuth2 credentials (client_id, client_secret, refresh_token) are configured THEN the system SHALL authenticate using refresh token grant
2. IF access token is expired or within refresh buffer (60s) THEN the system SHALL automatically refresh the token before API calls
3. WHEN new refresh token is received THEN the system SHALL update `.env` file using `dotenv.set_key()` for persistence
4. IF token refresh fails after retries (max 2 attempts, exponential backoff) THEN the system SHALL return None and log error
5. WHEN authentication errors occur (401, 403) THEN the system SHALL NOT retry and return False immediately

### Requirement 3: High-Quality OAuth2 Downloads with API Key Fallback

**User Story:** As a video producer, I want the system to download full-quality music files when possible, with automatic fallback to preview quality, so that videos have the best audio quality available.

#### Acceptance Criteria

1. WHEN OAuth2 is configured AND track selected THEN the system SHALL attempt full-quality download via OAuth2
2. IF OAuth2 download fails OR OAuth2 not configured THEN the system SHALL fallback to preview download using API key
3. WHEN downloading THEN the system SHALL save file with sanitized filename to configured output directory
4. IF download times out (default 300s) THEN the system SHALL retry once with exponential backoff (1s, 2s)
5. WHEN download completes THEN the system SHALL return (file_path, attribution_metadata) tuple with license, author, URL

### Requirement 4: Circuit Breaker Pattern for API Resilience

**User Story:** As a system operator, I want the client to fast-fail on repeated API errors, so that batch processing doesn't waste time on unavailable services and falls back to local files quickly.

#### Acceptance Criteria

1. WHEN API calls fail repeatedly THEN the circuit breaker SHALL open after threshold failures
2. IF circuit breaker is open THEN the system SHALL skip API calls immediately without delay
3. WHEN circuit breaker opens THEN the system SHALL log warning and allow fallback mechanisms to proceed
4. IF circuit breaker is half-open AND call succeeds THEN the system SHALL close the circuit breaker
5. WHEN configured THEN the circuit breaker SHALL use configurable failure threshold and timeout window

### Requirement 5: Local Fallback with Memory-Mapped I/O

**User Story:** As a video producer, I want the system to use local stock music files when Freesound is unavailable, so that video production continues without service dependencies.

#### Acceptance Criteria

1. WHEN Freesound API unavailable OR no suitable tracks found THEN the system SHALL fallback to local music files
2. IF local music paths configured THEN the system SHALL randomly select from existing files
3. WHEN copying large files (>1MB) THEN the system SHALL use memory-mapped I/O for efficient transfer
4. IF memory-mapped copy fails THEN the system SHALL fallback to standard file copy
5. WHEN local file used THEN the system SHALL return (file_path, attribution_metadata) with source="Local"

### Requirement 6: Attribution Metadata Tracking

**User Story:** As a content creator, I want automatic attribution data for all music tracks, so that I comply with Creative Commons licensing requirements.

#### Acceptance Criteria

1. WHEN track downloaded THEN the system SHALL extract and store metadata (source, type, author, url, license, name, id, path)
2. IF license information available THEN the system SHALL include full license string (e.g., "Attribution 3.0")
3. WHEN metadata returned THEN the system SHALL format as dictionary with standard keys
4. IF metadata incomplete THEN the system SHALL use fallback values (e.g., username="Unknown")
5. WHEN local file used THEN the system SHALL generate attribution metadata with source="Local"

### Requirement 7: Configurable Search Parameters

**User Story:** As a system administrator, I want configurable search parameters, so that I can customize music selection criteria for different video profiles or campaigns.

#### Acceptance Criteria

1. WHEN searching THEN the system SHALL accept query string, filters, sort order, max results, fields, timeout parameters
2. IF parameters not provided THEN the system SHALL use configured defaults from YAML config
3. WHEN sort order specified THEN the system SHALL pass to Freesound API (e.g., "rating_desc", "duration_asc")
4. IF max_results configured THEN the system SHALL limit API response to specified count
5. WHEN fields parameter provided THEN the system SHALL request only specified fields to reduce response size

### Requirement 8: Session Management and Connection Pooling

**User Story:** As a system operator, I want HTTP sessions reused across API calls, so that batch processing benefits from connection pooling and reduced latency.

#### Acceptance Criteria

1. WHEN client instantiated THEN the system SHALL accept `aiohttp.ClientSession` parameter
2. IF session provided THEN the system SHALL reuse for all HTTP requests (OAuth2, downloads, API calls)
3. WHEN session closed errors occur THEN the system SHALL attempt to get new session from connection pool
4. IF session recovery succeeds THEN the system SHALL retry failed operation once
5. WHEN retries exhausted THEN the system SHALL return None/empty and log error

## Non-Functional Requirements

### Code Architecture and Modularity

- **Single Responsibility**: `FreesoundClient` class handles only Freesound API integration, no video assembly or scraping logic
- **Modular Design**: Separate methods for search, OAuth2, downloads, token refresh for independent testing and reuse
- **Dependency Management**: Minimal external dependencies (aiohttp, freesound-api, dotenv), no video-specific imports
- **Clear Interfaces**: Public async methods with typed parameters and return values, private sync helpers prefixed with `_`
- **Extensibility**: Design allows future `BaseStockMusicClient` abstraction for multi-platform support

### Performance

- **Search Latency**: Complete search requests in <30 seconds (configurable timeout)
- **Download Speed**: Stream files in 32KB chunks to avoid memory spikes, support files up to 50MB
- **Token Refresh**: Complete OAuth2 token refresh in <5 seconds
- **Async Operations**: All I/O operations non-blocking, support concurrent downloads for batch processing
- **Memory Efficiency**: Use memory-mapped I/O for local file copies >1MB to reduce memory usage

### Security

- **Credential Storage**: All API keys, client secrets, refresh tokens stored in `.env` file, never hardcoded
- **Secret Handling**: Accept credentials via kwargs, validate presence before API calls
- **Token Persistence**: Update `.env` securely using `dotenv.set_key()` with proper error handling
- **Input Validation**: Sanitize all filenames using existing `sanitize_filename()` utility
- **Error Exposure**: Log errors without exposing full credentials (mask sensitive values in logs)

### Reliability

- **Circuit Breaker**: Use `@freesound_circuit_breaker` decorator from `src.utils.circuit_breaker` for all API methods
- **Retry Logic**: Max 2 retry attempts with exponential backoff (0.5s, 1s for OAuth; 1s, 2s for downloads)
- **Timeout Handling**: Respect configured timeouts for all operations, catch `TimeoutError` and `aiohttp.ServerTimeoutError`
- **Graceful Degradation**: Never raise exceptions on API failures, return None/empty list and log warnings
- **Fallback Chain**: OAuth2 → API key preview → local files ensure music always available

### Usability

- **Logging**: Structured logging at INFO level for successes, WARNING for fallbacks, ERROR for failures
- **Error Messages**: Clear, actionable error messages with context (e.g., "OAuth token refresh failed due to timeouts - failing fast")
- **Configuration**: All settings via YAML config (`config/video_production.yaml` → `audio_settings` section)
- **Documentation**: Comprehensive docstrings with Args, Returns, Raises sections for all public methods
- **Debug Mode**: Detailed logging when debug flag enabled, without exposing secrets
