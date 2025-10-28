# Design Document

## Overview

The Freesound Client is a production-grade async Python client that provides robust integration with Freesound.org's API for automated music discovery and download. It implements a multi-layered approach to reliability (OAuth2 → API key → local files) with circuit breaker resilience, automatic token management, and intelligent track selection based on video duration requirements.

**System Role**: The client acts as a specialized service layer between ContentEngineAI's video production pipeline (`src/video/producer.py`) and external music sources, encapsulating all Freesound API complexity while providing simple async interfaces for music search and download operations.

**Design Philosophy**: Following ContentEngineAI's "Fail Gracefully" principle, the client prioritizes pipeline continuity over perfect music matching—always providing a usable music track through progressive fallback strategies rather than halting video production.

## Steering Document Alignment

### Technical Standards (tech.md)

**Async-First Architecture**:
- All public methods use `async def` for non-blocking I/O
- Leverages `aiohttp.ClientSession` for connection pooling and concurrent requests
- Integrates with existing async pipeline context (`PipelineContext` from `src/video/producer.py`)

**Dependency Management**:
- Uses existing dependencies: `aiohttp` (HTTP), `freesound-api` (API wrapper), `python-dotenv` (env management)
- No new third-party libraries required
- Leverages Poetry for version pinning and reproducible builds

**Error Handling Standards**:
- Never raises exceptions to calling code on API failures
- Returns `None`, empty lists, or fallback values with structured logging
- Uses existing `@freesound_circuit_breaker` decorator from `src.utils.circuit_breaker`
- Follows tenacity retry patterns with exponential backoff (max 2 attempts)

**Security Compliance**:
- All credentials via environment variables (`.env` file)
- Uses `dotenv.set_key()` for secure token persistence
- Sanitizes all user-provided strings with `src.utils.sanitize_filename()`
- Never logs full credentials (masks sensitive values)

### Project Structure (structure.md)

**Module Location**: `src/audio/freesound_client.py` (already exists, will be refactored)

**Naming Conventions**:
- Class: `FreesoundClient` (PascalCase)
- Methods: `search_music()`, `download_full_sound_oauth2()` (snake_case)
- Private helpers: `_refresh_oauth2_token()`, `_search_sync()` (leading underscore)
- Constants: `DEFAULT_TIMEOUT`, `MAX_RETRIES` (from `src/video/video_config.py`)

**Import Order** (enforced by Ruff):
```python
# 1. Standard library
import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Any

# 2. Third-party
import aiohttp
import freesound
from dotenv import set_key

# 3. First-party (absolute imports)
from src.utils import download_file, ensure_dirs_exist, sanitize_filename
from src.utils.circuit_breaker import freesound_circuit_breaker
from src.video.video_config import (
    FREESOUND_DEFAULT_DOWNLOAD_TIMEOUT_SEC,
    FREESOUND_DOWNLOAD_CHUNK_SIZE,
    ...
)
```

**File Organization**:
- Single responsibility: Freesound API integration only
- One primary class (`FreesoundClient`) with focused methods
- Helper function (`update_env_file()`) for token persistence
- Target file size: <500 lines (currently ~387 lines, well within target)

## Code Reuse Analysis

### Existing Components to Leverage

- **`src.utils.circuit_breaker.freesound_circuit_breaker`**: Pre-configured circuit breaker with 3-failure threshold, 60s timeout
  - **Usage**: Decorate all public async methods (`search_music`, `download_full_sound_oauth2`, `download_sound_preview_with_api_key`)
  - **Benefit**: Automatic fast-fail on repeated API errors without custom retry logic

- **`src.utils.download_file(url, path, session, timeout_sec)`**: Async file downloader with retry logic
  - **Usage**: Download preview MP3 files in `download_sound_preview_with_api_key()`
  - **Benefit**: Reuses existing chunked streaming, timeout handling, error recovery

- **`src.utils.sanitize_filename(name)`**: Filename sanitization for cross-platform compatibility
  - **Usage**: Sanitize track names before saving files (`f"{sanitize_filename(sound.name)}.mp3"`)
  - **Benefit**: Prevents path traversal, invalid characters, filesystem errors

- **`src.utils.ensure_dirs_exist(path)`**: Directory creation with proper error handling
  - **Usage**: Create output directories before downloads
  - **Benefit**: Consistent directory management across codebase

- **`src.utils.connection_pool.get_http_session()`**: Centralized aiohttp session management
  - **Usage**: Recover from session-closed errors during downloads/OAuth2
  - **Benefit**: Connection pooling, proper resource cleanup

- **`src.utils.memory_mapped_io.copy_file_mmap()`**: Efficient large file copying
  - **Usage**: Copy local fallback music files (>1MB threshold)
  - **Benefit**: Reduced memory usage for large audio files

- **`src.video.video_config` Constants**: Centralized configuration values
  - **Usage**: Import timeouts, chunk sizes, token expiry settings
  - **Benefit**: Single source of truth, easy configuration changes

### Integration Points

- **`src/video/producer.py` (Pipeline Step 5b)**:
  ```python
  async def step_download_music(ctx: PipelineContext):
      fs_client = FreesoundClient(**ctx.secrets)
      tracks = await fs_client.search_music(...)
      file_path, music_info = await fs_client.download_full_sound_oauth2(...)
  ```
  - **Integration**: Pass voiceover duration to `search_music()` for filtering
  - **Data Flow**: Returns `(Path, dict)` tuple with file path and attribution metadata

- **Configuration (`config/video_production.yaml`)**:
  - **Section**: `audio_settings` (lines 154-198)
  - **Keys Used**: `freesound_*` settings (API key env vars, search params, timeouts)
  - **Integration**: `VideoConfig` Pydantic model validates and provides typed access

- **Environment Variables (`.env`)**:
  ```
  FREESOUND_API_KEY=your_api_key_here
  FREESOUND_CLIENT_ID=your_oauth_client_id
  FREESOUND_CLIENT_SECRET=your_oauth_client_secret
  FREESOUND_REFRESH_TOKEN=your_refresh_token
  ```
  - **Integration**: Loaded via `python-dotenv`, passed to `FreesoundClient(**kwargs)`
  - **Persistence**: Updated via `set_key()` when new refresh tokens received

## Architecture

The Freesound Client implements a **tiered reliability architecture** with three progressive fallback layers, orchestrated through async methods protected by circuit breaker patterns.

### Modular Design Principles

- **Single File Responsibility**: `src/audio/freesound_client.py` handles only Freesound API integration, no video assembly or scraping
- **Component Isolation**: Self-contained class with minimal external dependencies (only utilities, no domain logic)
- **Service Layer Separation**:
  - **Data Access**: API calls to Freesound (search, OAuth2, downloads)
  - **Business Logic**: Track selection, fallback decisions, duration matching
  - **Presentation**: Returns structured data (`dict`) for attribution display
- **Utility Modularity**: Delegates to focused utilities (`sanitize_filename`, `download_file`, `circuit_breaker`)

```mermaid
graph TD
    Producer[Producer: step_download_music] --> Client[FreesoundClient]
    Client --> Search[search_music: Track Discovery]
    Client --> OAuth[download_full_sound_oauth2]
    Client --> Preview[download_sound_preview_with_api_key]
    Client --> Local[Local Fallback]

    Search --> CB1[@freesound_circuit_breaker]
    OAuth --> CB2[@freesound_circuit_breaker]
    Preview --> CB3[@freesound_circuit_breaker]

    OAuth --> Refresh[_refresh_oauth2_token]
    OAuth --> UpdateEnv[update_env_file]

    Preview --> UtilDownload[src.utils.download_file]
    OAuth --> ChunkedStream[aiohttp chunked download]

    Local --> MMap[copy_file_mmap]
    Local --> Shutil[shutil.copy fallback]

    CB1 --> FastFail{Circuit Open?}
    FastFail -->|Yes| Return[Return empty/None]
    FastFail -->|No| APICall[Execute API call]

    style CB1 fill:#ffe6e6
    style CB2 fill:#ffe6e6
    style CB3 fill:#ffe6e6
    style FastFail fill:#fff4e6
```

### Reliability Layers

**Layer 1: OAuth2 Full-Quality Downloads**
- **Preferred**: Highest audio quality, full track length
- **Requirement**: OAuth2 credentials configured, valid refresh token
- **Fallback Trigger**: Token refresh failure, 401/403 responses, download timeout

**Layer 2: API Key Preview Downloads**
- **Alternative**: MP3 previews (lower quality, sufficient for background music)
- **Requirement**: API key configured
- **Fallback Trigger**: OAuth2 unavailable, full download failed

**Layer 3: Local Stock Files**
- **Last Resort**: Pre-selected MP3 files in `static/` directory
- **Requirement**: `background_music_paths` configured with existing files
- **Activation**: All API methods failed or circuit breaker open

## Components and Interfaces

### Component 1: FreesoundClient (Main Service Class)

- **Purpose**: Orchestrate music search, download, and fallback strategies
- **Interfaces**:
  ```python
  class FreesoundClient:
      def __init__(self, **kwargs: str):
          """Initialize with API credentials from environment variables."""

      async def search_music(
          query: str,
          filters: str | None = None,
          max_results: int = None,
          sort_order: str = "rating_desc",
          fields: str = "id,name,previews,license,username,url,duration",
          timeout_sec: int = None,
      ) -> list:
          """Search for music tracks with timeout support.

          Returns:
              List of track objects (freesound.Sound), empty on error/timeout
          """

      async def download_full_sound_oauth2(
          sound_id: int,
          output_dir: Path,
          session: aiohttp.ClientSession,
          timeout_sec: int = 300,
      ) -> tuple[Path, dict[str, Any]] | None:
          """Download full-quality track via OAuth2.

          Returns:
              (file_path, attribution_metadata) or None on failure
          """

      async def download_sound_preview_with_api_key(
          sound: Any,
          output_dir: Path,
          session: aiohttp.ClientSession,
          timeout_sec: int = None,
      ) -> tuple[Path, dict[str, Any]] | None:
          """Download preview MP3 using API key.

          Returns:
              (file_path, attribution_metadata) or None on failure
          """
  ```

- **Dependencies**:
  - `freesound.FreesoundClient()` (sync API wrapper)
  - `aiohttp.ClientSession` (async HTTP)
  - Circuit breaker decorator (`@freesound_circuit_breaker`)
  - Utilities (`download_file`, `sanitize_filename`, `ensure_dirs_exist`)

- **Reuses**:
  - `src.utils.circuit_breaker.freesound_circuit_breaker` (resilience)
  - `src.utils.download_file()` (preview downloads)
  - `src.utils.connection_pool.get_http_session()` (session recovery)

### Component 2: Token Management (OAuth2 Helper)

- **Purpose**: Manage OAuth2 access token lifecycle and refresh token persistence
- **Interfaces**:
  ```python
  async def _refresh_oauth2_token(
      session: aiohttp.ClientSession
  ) -> bool:
      """Refresh OAuth2 access token using refresh token.

      Returns:
          True if successful, False on failure
      """

  async def _get_valid_oauth2_token(
      session: aiohttp.ClientSession
  ) -> str | None:
      """Get valid access token, refreshing if needed.

      Returns:
          Access token string or None if refresh failed
      """

  def update_env_file(key_to_update: str, new_value: str):
      """Safely update .env file with new refresh token."""
  ```

- **Dependencies**:
  - `dotenv.set_key()` (environment file updates)
  - `aiohttp` for token refresh HTTP POST
  - `time.time()` for expiry tracking

- **Reuses**:
  - Token expiry constants from `video_config` (buffer, expiry duration)
  - Retry logic pattern (2 attempts, exponential backoff)

### Component 3: Local Fallback Manager (Integrated into Producer)

- **Purpose**: Copy local stock music files when API unavailable
- **Interfaces** (in `src/video/producer.py`):
  ```python
  # Within step_download_music():
  if not music_info:
      local_path = random.choice(config.audio_settings.background_music_paths)
      dest_path = ctx.run_paths["assets_dir"] / local_path.name

      if is_file_suitable_for_mmap(local_path):
          copy_file_mmap(local_path, dest_path)
      else:
          shutil.copy(local_path, dest_path)
  ```

- **Dependencies**:
  - `src.utils.memory_mapped_io` (efficient copying)
  - `random.choice()` for track selection
  - `shutil.copy()` fallback

- **Reuses**:
  - `copy_file_mmap()` for large files (>1MB)
  - Existing attribution metadata structure

## Data Models

### Track Search Result (freesound.Sound)

External data model from `freesound-api` library, accessed via attributes:

```python
# freesound.Sound object (from API)
sound.id               # int - Freesound track ID
sound.name             # str - Track title
sound.duration         # float - Length in seconds
sound.previews         # object with preview URLs
  .preview_hq_mp3      # str - High-quality preview URL
  .preview_lq_mp3      # str - Low-quality preview URL
sound.license          # str - License type (e.g., "Attribution 3.0")
sound.username         # str - Uploader username
sound.url              # str - Freesound page URL
```

### Attribution Metadata (Internal)

Returned by download methods as `dict[str, Any]`:

```python
{
    "source": str,      # "Freesound" or "Local"
    "type": str,        # "Music"
    "author": str,      # Track creator username
    "url": str,         # Freesound page or local path
    "license": str,     # License type (e.g., "Attribution 3.0")
    "name": str,        # Track title
    "id": str,          # Freesound ID or "local"
    "path": str,        # Absolute file path
}
```

### OAuth2 Token State (Internal)

Class instance attributes for token management:

```python
class FreesoundClient:
    oauth_client_id: str | None         # OAuth2 client ID
    oauth_client_secret: str | None     # OAuth2 client secret
    oauth_refresh_token: str | None     # Persistent refresh token
    oauth_access_token: str | None      # Temporary access token
    oauth_token_expiry: float | None    # Unix timestamp of expiry
```

### Configuration Model (Pydantic - video_config.py)

Typed configuration from `VideoConfig.audio_settings`:

```python
class AudioSettings(BaseModel):
    freesound_api_key_env_var: str = "FREESOUND_API_KEY"
    freesound_client_id_env_var: str = "FREESOUND_CLIENT_ID"
    freesound_client_secret_env_var: str = "FREESOUND_CLIENT_SECRET"
    freesound_refresh_token_env_var: str = "FREESOUND_REFRESH_TOKEN"

    freesound_search_query: str = "upbeat instrumental corporate"
    freesound_filters: str = "duration:[60 TO 180]"
    freesound_sort: str = "rating_desc"
    freesound_max_results: int = 15

    freesound_api_timeout_sec: int = 30
    freesound_download_timeout_sec: int = 300
    freesound_token_expiry_sec: int = 3600
    freesound_token_refresh_buffer_sec: int = 60
    freesound_download_chunk_size: int = 32768

    background_music_paths: list[Path] = [
        "static/background-music-calm-soft-334182.mp3",
        "static/background-music-happy-333014.mp3",
        "static/background-music-upbeat-energetic-333016.mp3"
    ]
```

## Error Handling

### Error Scenarios

1. **Circuit Breaker Open (Repeated API Failures)**
   - **Handling**:
     - `@freesound_circuit_breaker` decorator prevents API call
     - Methods return `None` or empty list immediately
     - Log warning: "Circuit breaker Freesound is OPEN, failing fast"
   - **User Impact**: Fallback to local music files, video production continues
   - **Recovery**: Auto-retry after 60s timeout, circuit closes on success

2. **OAuth2 Token Refresh Failure**
   - **Handling**:
     - Max 2 retry attempts with exponential backoff (0.5s, 1s)
     - Log error on 401/403 (auth failure, don't retry)
     - Return `False`, caller falls back to API key preview
   - **User Impact**: Lower quality preview MP3 used instead of full quality
   - **Recovery**: Next run may succeed if credentials updated in `.env`

3. **Search Timeout (>30 seconds)**
   - **Handling**:
     - `asyncio.wait_for()` raises `TimeoutError`
     - Catch exception, log warning, return empty list
     - No retry (circuit breaker may open if repeated)
   - **User Impact**: Fallback to local music files
   - **Recovery**: Next search with valid duration may succeed

4. **Download Timeout (>300 seconds)**
   - **Handling**:
     - Catch `TimeoutError` and `aiohttp.ServerTimeoutError`
     - Retry once with exponential backoff (1s, 2s)
     - Return `None` if both attempts fail
   - **User Impact**: Try next fallback (OAuth2 → preview → local)
   - **Recovery**: Smaller files or better network may succeed

5. **Session Closed Error (Connection Lost)**
   - **Handling**:
     - Detect "Session is closed" in exception message
     - Call `get_http_session()` to get new session
     - Retry operation once with new session
   - **User Impact**: Transparent recovery, no visible effect
   - **Recovery**: Operation completes with new session

6. **File Write Failure (Disk Full, Permissions)**
   - **Handling**:
     - Catch exceptions during file write operations
     - Log error with full traceback
     - Return `None`, proceed to next fallback
   - **User Impact**: Skip current track, try different download method or local file
   - **Recovery**: Resolve disk space/permissions, re-run pipeline

7. **Invalid Track Metadata (Missing Preview URLs)**
   - **Handling**:
     - Check `sound.previews` existence before accessing
     - Log warning: "No suitable MP3 preview available"
     - Return `None`, skip track
   - **User Impact**: Try next search result or fallback
   - **Recovery**: Better search filters may return valid tracks

8. **Network Errors (Connection Refused, DNS Failure)**
   - **Handling**:
     - Circuit breaker counts toward failure threshold
     - Log error with exception context
     - After 3 failures, circuit opens (fast-fail for 60s)
   - **User Impact**: Immediate fallback to local files after threshold
   - **Recovery**: Network restoration closes circuit on success

## Testing Strategy

### Unit Testing

**Test File**: `tests/test_freesound_client.py` (to be created/enhanced)

**Key Components to Test**:

1. **OAuth2 Token Management**:
   - Test token refresh with valid credentials
   - Test token expiry detection (within buffer window)
   - Test `.env` file update with `set_key()` mock
   - Test fast-fail on 401/403 auth errors
   - Mock `aiohttp.ClientSession.post()` for token endpoint

2. **Search Functionality**:
   - Test duration filter construction: `f"duration:[{duration} TO {max}]"`
   - Test fallback to general search when no results
   - Test timeout handling with `asyncio.wait_for()` mock
   - Test empty list return on circuit breaker open
   - Mock `freesound.FreesoundClient.text_search()`

3. **Download Methods**:
   - Test OAuth2 download with valid token
   - Test preview download with API key
   - Test filename sanitization
   - Test attribution metadata structure
   - Mock `aiohttp` responses and `download_file()`

4. **Circuit Breaker Integration**:
   - Test decorator application on all public methods
   - Test fast-fail behavior when circuit open
   - Test state transitions (CLOSED → OPEN → HALF_OPEN → CLOSED)
   - Use `freesound_circuit_breaker.reset()` in test setup

5. **Error Handling**:
   - Test retry logic with exponential backoff
   - Test session recovery on "Session is closed" errors
   - Test graceful None return on all failure paths
   - Verify no exceptions propagate to caller

**Mock Strategy**:
```python
import pytest
from aioresponses import aioresponses
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_search_music_duration_filter():
    with aioresponses() as mock_aiohttp:
        mock_freesound = AsyncMock()
        mock_freesound.text_search.return_value = [mock_sound]

        client = FreesoundClient(FREESOUND_API_KEY="test_key")
        client.fs_api_client = mock_freesound

        tracks = await client.search_music(
            query="upbeat",
            filters="duration:[60 TO 180]",
            timeout_sec=10
        )

        assert len(tracks) > 0
        mock_freesound.text_search.assert_called_once()
```

### Integration Testing

**Test File**: `tests/integration/test_freesound_integration.py` (optional, requires API key)

**Key Flows to Test**:

1. **End-to-End Search and Download**:
   - Search for tracks matching 30s duration
   - Download preview with real API key
   - Verify file saved to disk
   - Validate attribution metadata completeness
   - **Requires**: `FREESOUND_API_KEY` in test environment

2. **Token Refresh Flow**:
   - Create client with expired access token
   - Trigger OAuth2 download
   - Verify automatic token refresh
   - Check `.env` file updated (use temp file)
   - **Requires**: OAuth2 credentials in test environment

3. **Fallback Chain**:
   - Simulate OAuth2 failure (invalid credentials)
   - Verify preview download attempted
   - Simulate preview failure
   - Verify local file used as fallback
   - **Requires**: Local music files in `static/`

4. **Producer Integration**:
   - Call `step_download_music()` from producer
   - Verify music file downloaded to correct path
   - Validate attribution saved to `outputs/<ASIN>/attribution.txt`
   - Check performance metrics logged
   - **Requires**: Full pipeline context setup

**Integration Test Strategy**:
```python
@pytest.mark.integration
@pytest.mark.skipif(not has_api_key(), reason="API key required")
async def test_real_api_search_and_download(tmp_path):
    client = FreesoundClient(FREESOUND_API_KEY=os.getenv("FREESOUND_API_KEY"))

    async with aiohttp.ClientSession() as session:
        tracks = await client.search_music(
            query="upbeat instrumental",
            filters="duration:[30 TO 60]",
            max_results=5
        )

        assert len(tracks) > 0, "Search should return results"

        result = await client.download_sound_preview_with_api_key(
            tracks[0], tmp_path, session
        )

        assert result is not None, "Download should succeed"
        file_path, metadata = result
        assert file_path.exists(), "File should be saved"
        assert metadata["license"], "License should be included"
```

### End-to-End Testing

**Test Scenario**: Complete video production with Freesound music

**User Scenarios to Test**:

1. **Happy Path: OAuth2 Success**:
   ```bash
   poetry run python -m src.scraper.amazon.scraper --keywords "B082F13J55" --debug --clean
   poetry run python -m src.video.producer outputs/B082F13J55/data.json slideshow_images1 --debug
   ```
   - **Verify**: Music downloaded via OAuth2 (full quality)
   - **Check**: `outputs/B082F13J55/assets/` contains `.wav` file
   - **Validate**: Video has background music track
   - **Assert**: Attribution in `outputs/B082F13J55/attribution.txt`

2. **Fallback Path: API Key Preview**:
   - **Setup**: Remove OAuth2 credentials from `.env`
   - **Run**: Same production command
   - **Verify**: Music downloaded via API key (preview MP3)
   - **Check**: Log shows "OAuth2 not configured" warning
   - **Assert**: Video generation succeeds with preview quality

3. **Final Fallback: Local Files**:
   - **Setup**: Remove all Freesound credentials
   - **Run**: Same production command
   - **Verify**: Random local file copied from `static/`
   - **Check**: Log shows "Freesound API unavailable" message
   - **Assert**: Video uses local fallback music

4. **Circuit Breaker Activation**:
   - **Setup**: Invalid API credentials (trigger repeated failures)
   - **Run**: Batch processing (3+ products)
   - **Verify**: First product attempts API, circuit opens
   - **Check**: Subsequent products skip API (fast-fail)
   - **Assert**: All videos complete with local music

**E2E Test Validation**:
- Audio track present in final MP4 (use `ffprobe`)
- Volume normalization applied (-20dB)
- Fade in/out effects present
- Attribution file generated
- No pipeline crashes or hung processes
