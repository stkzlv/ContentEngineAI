# Scraper Async Downloader Refactoring (Deferred)

**Goal:** Unify the downloader logic, eliminate synchronous I/O in the Amazon scraper, and improve performance and maintainability.

## Current State

- `src/scraper/base/downloader.py`: Has both `download_file_sync` and `download_file_async` methods
- `src/scraper/base/utils.py`: Contains `exponential_backoff_retry` decorator (lines 21-50)
- `src/scraper/amazon/downloader.py`: Uses sync `download_file_sync` (906 lines, calls at lines 270, 371)
- `convert_m3u8_to_mp4`: Uses blocking `subprocess.run` (line 94)

## Implementation Steps

1.  **~~Create a Shared Async Download Utility~~** (NOT NEEDED)
    *   The `exponential_backoff_retry` decorator is already centralized in `src/scraper/base/utils.py`
    *   The `download_file_async` method already exists in `BaseDownloader`
    *   No need to create new file - refactor existing structure

2.  **Refactor `src/scraper/base/downloader.py`:**
    *   Mark `download_file_sync` as deprecated with warning
    *   Add deprecation docstring and runtime warning
    *   Eventually remove after Amazon downloader migration

3.  **Refactor `src/scraper/amazon/downloader.py` (906 lines):**
    *   Convert `download_media_files` function to `async def`
    *   Replace `download_file_sync` calls (lines 270, 371) with `BaseDownloader.download_file_async`
    *   Manage `aiohttp.ClientSession` lifecycle properly
    *   Adapt `@task` decorator to handle async function (use `asyncio.run()` wrapper)
    *   Refactor `convert_m3u8_to_mp4` to use `asyncio.create_subprocess_exec` (line 54-113)
    *   Ensure `_validate_image_size_before_download` and validators work in async context

4.  **Testing:**
    *   Run existing scraper tests (check `tests/scraper/` directory)
    *   Add async-specific tests if needed
    *   Verify no regressions in media download functionality

## Risk Analysis

| Work Item | Risk | Notes |
|-----------|------|-------|
| Async Downloader | 🟡 Medium | Botasaurus integration complexity |

## Implementation Timeline

**Estimated**: 3-5 days
**Status**: Deferred
**Risk**: Botasaurus integration unknowns
