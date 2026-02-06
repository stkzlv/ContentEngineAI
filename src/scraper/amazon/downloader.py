"""Media download utilities for Amazon scraper.

This module handles downloading of images and videos using Botasaurus tasks
with proper error handling and file management.
"""

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Any

import requests
from botasaurus import bt
from botasaurus.task import task

from .botasaurus_output import get_task_config_for_outputs
from .config import CONFIG
from .download_async import (
    _download_media_async,
    convert_m3u8_to_mp4,  # noqa: F401
    download_file_async,  # noqa: F401
)
from .download_validators import (  # noqa: F401
    _validate_image_size_before_download,
)
from .media_validator import generate_validation_report

logger = logging.getLogger(__name__)

# Get task configuration with custom output function
_task_config = {
    "parallel": max(
        1, bt.calc_max_parallel_browsers() // 2
    ),  # Dynamic calculation - use half for downloads
    "cache": False,  # Disable cache to ensure actual downloads happen
    "max_retry": 3,  # Standard retry count
    "close_on_crash": True,  # Will be updated based on debug mode
}

# Add custom output configuration
_task_config.update(get_task_config_for_outputs())

# NOTE: Botasaurus manages its own output directory structure by design.
# We use custom output functions instead of trying to override the output directory.


# Enhanced task configuration for debugging
_enhanced_task_config = {
    **_task_config,
    "raise_exception": True,  # Raise exception to see actual errors
    "create_error_logs": False,  # Disabled - botasaurus can't relocate to outputs/
    "close_on_crash": False,  # Keep task open on crash for debugging
    "max_retry": 3,  # Reasonable retry count
    # Output handled by custom output function in get_task_config_for_outputs()
}


@task(**_enhanced_task_config)
def download_media_files(data: dict[str, Any]) -> dict[str, Any]:
    """Download product media files (images and videos) using Botasaurus task.

    This function serves as a sync wrapper around the async download logic to maintain
    compatibility with Botasaurus's task decorator system.

    Args:
    ----
        data: Dictionary containing:
            - asin: Product ASIN
            - images: List of image URLs
            - videos: List of video URLs
            - platform: Platform name (default: "amazon")
            - debug_mode: Debug mode flag

    Returns:
    -------
        Dictionary with download results and file paths

    """
    asin = data["asin"]
    image_urls = data.get("images", [])
    video_urls = data.get("videos", [])
    platform = data.get("platform", "amazon")
    debug_mode = data.get("debug_mode", False)
    output_dir = data.get("output_dir")

    if debug_mode:
        logger.info("📥 [MEDIA DOWNLOAD] Starting async download for ASIN: %s", asin)
        logger.info(
            "📥 [MEDIA DOWNLOAD] Images: %d, Videos: %d",
            len(image_urls),
            len(video_urls),
        )

    # Run async download helper
    try:
        # Check if we're already in an async context
        try:
            asyncio.get_running_loop()
            # We're already in an async context, create a task
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(
                        _download_media_async(
                            asin,
                            image_urls,
                            video_urls,
                            platform,
                            debug_mode,
                            output_dir=output_dir,
                        )
                    )
                )
                download_result = future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            download_result = asyncio.run(
                _download_media_async(
                    asin,
                    image_urls,
                    video_urls,
                    platform,
                    debug_mode,
                    output_dir=output_dir,
                )
            )

        downloaded_images = download_result["downloaded_images"]
        downloaded_videos = download_result["downloaded_videos"]
        outputs_root = download_result["outputs_root"]
        product_dir = download_result["product_dir"]

    except Exception as e:
        logger.error("❌ [MEDIA DOWNLOAD] Async download failed: %s", e)
        return {
            "asin": asin,
            "downloaded_images": [],
            "downloaded_videos": [],
            "total_images": 0,
            "total_videos": 0,
            "error": f"Download failed: {e}",
        }

    # Generate validation report if enabled
    validation_report = None
    create_reports = True
    try:
        create_reports = (
            CONFIG.get("global_settings", {})
            .get("debug_settings", {})
            .get("create_media_validation_reports", True)
        )
    except Exception:
        create_reports = True

    if create_reports and debug_mode:
        try:
            all_files = []
            for img_path in downloaded_images:
                all_files.append(outputs_root / img_path)
            for vid_path in downloaded_videos:
                all_files.append(outputs_root / vid_path)

            if all_files:
                from .media_validator import validate_media_batch

                validation_results = validate_media_batch(all_files)

                report_path = product_dir / f"{asin}_media_validation_report.json"
                validation_report = generate_validation_report(
                    validation_results, report_path
                )

                logger.info(
                    "📋 [VALIDATION REPORT] Generated for %d files:", len(all_files)
                )
                logger.info(
                    "   • Valid files: %s", validation_report["summary"]["valid_files"]
                )
                logger.info(
                    "   • Invalid files: %s",
                    validation_report["summary"]["invalid_files"],
                )
                logger.info(
                    "   • Success rate: %.1f%%",
                    validation_report["summary"]["success_rate"],
                )
                logger.info("   • Report saved: %s", report_path)

        except Exception as e:
            logger.warning("⚠️ [VALIDATION REPORT] Failed to generate report: %s", e)

    # Final results summary
    result = {
        "asin": asin,
        "downloaded_images": downloaded_images,
        "downloaded_videos": downloaded_videos,
        "total_images": len(downloaded_images),
        "total_videos": len(downloaded_videos),
        "validation_report": validation_report.get("summary")
        if validation_report
        else None,
    }

    if debug_mode:
        logger.info("📊 [MEDIA DOWNLOAD] Download summary for ASIN %s:", asin)
        logger.info(
            "   • Images: %d/%d downloaded and validated successfully",
            len(downloaded_images),
            len(image_urls),
        )
        logger.info(
            "   • Videos: %d/%d downloaded and validated successfully",
            len(downloaded_videos),
            len(video_urls),
        )
        if downloaded_images:
            logger.info(
                "   • Image files: %s",
                [img.split("/")[-1] for img in downloaded_images[:3]],
            )
        if downloaded_videos:
            logger.info(
                "   • Video files: %s",
                [vid.split("/")[-1] for vid in downloaded_videos[:3]],
            )

    return result


def download_file_sync(
    url: str, file_path: Path, timeout: int | None = None, max_retries: int = 2
) -> bool:
    """Synchronous file download utility using requests with retry logic.

    Args:
    ----
        url: URL to download
        file_path: Path to save the file
        timeout: Request timeout in seconds (default from config)
        max_retries: Maximum number of retry attempts on failure

    Returns:
    -------
        True if successful, False otherwise

    """
    import time

    # Get config values for download
    try:
        download_config = CONFIG.get("global_settings", {}).get("download_config", {})
        amazon_config = CONFIG.get("scrapers", {}).get("amazon", {})
        download_headers = amazon_config.get("http_headers", {}).get(
            "media_download", {}
        )

        default_timeout = download_config.get("download_timeout", 30)
        chunk_size = download_config.get("download_chunk_size", 8192)
    except Exception:
        # Fallback values from config
        default_timeout = (
            CONFIG.get("global_settings", {})
            .get("download_config", {})
            .get("download_timeout", 30)
        )
        chunk_size = (
            CONFIG.get("global_settings", {})
            .get("download_config", {})
            .get("download_chunk_size", 8192)
        )
        download_headers = (
            CONFIG.get("scrapers", {})
            .get("amazon", {})
            .get("http_headers", {})
            .get(
                "media_download",
                {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/125.0.0.0 Safari/537.36"
                    ),
                    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": "https://www.amazon.com/",
                },
            )
        )

    # Use provided timeout or default
    effective_timeout = timeout if timeout is not None else default_timeout

    # Retry loop with exponential backoff
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                logger.debug("Retry attempt %d/%d for: %s", attempt, max_retries, url)
            else:
                logger.debug("Downloading: %s", url)

            response = requests.get(
                url,
                headers=download_headers,
                timeout=effective_timeout,
                stream=True,
            )
            response.raise_for_status()

            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)

            # Verify file was created and has content
            if file_path.exists() and file_path.stat().st_size > 0:
                file_size = file_path.stat().st_size
                logger.debug("Downloaded %d bytes to %s", file_size, file_path.name)
                return True
            else:
                logger.debug("File not created or empty: %s", file_path)
                # Don't retry if file is empty
                return False

        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ) as e:
            # These are transient errors worth retrying
            logger.debug(
                "Transient error on attempt %d/%d: %s", attempt + 1, max_retries + 1, e
            )

            # Clean up partial file
            if file_path.exists():
                with contextlib.suppress(Exception):
                    file_path.unlink()

            # If this was the last attempt, fail
            if attempt >= max_retries:
                logger.debug(
                    "Download failed after %d attempts: %s", max_retries + 1, url
                )
                return False

            # Exponential backoff: 1s, 2s, 4s...
            backoff_time = 2**attempt
            logger.debug("Waiting %ds before retry...", backoff_time)
            time.sleep(backoff_time)

        except Exception as e:
            # Non-transient errors - don't retry
            logger.debug("Download failed for %s: %s", url, e)
            # Clean up partial file
            if file_path.exists():
                with contextlib.suppress(Exception):
                    file_path.unlink()
            return False

    return False
