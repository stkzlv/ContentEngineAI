"""Media download utilities for Amazon scraper.

This module handles downloading of images and videos using Botasaurus tasks
with proper error handling and file management.
"""

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Any

import aiohttp
import requests
from botasaurus import bt
from botasaurus.task import task

from .botasaurus_output import get_task_config_for_outputs
from .config import CONFIG, get_filename_pattern
from .media_validator import (
    generate_validation_report,
    verify_image_file,
    verify_video_file,
)

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


async def convert_m3u8_to_mp4(
    m3u8_url: str, output_path: Path, timeout: int = 120
) -> bool:
    """Convert M3U8 HLS stream to MP4 file using ffmpeg asynchronously.

    Args:
    ----
        m3u8_url: URL of the m3u8 playlist
        output_path: Path where to save the converted MP4
        timeout: Maximum time to wait for conversion (seconds)

    Returns:
    -------
        True if conversion successful, False otherwise

    """
    import asyncio

    logger = logging.getLogger(__name__)

    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # FFmpeg command to download and convert m3u8 to mp4
        cmd = [
            "ffmpeg",
            "-protocol_whitelist",
            "file,http,https,tcp,tls,crypto",
            "-i",
            m3u8_url,
            "-c",
            "copy",  # Copy streams without re-encoding (faster)
            "-bsf:a",
            "aac_adtstoasc",  # Fix audio stream format
            "-y",  # Overwrite output file if exists
            str(output_path),
        ]

        logger.info(f"🎬 Converting m3u8 to mp4: {output_path.name}")
        logger.debug(f"   Command: {' '.join(cmd)}")

        # Run ffmpeg with timeout using async subprocess
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )

            if process.returncode == 0:
                logger.info(f"✅ Successfully converted to MP4: {output_path.name}")
                return True
            else:
                logger.error(
                    f"❌ FFmpeg conversion failed with code {process.returncode}"
                )
                stderr_text = stderr.decode("utf-8", errors="ignore")[:500]
                logger.error(f"   stderr: {stderr_text}")
                return False

        except TimeoutError:
            logger.error(f"❌ FFmpeg conversion timed out after {timeout}s")
            process.kill()
            await process.wait()
            return False

    except FileNotFoundError:
        logger.error("❌ FFmpeg not found. Please install: sudo apt install ffmpeg")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during m3u8 conversion: {e}")
        return False


async def _download_media_async(
    asin: str,
    image_urls: list[str],
    video_urls: list[str],
    platform: str,
    debug_mode: bool,
) -> dict[str, Any]:
    """Async helper function for downloading media files.

    Args:
    ----
        asin: Product ASIN
        image_urls: List of image URLs
        video_urls: List of video URLs
        platform: Platform name
        debug_mode: Debug mode flag

    Returns:
    -------
        Dictionary with download results

    """
    logger = logging.getLogger(__name__)

    # Create aiohttp session for async downloads
    async with aiohttp.ClientSession() as session:
        downloaded_images = []
        downloaded_videos = []

        # Get download configuration
        global_settings = CONFIG.get("global_settings", {})
        download_config = global_settings.get("download_config", {})
        min_image_file_size = download_config.get("min_image_file_size", 10000)

        # Setup output directories
        from .botasaurus_output import get_outputs_root

        outputs_root = get_outputs_root()
        product_dir = outputs_root / asin
        images_dir = product_dir / "images"
        videos_dir = product_dir / "videos"
        images_dir.mkdir(parents=True, exist_ok=True)
        videos_dir.mkdir(parents=True, exist_ok=True)

        # Download images concurrently
        if image_urls:
            if debug_mode:
                logger.info(f"🖼️ [IMAGE DOWNLOAD] Processing {len(image_urls)} images")

            async def download_single_image(i: int, url: str) -> str | None:
                try:
                    if not url or not url.startswith("http"):
                        return None

                    # Validate image before download
                    if not _validate_image_size_before_download(
                        url, min_image_file_size, debug_mode, logger
                    ):
                        return None

                    # Generate filename
                    supported_exts = [".jpg", ".jpeg", ".png", ".webp", ".gif"]
                    default_ext = "jpg"
                    ext = default_ext
                    for extension in supported_exts:
                        if url.endswith(extension):
                            ext = extension.lstrip(".")
                            break

                    filename = get_filename_pattern(
                        "image", asin=asin, index=i, ext=ext
                    )
                    file_path = images_dir / filename

                    # Download file async
                    success = await download_file_async(session, url, file_path)
                    if success:
                        # Validate downloaded file
                        validation_result = verify_image_file(file_path)
                        if validation_result.is_valid:
                            relative_path = str(file_path.relative_to(outputs_root))
                            if debug_mode:
                                file_size = validation_result.validation_data.get(
                                    "actual_file_size", 0
                                )
                                dimensions = (
                                    validation_result.validation_data.get("width", 0),
                                    validation_result.validation_data.get("height", 0),
                                )
                                dim_str = f"{dimensions[0]}x{dimensions[1]}"
                                logger.info(
                                    f"✅ [IMAGE] {filename} "
                                    f"({file_size} bytes, {dim_str})"
                                )
                            return relative_path
                        else:
                            with contextlib.suppress(Exception):
                                file_path.unlink()
                except Exception as e:
                    logger.warning(f"❌ [IMAGE] Failed {i+1}: {e}")
                return None

            # Download images concurrently with semaphore
            download_config = global_settings.get("download_config", {})
            max_concurrent = download_config.get("concurrent_image_downloads", 5)
            semaphore = asyncio.Semaphore(max_concurrent)

            async def download_with_semaphore(i: int, url: str) -> str | None:
                async with semaphore:
                    return await download_single_image(i, url)

            tasks = [
                download_with_semaphore(i, url) for i, url in enumerate(image_urls)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, str):
                    downloaded_images.append(result)

        # Download videos concurrently
        if video_urls:
            if debug_mode:
                m3u8_count = sum(1 for url in video_urls if url and ".m3u8" in url)
                mp4_count = len(video_urls) - m3u8_count
                logger.info(
                    f"🎥 [VIDEO] Processing {len(video_urls)} videos "
                    f"(M3U8: {m3u8_count}, MP4: {mp4_count})"
                )

            async def download_single_video(i: int, url: str) -> str | None:
                try:
                    if not url or not url.startswith("http"):
                        return None

                    filename = get_filename_pattern(
                        "video", asin=asin, index=i, ext="mp4"
                    )
                    file_path = videos_dir / filename

                    # Handle M3U8 streams or direct MP4
                    is_m3u8 = ".m3u8" in url
                    if is_m3u8:
                        video_config = global_settings.get("video_config", {})
                        m3u8_timeout = video_config.get("m3u8_download_timeout", 120)
                        success = await convert_m3u8_to_mp4(
                            url, file_path, timeout=m3u8_timeout
                        )
                    else:
                        download_config = global_settings.get("download_config", {})
                        video_timeout = download_config.get(
                            "video_download_timeout", 300
                        )
                        success = await download_file_async(
                            session, url, file_path, timeout=video_timeout
                        )

                    if success:
                        # Validate downloaded file
                        validation_result = verify_video_file(file_path)
                        if validation_result.is_valid:
                            relative_path = str(file_path.relative_to(outputs_root))
                            if debug_mode:
                                file_size = validation_result.validation_data.get(
                                    "actual_file_size", 0
                                )
                                duration = validation_result.validation_data.get(
                                    "duration", 0
                                )
                                logger.info(
                                    f"✅ [VIDEO] {filename} "
                                    f"({file_size} bytes, {duration}s)"
                                )
                            return relative_path
                        else:
                            with contextlib.suppress(Exception):
                                file_path.unlink()
                except Exception as e:
                    logger.warning(f"❌ [VIDEO] Failed {i+1}: {e}")
                return None

            # Download videos concurrently with semaphore
            download_config = global_settings.get("download_config", {})
            max_concurrent_videos = download_config.get("concurrent_video_downloads", 3)
            semaphore_video = asyncio.Semaphore(max_concurrent_videos)

            async def download_video_with_semaphore(i: int, url: str) -> str | None:
                async with semaphore_video:
                    return await download_single_video(i, url)

            video_tasks = [
                download_video_with_semaphore(i, url)
                for i, url in enumerate(video_urls)
            ]
            video_results = await asyncio.gather(*video_tasks, return_exceptions=True)

            for result in video_results:
                if isinstance(result, str):
                    downloaded_videos.append(result)

        return {
            "downloaded_images": downloaded_images,
            "downloaded_videos": downloaded_videos,
            "outputs_root": outputs_root,
            "product_dir": product_dir,
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
    logger = logging.getLogger(__name__)

    asin = data["asin"]
    image_urls = data.get("images", [])
    video_urls = data.get("videos", [])
    platform = data.get("platform", "amazon")
    debug_mode = data.get("debug_mode", False)

    if debug_mode:
        logger.info(f"📥 [MEDIA DOWNLOAD] Starting async download for ASIN: {asin}")
        logger.info(
            f"📥 [MEDIA DOWNLOAD] Images: {len(image_urls)}, "
            f"Videos: {len(video_urls)}"
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
                            asin, image_urls, video_urls, platform, debug_mode
                        )
                    )
                )
                download_result = future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            download_result = asyncio.run(
                _download_media_async(
                    asin, image_urls, video_urls, platform, debug_mode
                )
            )

        downloaded_images = download_result["downloaded_images"]
        downloaded_videos = download_result["downloaded_videos"]
        outputs_root = download_result["outputs_root"]
        product_dir = download_result["product_dir"]

    except Exception as e:
        logger.error(f"❌ [MEDIA DOWNLOAD] Async download failed: {e}")
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
                    f"📋 [VALIDATION REPORT] Generated for {len(all_files)} files:"
                )
                logger.info(
                    f"   • Valid files: {validation_report['summary']['valid_files']}"
                )
                logger.info(
                    f"   • Invalid files: "
                    f"{validation_report['summary']['invalid_files']}"
                )
                logger.info(
                    f"   • Success rate: "
                    f"{validation_report['summary']['success_rate']:.1f}%"
                )
                logger.info(f"   • Report saved: {report_path}")

        except Exception as e:
            logger.warning(f"⚠️ [VALIDATION REPORT] Failed to generate report: {e}")

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
        logger.info(f"📊 [MEDIA DOWNLOAD] Download summary for ASIN {asin}:")
        logger.info(
            f"   • Images: {len(downloaded_images)}/{len(image_urls)} "
            f"downloaded and validated successfully"
        )
        logger.info(
            f"   • Videos: {len(downloaded_videos)}/{len(video_urls)} "
            f"downloaded and validated successfully"
        )
        if downloaded_images:
            logger.info(
                f"   • Image files: "
                f"{[img.split('/')[-1] for img in downloaded_images[:3]]}"
            )
        if downloaded_videos:
            logger.info(
                f"   • Video files: "
                f"{[vid.split('/')[-1] for vid in downloaded_videos[:3]]}"
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
        timeout: Request timeout in seconds (default: 30s for images, 300s for videos)
        max_retries: Maximum number of retry attempts on failure (default: 2)

    Returns:
    -------
        True if successful, False otherwise

    """
    import time

    # Import DEBUG_MODE from main module
    try:
        from . import scraper

        DEBUG_MODE = scraper.DEBUG_MODE
    except Exception:
        DEBUG_MODE = False

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
            if DEBUG_MODE and attempt > 0:
                logging.getLogger(__name__).debug(
                    f"🔄 Retry attempt {attempt}/{max_retries} for: {url}"
                )
            elif DEBUG_MODE:
                logging.getLogger(__name__).debug(f"📥 Downloading: {url}")

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
                if DEBUG_MODE:
                    file_size = file_path.stat().st_size
                    logging.getLogger(__name__).debug(
                        f"✅ Downloaded {file_size} bytes to {file_path.name}"
                    )
                return True
            else:
                if DEBUG_MODE:
                    logging.getLogger(__name__).warning(
                        f"❌ File not created or empty: {file_path}"
                    )
                # Don't retry if file is empty
                return False

        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ) as e:
            # These are transient errors worth retrying
            if DEBUG_MODE:
                logging.getLogger(__name__).warning(
                    f"⚠️ Transient error on attempt {attempt + 1}/{max_retries + 1}: {e}"
                )

            # Clean up partial file
            if file_path.exists():
                with contextlib.suppress(Exception):
                    file_path.unlink()

            # If this was the last attempt, fail
            if attempt >= max_retries:
                if DEBUG_MODE:
                    logging.getLogger(__name__).error(
                        f"❌ Download failed after {max_retries + 1} attempts: {url}"
                    )
                return False

            # Exponential backoff: 1s, 2s, 4s...
            backoff_time = 2**attempt
            if DEBUG_MODE:
                logging.getLogger(__name__).debug(
                    f"⏳ Waiting {backoff_time}s before retry..."
                )
            time.sleep(backoff_time)

        except Exception as e:
            # Non-transient errors - don't retry
            if DEBUG_MODE:
                logging.getLogger(__name__).error(f"❌ Download failed for {url}: {e}")
            # Clean up partial file
            if file_path.exists():
                with contextlib.suppress(Exception):
                    file_path.unlink()
            return False

    return False


async def download_file_async(
    session: aiohttp.ClientSession,
    url: str,
    file_path: Path,
    timeout: int | None = None,
    max_retries: int = 2,
) -> bool:
    """Asynchronous file download utility using aiohttp with retry logic.

    Args:
    ----
        session: Aiohttp session for downloads
        url: URL to download
        file_path: Path to save the file
        timeout: Request timeout in seconds (default: 30s for images, 300s for videos)
        max_retries: Maximum number of retry attempts on failure (default: 2)

    Returns:
    -------
        True if successful, False otherwise

    """
    # Import DEBUG_MODE from main module
    try:
        from . import scraper

        DEBUG_MODE = scraper.DEBUG_MODE
    except Exception:
        DEBUG_MODE = False

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
            if DEBUG_MODE and attempt > 0:
                logging.getLogger(__name__).debug(
                    f"🔄 Retry attempt {attempt}/{max_retries} for: {url}"
                )
            elif DEBUG_MODE:
                logging.getLogger(__name__).debug(f"📥 Downloading: {url}")

            async with session.get(  # type: ignore[attr-defined]
                url,
                headers=download_headers,
                timeout=aiohttp.ClientTimeout(total=effective_timeout),
            ) as response:
                response.raise_for_status()

                # Ensure parent directory exists
                file_path.parent.mkdir(parents=True, exist_ok=True)

                with open(file_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(chunk_size):
                        if chunk:
                            f.write(chunk)

            # Verify file was created and has content
            if file_path.exists() and file_path.stat().st_size > 0:
                if DEBUG_MODE:
                    file_size = file_path.stat().st_size
                    logging.getLogger(__name__).debug(
                        f"✅ Downloaded {file_size} bytes to {file_path.name}"
                    )
                return True
            else:
                if DEBUG_MODE:
                    logging.getLogger(__name__).warning(
                        f"❌ File not created or empty: {file_path}"
                    )
                return False

        except (TimeoutError, aiohttp.ClientError) as e:
            # These are transient errors worth retrying
            if DEBUG_MODE:
                logging.getLogger(__name__).warning(
                    f"⚠️ Transient error on attempt {attempt + 1}/{max_retries + 1}: {e}"
                )

            # Clean up partial file
            if file_path.exists():
                with contextlib.suppress(Exception):
                    file_path.unlink()

            # If this was the last attempt, fail
            if attempt >= max_retries:
                if DEBUG_MODE:
                    logging.getLogger(__name__).error(
                        f"❌ Download failed after {max_retries + 1} attempts: {url}"
                    )
                return False

            # Exponential backoff: 1s, 2s, 4s...
            backoff_time = 2**attempt
            if DEBUG_MODE:
                logging.getLogger(__name__).debug(
                    f"⏳ Waiting {backoff_time}s before retry..."
                )
            await asyncio.sleep(backoff_time)

        except Exception as e:
            # Non-transient errors - don't retry
            if DEBUG_MODE:
                logging.getLogger(__name__).error(f"❌ Download failed for {url}: {e}")
            # Clean up partial file
            if file_path.exists():
                with contextlib.suppress(Exception):
                    file_path.unlink()
            return False

    return False


def _validate_image_size_before_download(
    url: str, min_file_size: int, debug_mode: bool = False, logger=None
) -> bool:
    """Intelligent image validation via HEAD request before downloading

    Uses multiple criteria to distinguish between thumbnails and product images:
    1. File size threshold (configurable)
    2. URL pattern analysis (Amazon-specific heuristics)
    3. Content-Type verification
    4. Smart fallback for edge cases

    Args:
    ----
        url: Image URL to validate
        min_file_size: Minimum file size in bytes
        debug_mode: Whether to log debug information
        logger: Logger instance

    Returns:
    -------
        True if image meets quality requirements, False otherwise

    """
    try:
        # Get config values for validation
        try:
            download_config = CONFIG.get("global_settings", {}).get(
                "download_config", {}
            )
            amazon_config = CONFIG.get("scrapers", {}).get("amazon", {})
            validation_headers = amazon_config.get("http_headers", {}).get(
                "media_download", {}
            )

            validation_timeout = download_config.get(
                "validation_timeout",
                CONFIG.get("global_settings", {})
                .get("system_timeouts", {})
                .get("head_request_timeout", 10),
            )
        except Exception:
            # Fallback values
            validation_timeout = 10
            try:
                # Try to get user agent from config
                standard_headers = (
                    CONFIG.get("scrapers", {})
                    .get("amazon", {})
                    .get("http_headers", {})
                    .get("standard", {})
                )
                default_ua = (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
                )
                user_agent = standard_headers.get("User-Agent", default_ua)
            except Exception:
                user_agent = (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
                )

            validation_headers = {
                "User-Agent": user_agent,
                "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
            }

        # INTELLIGENT VALIDATION STEP 1: URL Pattern Analysis
        # Amazon thumbnail patterns we want to avoid
        thumbnail_indicators = [
            "._SL75_",
            "._SY75_",
            "._SX75_",  # 75px thumbnails
            "._SL64_",
            "._SY64_",
            "._SX64_",  # 64px thumbnails
            "._SL40_",
            "._SY40_",
            "._SX40_",  # 40px thumbnails
            "._AC_UX60_",
            "._AC_UY60_",  # 60px thumbnails
            "._SS40_",
            "._SS64_",
            "._SS75_",  # Square small thumbnails
        ]

        # Check if URL contains obvious thumbnail indicators
        url_lower = url.lower()
        is_obvious_thumbnail = any(
            indicator.lower() in url_lower for indicator in thumbnail_indicators
        )

        if is_obvious_thumbnail:
            if debug_mode and logger:
                logger.debug(
                    "❌ [SMART-VALIDATION] Obvious thumbnail pattern detected in URL"
                )
            return False

        # INTELLIGENT VALIDATION STEP 2: High-quality indicators
        # Amazon high-quality image patterns
        high_quality_indicators = [
            "._AC_UX522_",
            "._AC_UY522_",  # 522px+ images
            "._SL1000_",
            "._SY1000_",
            "._SX1000_",  # 1000px+ images
            "._SL1500_",
            "._SY1500_",
            "._SX1500_",  # 1500px+ images
            "._AC_UX679_",
            "._AC_UY679_",  # 679px+ images
        ]

        is_high_quality = any(
            indicator.lower() in url_lower for indicator in high_quality_indicators
        )

        # If it's obviously high quality, skip size check
        if is_high_quality:
            if debug_mode and logger:
                logger.info(
                    "✅ [SMART-VALIDATION] High-quality image pattern detected, "
                    "skipping size check"
                )
            return True

        # INTELLIGENT VALIDATION STEP 3: HTTP HEAD Request with smart interpretation
        response = requests.head(
            url,
            headers=validation_headers,
            timeout=validation_timeout,
            allow_redirects=True,
        )

        if response.status_code == 200:
            content_length = response.headers.get("content-length")
            content_type = response.headers.get("content-type", "")

            # Verify it's actually an image
            if content_type and not content_type.startswith("image/"):
                if debug_mode and logger:
                    logger.debug(f"❌ [SMART-VALIDATION] Not an image: {content_type}")
                return False

            if content_length:
                file_size = int(content_length)

                # INTELLIGENT VALIDATION STEP 4: Smart size thresholds
                # Use different thresholds based on image format
                if "webp" in content_type.lower():
                    # WebP is more compressed, use lower threshold
                    effective_min_size = max(min_file_size // 2, 1000)  # At least 1KB
                elif "png" in content_type.lower():
                    # PNG can be larger for same content, be more lenient
                    effective_min_size = min_file_size
                else:
                    # JPEG and others - use standard threshold
                    effective_min_size = min_file_size

                is_valid = file_size >= effective_min_size

                # INTELLIGENT VALIDATION STEP 5: Smart fallback for borderline cases
                if not is_valid and file_size > (effective_min_size * 0.7):
                    # If image is close to threshold (within 70%), check URL for
                    # quality hints
                    quality_hints = [
                        "_SL300_",
                        "_SY300_",
                        "_SX300_",
                        "_AC_UX300_",
                        "_AC_UY300_",
                    ]
                    has_quality_hint = any(
                        hint.lower() in url_lower for hint in quality_hints
                    )

                    if has_quality_hint:
                        if debug_mode and logger:
                            logger.info(
                                f"✅ [SMART-VALIDATION] Borderline size ({file_size} "
                                f"bytes) but quality hint detected"
                            )
                        return True

                if debug_mode and logger:
                    if is_valid:
                        logger.info(
                            f"✅ [SMART-VALIDATION] Image size OK: {file_size} bytes "
                            f"(>= {effective_min_size}, format: {content_type})"
                        )
                    else:
                        logger.debug(
                            f"❌ [SMART-VALIDATION] Image too small: {file_size} bytes "
                            f"(< {effective_min_size}, format: {content_type})"
                        )

                return is_valid
            else:
                # No content-length header - use URL analysis as fallback
                if debug_mode and logger:
                    logger.debug(
                        "⚠️ [SMART-VALIDATION] No content-length header, "
                        "using URL analysis"
                    )
                # Already checked for thumbnail patterns above, so assume valid
                return True
        else:
            if debug_mode and logger:
                logger.debug(
                    f"❌ [SMART-VALIDATION] HTTP {response.status_code} for URL "
                    f"validation"
                )
            return False

    except requests.exceptions.Timeout:
        if debug_mode and logger:
            logger.debug(
                "⏰ [SMART-VALIDATION] Timeout during validation, assuming valid"
            )
        return True  # Assume valid on timeout to avoid missing images
    except Exception as e:
        if debug_mode and logger:
            logger.debug(f"❌ [SMART-VALIDATION] Validation error: {e}, assuming valid")
        return True  # Assume valid on error to avoid missing images
