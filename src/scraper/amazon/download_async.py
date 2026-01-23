"""Async media download utilities for Amazon scraper.

This module handles asynchronous downloading of images and videos using aiohttp,
including HLS stream conversion and concurrent download orchestration.
"""

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Any

import aiohttp

from .config import CONFIG, get_filename_pattern
from .download_validators import _validate_image_size_before_download
from .media_validator import verify_image_file, verify_video_file

logger = logging.getLogger(__name__)


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

        logger.info("🎬 Converting m3u8 to mp4: %s", output_path.name)
        logger.debug("   Command: %s", " ".join(cmd))

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
                logger.info("✅ Successfully converted to MP4: %s", output_path.name)
                return True
            else:
                logger.error(
                    "❌ FFmpeg conversion failed with code %s", process.returncode
                )
                stderr_text = stderr.decode("utf-8", errors="ignore")[:500]
                logger.error("   stderr: %s", stderr_text)
                return False

        except TimeoutError:
            logger.error("❌ FFmpeg conversion timed out after %ds", timeout)
            process.kill()
            await process.wait()
            return False

    except FileNotFoundError:
        logger.error("❌ FFmpeg not found. Please install: sudo apt install ffmpeg")
        return False
    except Exception as e:
        logger.error("❌ Unexpected error during m3u8 conversion: %s", e)
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
        timeout: Request timeout in seconds (default from config)
        max_retries: Maximum number of retry attempts on failure

    Returns:
    -------
        True if successful, False otherwise

    """
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
                file_size = file_path.stat().st_size
                logger.debug("Downloaded %d bytes to %s", file_size, file_path.name)
                return True
            else:
                logger.debug("File not created or empty: %s", file_path)
                return False

        except (TimeoutError, aiohttp.ClientError) as e:
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
            await asyncio.sleep(backoff_time)

        except Exception as e:
            # Non-transient errors - don't retry
            logger.debug("Download failed for %s: %s", url, e)
            # Clean up partial file
            if file_path.exists():
                with contextlib.suppress(Exception):
                    file_path.unlink()
            return False

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
                logger.info("🖼️ [IMAGE DOWNLOAD] Processing %d images", len(image_urls))

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
                                    "✅ [IMAGE] %s (%s bytes, %s)",
                                    filename,
                                    file_size,
                                    dim_str,
                                )
                            return relative_path
                        else:
                            with contextlib.suppress(Exception):
                                file_path.unlink()
                except Exception as e:
                    logger.warning("❌ [IMAGE] Failed %d: %s", i + 1, e)
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
                    "🎥 [VIDEO] Processing %d videos (M3U8: %d, MP4: %d)",
                    len(video_urls),
                    m3u8_count,
                    mp4_count,
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
                                    "✅ [VIDEO] %s (%s bytes, %ss)",
                                    filename,
                                    file_size,
                                    duration,
                                )
                            return relative_path
                        else:
                            with contextlib.suppress(Exception):
                                file_path.unlink()
                except Exception as e:
                    logger.warning("❌ [VIDEO] Failed %d: %s", i + 1, e)
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
