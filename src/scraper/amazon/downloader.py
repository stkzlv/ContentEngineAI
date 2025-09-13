"""Media download utilities for Amazon scraper.

This module handles downloading of images and videos using Botasaurus tasks
with proper error handling and file management.
"""

import contextlib
import logging
from pathlib import Path as PathLib
from typing import Any

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
    "create_error_logs": True,  # Enable error logging for debugging
    "close_on_crash": False,  # Keep task open on crash for debugging
    "max_retry": 3,  # Reasonable retry count
    # Output handled by custom output function in get_task_config_for_outputs()
}


@task(**_enhanced_task_config)
def download_media_files(data: dict[str, Any]) -> dict[str, Any]:
    """Download product media files (images and videos) using Botasaurus task

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

    # Get debug mode from data parameter to avoid circular imports
    DEBUG_MODE = data.get("debug_mode", False)

    if DEBUG_MODE:
        logger.info(f"📥 [MEDIA DOWNLOAD] Starting media download for ASIN: {asin}")
        logger.info(f"📥 [MEDIA DOWNLOAD] Images to download: {len(image_urls)}")
        logger.info(f"📥 [MEDIA DOWNLOAD] Videos to download: {len(video_urls)}")
        logger.info(f"📥 [MEDIA DOWNLOAD] Platform: {platform}")

        if image_urls:
            logger.info("📥 [MEDIA DOWNLOAD] Image URLs:")
            for i, url in enumerate(image_urls[:3]):  # Show first 3
                logger.info(f"   {i+1}. {url[:80]}...")
            if len(image_urls) > 3:
                logger.info(f"   ... and {len(image_urls) - 3} more images")

        if video_urls:
            logger.info("📥 [MEDIA DOWNLOAD] Video URLs:")
            for i, url in enumerate(video_urls[:3]):  # Show first 3
                logger.info(f"   {i+1}. {url[:80]}...")
            if len(video_urls) > 3:
                logger.info(f"   ... and {len(video_urls) - 3} more videos")

    # Create directories using simplified product-oriented structure
    # Use product-centric directory structure for images and videos
    from ...utils.outputs_paths import (
        get_product_directory,
        get_product_images_directory,
        get_product_videos_directory,
    )

    product_dir = get_product_directory(asin)
    images_dir = get_product_images_directory(asin)
    videos_dir = get_product_videos_directory(asin)

    if DEBUG_MODE:
        logger.info("📁 [MEDIA DOWNLOAD] Creating directories:")
        logger.info(f"   • Product dir: {product_dir}")
        logger.info(f"   • Images dir: {images_dir}")
        logger.info(f"   • Videos dir: {videos_dir}")

    try:
        # Directories are already created by the utility functions
        pass
        if DEBUG_MODE:
            logger.info("✅ [MEDIA DOWNLOAD] Directories created successfully")
    except Exception as e:
        logger.error(f"❌ [MEDIA DOWNLOAD] Failed to create directories: {e}")
        return {
            "asin": asin,
            "downloaded_images": [],
            "downloaded_videos": [],
            "total_images": 0,
            "total_videos": 0,
            "error": f"Directory creation failed: {e}",
        }

    downloaded_images = []
    downloaded_videos = []

    # Download images with pre-validation
    if DEBUG_MODE:
        logger.info(
            f"🖼️ [IMAGE DOWNLOAD] Starting image downloads with pre-validation for "
            f"ASIN: {asin}"
        )

    # Get minimum file size threshold for high-res images (from config)
    min_file_size = (
        CONFIG.get("global_settings", {})
        .get("image_config", {})
        .get("min_high_res_file_size", 10000)  # 10KB minimum for high-res images
    )

    for i, url in enumerate(image_urls):
        try:
            if not url or not url.startswith("http"):
                if DEBUG_MODE:
                    logger.warning(
                        f"🖼️ [IMAGE DOWNLOAD] Skipping invalid URL {i+1}: {url}"
                    )
                continue

            # Pre-download validation: check file size via HEAD request
            if DEBUG_MODE:
                logger.info(
                    f"🖼️ [PRE-VALIDATION] Checking image {i+1}/{len(image_urls)}: "
                    f"{url[:80]}..."
                )

            if not _validate_image_size_before_download(
                url, min_file_size, DEBUG_MODE, logger
            ):
                if DEBUG_MODE:
                    logger.warning(
                        f"⏭️ [SMART-VALIDATION] Skip image {i+1}: validation failed"
                    )
                continue

            # Generate filename using configurable pattern - get extensions from config
            supported_exts = (
                CONFIG.get("global_settings", {})
                .get("media_config", {})
                .get("supported_image_extensions", [".jpg", ".jpeg", ".png", ".webp"])
            )
            default_ext = (
                CONFIG.get("global_settings", {})
                .get("media_config", {})
                .get("default_image_extension", ".jpg")
                .lstrip(".")
            )

            ext = default_ext
            for extension in supported_exts:
                if url.endswith(extension):
                    ext = extension.lstrip(".")
                    break

            filename = get_filename_pattern("image", asin=asin, index=i, ext=ext)
            file_path = images_dir / filename

            if DEBUG_MODE:
                logger.info(
                    f"🖼️ [IMAGE DOWNLOAD] Downloading validated image {i+1}/"
                    f"{len(image_urls)}: {filename}"
                )
                logger.info(f"   • URL: {url[:100]}...")
                logger.info(f"   • Path: {file_path}")

            # Use download_file_sync for file download
            success = download_file_sync(url, file_path)
            if success:
                # Post-download validation using comprehensive validator
                if DEBUG_MODE:
                    logger.info(
                        f"🔍 [VALIDATION] Validating downloaded image: {filename}"
                    )

                validation_result = verify_image_file(file_path)

                if validation_result.is_valid:
                    # Store relative path from outputs root for simplified structure
                    from ..amazon.botasaurus_output import get_outputs_root

                    outputs_root = get_outputs_root()
                    relative_path = str(file_path.relative_to(outputs_root))
                    downloaded_images.append(relative_path)

                    if DEBUG_MODE:
                        file_size = validation_result.validation_data.get(
                            "actual_file_size", 0
                        )
                        dimensions = (
                            validation_result.validation_data.get("width", 0),
                            validation_result.validation_data.get("height", 0),
                        )
                        logger.info(
                            f"✅ [VALIDATION] Image validation passed: {filename} "
                            f"({file_size} bytes, {dimensions[0]}x{dimensions[1]})"
                        )
                else:
                    # Remove invalid file
                    with contextlib.suppress(Exception):
                        file_path.unlink()

                    if DEBUG_MODE:
                        issues = ", ".join(
                            validation_result.issues[:3]
                        )  # Show first 3 issues
                        logger.warning(
                            f"❌ [VALIDATION] Failed: {filename} - {issues}"
                        )
            else:
                if DEBUG_MODE:
                    logger.warning(f"❌ [IMAGE DOWNLOAD] Failed to download {filename}")

        except Exception as e:
            logger.warning(
                f"❌ [IMAGE DOWNLOAD] Exception downloading image {i+1} from "
                f"{url[:50]}...: {e}"
            )
            if DEBUG_MODE:
                import traceback

                logger.debug(f"   • Full traceback: {traceback.format_exc()}")
            continue

    # Filter out M3U8 playlist files (they're not actual videos)
    filtered_video_urls = []
    m3u8_count = 0

    for url in video_urls:
        if url and ".m3u8" in url:
            m3u8_count += 1
            if DEBUG_MODE:
                logger.debug(
                    f"🎥 [VIDEO FILTER] Skipping M3U8 playlist file: {url[:80]}..."
                )
        else:
            filtered_video_urls.append(url)

    if DEBUG_MODE:
        logger.info(f"🎥 [VIDEO DOWNLOAD] Starting video downloads for ASIN: {asin}")
        logger.info(f"🎥 [VIDEO FILTER] Filtered out {m3u8_count} M3U8 playlist files")
        logger.info(
            f"🎥 [VIDEO FILTER] Downloading {len(filtered_video_urls)} "
            f"actual video files"
        )

    for i, url in enumerate(filtered_video_urls):
        try:
            if not url or not url.startswith("http"):
                if DEBUG_MODE:
                    logger.warning(
                        f"🎥 [VIDEO DOWNLOAD] Skipping invalid URL {i+1}: {url}"
                    )
                continue

            # Generate filename using configurable pattern
            filename = get_filename_pattern("video", asin=asin, index=i, ext="mp4")
            file_path = videos_dir / filename

            if DEBUG_MODE:
                logger.info(
                    f"🎥 [VIDEO DOWNLOAD] Downloading video "
                    f"{i+1}/{len(filtered_video_urls)}: {filename}"
                )
                logger.info(f"   • URL: {url[:100]}...")
                logger.info(f"   • Path: {file_path}")

            success = download_file_sync(url, file_path)
            if success:
                # Post-download validation using comprehensive validator
                if DEBUG_MODE:
                    logger.info(
                        f"🔍 [VALIDATION] Validating downloaded video: {filename}"
                    )

                validation_result = verify_video_file(file_path)

                if validation_result.is_valid:
                    # Store relative path from outputs root for simplified structure
                    from ..amazon.botasaurus_output import get_outputs_root

                    outputs_root = get_outputs_root()
                    relative_path = str(file_path.relative_to(outputs_root))
                    downloaded_videos.append(relative_path)

                    if DEBUG_MODE:
                        file_size = validation_result.validation_data.get(
                            "actual_file_size", 0
                        )
                        duration = validation_result.validation_data.get("duration", 0)
                        dimensions = (
                            validation_result.validation_data.get("width", 0),
                            validation_result.validation_data.get("height", 0),
                        )
                        logger.info(
                            f"✅ [VALIDATION] Passed: {filename} "
                            f"({file_size//1024}KB,{duration:.1f}s,{dimensions[0]}x{dimensions[1]})"
                        )
                else:
                    # Remove invalid file
                    with contextlib.suppress(Exception):
                        file_path.unlink()

                    if DEBUG_MODE:
                        issues = ", ".join(
                            validation_result.issues[:3]
                        )  # Show first 3 issues
                        logger.warning(
                            f"❌ [VALIDATION] Video failed: {filename} - {issues}"
                        )
            else:
                if DEBUG_MODE:
                    logger.warning(f"❌ [VIDEO DOWNLOAD] Failed to download {filename}")

        except Exception as e:
            logger.warning(
                f"❌ [VIDEO DOWNLOAD] Exception downloading video {i+1} "
                f"from {url[:50]}...: {e}"
            )
            if DEBUG_MODE:
                import traceback

                logger.debug(f"   • Full traceback: {traceback.format_exc()}")
            continue

    # Generate final validation report based on configuration
    validation_report = None

    # Check if validation reports should be created
    create_reports = True
    try:
        create_reports = (
            CONFIG.get("global_settings", {})
            .get("debug_settings", {})
            .get("create_media_validation_reports", True)
        )
    except Exception:
        create_reports = True

    if create_reports and DEBUG_MODE:
        try:
            # Collect all downloaded files for final validation report
            from ..amazon.botasaurus_output import get_outputs_root

            outputs_root = get_outputs_root()

            all_files = []
            for img_path in downloaded_images:
                all_files.append(outputs_root / img_path)
            for vid_path in downloaded_videos:
                all_files.append(outputs_root / vid_path)

            if all_files:
                # Generate validation report for all downloaded files
                from .media_validator import validate_media_batch

                validation_results = validate_media_batch(all_files)

                # Save validation report to product directory
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

    if DEBUG_MODE:
        logger.info(f"📊 [MEDIA DOWNLOAD] Download summary for ASIN {asin}:")
        logger.info(
            f"   • Images: {len(downloaded_images)}/{len(image_urls)} "
            f"downloaded and validated successfully"
        )
        logger.info(
            f"   • Videos: {len(downloaded_videos)}/{len(filtered_video_urls)} "
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


def download_file_sync(url: str, file_path: PathLib) -> bool:
    """Synchronous file download utility using requests

    Args:
    ----
        url: URL to download
        file_path: Path to save the file

    Returns:
    -------
        True if successful, False otherwise

    """
    try:
        # Import DEBUG_MODE from main module
        try:
            from . import scraper

            DEBUG_MODE = scraper.DEBUG_MODE
        except Exception:
            DEBUG_MODE = False

        # Get config values for download
        try:
            download_config = CONFIG.get("global_settings", {}).get(
                "download_config", {}
            )
            amazon_config = CONFIG.get("scrapers", {}).get("amazon", {})
            download_headers = amazon_config.get("http_headers", {}).get(
                "media_download", {}
            )

            download_timeout = download_config.get(
                "download_timeout",
                CONFIG.get("global_settings", {})
                .get("download_config", {})
                .get("download_timeout", 30),
            )
            chunk_size = download_config.get("download_chunk_size", 8192)
        except Exception:
            # Fallback values from config
            download_timeout = (
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

        if DEBUG_MODE:
            logging.getLogger(__name__).debug(f"📥 Downloading: {url}")

        response = requests.get(
            url, headers=download_headers, timeout=download_timeout, stream=True
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
            return False

    except Exception as e:
        if DEBUG_MODE:
            logging.getLogger(__name__).error(f"❌ Download failed for {url}: {e}")
        # Clean up partial file
        if file_path.exists():
            with contextlib.suppress(Exception):
                file_path.unlink()
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
