"""Download validation utilities for Amazon scraper.

This module handles pre-download validation of media URLs using HEAD requests
and URL pattern analysis to filter out thumbnails and invalid images.
"""

import logging

import requests

from .config import CONFIG

logger = logging.getLogger(__name__)


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
                    "❌ [SMART-VALIDATION] Obvious thumbnail pattern "
                    "detected in URL"
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
            indicator.lower() in url_lower
            for indicator in high_quality_indicators
        )

        # If it's obviously high quality, skip size check
        if is_high_quality:
            if debug_mode and logger:
                logger.info(
                    "✅ [SMART-VALIDATION] High-quality image pattern "
                    "detected, skipping size check"
                )
            return True

        # INTELLIGENT VALIDATION STEP 3: HTTP HEAD Request
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
                    logger.debug(
                        f"❌ [SMART-VALIDATION] Not an image: {content_type}"
                    )
                return False

            if content_length:
                file_size = int(content_length)

                # INTELLIGENT VALIDATION STEP 4: Smart size thresholds
                # Use different thresholds based on image format
                if "webp" in content_type.lower():
                    # WebP is more compressed, use lower threshold
                    effective_min_size = max(
                        min_file_size // 2, 1000
                    )  # At least 1KB
                elif "png" in content_type.lower():
                    # PNG can be larger, be more lenient
                    effective_min_size = min_file_size
                else:
                    # JPEG and others - use standard threshold
                    effective_min_size = min_file_size

                is_valid = file_size >= effective_min_size

                # STEP 5: Smart fallback for borderline cases
                if not is_valid and file_size > (effective_min_size * 0.7):
                    # If image is close to threshold (within 70%),
                    # check URL for quality hints
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
                                f"✅ [SMART-VALIDATION] Borderline size "
                                f"({file_size} bytes) but quality hint "
                                f"detected"
                            )
                        return True

                if debug_mode and logger:
                    if is_valid:
                        logger.info(
                            f"✅ [SMART-VALIDATION] Image size OK: "
                            f"{file_size} bytes "
                            f"(>= {effective_min_size}, "
                            f"format: {content_type})"
                        )
                    else:
                        logger.debug(
                            f"❌ [SMART-VALIDATION] Image too small: "
                            f"{file_size} bytes "
                            f"(< {effective_min_size}, "
                            f"format: {content_type})"
                        )

                return is_valid
            else:
                # No content-length header - use URL analysis as fallback
                if debug_mode and logger:
                    logger.debug(
                        "⚠️ [SMART-VALIDATION] No content-length header, "
                        "using URL analysis"
                    )
                # Already checked for thumbnail patterns, so assume valid
                return True
        else:
            if debug_mode and logger:
                logger.debug(
                    f"❌ [SMART-VALIDATION] HTTP {response.status_code} "
                    f"for URL validation"
                )
            return False

    except requests.exceptions.Timeout:
        if debug_mode and logger:
            logger.debug(
                "⏰ [SMART-VALIDATION] Timeout during validation, "
                "assuming valid"
            )
        return True  # Assume valid on timeout to avoid missing images
    except Exception as e:
        if debug_mode and logger:
            logger.debug(
                f"❌ [SMART-VALIDATION] Validation error: {e}, "
                f"assuming valid"
            )
        return True  # Assume valid on error to avoid missing images
