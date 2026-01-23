"""Image and video URL utility functions for Amazon scraper."""

import logging
import re

import requests

from .config import CONFIG, get_config_value
from .constants import (
    HIGH_RES_DIMENSION,
    HIGH_RES_UPGRADE_DIMENSION,
    VERY_HIGH_RES_DIMENSION,
)

logger = logging.getLogger(__name__)


def modify_amazon_image_for_high_res(url: str) -> str:
    """Convert Amazon image URL to high-resolution version"""
    if not url or not isinstance(url, str):
        return url

    # Replace small size indicators with large ones
    # Pattern for _SL{size}_, _SX{size}_, _SY{size}_, etc.
    size_pattern = r"\._(?:AC_)?(SL|SX|SY)(\d+)_\."
    match = re.search(size_pattern, url)

    if match:
        # Replace with high-res version from config
        try:
            high_res_dimension = (
                CONFIG.get("global_settings", {})
                .get("media_config", {})
                .get("high_res_upgrade_dimension", HIGH_RES_UPGRADE_DIMENSION)
            )
        except Exception:
            high_res_dimension = HIGH_RES_UPGRADE_DIMENSION

        new_url = re.sub(
            size_pattern, f"._AC_{match.group(1)}{high_res_dimension}_.", url
        )
        return new_url

    # If no size pattern found, try adding high-res suffix
    # Get high-res suffix from config
    high_res_suffix = (
        CONFIG.get("global_settings", {})
        .get("media_config", {})
        .get("amazon_high_res_suffix", "._AC_SL2000_.jpg")
    )
    default_ext = (
        CONFIG.get("global_settings", {})
        .get("media_config", {})
        .get("default_image_extension", ".jpg")
    )

    if "._" in url and url.endswith(default_ext):
        return url.replace(default_ext, high_res_suffix)

    return url


def is_amazon_product_image(url: str) -> bool:
    """Check if URL appears to be an Amazon product image"""
    if not url or not isinstance(url, str):
        return False

    # Amazon image domain patterns
    # Get Amazon domains from config
    amazon_domains = (
        CONFIG.get("global_settings", {})
        .get("media_config", {})
        .get("amazon_media_domains", ["images-amazon.com", "m.media-amazon.com"])
    )

    # Check for Amazon image domains
    return any(domain in url for domain in amazon_domains)


def is_valid_high_res_image(url: str) -> bool:
    """Check if image URL is valid and high-resolution"""
    if not url or not isinstance(url, str):
        return False

    # Must be HTTP URL
    if not url.startswith("http"):
        return False

    # Get high-res threshold from config
    high_res_threshold = int(
        get_config_value(
            "global_settings",
            "image_config",
            "min_high_res_dimension",
            default=HIGH_RES_DIMENSION,
        )
    )

    # Check for Amazon high-res patterns
    # Look for size indicators
    size_match = re.search(r"\._(?:AC_)?(SL|SX|SY)(\d+)_", url)
    if size_match:
        size = int(size_match.group(2))
        return size >= high_res_threshold

    # If no specific size pattern, assume it's valid if from Amazon
    return is_amazon_product_image(url)


def is_valid_video_url(url: str) -> bool:
    """Check if URL appears to be a valid video URL"""
    if not url or not isinstance(url, str):
        return False

    # Must be HTTP URL
    if not url.startswith("http"):
        return False

    # Exclude VDP (Video Detail Page) links as they are navigation URLs, not direct
    # video files
    if "/vdp/" in url:
        return False

    # Check for video file extensions or streaming formats
    video_patterns = [
        r"\.mp4(\?|$)",
        r"\.m3u8(\?|$)",
        r"\.webm(\?|$)",
        r"\.mov(\?|$)",
        r"default\.mp4",
        r"media-amazon\.com.*\/.*\.mp4",
        r"vse-vms-transcoding",
    ]

    return any(re.search(pattern, url, re.IGNORECASE) for pattern in video_patterns)


def validate_video_url_accessibility(url: str) -> bool:
    """Validate that a video URL is accessible with proper rate limiting and
    error handling
    """
    if not url or not isinstance(url, str):
        return False

    # Skip problematic URL types early
    if url.startswith(("blob:", "data:")) or not url.startswith("http"):
        logger.debug("Skipping invalid URL type: %s...", url[:50])
        return False

    try:
        import random
        import time

        # Get config values for delays and headers
        try:
            rate_config = CONFIG.get("global_settings", {}).get("rate_limiting", {})
            download_config = CONFIG.get("global_settings", {}).get(
                "download_config", {}
            )
            amazon_config = CONFIG.get("scrapers", {}).get("amazon", {})
            video_headers = amazon_config.get("http_headers", {}).get(
                "video_validation", {}
            )

            delay_range = rate_config.get("video_validation_delay", [0.5, 1.5])
            validation_timeout = download_config.get(
                "validation_timeout",
                CONFIG.get("global_settings", {})
                .get("system_timeouts", {})
                .get("head_request_timeout", 10),
            )
            range_bytes = download_config.get("validation_range_bytes", "0-1023")
        except Exception:
            # Fallback values
            delay_range = [0.5, 1.5]
            validation_timeout = (
                CONFIG.get("global_settings", {})
                .get("system_timeouts", {})
                .get("head_request_timeout", 10)
            )
            range_bytes = "0-1023"
            # Use fallback headers from config
            video_headers = (
                CONFIG.get("scrapers", {})
                .get("amazon", {})
                .get("http_headers", {})
                .get(
                    "video_validation",
                    {
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/120.0.0.0 Safari/537.36"
                        ),
                        "Accept": (
                            "video/webm,video/ogg,video/*;q=0.9,"
                            "application/ogg;q=0.7,audio/*;q=0.6,*/*;q=0.5"
                        ),
                        "Accept-Language": "en-US,en;q=0.9",
                        "Accept-Encoding": "identity",
                        "Referer": "https://www.amazon.com/",
                    },
                )
            )

        # Add random delay to prevent rate limiting
        delay = random.uniform(delay_range[0], delay_range[1])  # noqa: S311
        time.sleep(delay)

        # Enhanced headers to appear more like a real browser
        headers = video_headers.copy()
        headers["Range"] = (
            f"bytes={range_bytes}"  # Only request first portion to minimize bandwidth
        )

        # Make HEAD request with enhanced error handling
        response = requests.head(
            url, timeout=validation_timeout, allow_redirects=True, headers=headers
        )
        is_accessible = response.status_code < 400

        status_msg = f"HTTP {response.status_code}"
        content_type = response.headers.get("content-type", "unknown")
        if is_accessible:
            logger.debug(
                "Video URL accessible (%s, %s): %s...",
                status_msg,
                content_type,
                url[:60],
            )
        else:
            logger.debug("Video URL failed (%s): %s...", status_msg, url[:60])

        return is_accessible

    except requests.exceptions.Timeout:
        logger.debug("Video URL timeout: %s...", url[:60])
        return False
    except requests.exceptions.RequestException as e:
        logger.debug("Video URL request failed (%s): %s...", type(e).__name__, url[:60])
        return False
    except Exception as e:
        logger.debug("Video URL validation error (%s): %s...", e, url[:60])
        return False


def check_amazon_high_res_pattern(url: str, min_sl_size: int = None) -> bool:
    """Check if an image URL matches Amazon's high-resolution pattern"""
    if not url or not isinstance(url, str):
        return False

    # Get min_sl_size from config if not provided
    if min_sl_size is None:
        min_sl_size = get_config_value(
            "global_settings",
            "image_config",
            "min_high_res_dimension",
            default=HIGH_RES_DIMENSION,
        )

    # Check for Amazon high-res pattern ._SL{size}_
    match = re.search(r"\._(?:AC_)?(SL|SX|SY)([1-9]\d{2,})_", url)
    if match:
        try:
            size = int(match.group(2))
            return size >= min_sl_size
        except (ValueError, IndexError):
            logger.debug("Regex matched but size extraction failed for URL: %s", url)
    return False


def filter_amazon_fallback_image(url: str, min_sl_size: int = None) -> bool:
    """Filter Amazon fallback images based on quality checks (matching old scraper)

    Args:
    ----
        url: Image URL to check
        min_sl_size: Minimum size threshold

    Returns:
    -------
        True if the URL should be kept, False if it should be filtered out

    """
    if not url or not isinstance(url, str) or not url.lower().startswith("http"):
        return False

    # Get min_sl_size from config if not provided
    if min_sl_size is None:
        min_sl_size = get_config_value(
            "global_settings",
            "image_config",
            "min_high_res_dimension",
            default=HIGH_RES_DIMENSION,
        )

    if is_placeholder_image(url):
        return False

    if check_amazon_high_res_pattern(url, min_sl_size):
        return True

    # Exclude low-res patterns
    if re.search(r"\._(?:S[XYR]|UX|US|AC)\d{1,3}[_,.]", url):
        logger.debug("Excluding low-res pattern image: %s", url)
        return False

    # Must be a valid image extension
    if not re.search(r"\.(jpg|jpeg|png|webp)$", url, re.IGNORECASE):
        logger.debug("Excluding non-image file extension: %s", url)
        return False

    logger.debug("Keeping filtered fallback image: %s", url)
    return True


def is_placeholder_image(url: str) -> bool:
    """Check if an image URL is a placeholder or low-quality image

    Args:
    ----
        url: Image URL to check

    Returns:
    -------
        True if the URL appears to be a placeholder

    """
    if not url or not isinstance(url, str):
        return True

    url_lower = url.lower()

    # Check for common placeholder patterns (matching old scraper exactly)
    placeholder_patterns = [
        "pixel",
        "spinner",
        "loading",
        "grey-pixel",
        "adsystem",
        "transparent",
        "csgid=",
    ]

    if any(pattern in url_lower for pattern in placeholder_patterns):
        return True

    if "placeholder.com" in url_lower or "placehold.it" in url_lower:
        return True

    # Check for small thumbnail patterns
    if re.search(r"\._(?:SS|SR|SX|SY)\d{2,3}_", url):
        return True

    # Check for 1x1 pixel images (note: double backslash in old scraper)
    return bool(re.search("1x1\\.(png|gif|jpg)", url_lower))


def _validate_image_dimensions(
    url: str, min_dimension: int, debug_mode: bool = False, logger=None
) -> bool:
    """Validate that an image meets minimum dimension requirements.

    Since Amazon URL patterns like _SL1500_ don't guarantee actual dimensions,
    this function performs actual HTTP requests to verify image size.

    Args:
    ----
        url: Image URL to validate
        min_dimension: Minimum required dimension (width or height)
        debug_mode: Whether to log debug information
        logger: Logger instance for debug output

    Returns:
    -------
        True if image meets dimension requirements, False otherwise

    """
    import io

    from PIL import Image

    try:
        # Quick URL pattern check first for obvious cases
        size_match = re.search(r"\._(?:AC_)?(SL|SX|SY)(\d+)_", url)
        if size_match:
            size = int(size_match.group(2))
            # If URL pattern indicates very high resolution, trust it
            try:
                very_high_res_threshold = (
                    CONFIG.get("global_settings", {})
                    .get("image_config", {})
                    .get("very_high_res_dimension", VERY_HIGH_RES_DIMENSION)
                )
            except Exception:
                very_high_res_threshold = VERY_HIGH_RES_DIMENSION

            if size >= very_high_res_threshold:
                if debug_mode and logger:
                    logger.debug(
                        "URL pattern indicates very high-res (%dpx), trusting: %s...",
                        size,
                        url[:80],
                    )
                return True

        # For other cases, check actual dimensions by downloading image headers
        if debug_mode and logger:
            logger.debug("Checking actual dimensions for: %s...", url[:80])

        # Get headers from config for realistic requests
        headers = (
            CONFIG.get("scrapers", {})
            .get("amazon", {})
            .get("http_headers", {})
            .get(
                "media_download",
                {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
                    ),
                    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
                },
            )
        )

        timeout = (
            CONFIG.get("global_settings", {})
            .get("system_timeouts", {})
            .get("head_request_timeout", 10)
        )

        # Download first 4KB to get image headers and determine dimensions
        range_headers = headers.copy()
        range_headers["Range"] = "bytes=0-4095"

        response = requests.get(url, headers=range_headers, timeout=timeout)
        if response.status_code in [200, 206]:  # 206 = Partial Content
            try:
                # Try to get dimensions from partial image data
                img = Image.open(io.BytesIO(response.content))
                width, height = img.size
                max_dimension = max(width, height)

                result = max_dimension >= min_dimension
                if debug_mode and logger:
                    logger.debug(
                        f"Actual dimensions: {width}x{height}, max: {max_dimension}px, "
                        f"required: {min_dimension}px -> {'PASS' if result else 'FAIL'}"
                    )

                return result

            except Exception as e:
                # If we can't determine dimensions from partial data,
                # fall back to URL pattern if available
                if size_match:
                    size = int(size_match.group(2))
                    result = size >= min_dimension
                    if debug_mode and logger:
                        logger.debug(
                            "Fallback to URL pattern %dpx: %s",
                            size,
                            "PASS" if result else "FAIL",
                        )
                    return result
                else:
                    if debug_mode and logger:
                        logger.debug("Cannot determine dimensions, rejecting: %s", e)
                    return False
        else:
            if debug_mode and logger:
                logger.debug("HTTP error %d, rejecting", response.status_code)
            return False

    except Exception as e:
        if debug_mode and logger:
            logger.warning("Error validating image dimensions: %s", e)
        return False


def _is_irrelevant_image(url: str) -> bool:
    """Check if image URL appears to be irrelevant (sprites, icons, ads, etc.)"""
    if not url or not isinstance(url, str):
        return True

    url_lower = url.lower()

    # Skip obvious non-product images
    irrelevant_patterns = [
        # UI elements
        "sprite",
        "nav-sprite",
        "icon",
        "button",
        "arrow",
        "logo",
        # Navigation/UI
        "gno/sprites",
        "navbar",
        "header",
        "footer",
        "ui-",
        # Ads and tracking
        "adsystem",
        "adnxs",
        "doubleclick",
        "amazon-adsystem",
        # Small/placeholder images
        "1x1",
        "pixel",
        "transparent",
        "loading",
        "spinner",
        # Size indicators for very small images
        "_sx38_",
        "_sy38_",
        "_sx50_",
        "_sy50_",
        "_ac_ux",
        "_sx75_",
        "_sy75_",
        # Amazon UI elements
        "homecustomproduct/360_icon",
        "g/01/gno",
        "g/01/ui",
    ]

    return any(pattern in url_lower for pattern in irrelevant_patterns)


def _meets_dimension_requirements(url: str, min_dimension: int) -> bool:
    """Quick check if URL pattern indicates it meets dimension requirements"""
    if not url or not isinstance(url, str):
        return False

    # Check URL pattern for size indicators
    size_match = re.search(r"_(?:AC_)?(SL|SX|SY)(\d+)_", url)
    if size_match:
        try:
            size = int(size_match.group(2))
            # If URL indicates size >= minimum, likely meets requirements
            return size >= min_dimension
        except (ValueError, IndexError):
            pass

    # If no size pattern, check if it's a main product image (usually high-res)
    # Main product images on Amazon typically have these patterns
    main_image_patterns = [
        r"/I/[A-Z0-9]+\._AC_",  # Main product image pattern
        r"/I/[A-Z0-9]+\.jpg$",  # Direct product image
    ]

    return any(re.search(pattern, url) for pattern in main_image_patterns)


def _is_product_related_video(url: str, page_title: str = "") -> bool:
    """Strict filtering for actual product videos only"""
    if not url or not isinstance(url, str):
        return False

    url_lower = url.lower()

    # STRICT: Must be Amazon product video domain
    if not any(domain in url_lower for domain in ["media-amazon.com"]):
        return False

    # STRICT: Must be MP4 video file
    if not (".mp4" in url_lower or "video" in url_lower):
        return False

    # CRITICAL: Must be official Amazon product video (al-na namespace), NOT
    # influencer videos
    if not ("al-na-" in url_lower and "productvideooptimized" in url_lower):
        return False

    # EXCLUDE: Any ad-related, promotional, or influencer content
    exclude_patterns = [
        # Ad services and tracking
        "adsystem",
        "adnxs",
        "doubleclick",
        "amazon-adsystem",
        "ads/",
        # CRITICAL: Influencer/VSE video services (main problem source)
        "vse-vms-transcoding",
        "vse-vms-closed-captions",
        "videopreview.jobtemplate",
        "default.jobtemplate.hls",
        "default.vertical.jobtemplate",
        "gandalf_preview",
        # Promotional/marketing videos
        "promo",
        "advertisement",
        "commercial",
        "marketing",
        "brand-video",
        # Generic content
        "howto",
        "tutorial",
        "generic",
        "demo",
        "training",
        # Amazon internal/UI videos
        "amazon-internal",
        "ui-video",
        "template",
        "widget",
        # Third-party content
        "youtube",
        "vimeo",
        "facebook",
        "instagram",
        "tiktok",
        # Non-product specific
        "category",
        "browse",
        "search",
        "recommendation",
    ]

    if any(pattern in url_lower for pattern in exclude_patterns):
        return False

    # REQUIRE: Official Amazon product video indicators ONLY
    required_patterns = [
        # Amazon product video paths (very specific) - EXCLUDE VSE
        "/al-na-",  # Amazon Labs video namespace (official product videos)
        "productvideooptimized",  # Optimized product videos
        # NOTE: Removed "/vse-vms/" as it's for influencer videos
    ]

    has_required = any(pattern in url_lower for pattern in required_patterns)

    # ADDITIONAL: Check for product-specific video patterns (OFFICIAL ONLY)
    product_video_patterns = [
        "/s/al-na-",  # Amazon product video storage (official)
        "item-video",  # Item-specific videos (official)
        "product-video",  # Product demonstration videos (official)
        # NOTE: Must still have al-na- pattern to be considered official
    ]

    has_product_pattern = any(
        pattern in url_lower for pattern in product_video_patterns
    )

    # Must have either required pattern OR product pattern
    return has_required or has_product_pattern
