"""Utility functions for Amazon scraper.

This module contains validation functions, retry logic, and other standalone
utilities used throughout the scraper.
"""

import logging
import re
import subprocess
from typing import Any

from botasaurus.browser import Driver

from ..base.utils import exponential_backoff_retry  # noqa: F401

logger = logging.getLogger(__name__)


def validate_asin_format(asin: str) -> bool:
    """Validate proper ASIN format using configured patterns"""
    if not asin:
        return False

    try:
        from .config import CONFIG

        asin_patterns = CONFIG.get("global_settings", {}).get("asin_patterns", {})
        modern_pattern = asin_patterns.get("modern_asin_pattern", r"^B0[A-Z0-9]{8}$")
        legacy_pattern = asin_patterns.get("legacy_asin_pattern", r"^[A-Z0-9]{10}$")
    except Exception:
        # Fallback patterns
        modern_pattern = r"^B0[A-Z0-9]{8}$"
        legacy_pattern = r"^[A-Z0-9]{10}$"

    # Check modern ASIN format (B0 + 8 chars)
    if re.match(modern_pattern, asin):
        return True

    # Check legacy ASIN format (10 chars)
    return bool(re.match(legacy_pattern, asin))


def is_valid_product_data(
    title: str,
    price: str,
    description: str = None,
    asin: str = None,
    rating: str = None,
    essential_fields: list[str] = None,
) -> bool:
    """Validate critical product data (requirement #16)

    Args:
    ----
        title: Product title (always required)
        price: Product price (always required)
        description: Product description (optional validation)
        asin: Product ASIN (optional validation)
        rating: Product rating (optional validation)
        essential_fields: List of fields to validate
            ['title', 'price', 'description', 'asin', 'rating']

    Returns:
    -------
        bool: True if all essential fields are valid

    """
    # Configurable core validation - only validate if essential_fields is not empty
    if essential_fields:
        # Only validate specified fields
        pass
    else:
        # Default: require title only (minimal validation)
        # Allow products without price when validation is disabled
        return bool(title and title.strip())

    # Legacy validation for when essential_fields contains 'title' or 'price'
    if "title" in essential_fields and (not title or not title.strip()):
        return False

    if "price" in essential_fields:
        if not price or not price.strip():
            return False
        # Basic price format validation only when price is required
        price_valid = (
            "$" in price
            or "£" in price
            or "€" in price
            or any(char.isdigit() for char in price)
        )
        if not price_valid:
            return False

    # Enhanced validation for additional fields if specified
    if essential_fields:
        for field in essential_fields:
            if (
                (
                    field == "description"
                    and (not description or not description.strip())
                )
                or (field == "asin" and (not asin or not validate_asin_format(asin)))
                or (field == "rating" and (not rating or not rating.strip()))
            ):
                return False

    return True


def detect_regional_redirect(
    driver: Driver, original_url: str
) -> tuple[bool, str | None]:
    """Detect if Amazon redirected to a different regional site (requirement #8)"""
    try:
        current_url = driver.current_url

        # Extract domain from URLs for comparison
        original_domain = original_url.split("/")[2] if "//" in original_url else None
        current_domain = current_url.split("/")[2] if "//" in current_url else None

        if original_domain and current_domain and original_domain != current_domain:
            return True, current_domain

        # Check for common redirect indicators in the page
        redirect_indicators = [
            "not available in your country",
            "product is not available",
            "unavailable in your location",
            "redirected you to",
        ]

        page_text = driver.get_text("body").lower() if driver.select("body") else ""

        for indicator in redirect_indicators:
            if indicator in page_text:
                return True, f"Product unavailable: {indicator}"
    except Exception as e:
        # If we can't detect redirect, assume no redirect occurred
        import logging

        logging.getLogger(__name__).debug(
            f"Could not detect redirect for {original_url}: {e}"
        )

    return False, None


def detect_monitors() -> list[dict[str, Any]]:
    """Detect multi-monitor setup and return monitor information

    This function uses system commands to detect available monitors
    and their resolutions for optimal browser window positioning.

    Returns
    -------
        List of monitor dictionaries with width, height, x, y coordinates

    """
    monitors = []

    try:
        # Try xrandr first (Linux/Unix)
        # Get timeout from config
        from .config import CONFIG

        timeout = (
            CONFIG.get("global_settings", {})
            .get("system_timeouts", {})
            .get("system_command_timeout", 5)
        )
        result = subprocess.run(
            ["xrandr"], capture_output=True, text=True, timeout=timeout
        )
        if result.returncode == 0:
            lines = result.stdout.split("\n")
            for line in lines:
                if " connected" in line and "primary" in line:
                    # Parse primary monitor: "HDMI-1 connected primary 1920x1080+0+0"
                    parts = line.split()
                    for part in parts:
                        if "x" in part and "+" in part:
                            # Extract resolution and position: "1920x1080+0+0"
                            resolution_pos = part.split("+")
                            if len(resolution_pos) >= 3:
                                width_height = resolution_pos[0].split("x")
                                if len(width_height) == 2:
                                    monitors.append(
                                        {
                                            "width": int(width_height[0]),
                                            "height": int(width_height[1]),
                                            "x": int(resolution_pos[1]),
                                            "y": int(resolution_pos[2]),
                                            "primary": True,
                                        }
                                    )
                                    break
                elif " connected" in line and "primary" not in line:
                    # Parse secondary monitors
                    parts = line.split()
                    for part in parts:
                        if "x" in part and "+" in part:
                            resolution_pos = part.split("+")
                            if len(resolution_pos) >= 3:
                                width_height = resolution_pos[0].split("x")
                                if len(width_height) == 2:
                                    monitors.append(
                                        {
                                            "width": int(width_height[0]),
                                            "height": int(width_height[1]),
                                            "x": int(resolution_pos[1]),
                                            "y": int(resolution_pos[2]),
                                            "primary": False,
                                        }
                                    )
                                    break
    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        FileNotFoundError,
        ValueError,
    ):
        pass

    # Fallback: assume single monitor with common resolution
    if not monitors:
        # Get fallback monitor dimensions from config
        try:
            browser_config = CONFIG.get("global_settings", {}).get("browser_config", {})
            fallback = browser_config.get(
                "fallback_window_position", [0, 0, 1920, 1080]
            )
            monitors.append(
                {
                    "width": fallback[2],
                    "height": fallback[3],
                    "x": fallback[0],
                    "y": fallback[1],
                    "primary": True,
                }
            )
        except Exception:
            monitors.append(
                {"width": 1920, "height": 1080, "x": 0, "y": 0, "primary": True}
            )

    return monitors


def get_optimal_browser_position(
    monitors: list[dict[str, Any]],
) -> tuple[int, int, int, int]:
    """Calculate optimal browser window position and size

    Places the browser on the primary monitor and maximizes it.

    Args:
    ----
        monitors: List of monitor information from detect_monitors()

    Returns:
    -------
        Tuple of (x, y, width, height) for browser window

    """
    if not monitors:
        # Get fallback position from config
        try:
            from .config import CONFIG

            browser_config = CONFIG.get("global_settings", {}).get("browser_config", {})
            fallback = browser_config.get(
                "fallback_window_position", [0, 0, 1920, 1080]
            )
            return tuple(fallback)
        except Exception:
            return (0, 0, 1920, 1080)

    # Always use primary monitor
    primary_monitor = None
    for monitor in monitors:
        if monitor.get("primary", False):
            primary_monitor = monitor
            break

    # If no primary monitor found, use first monitor
    if not primary_monitor:
        primary_monitor = monitors[0]

    # Calculate window position on primary monitor
    window_x = primary_monitor["x"]
    window_y = primary_monitor["y"]
    window_width = primary_monitor["width"]
    window_height = primary_monitor["height"]

    return (window_x, window_y, window_width, window_height)


def build_affiliate_url(url: str, associate_tag: str = None) -> str:
    """Build Amazon affiliate URL with associate tag parameter.

    Extracts ASIN from URL and builds clean affiliate link optimized for
    URL shortening services. Removes search parameters and tracking tokens
    to minimize URL length.

    Args:
    ----
        url: Amazon URL (e.g., "https://www.amazon.com/dp/B0BTYCRJSS")
        associate_tag: Amazon Associates tag (loads from config if None)

    Returns:
    -------
        Clean URL with tag parameter (e.g., "https://www.amazon.com/dp/B0BTYCRJSS?tag=stealtech06-20")

    Examples:
    --------
        >>> build_affiliate_url("https://www.amazon.com/dp/B0BTYCRJSS")
        "https://www.amazon.com/dp/B0BTYCRJSS?tag=stealtech06-20"

        >>> build_affiliate_url("https://www.amazon.com/product/dp/B0BTYCRJSS?dib=...")
        "https://www.amazon.com/dp/B0BTYCRJSS?tag=stealtech06-20"

    """
    if not url:
        return url

    # Load associate tag: env var > config > None
    if associate_tag is None:
        import os

        associate_tag = os.environ.get("AMAZON_ASSOCIATE_TAG")

    if associate_tag is None:
        try:
            from .config import CONFIG

            associate_tag = (
                CONFIG.get("scrapers", {}).get("amazon", {}).get("associate_tag")
            )
        except Exception:
            associate_tag = None

    # If no associate tag configured, return original URL
    if not associate_tag:
        return url

    # Extract ASIN and build clean URL
    import re
    from urllib.parse import urlparse

    # Extract ASIN from /dp/{ASIN} pattern
    asin_match = re.search(r"/dp/([A-Z0-9]{10})", url)
    if not asin_match:
        # Fallback: add tag to original URL if ASIN not found
        if "?" in url:
            if "?tag=" in url:
                url = re.sub(r"\?tag=[^&]*", f"?tag={associate_tag}", url)
            elif "&tag=" in url:
                url = re.sub(r"&tag=[^&]*", f"&tag={associate_tag}", url)
            else:
                url = f"{url}&tag={associate_tag}"
        else:
            url = f"{url}?tag={associate_tag}"
        return url

    # Build clean URL with just domain, /dp/{ASIN}, and associate tag
    asin = asin_match.group(1)
    parsed = urlparse(url)
    domain = f"{parsed.scheme}://{parsed.netloc}"

    # Build clean affiliate URL
    clean_url = f"{domain}/dp/{asin}?tag={associate_tag}"

    return clean_url
