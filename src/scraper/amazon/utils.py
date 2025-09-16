"""Utility functions for Amazon scraper.

This module contains validation functions, retry logic, and other standalone
utilities used throughout the scraper.
"""

import logging
import random
import re
import subprocess
import time
from collections.abc import Callable
from typing import Any

from botasaurus.browser import Driver  # type: ignore[import-untyped]


def exponential_backoff_retry(
    func: Callable,
    max_retries: int = None,
    base_delay: float = None,
    max_delay: float = None,
    backoff_factor: float = None,
    jitter: bool = None,
) -> Callable:
    """Decorator to add exponential backoff retry logic to functions

    Args:
    ----
        func: Function to retry
        max_retries: Maximum number of retry attempts (uses config if None)
        base_delay: Initial delay in seconds (uses config if None)
        max_delay: Maximum delay in seconds (uses config if None)
        backoff_factor: Multiplier for each retry (uses config if None)
        jitter: Add random jitter to prevent thundering herd (uses config if None)

    """

    def wrapper(*args, **kwargs):
        # Import here to avoid circular imports
        from .config import CONFIG

        # Load config values if not provided
        try:
            global_config = CONFIG.get("global_settings", {}).get("retry_config", {})
        except Exception:
            global_config = {}

        actual_max_retries = (
            max_retries
            if max_retries is not None
            else global_config.get("default_max_retries", 3)
        )
        actual_base_delay = (
            base_delay
            if base_delay is not None
            else global_config.get("base_delay", 1.0)
        )
        actual_max_delay = (
            max_delay if max_delay is not None else global_config.get("max_delay", 60.0)
        )
        actual_backoff_factor = (
            backoff_factor
            if backoff_factor is not None
            else global_config.get("backoff_factor", 2.0)
        )
        actual_jitter = (
            jitter if jitter is not None else global_config.get("use_jitter", True)
        )
        jitter_factor = global_config.get("jitter_factor", 0.5)

        # Import DEBUG_MODE from main module
        try:
            from . import scraper

            DEBUG_MODE = scraper.DEBUG_MODE
        except Exception:
            DEBUG_MODE = False

        last_exception: Exception | None = None

        for attempt in range(actual_max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt == actual_max_retries:
                    if DEBUG_MODE:
                        logging.getLogger(__name__).warning(
                            f"❌ Final retry failed for {func.__name__}: {e}"
                        )
                    raise last_exception from None

                # Calculate delay with exponential backoff
                delay = min(
                    actual_base_delay * (actual_backoff_factor**attempt),
                    actual_max_delay,
                )

                # Add jitter to prevent thundering herd
                if actual_jitter:
                    delay *= jitter_factor + random.random() * (1.0 - jitter_factor)  # noqa: S311

                if DEBUG_MODE:
                    logging.getLogger(__name__).debug(
                        f"🔄 Retry {attempt + 1}/{actual_max_retries} for "
                        f"{func.__name__} in {delay:.2f}s: {e}"
                    )

                time.sleep(delay)

        # This should never be reached since we would have returned or raised in
        # the loop
        raise RuntimeError(
            f"Retry logic failed for {func.__name__}"
        ) from last_exception

    return wrapper


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
            return (0, 0, 1920, 1080)  # Fallback position - maximized on default screen

    # Always use primary monitor
    primary_monitor = None
    for monitor in monitors:
        if monitor.get("primary", False):
            primary_monitor = monitor
            break

    # If no primary monitor found, use first monitor
    if not primary_monitor:
        primary_monitor = monitors[0]

    # Maximize the window on the primary monitor
    window_x = primary_monitor["x"]
    window_y = primary_monitor["y"]
    window_width = primary_monitor["width"]
    window_height = primary_monitor["height"]

    return (window_x, window_y, window_width, window_height)
