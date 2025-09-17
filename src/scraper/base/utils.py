"""Platform-agnostic utility functions for the multi-platform scraper architecture.

This module contains shared utility functions that can be used across all platform
scrapers, including retry logic, validation helpers, and common browser utilities.
"""

import logging
import os
import platform
import random
import re
import subprocess
import time
from collections.abc import Callable
from typing import Any

from .config import get_config_manager
from .models import BaseProductData


def exponential_backoff_retry(
    func: Callable | None = None,
    max_retries: int | None = None,
    base_delay: float | None = None,
    backoff_factor: float | None = None,
    max_delay: float | None = None,
    use_jitter: bool | None = None,
    jitter_factor: float | None = None,
) -> Callable:
    """Decorator for exponential backoff retry logic.

    This decorator provides automatic retry functionality with exponential backoff
    for functions that may fail due to transient issues (network timeouts, etc.).
    Configuration is loaded from the global config if not explicitly provided.

    Args:
    ----
        func: Function to wrap with retry logic
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds before first retry
        backoff_factor: Multiplier for each subsequent retry
        max_delay: Maximum delay between retries
        use_jitter: Whether to add random jitter to delays
        jitter_factor: Factor for jitter calculation (0.0-1.0)

    Returns:
    -------
        Wrapped function with retry logic

    """

    def decorator(wrapped_func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Load retry configuration
            config_manager = get_config_manager()
            retry_config = config_manager.get_retry_config()

            # Use provided values or fall back to config
            actual_max_retries = (
                max_retries
                if max_retries is not None
                else retry_config.get("default_max_retries", 3)
            )
            actual_base_delay = (
                base_delay
                if base_delay is not None
                else retry_config.get("base_delay", 1.0)
            )
            actual_backoff_factor = (
                backoff_factor
                if backoff_factor is not None
                else retry_config.get("backoff_factor", 2.0)
            )
            actual_max_delay = (
                max_delay
                if max_delay is not None
                else retry_config.get("max_delay", 60.0)
            )
            actual_use_jitter = (
                use_jitter
                if use_jitter is not None
                else retry_config.get("use_jitter", True)
            )
            actual_jitter_factor = (
                jitter_factor
                if jitter_factor is not None
                else retry_config.get("jitter_factor", 0.5)
            )

            last_exception = None

            for attempt in range(actual_max_retries + 1):
                try:
                    return wrapped_func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt == actual_max_retries:
                        logging.getLogger(__name__).error(
                            f"❌ Final retry failed for {wrapped_func.__name__}: {e}"
                        )
                        raise last_exception from None

                # Calculate delay with exponential backoff
                delay = actual_base_delay * (actual_backoff_factor**attempt)
                delay = min(delay, actual_max_delay)

                # Add jitter to prevent thundering herd
                if actual_use_jitter:
                    delay *= actual_jitter_factor + random.random() * (  # noqa: S311
                        1.0 - actual_jitter_factor
                    )

                logging.getLogger(__name__).debug(
                    f"🔄 Retry {attempt + 1}/{actual_max_retries} for "
                    f"{wrapped_func.__name__} in {delay:.2f}s: {last_exception}"
                )

                time.sleep(delay)

        return wrapper

    if func is None:
        # Called as @exponential_backoff_retry() with parentheses
        return decorator
    else:
        # Called as @exponential_backoff_retry without parentheses
        return decorator(func)


def is_valid_product_data(product: BaseProductData) -> bool:
    """Validate product data completeness and quality.

    Args:
    ----
        product: Product data to validate

    Returns:
    -------
        True if product data is valid, False otherwise

    """
    # Check required fields
    if not product.title or product.title.strip() == "":
        return False

    if not product.price or product.price.strip() == "":
        return False

    if not product.url or product.url.strip() == "":
        return False

    # Check for common invalid indicators
    invalid_indicators = [
        "N/A",
        "n/a",
        "unavailable",
        "out of stock",
        "price not available",
        "currently unavailable",
    ]

    title_lower = product.title.lower()
    price_lower = product.price.lower()

    for indicator in invalid_indicators:
        if indicator in title_lower or indicator in price_lower:
            return False

    # Basic price format validation
    return (
        "$" in product.price
        or "£" in product.price
        or "€" in product.price
        or any(char.isdigit() for char in product.price)
    )


def is_valid_price(price: str) -> bool:
    """Validate price string format.

    Args:
    ----
        price: Price string to validate

    Returns:
    -------
        True if price format is valid, False otherwise

    """
    if not price or price.strip() == "":
        return False

    # Check for currency symbols or digits
    return (
        "$" in price
        or "£" in price
        or "€" in price
        or "¥" in price
        or any(char.isdigit() for char in price)
    )


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for cross-platform compatibility.

    Args:
    ----
        filename: Original filename

    Returns:
    -------
        Sanitized filename safe for all platforms

    """
    # Remove or replace invalid characters
    invalid_chars = '<>:"|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")

    # Replace path separators
    filename = filename.replace("/", "_").replace("\\", "_")

    # Limit length
    if len(filename) > 200:
        name, ext = os.path.splitext(filename)
        filename = name[: 200 - len(ext)] + ext

    return filename


def detect_monitors() -> list[dict[str, Any]]:
    """Detect available monitors for browser positioning.

    Returns
    -------
        List of monitor information dictionaries

    """
    monitors = []

    try:
        if platform.system() == "Linux":
            # Try xrandr first
            try:
                result = subprocess.run(
                    ["xrandr", "--query"], capture_output=True, text=True, timeout=5
                )

                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if " connected" in line and "x" in line:
                            # Parse resolution and position
                            parts = line.split()
                            for part in parts:
                                if "x" in part and "+" in part:
                                    # Format: 1920x1080+0+0
                                    res_pos = part.split("+")
                                    if len(res_pos) >= 3:
                                        width, height = res_pos[0].split("x")
                                        x, y = int(res_pos[1]), int(res_pos[2])

                                        monitors.append(
                                            {
                                                "width": int(width),
                                                "height": int(height),
                                                "x": x,
                                                "y": y,
                                                "primary": "primary" in line,
                                            }
                                        )

                    if monitors:
                        return monitors
            except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
                pass

            # Fallback: try DISPLAY environment variable
            display = os.environ.get("DISPLAY")
            if display:
                monitors.append(
                    {"width": 1920, "height": 1080, "x": 0, "y": 0, "primary": True}
                )

        elif platform.system() == "Darwin":  # macOS
            try:
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode == 0:
                    # Parse macOS display info (simplified)
                    monitors.append(
                        {"width": 1920, "height": 1080, "x": 0, "y": 0, "primary": True}
                    )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        elif platform.system() == "Windows":
            # Windows fallback
            monitors.append(
                {"width": 1920, "height": 1080, "x": 0, "y": 0, "primary": True}
            )

    except Exception as e:
        # Ultimate fallback
        import logging

        logging.getLogger(__name__).debug(f"Monitor detection failed: {e}")

    # Default single monitor if detection failed
    if not monitors:
        monitors.append(
            {"width": 1920, "height": 1080, "x": 0, "y": 0, "primary": True}
        )

    return monitors


def get_optimal_browser_position(
    monitors: list[dict[str, Any]],
) -> tuple[int, int, int, int]:
    """Calculate optimal browser window position and size.

    Args:
    ----
        monitors: List of monitor information from detect_monitors()

    Returns:
    -------
        Tuple of (x, y, width, height) for browser window

    """
    if not monitors:
        return (100, 100, 1200, 800)

    # Use primary monitor if available, otherwise first monitor
    primary_monitor = None
    for monitor in monitors:
        if monitor.get("primary", False):
            primary_monitor = monitor
            break

    if primary_monitor is None:
        primary_monitor = monitors[0]

    # Calculate browser size (80% of monitor size)
    browser_width = int(primary_monitor["width"] * 0.8)
    browser_height = int(primary_monitor["height"] * 0.8)

    # Calculate position (centered)
    browser_x = primary_monitor["x"] + (primary_monitor["width"] - browser_width) // 2
    browser_y = primary_monitor["y"] + (primary_monitor["height"] - browser_height) // 2

    return (browser_x, browser_y, browser_width, browser_height)


def extract_product_id_from_url(
    url: str, platform_patterns: dict[str, str]
) -> str | None:
    """Extract platform-specific product ID from URL.

    Args:
    ----
        url: Product URL
        platform_patterns: Dictionary of regex patterns for each platform

    Returns:
    -------
        Extracted product ID or None if not found

    """
    for _platform_name, pattern in platform_patterns.items():
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def create_user_agent() -> str:
    """Create a realistic user agent string.

    Returns
    -------
        User agent string for HTTP requests

    """
    # Use a recent Chrome user agent
    return (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )


def human_delay(min_seconds: float = 0.5, max_seconds: float = 2.0) -> None:
    """Add human-like delay to avoid detection.

    Args:
    ----
        min_seconds: Minimum delay in seconds
        max_seconds: Maximum delay in seconds

    """
    delay = random.uniform(min_seconds, max_seconds)  # noqa: S311
    time.sleep(delay)


def normalize_price(price_str: str) -> float | None:
    """Normalize price string to float value.

    Args:
    ----
        price_str: Price string (e.g., "$19.99", "£15.50")

    Returns:
    -------
        Float value or None if cannot be parsed

    """
    if not price_str:
        return None

    # Remove currency symbols and spaces
    normalized = re.sub(r"[^\d.,]", "", price_str)

    # Handle comma as decimal separator (European format)
    if "," in normalized and "." not in normalized:
        normalized = normalized.replace(",", ".")
    elif "," in normalized and "." in normalized:
        # Assume comma is thousands separator
        normalized = normalized.replace(",", "")

    try:
        return float(normalized)
    except ValueError:
        return None


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format.

    Args:
    ----
        seconds: Duration in seconds

    Returns:
    -------
        Formatted duration string

    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"
