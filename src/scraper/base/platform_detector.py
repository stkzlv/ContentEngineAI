"""Platform detection from product ID patterns.

Provides extensible platform detection based on product ID format patterns.
Each e-commerce platform has distinct product ID formats that can be used
to automatically route products to the appropriate scraper.
"""

import re
from collections.abc import Callable

# Type alias for platform detector functions
PlatformDetector = Callable[[str], bool]

# Registry of platform detectors: platform_name -> detector_function
_PLATFORM_DETECTORS: dict[str, PlatformDetector] = {}


def register_platform(
    platform_name: str,
) -> Callable[[PlatformDetector], PlatformDetector]:
    """Decorator to register a platform detector function.

    Args:
    ----
        platform_name: The platform identifier (e.g., "amazon", "ebay").

    Returns:
    -------
        Decorator function that registers the detector.

    """

    def decorator(func: PlatformDetector) -> PlatformDetector:
        _PLATFORM_DETECTORS[platform_name] = func
        return func

    return decorator


@register_platform("amazon")
def _is_amazon_asin(product_id: str) -> bool:
    """Check if product ID matches Amazon ASIN format.

    Amazon ASINs are 10-character alphanumeric identifiers that start with B0 or B1.

    Args:
    ----
        product_id: The product identifier to check.

    Returns:
    -------
        True if the product ID matches Amazon ASIN format.

    """
    if not product_id or len(product_id) != 10:
        return False

    # Modern ASIN pattern: starts with B0 or B1, followed by 8 alphanumeric chars
    modern_pattern = r"^B[01][A-Z0-9]{8}$"
    return bool(re.match(modern_pattern, product_id, re.IGNORECASE))


def detect_platform(product_id: str) -> str | None:
    """Detect e-commerce platform from product ID pattern.

    Iterates through registered platform detectors and returns the first
    matching platform. Order of detection is based on registration order.

    Args:
    ----
        product_id: The product identifier to analyze.

    Returns:
    -------
        Platform name (e.g., "amazon") if detected, None if unknown.

    """
    if not product_id:
        return None

    product_id = product_id.strip()

    for platform_name, detector in _PLATFORM_DETECTORS.items():
        if detector(product_id):
            return platform_name

    return None


def get_registered_platforms() -> list[str]:
    """Get list of all registered platform names.

    Returns
    -------
        List of platform identifiers that have registered detectors.

    """
    return list(_PLATFORM_DETECTORS.keys())
