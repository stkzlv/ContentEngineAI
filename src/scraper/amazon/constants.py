"""Constants for the Amazon scraper module.

Centralizes magic numbers, filter codes, and threshold values
used across scraper components.
"""

# Image resolution thresholds
HIGH_RES_DIMENSION = 1500
VERY_HIGH_RES_DIMENSION = 2000
HIGH_RES_UPGRADE_DIMENSION = 2000

# Default media limits per product
DEFAULT_MAX_IMAGES_PER_PRODUCT = 10
DEFAULT_MAX_VIDEOS_PER_PRODUCT = 10

# Amazon high-res URL suffixes
AMAZON_HIGH_RES_SUFFIXES = ("._SL1500_", "._SY1500_", "._SX1500_")
AMAZON_HIGH_RES_SUFFIX_TEMPLATE = "._AC_SL{size}_.jpg"
AMAZON_HIGH_RES_URL_PATTERNS = ("_SL1500_", "_SL2000_", "_SL1600_")

# Amazon filter codes
RATING_FILTER_CODES: dict[float, str] = {
    4.0: "2661618011",  # 4 stars & up
    3.0: "2661617011",  # 3 stars & up
    2.0: "2661616011",  # 2 stars & up
    1.0: "2661615011",  # 1 star & up
}
PRIME_FILTER_CODE = "p_85:2470955011"
FREE_SHIPPING_FILTER_CODE = "p_76:419122011"

# Media validation defaults
DEFAULT_MIN_TOTAL_MEDIA = 3
DEFAULT_MIN_IMAGES_IF_NO_VIDEO = 5
DEFAULT_MIN_IMAGES_WITH_VIDEO = 2

# Batch processing defaults
DEFAULT_MAX_SCRAPE_ATTEMPTS = 50
DEFAULT_PREFETCH_MULTIPLIER = 3
DEFAULT_MAX_BATCH_SIZE = 15
