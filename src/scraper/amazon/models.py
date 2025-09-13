"""Amazon-specific data models extending the base scraper models.

This module contains Amazon-specific data structures that extend the base
platform-agnostic models with Amazon-specific fields and functionality.
"""

from dataclasses import dataclass
from typing import Any

from ..base import BaseProductData, BaseSearchParameters, Platform


@dataclass
class ProductData(BaseProductData):
    """Amazon-specific product data extending BaseProductData.

    Adds Amazon-specific fields like ASIN while inheriting all
    common product fields from the base class.
    """

    # Amazon-specific fields
    asin: str | None = None
    serp_rating: str | None = None
    serp_reviews_count: str | None = None

    def __post_init__(self):
        """Initialize Amazon product with Platform.AMAZON."""
        super().__post_init__()
        # Set platform and platform_id from ASIN
        if not hasattr(self, "platform") or self.platform is None:
            self.platform = Platform.AMAZON
        if self.asin and (not hasattr(self, "platform_id") or self.platform_id is None):
            self.platform_id = self.asin

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with Amazon-specific fields."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "asin": self.asin,
                "serp_rating": self.serp_rating,
                "serp_reviews_count": self.serp_reviews_count,
            }
        )
        return base_dict


@dataclass
class SerpProductInfo:
    """Search result product information"""

    url: str
    rating: str | None = None
    reviews_count: str | None = None
    asin: str | None = None
    keyword: str = ""


@dataclass
class SearchParameters(BaseSearchParameters):
    """Amazon-specific search parameters extending BaseSearchParameters.

    Adds Amazon-specific filtering options while inheriting common
    search parameters from the base class.
    """

    # Amazon-specific fields
    prime_only: bool = False

    def __post_init__(self):
        """Initialize Amazon search parameters."""
        super().__post_init__()

    def validate(self) -> list[str]:
        """Validate Amazon search parameters."""
        # Use base validation and add Amazon-specific checks
        errors = super().validate()

        # Amazon-specific sort order validation
        valid_sorts = {
            "relevanceblender",
            "price-asc-rank",
            "price-desc-rank",
            "review-rank",
            "date-desc-rank",
            "featured-rank",
        }
        if self.sort_order not in valid_sorts:
            errors.append(f"sort_order must be one of: {', '.join(valid_sorts)}")

        return errors

    def to_cents(self, price: float) -> int:
        """Convert dollar amount to cents for Amazon URL encoding."""
        # Import CONFIG here to avoid circular imports
        try:
            from .config import CONFIG

            multiplier = (
                CONFIG.get("scrapers", {})
                .get("amazon", {})
                .get("filter_parameters", {})
                .get("price_to_cents_multiplier", 100)
            )
        except Exception:
            multiplier = 100
        return int(price * multiplier)

    def encode_price_range(self) -> str | None:
        """Encode price range for Amazon p_36 parameter."""
        if self.min_price is None and self.max_price is None:
            return None

        min_cents = self.to_cents(self.min_price) if self.min_price is not None else ""
        max_cents = self.to_cents(self.max_price) if self.max_price is not None else ""

        return f"p_36:{min_cents}-{max_cents}"

    def encode_rating_filter(self) -> str | None:
        """Encode rating filter for Amazon p_72 parameter."""
        if self.min_rating is None:
            return None

        # Get rating codes from config
        try:
            from .config import CONFIG

            rating_codes = (
                CONFIG.get("scrapers", {})
                .get("amazon", {})
                .get("filter_parameters", {})
                .get("rating_codes", {})
            )
        except Exception:
            # Fallback rating codes
            rating_codes = {
                4.0: "2661618011",  # 4 stars & up
                3.0: "2661617011",  # 3 stars & up
                2.0: "2661616011",  # 2 stars & up
                1.0: "2661615011",  # 1 star & up
            }

        # Find the appropriate rating code
        for rating, code in rating_codes.items():
            if self.min_rating >= rating:
                return f"p_72:{code}"

        return None

    def encode_prime_filter(self) -> str | None:
        """Encode Prime shipping filter."""
        if not self.prime_only:
            return None
        try:
            from .config import CONFIG

            prime_code = (
                CONFIG.get("scrapers", {})
                .get("amazon", {})
                .get("filter_parameters", {})
                .get("prime_filter_code", "p_85:2470955011")
            )
        except Exception:
            prime_code = "p_85:2470955011"
        return prime_code

    def encode_free_shipping_filter(self) -> str | None:
        """Encode free shipping filter."""
        if not self.free_shipping:
            return None
        try:
            from .config import CONFIG

            shipping_code = (
                CONFIG.get("scrapers", {})
                .get("amazon", {})
                .get("filter_parameters", {})
                .get("free_shipping_filter_code", "p_76:419122011")
            )
        except Exception:
            shipping_code = "p_76:419122011"
        return shipping_code

    def encode_brand_filter(self) -> list[str]:
        """Encode brand filters for Amazon p_89 parameter."""
        return [f"p_89:{brand.replace(' ', '+')}" for brand in self.brands]
