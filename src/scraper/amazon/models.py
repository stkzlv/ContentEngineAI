"""Amazon-specific data models extending the base scraper models.

This module contains Amazon-specific data structures that extend the base
platform-agnostic models with Amazon-specific fields and functionality.
"""

from dataclasses import dataclass, field
from typing import Any

from src.scraper.base.keyword_pillars import normalize_keyword
from src.scraper.base.keyword_pillars import pillar_for as keyword_pillar_for

from ..base import BaseProductData, BaseSearchParameters, Platform
from .constants import (
    FREE_SHIPPING_FILTER_CODE,
    PRIME_FILTER_CODE,
    RATING_FILTER_CODES,
)


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
        # The detail page is read first and supplies `rating` on every arm; the
        # search-results card is the fallback for a page whose star widget
        # could not be parsed, which is why this stays conditional. Making it
        # unconditional would overwrite a correct detail-page rating with the
        # card's on every keyword scrape.
        if not self.rating and self.serp_rating:
            self.rating = self.serp_rating

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
    # Override the generic base default ("relevance") with the Amazon sort
    # token. validate() and build_search_url() expect an Amazon token; the
    # CLI --sort maps friendly names, but the global batch uses this default.
    sort_order: str = "relevanceblender"

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
            rating_codes = RATING_FILTER_CODES

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
                .get("prime_filter_code", PRIME_FILTER_CODE)
            )
        except Exception:
            prime_code = PRIME_FILTER_CODE
        return str(prime_code)

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
                .get("free_shipping_filter_code", FREE_SHIPPING_FILTER_CODE)
            )
        except Exception:
            shipping_code = FREE_SHIPPING_FILTER_CODE
        return str(shipping_code) if shipping_code is not None else None

    def encode_brand_filter(self) -> list[str]:
        """Encode brand filters for Amazon p_89 parameter."""
        return [f"p_89:{brand.replace(' ', '+')}" for brand in self.brands]


# Batch Processing Data Models


@dataclass
class BatchConfig:
    """Batch processing configuration.

    Defines configuration for batch scraping operations including
    product IDs, keywords, error handling, and search parameters.
    """

    product_ids: list[str]  # Product IDs (ASINs) to scrape
    keywords: list[str]  # Keywords to search (flattened from pillar dict)
    fail_fast: bool  # Stop on first failure
    search_params: SearchParameters  # Filters for keyword searches
    max_products: int  # Max products across all sources
    products_per_keyword: int  # Max products per keyword/product ID
    keyword_pillar_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # The map is looked up by normalized keyword, so normalize whatever it
        # was built from. A caller handing over raw keys -- a test, or a config
        # path that predates the shared reader -- would otherwise build a map
        # that silently never matches.
        self.keyword_pillar_map = {
            normalize_keyword(key): value
            for key, value in (self.keyword_pillar_map or {}).items()
        }

    def pillar_for(self, keyword: str) -> str | None:
        """Return the pillar a keyword belongs to, or None.

        Normalized on both sides: the map is keyed by the matching form, so a
        byte-exact lookup would miss a keyword differing only in case or
        spacing.
        """
        return keyword_pillar_for(keyword, self.keyword_pillar_map)


@dataclass
class BatchSummary:
    """Batch execution summary.

    Contains comprehensive statistics about a batch scraping operation
    including counts, failures, and media collection statistics.
    """

    total_attempted: int  # Total products attempted
    product_ids_attempted: int  # Product IDs attempted
    keywords_attempted: int  # Keywords attempted
    successful: int  # Successfully scraped
    failed: int  # Failed scrapes
    successful_products: list[str]  # ASINs of successful products
    failed_products: list[str]  # ASINs of failed products
    media_stats: dict[str, int | float]  # Media collection statistics
    duration_sec: float  # Total batch duration
    # Keywords that produced nothing, tracked apart from the ASIN-shaped
    # fields above: a keyword that returns no product or whose search raises
    # is a lost input with no ASIN to name it by, and folding it into
    # `failed_products` would put a search phrase in a list of identifiers.
    failed_keywords: list[str] = field(default_factory=list)
    # Why an input was lost, when the reason was Amazon's error page. That
    # page means two different things and the run can tell them apart from
    # what its other inputs did, so the summary reports which: a dead query
    # needs the keyword replacing, a rate limit needs only a longer gap.
    dead_queries: list[str] = field(default_factory=list)
    throttled_inputs: list[str] = field(default_factory=list)


@dataclass
class ProductResult:
    """Individual product scraping result.

    Represents the outcome of scraping a single product,
    including success status and any error information.
    """

    product_id: str  # ASIN or keyword
    success: bool  # Scraping succeeded
    data: ProductData | None  # Product data if successful
    error: str | None  # Error message if failed
    source: str  # "product_id" or "keyword"
