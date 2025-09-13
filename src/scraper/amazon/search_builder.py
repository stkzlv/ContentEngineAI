"""Amazon search URL construction utilities.

This module handles building and encoding Amazon search URLs with advanced
filtering parameters.
"""

import logging
import urllib.parse

from .models import SearchParameters


class SearchParameterBuilder:
    """Builds Amazon search URLs with advanced filtering parameters."""

    def __init__(self, base_url: str):
        self.base_url = base_url

    def build_search_url(
        self, keyword: str, parameters: SearchParameters | None = None
    ) -> str:
        """Build Amazon search URL with keyword and optional filters."""
        if parameters is None:
            parameters = SearchParameters()

        # Validate parameters
        validation_errors = parameters.validate()
        if validation_errors:
            logging.getLogger(__name__).warning(
                f"Search parameter validation errors: {validation_errors}"
            )

        # Start with basic search parameters
        url_params = {"k": keyword.replace(" ", "+")}

        # Build refinement hierarchy (rh) parameter
        refinements = []

        # Price range
        price_filter = parameters.encode_price_range()
        if price_filter:
            refinements.append(price_filter)

        # Rating filter
        rating_filter = parameters.encode_rating_filter()
        if rating_filter:
            refinements.append(rating_filter)

        # Prime shipping
        prime_filter = parameters.encode_prime_filter()
        if prime_filter:
            refinements.append(prime_filter)

        # Free shipping
        shipping_filter = parameters.encode_free_shipping_filter()
        if shipping_filter:
            refinements.append(shipping_filter)

        # Brand filters
        brand_filters = parameters.encode_brand_filter()
        refinements.extend(brand_filters)

        # Add refinements to URL if any exist
        if refinements:
            url_params["rh"] = ",".join(refinements)

        # Sort order
        if parameters.sort_order != "relevanceblender":
            url_params["s"] = parameters.sort_order

        # Build final URL
        query_string = urllib.parse.urlencode(url_params)
        return f"{self.base_url}/s?{query_string}"

    def log_search_parameters(
        self, keyword: str, parameters: SearchParameters | None = None
    ):
        """Log search parameters for debugging."""
        logger = logging.getLogger(__name__)
        if parameters is None:
            logger.info(f"Search URL for '{keyword}': basic search (no filters)")
            return

        filters = []
        if parameters.min_price is not None or parameters.max_price is not None:
            price_range = (
                f"${parameters.min_price or 0:.2f}-${parameters.max_price or '∞'}"
            )
            filters.append(f"price: {price_range}")

        if parameters.min_rating is not None:
            filters.append(f"rating: {parameters.min_rating}+ stars")

        if parameters.prime_only:
            filters.append("Prime only")

        if parameters.free_shipping:
            filters.append("Free shipping")

        if parameters.brands:
            filters.append(f"brands: {', '.join(parameters.brands)}")

        if parameters.sort_order != "relevanceblender":
            filters.append(f"sort: {parameters.sort_order}")

        if filters:
            logger.info(
                f"Search URL for '{keyword}' with filters: {'; '.join(filters)}"
            )
        else:
            logger.info(f"Search URL for '{keyword}': basic search (no filters)")
