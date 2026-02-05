"""Link-in-bio manager — orchestrates provider calls after publish."""

import json
import logging
from pathlib import Path

from src.publisher.link_in_bio.base import BaseLinkInBioProvider

logger = logging.getLogger(__name__)

DEFAULT_MAX_TITLE_LENGTH = 80


class LinkInBioManager:
    """Manages link-in-bio updates after video publishing."""

    def __init__(
        self,
        provider: BaseLinkInBioProvider,
        max_links: int = 25,
        max_title_length: int = DEFAULT_MAX_TITLE_LENGTH,
    ) -> None:
        self.provider = provider
        self.max_links = max_links
        self.max_title_length = max_title_length

    async def update(self, product_id: str, outputs_dir: Path) -> dict:
        """Add product link to bio page after successful publish.

        Reads product data from outputs/<product_id>/data.json,
        creates a link, and rotates oldest if max_links exceeded.
        """
        data_path = outputs_dir / product_id / "data.json"
        if not data_path.exists():
            logger.warning(
                "No data.json found for %s, skipping link-in-bio", product_id
            )
            return {"success": False, "reason": "no_data"}

        with open(data_path, encoding="utf-8") as f:
            raw = json.load(f)

        product = raw[0] if isinstance(raw, list) else raw
        title = product.get("title", "")
        url = product.get("url", "")

        if not title or not url:
            logger.warning("Missing title or url for %s, skipping", product_id)
            return {"success": False, "reason": "missing_fields"}

        # Authenticate
        await self.provider.authenticate()

        # Check for duplicates
        existing = await self.provider.list_links()
        for link in existing:
            link_url = link.get("url", link.get("link", ""))
            if product_id in link_url:
                logger.info("Link for %s already exists, skipping", product_id)
                return {"success": True, "reason": "duplicate", "existing": True}

        # Rotate oldest if at capacity
        if self.max_links > 0 and len(existing) >= self.max_links:
            oldest = existing[-1]
            oldest_id = oldest.get("id", oldest.get("link_id"))
            if oldest_id:
                logger.info(
                    "Rotating oldest link (id=%s) to stay under %d",
                    oldest_id,
                    self.max_links,
                )
                await self.provider.delete_link(oldest_id)

        # Truncate title for readability
        if len(title) <= self.max_title_length:
            display_title = title
        else:
            display_title = title[: self.max_title_length - 3] + "..."

        image = product.get("main_image")
        result = await self.provider.add_link(
            title=display_title,
            url=url,
            image=image,
        )

        logger.info("Link-in-bio updated for %s", product_id)
        return {"success": True, "result": result}


def create_link_in_bio_manager(
    provider_name: str,
    max_links: int = 25,
    max_title_length: int = DEFAULT_MAX_TITLE_LENGTH,
) -> LinkInBioManager:
    """Factory to create a LinkInBioManager for the given provider."""
    if provider_name == "lnkbio":
        from src.publisher.link_in_bio.lnkbio import LnkBioProvider

        return LinkInBioManager(
            provider=LnkBioProvider(),
            max_links=max_links,
            max_title_length=max_title_length,
        )

    raise ValueError(f"Unknown link-in-bio provider: {provider_name}")
