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
        max_links: int = 0,
        max_title_length: int = DEFAULT_MAX_TITLE_LENGTH,
    ) -> None:
        self.provider = provider
        self.max_links = max_links
        self.max_title_length = max_title_length

    async def update(self, product_id: str, outputs_dir: Path) -> dict[str, object]:
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
        url = product.get("affiliate_link") or product.get("url", "")

        if not title or not url:
            logger.warning("Missing title or url for %s, skipping", product_id)
            return {"success": False, "reason": "missing_fields"}

        # Authenticate
        await self.provider.authenticate()

        # Check for duplicates
        existing = await self.provider.list_links()
        logger.debug(
            "Duplicate check: %d existing links for %s", len(existing), product_id
        )
        for link in existing:
            link_url = str(link.get("url", link.get("link", "")))
            if product_id in link_url:
                logger.info("Link-in-bio skipped %s (duplicate)", product_id)
                return {"success": True, "reason": "duplicate", "existing": True}

        # Rotate oldest if at capacity
        if self.max_links > 0 and len(existing) >= self.max_links:
            oldest = existing[-1]
            oldest_id = str(oldest.get("id", oldest.get("link_id", "")))
            if oldest_id:
                logger.info(
                    "Link-in-bio rotated oldest (id=%s, cap=%d)",
                    oldest_id,
                    self.max_links,
                )
                await self.provider.delete_link(oldest_id)
        elif self.max_links > 0:
            logger.debug("Under link cap: %d/%d", len(existing), self.max_links)

        # Truncate title for readability
        if len(title) <= self.max_title_length:
            display_title = title
        else:
            display_title = title[: self.max_title_length - 3] + "..."

        # Resolve image: URL from images array, local file as fallback
        image: str | None = None
        images = product.get("images", [])
        if images:
            image = images[0]

        image_file: Path | None = None
        downloaded = product.get("downloaded_images", [])
        if downloaded:
            candidate = outputs_dir / downloaded[0]
            if candidate.exists():
                image_file = candidate

        logger.debug(
            "Link details: url=%s image=%s image_file=%s",
            url,
            image or "none",
            image_file or "none",
        )
        result = await self.provider.add_link(
            title=display_title,
            url=url,
            image=image,
            image_file=image_file,
        )

        logger.info("Link-in-bio added %s: %s", product_id, display_title[:50])
        return {"success": True, "result": result}


def create_link_in_bio_manager(
    provider_name: str,
    max_links: int = 0,
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
