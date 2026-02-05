"""Lnk.Bio provider for link-in-bio integration."""

import logging
from os import environ

import httpx

from src.publisher.link_in_bio.base import BaseLinkInBioProvider

logger = logging.getLogger(__name__)

TOKEN_URL = "https://lnk.bio/oauth/token"  # noqa: S105
BASE_URL = "https://lnk.bio/oauth/v1"


class LnkBioProvider(BaseLinkInBioProvider):
    """Lnk.Bio REST API provider using OAuth2 client_credentials."""

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.client_id = client_id or environ.get("LNKBIO_CLIENT_ID", "")
        self.client_secret = client_secret or environ.get("LNKBIO_CLIENT_SECRET", "")
        self.timeout = timeout
        self._access_token: str | None = None

    async def authenticate(self) -> bool:
        if not self.client_id or not self.client_secret:
            raise ValueError("LNKBIO_CLIENT_ID and LNKBIO_CLIENT_SECRET must be set")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            self._access_token = data["access_token"]
            logger.info("Lnk.Bio authentication successful")
            return True

    @property
    def _headers(self) -> dict[str, str]:
        if not self._access_token:
            raise RuntimeError("Not authenticated — call authenticate() first")
        return {"Authorization": f"Bearer {self._access_token}"}

    async def _request(
        self,
        method: str,
        path: str,
        data: dict | None = None,
    ) -> dict[str, object]:
        """Make an API request with auto-retry on 401."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.request(
                method,
                f"{BASE_URL}{path}",
                headers=self._headers,
                data=data,
            )

            if resp.status_code == 401:
                logger.debug("Token expired, re-authenticating")
                await self.authenticate()
                resp = await client.request(
                    method,
                    f"{BASE_URL}{path}",
                    headers=self._headers,
                    data=data,
                )

            resp.raise_for_status()
            result: dict[str, object] = resp.json()
            return result

    async def add_link(
        self,
        title: str,
        url: str,
        image: str | None = None,
    ) -> dict[str, object]:
        data: dict[str, str] = {"title": title, "link": url}
        if image:
            data["image"] = image

        result = await self._request("POST", "/lnk/add", data=data)
        logger.info("Created link: %s", title[:50])
        return result

    async def list_links(self) -> list[dict[str, object]]:
        result = await self._request("GET", "/lnks")
        links: list[dict[str, object]] = result.get("data", result.get("lnks", []))  # type: ignore[assignment]
        return links

    async def delete_link(self, link_id: str | int) -> bool:
        await self._request("POST", "/lnk/delete", data={"link_id": str(link_id)})
        logger.info("Deleted link: %s", link_id)
        return True
