"""Lnk.Bio provider for link-in-bio integration."""

import logging
from os import environ
from pathlib import Path

import httpx

from src.publisher.link_in_bio.base import BaseLinkInBioProvider

logger = logging.getLogger(__name__)

TOKEN_URL = "https://lnk.bio/oauth/token"  # noqa: S105
BASE_URL = "https://lnk.bio/oauth/v1"
_DEFAULT_HEADERS = {"User-Agent": "ContentEngineAI/1.0"}


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

        async with httpx.AsyncClient(
            timeout=self.timeout, headers=_DEFAULT_HEADERS
        ) as client:
            resp = await client.post(
                TOKEN_URL,
                auth=(self.client_id, self.client_secret),
                data={"grant_type": "client_credentials"},
            )
            resp.raise_for_status()
            data = resp.json()
            self._access_token = data["access_token"]
            logger.debug("Lnk.Bio auth token obtained (status=%d)", resp.status_code)
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
        logger.debug("API request: %s %s", method, path)
        async with httpx.AsyncClient(
            timeout=self.timeout, headers=_DEFAULT_HEADERS
        ) as client:
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
            logger.debug("API response: %s %s -> %d", method, path, resp.status_code)
            return result

    async def add_link(
        self,
        title: str,
        url: str,
        image: str | None = None,
        image_file: Path | None = None,
    ) -> dict[str, object]:
        data: dict[str, str] = {"title": title, "link": url}
        files: dict[str, tuple[str, bytes, str]] | None = None

        if image:
            data["image"] = image
        elif image_file and image_file.exists():
            img_bytes = image_file.read_bytes()
            files = {"image": (image_file.name, img_bytes, "image/jpeg")}
            logger.debug("Using local image fallback: %s", image_file)

        if files:
            result = await self._request_multipart("/lnk/add", data=data, files=files)
        else:
            result = await self._request("POST", "/lnk/add", data=data)

        logger.debug(
            "Link created: title=%s url=%s image=%s",
            title[:50],
            url,
            image or (str(image_file) if image_file else "none"),
        )
        return result

    async def _request_multipart(
        self,
        path: str,
        data: dict[str, str],
        files: dict[str, tuple[str, bytes, str]],
    ) -> dict[str, object]:
        """POST multipart form data (for file uploads)."""
        logger.debug("API multipart request: POST %s", path)
        async with httpx.AsyncClient(
            timeout=self.timeout, headers=_DEFAULT_HEADERS
        ) as client:
            resp = await client.post(
                f"{BASE_URL}{path}",
                headers=self._headers,
                data=data,
                files=files,
            )

            if resp.status_code == 401:
                logger.debug("Token expired, re-authenticating")
                await self.authenticate()
                resp = await client.post(
                    f"{BASE_URL}{path}",
                    headers=self._headers,
                    data=data,
                    files=files,
                )

            resp.raise_for_status()
            result: dict[str, object] = resp.json()
            logger.debug("API response: POST %s -> %d", path, resp.status_code)
            return result

    async def list_links(self) -> list[dict[str, object]]:
        result = await self._request("GET", "/lnk/list")
        links: list[dict[str, object]] = result.get("data", [])  # type: ignore[assignment]
        logger.debug("Listed %d existing links", len(links))
        return links

    async def delete_link(self, link_id: str | int) -> bool:
        await self._request("POST", "/lnk/delete", data={"link_id": str(link_id)})
        logger.debug("Deleted link id=%s", link_id)
        return True
