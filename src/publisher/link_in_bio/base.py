"""Abstract base class for link-in-bio providers."""

from abc import ABC, abstractmethod


class BaseLinkInBioProvider(ABC):
    """Interface for link-in-bio services (Lnk.Bio, Linktree, etc.)."""

    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with the provider. Returns True on success."""

    @abstractmethod
    async def add_link(
        self,
        title: str,
        url: str,
        image: str | None = None,
    ) -> dict[str, object]:
        """Add a link to the bio page. Returns provider response."""

    @abstractmethod
    async def list_links(self) -> list[dict[str, object]]:
        """List all current links. Each dict has at least 'id', 'title', 'url'."""

    @abstractmethod
    async def delete_link(self, link_id: str | int) -> bool:
        """Delete a link by ID. Returns True on success."""
