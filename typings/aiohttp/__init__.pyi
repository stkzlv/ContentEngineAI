from types import TracebackType
from typing import Any, TypeVar

_T = TypeVar("_T")
_RetValT = TypeVar("_RetValT")

class ClientSession:
    def __init__(self) -> None: ...
    async def __aenter__(self) -> ClientSession: ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...
    def post(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json: Any = None,
        timeout: int | None = None,
    ) -> ClientResponse: ...

class ClientResponse:
    async def __aenter__(self) -> ClientResponse: ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...
    async def json(self) -> Any: ...
    async def text(self) -> str: ...
    def raise_for_status(self) -> None: ...
    @property
    def status(self) -> int: ...

class ClientError(Exception): ...

class ClientResponseError(ClientError):
    status: int
    message: str

class ContentTypeError(ClientError): ...

class ClientTimeout:
    total: float | None
    connect: float | None
    sock_read: float | None
    sock_connect: float | None
    def __init__(
        self,
        total: float | None = None,
        connect: float | None = None,
        sock_read: float | None = None,
        sock_connect: float | None = None,
    ) -> None: ...
