"""Retry utilities for network operations with exponential backoff.

Provides decorators for adding retry logic to sync and async functions
that perform network operations (HTTP requests, API calls, etc.).
"""

import logging
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from tenacity import (  # type: ignore[attr-defined]
    AsyncRetrying,
    RetryError,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Status codes that should trigger a retry
RETRYABLE_STATUS_CODES = {429, 503}

# 4xx client errors that should NOT be retried (except 429)
NON_RETRYABLE_CLIENT_ERRORS = {400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410}


def _is_retryable_exception(exc: BaseException) -> bool:
    """Check if an exception should trigger a retry.

    Args:
    ----
        exc: The exception to check.

    Returns:
    -------
        True if the exception is retryable, False otherwise.

    """
    # Import here to avoid hard dependency if libraries not installed
    try:
        import requests

        if isinstance(exc, requests.Timeout | requests.ConnectionError):
            return True
    except ImportError:
        pass

    try:
        import httpx

        if isinstance(exc, httpx.TimeoutException | httpx.ConnectError):
            return True
    except ImportError:
        pass

    # Check for HTTP status code errors
    status_code = _get_status_code(exc)
    if status_code is not None:
        # Retry on 429 (rate limit) and 503 (service unavailable)
        if status_code in RETRYABLE_STATUS_CODES:
            return True
        # Don't retry on other 4xx client errors
        if 400 <= status_code < 500:
            return False
        # Retry on 5xx server errors (except those already handled)
        if 500 <= status_code < 600:
            return True

    # Check for generic network/connection errors
    exc_name = type(exc).__name__.lower()
    retryable_terms = ["timeout", "connection", "network", "socket", "refused"]
    return any(term in exc_name for term in retryable_terms)


def _get_status_code(exc: BaseException) -> int | None:
    """Extract HTTP status code from an exception if available.

    Args:
    ----
        exc: The exception to extract status code from.

    Returns:
    -------
        The HTTP status code, or None if not available.

    """
    # requests library
    if hasattr(exc, "response") and hasattr(exc.response, "status_code"):
        return int(exc.response.status_code)

    # httpx library
    if hasattr(exc, "response") and hasattr(exc.response, "status_code"):
        return int(exc.response.status_code)

    # Check for status_code attribute directly
    if hasattr(exc, "status_code"):
        return int(exc.status_code)

    # Check for status attribute (some libraries use this)
    if hasattr(exc, "status"):
        return int(exc.status)

    return None


def _log_retry(retry_state: Any) -> None:
    """Log retry attempts with useful context.

    Args:
    ----
        retry_state: Tenacity retry state object.

    """
    exc = retry_state.outcome.exception()
    attempt = retry_state.attempt_number
    wait_time = retry_state.next_action.sleep if retry_state.next_action else 0

    exc_type = type(exc).__name__
    exc_msg = str(exc)[:100]  # Truncate long messages

    logger.warning(
        "Retry attempt %d after %s: %s - waiting %.1fs before next attempt",
        attempt,
        exc_type,
        exc_msg,
        wait_time,
    )


def retry_network(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 30.0,
    multiplier: float = 1.0,
) -> Callable[[F], F]:
    """Decorator for retrying network operations with exponential backoff.

    Automatically retries on transient network failures including:
    - Connection timeouts and errors
    - Rate limiting (HTTP 429)
    - Service unavailable (HTTP 503)
    - Other 5xx server errors

    Does NOT retry on:
    - 4xx client errors (except 429)
    - Authentication failures (401, 403)
    - Not found errors (404)

    Works with both sync and async functions.

    Args:
    ----
        max_attempts: Maximum number of retry attempts (default: 3).
        min_wait: Minimum wait time between retries in seconds (default: 1.0).
        max_wait: Maximum wait time between retries in seconds (default: 30.0).
        multiplier: Multiplier for exponential backoff (default: 1.0).

    Returns:
    -------
        Decorated function with retry logic.

    Example:
    -------
        @retry_network()
        def fetch_data(url: str) -> dict:
            return requests.get(url).json()

        @retry_network(max_attempts=5, max_wait=60)
        async def fetch_async(url: str) -> dict:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                return response.json()

    """

    def decorator(func: F) -> F:
        retry_kwargs = {
            "stop": stop_after_attempt(max_attempts),
            "wait": wait_exponential(multiplier=multiplier, min=min_wait, max=max_wait),
            "retry": retry_if_exception(_is_retryable_exception),
            "before_sleep": _log_retry,
            "reraise": True,
        }

        if _is_async_function(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                async for attempt in AsyncRetrying(**retry_kwargs):
                    with attempt:
                        return await func(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]
        else:

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                for attempt in Retrying(**retry_kwargs):  # type: ignore[attr-defined]
                    with attempt:
                        return func(*args, **kwargs)

            return sync_wrapper  # type: ignore[return-value]

    return decorator


def _is_async_function(func: Callable[..., Any]) -> bool:
    """Check if a function is async.

    Args:
    ----
        func: The function to check.

    Returns:
    -------
        True if the function is async, False otherwise.

    """
    import asyncio

    return asyncio.iscoroutinefunction(func)


def is_retryable_status_code(status_code: int) -> bool:
    """Check if an HTTP status code should trigger a retry.

    Args:
    ----
        status_code: The HTTP status code to check.

    Returns:
    -------
        True if the status code is retryable, False otherwise.

    """
    if status_code in RETRYABLE_STATUS_CODES:
        return True
    return bool(500 <= status_code < 600 and status_code != 503)


class RetryableHTTPError(Exception):
    """Exception for HTTP errors that should trigger retries."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def raise_for_retry_status(status_code: int, message: str = "") -> None:
    """Raise RetryableHTTPError if status code warrants a retry.

    Use this helper when you want to trigger retry logic for specific
    HTTP status codes without relying on library-specific exceptions.

    Args:
    ----
        status_code: The HTTP status code to check.
        message: Optional error message.

    Raises:
    ------
        RetryableHTTPError: If the status code should trigger a retry.

    """
    if is_retryable_status_code(status_code):
        raise RetryableHTTPError(
            message or f"HTTP {status_code} - retryable error", status_code
        )
