"""Unit tests for retry utilities module.

Tests cover the @retry_network decorator for both sync and async functions,
exception filtering logic, and helper functions.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.utils.retry import (
    RETRYABLE_STATUS_CODES,
    RetryableHTTPError,
    _get_status_code,
    _is_async_function,
    _is_retryable_exception,
    is_retryable_status_code,
    raise_for_retry_status,
    retry_network,
)


class TestIsRetryableException:
    """Tests for _is_retryable_exception function."""

    def test_requests_timeout_is_retryable(self):
        """requests.Timeout should trigger retry."""
        exc = requests.Timeout("Connection timed out")
        assert _is_retryable_exception(exc) is True

    def test_requests_connection_error_is_retryable(self):
        """requests.ConnectionError should trigger retry."""
        exc = requests.ConnectionError("Connection refused")
        assert _is_retryable_exception(exc) is True

    def test_status_429_is_retryable(self):
        """HTTP 429 (rate limit) should trigger retry."""
        exc = RetryableHTTPError("Rate limited", 429)
        assert _is_retryable_exception(exc) is True

    def test_status_503_is_retryable(self):
        """HTTP 503 (service unavailable) should trigger retry."""
        exc = RetryableHTTPError("Service unavailable", 503)
        assert _is_retryable_exception(exc) is True

    def test_status_500_is_retryable(self):
        """HTTP 500 (internal server error) should trigger retry."""
        exc = RetryableHTTPError("Internal error", 500)
        assert _is_retryable_exception(exc) is True

    def test_status_401_not_retryable(self):
        """HTTP 401 (unauthorized) should NOT trigger retry."""
        exc = RetryableHTTPError("Unauthorized", 401)
        assert _is_retryable_exception(exc) is False

    def test_status_403_not_retryable(self):
        """HTTP 403 (forbidden) should NOT trigger retry."""
        exc = RetryableHTTPError("Forbidden", 403)
        assert _is_retryable_exception(exc) is False

    def test_status_404_not_retryable(self):
        """HTTP 404 (not found) should NOT trigger retry."""
        exc = RetryableHTTPError("Not found", 404)
        assert _is_retryable_exception(exc) is False

    def test_timeout_in_name_is_retryable(self):
        """Exceptions with 'timeout' in name should trigger retry."""

        class CustomTimeoutError(Exception):
            pass

        exc = CustomTimeoutError("Timed out")
        assert _is_retryable_exception(exc) is True

    def test_connection_in_name_is_retryable(self):
        """Exceptions with 'connection' in name should trigger retry."""

        class ConnectionFailedError(Exception):
            pass

        exc = ConnectionFailedError("Connection failed")
        assert _is_retryable_exception(exc) is True

    def test_generic_value_error_not_retryable(self):
        """Generic ValueError should NOT trigger retry."""
        exc = ValueError("Invalid value")
        assert _is_retryable_exception(exc) is False

    def test_generic_runtime_error_not_retryable(self):
        """Generic RuntimeError should NOT trigger retry."""
        exc = RuntimeError("Something went wrong")
        assert _is_retryable_exception(exc) is False


class TestGetStatusCode:
    """Tests for _get_status_code function."""

    def test_extracts_status_code_from_response_attribute(self):
        """Should extract status_code from response attribute."""
        exc = MagicMock()
        exc.response = MagicMock()
        exc.response.status_code = 503
        assert _get_status_code(exc) == 503

    def test_extracts_status_code_directly(self):
        """Should extract status_code from direct attribute."""
        exc = RetryableHTTPError("Error", 429)
        assert _get_status_code(exc) == 429

    def test_extracts_status_from_status_attribute(self):
        """Should extract from status attribute."""
        exc = MagicMock(spec=[])
        exc.status = 500
        assert _get_status_code(exc) == 500

    def test_returns_none_when_no_status(self):
        """Should return None when no status code available."""
        exc = ValueError("No status")
        assert _get_status_code(exc) is None


class TestIsRetryableStatusCode:
    """Tests for is_retryable_status_code function."""

    def test_429_is_retryable(self):
        assert is_retryable_status_code(429) is True

    def test_503_is_retryable(self):
        assert is_retryable_status_code(503) is True

    def test_500_is_retryable(self):
        assert is_retryable_status_code(500) is True

    def test_502_is_retryable(self):
        assert is_retryable_status_code(502) is True

    def test_200_not_retryable(self):
        assert is_retryable_status_code(200) is False

    def test_400_not_retryable(self):
        assert is_retryable_status_code(400) is False

    def test_401_not_retryable(self):
        assert is_retryable_status_code(401) is False

    def test_404_not_retryable(self):
        assert is_retryable_status_code(404) is False


class TestRaiseForRetryStatus:
    """Tests for raise_for_retry_status function."""

    def test_raises_for_429(self):
        with pytest.raises(RetryableHTTPError) as exc_info:
            raise_for_retry_status(429, "Rate limited")
        assert exc_info.value.status_code == 429

    def test_raises_for_503(self):
        with pytest.raises(RetryableHTTPError) as exc_info:
            raise_for_retry_status(503)
        assert exc_info.value.status_code == 503

    def test_raises_for_500(self):
        with pytest.raises(RetryableHTTPError) as exc_info:
            raise_for_retry_status(500, "Server error")
        assert exc_info.value.status_code == 500

    def test_no_raise_for_200(self):
        raise_for_retry_status(200)  # Should not raise

    def test_no_raise_for_404(self):
        raise_for_retry_status(404)  # Should not raise


class TestIsAsyncFunction:
    """Tests for _is_async_function helper."""

    def test_sync_function_returns_false(self):
        def sync_func():
            pass

        assert _is_async_function(sync_func) is False

    def test_async_function_returns_true(self):
        async def async_func():
            pass

        assert _is_async_function(async_func) is True

    def test_lambda_returns_false(self):
        assert _is_async_function(lambda: None) is False


class TestRetryNetworkSync:
    """Tests for @retry_network decorator with sync functions."""

    def test_successful_call_no_retry(self):
        """Successful call should not trigger any retries."""
        call_count = 0

        @retry_network(max_attempts=3, min_wait=0.01, max_wait=0.01)
        def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = success_func()
        assert result == "success"
        assert call_count == 1

    def test_transient_failure_then_success(self):
        """Should retry on transient failure and eventually succeed."""
        call_count = 0

        @retry_network(max_attempts=3, min_wait=0.01, max_wait=0.01)
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise requests.Timeout("Timed out")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert call_count == 3

    def test_max_attempts_exceeded_raises(self):
        """Should raise after max attempts exceeded."""
        call_count = 0

        @retry_network(max_attempts=3, min_wait=0.01, max_wait=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise requests.Timeout("Always times out")

        with pytest.raises(requests.Timeout):
            always_fails()
        assert call_count == 3

    def test_non_retryable_exception_raises_immediately(self):
        """Non-retryable exceptions should raise immediately without retry."""
        call_count = 0

        @retry_network(max_attempts=3, min_wait=0.01, max_wait=0.01)
        def auth_error_func():
            nonlocal call_count
            call_count += 1
            raise RetryableHTTPError("Unauthorized", 401)

        with pytest.raises(RetryableHTTPError):
            auth_error_func()
        assert call_count == 1  # No retries

    def test_value_error_raises_immediately(self):
        """ValueError should not trigger retry."""
        call_count = 0

        @retry_network(max_attempts=3, min_wait=0.01, max_wait=0.01)
        def bad_input_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Bad input")

        with pytest.raises(ValueError):
            bad_input_func()
        assert call_count == 1

    def test_connection_error_retries(self):
        """ConnectionError should trigger retries."""
        call_count = 0

        @retry_network(max_attempts=3, min_wait=0.01, max_wait=0.01)
        def connection_error_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise requests.ConnectionError("Connection refused")
            return "connected"

        result = connection_error_func()
        assert result == "connected"
        assert call_count == 2

    def test_preserves_function_metadata(self):
        """Decorator should preserve function name and docstring."""

        @retry_network()
        def documented_func():
            """Example docstring for testing metadata preservation."""
            return True

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ is not None
        assert "docstring" in documented_func.__doc__


class TestRetryNetworkAsync:
    """Tests for @retry_network decorator with async functions."""

    @pytest.mark.asyncio
    async def test_async_successful_call_no_retry(self):
        """Async successful call should not trigger any retries."""
        call_count = 0

        @retry_network(max_attempts=3, min_wait=0.01, max_wait=0.01)
        async def async_success():
            nonlocal call_count
            call_count += 1
            return "async success"

        result = await async_success()
        assert result == "async success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_transient_failure_then_success(self):
        """Async should retry on transient failure and eventually succeed."""
        call_count = 0

        @retry_network(max_attempts=3, min_wait=0.01, max_wait=0.01)
        async def async_flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise requests.Timeout("Async timeout")
            return "recovered"

        result = await async_flaky()
        assert result == "recovered"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_max_attempts_exceeded(self):
        """Async should raise after max attempts exceeded."""
        call_count = 0

        @retry_network(max_attempts=2, min_wait=0.01, max_wait=0.01)
        async def async_always_fails():
            nonlocal call_count
            call_count += 1
            raise requests.ConnectionError("Always fails")

        with pytest.raises(requests.ConnectionError):
            await async_always_fails()
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_non_retryable_raises_immediately(self):
        """Async non-retryable exceptions should raise immediately."""
        call_count = 0

        @retry_network(max_attempts=3, min_wait=0.01, max_wait=0.01)
        async def async_auth_error():
            nonlocal call_count
            call_count += 1
            raise RetryableHTTPError("Forbidden", 403)

        with pytest.raises(RetryableHTTPError):
            await async_auth_error()
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_preserves_function_metadata(self):
        """Async decorator should preserve function metadata."""

        @retry_network()
        async def async_documented():
            """Example async docstring for metadata test."""
            return True

        assert async_documented.__name__ == "async_documented"
        assert async_documented.__doc__ is not None
        assert "docstring" in async_documented.__doc__


class TestRetryNetworkParameters:
    """Tests for @retry_network decorator parameters."""

    def test_custom_max_attempts(self):
        """Should respect custom max_attempts."""
        call_count = 0

        @retry_network(max_attempts=5, min_wait=0.01, max_wait=0.01)
        def five_attempts():
            nonlocal call_count
            call_count += 1
            raise requests.Timeout("Timeout")

        with pytest.raises(requests.Timeout):
            five_attempts()
        assert call_count == 5

    def test_single_attempt(self):
        """max_attempts=1 should not retry."""
        call_count = 0

        @retry_network(max_attempts=1, min_wait=0.01, max_wait=0.01)
        def single_attempt():
            nonlocal call_count
            call_count += 1
            raise requests.Timeout("Timeout")

        with pytest.raises(requests.Timeout):
            single_attempt()
        assert call_count == 1


class TestRetryableHTTPError:
    """Tests for RetryableHTTPError exception class."""

    def test_stores_status_code(self):
        exc = RetryableHTTPError("Error message", 503)
        assert exc.status_code == 503

    def test_stores_message(self):
        exc = RetryableHTTPError("Custom error", 429)
        assert str(exc) == "Custom error"

    def test_is_exception_subclass(self):
        exc = RetryableHTTPError("Error", 500)
        assert isinstance(exc, Exception)


class TestRetryableStatusCodes:
    """Tests for RETRYABLE_STATUS_CODES constant."""

    def test_contains_429(self):
        assert 429 in RETRYABLE_STATUS_CODES

    def test_contains_503(self):
        assert 503 in RETRYABLE_STATUS_CODES

    def test_not_contains_401(self):
        assert 401 not in RETRYABLE_STATUS_CODES

    def test_not_contains_404(self):
        assert 404 not in RETRYABLE_STATUS_CODES
