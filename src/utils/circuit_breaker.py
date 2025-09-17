"""Circuit breaker pattern implementation for external service reliability.

This module provides circuit breaker functionality to prevent cascading failures
when external services (Google Cloud STT, Freesound API, Pexels API) are unavailable.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from enum import Enum
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "CLOSED"  # Normal operation, calls pass through
    OPEN = "OPEN"  # Failure mode, calls fail immediately
    HALF_OPEN = "HALF_OPEN"  # Recovery testing, limited calls allowed


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is in OPEN state."""

    pass


class CircuitBreaker:
    """Circuit breaker for external service calls.

    Prevents cascading failures by monitoring external service failures
    and temporarily blocking calls when failure threshold is exceeded.

    Args:
    ----
        failure_threshold: Number of consecutive failures before opening circuit
        timeout: Seconds to wait before attempting recovery (HALF_OPEN)
        expected_exceptions: Exception types that count as failures
        name: Human-readable name for logging

    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exceptions: tuple[type[Exception], ...] = (Exception,),
        name: str = "CircuitBreaker",
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exceptions = expected_exceptions
        self.name = name

        # State tracking
        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.state = CircuitState.CLOSED
        self.next_attempt_time = 0.0

        # Statistics
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0

    def __call__(self, func: F) -> F:
        """Decorator to wrap function with circuit breaker logic."""

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await self._call_with_circuit_breaker(func, *args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                raise ValueError("Use async wrapper for async functions")
            return self._call_with_circuit_breaker_sync(func, *args, **kwargs)

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore

    async def _call_with_circuit_breaker(self, func: Callable, *args, **kwargs):
        """Execute async function with circuit breaker protection."""
        self.total_calls += 1

        # Check circuit state before attempting call
        if self.state == CircuitState.OPEN:
            if time.time() < self.next_attempt_time:
                logger.warning(
                    f"Circuit breaker {self.name} is OPEN, failing fast. "
                    f"Next attempt in {self.next_attempt_time - time.time():.1f}s"
                )
                raise CircuitBreakerError(
                    f"Circuit breaker {self.name} is OPEN. Service temporarily "
                    f"unavailable."
                )
            else:
                # Transition to HALF_OPEN for recovery attempt
                self._transition_to_half_open()

        try:
            # Execute the protected function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Success - reset failure tracking
            self._on_success()
            return result

        except self.expected_exceptions as e:
            # Expected failure - track and potentially open circuit
            self._on_failure(e)
            raise
        except Exception:
            # Unexpected exception - let it propagate without affecting circuit
            raise

    def _call_with_circuit_breaker_sync(self, func: Callable, *args, **kwargs):
        """Execute sync function with circuit breaker protection."""
        self.total_calls += 1

        # Check circuit state before attempting call
        if self.state == CircuitState.OPEN:
            if time.time() < self.next_attempt_time:
                logger.warning(
                    f"Circuit breaker {self.name} is OPEN, failing fast. "
                    f"Next attempt in {self.next_attempt_time - time.time():.1f}s"
                )
                raise CircuitBreakerError(
                    f"Circuit breaker {self.name} is OPEN. Service temporarily "
                    f"unavailable."
                )
            else:
                # Transition to HALF_OPEN for recovery attempt
                self._transition_to_half_open()

        try:
            # Execute the protected function
            result = func(*args, **kwargs)

            # Success - reset failure tracking
            self._on_success()
            return result

        except self.expected_exceptions as e:
            # Expected failure - track and potentially open circuit
            self._on_failure(e)
            raise
        except Exception:
            # Unexpected exception - let it propagate without affecting circuit
            raise

    def _on_success(self):
        """Handle successful function execution."""
        self.successful_calls += 1
        self.failure_count = 0

        if self.state == CircuitState.HALF_OPEN:
            # Recovery successful, close the circuit
            self.state = CircuitState.CLOSED
            logger.info(f"Circuit breaker {self.name} recovered - state: CLOSED")

        logger.debug(f"Circuit breaker {self.name} - successful call")

    def _on_failure(self, exception: Exception):
        """Handle failed function execution."""
        self.failed_calls += 1
        self.failure_count += 1
        self.last_failure_time = time.time()

        logger.warning(
            f"Circuit breaker {self.name} - failure "
            f"{self.failure_count}/{self.failure_threshold}: {exception}"
        )

        if self.failure_count >= self.failure_threshold:
            # Open the circuit
            self.state = CircuitState.OPEN
            self.next_attempt_time = time.time() + self.timeout
            logger.error(
                f"Circuit breaker {self.name} OPENED - threshold reached. "
                f"Will retry after {self.timeout}s"
            )

    def _transition_to_half_open(self):
        """Transition circuit to HALF_OPEN state for recovery testing."""
        self.state = CircuitState.HALF_OPEN
        logger.info(
            f"Circuit breaker {self.name} attempting recovery - state: HALF_OPEN"
        )

    @property
    def is_closed(self) -> bool:
        """Check if circuit is in CLOSED state (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is in OPEN state (failing fast)."""
        return self.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is in HALF_OPEN state (recovery testing)."""
        return self.state == CircuitState.HALF_OPEN

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        success_rate = (
            self.successful_calls / self.total_calls if self.total_calls > 0 else 0.0
        )

        return {
            "name": self.name,
            "state": self.state.value,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": success_rate,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "timeout": self.timeout,
            "last_failure_time": self.last_failure_time,
            "next_attempt_time": self.next_attempt_time if self.is_open else None,
        }

    def reset(self):
        """Manually reset circuit breaker to CLOSED state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.next_attempt_time = 0.0
        logger.info(f"Circuit breaker {self.name} manually reset to CLOSED")


# Pre-configured circuit breakers for common external services
google_stt_circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    timeout=120,  # 2 minutes recovery time
    expected_exceptions=(
        ConnectionError,
        TimeoutError,
        OSError,  # Network-related errors
    ),
    name="GoogleSTT",
)

freesound_circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    timeout=60,  # 1 minute recovery time
    expected_exceptions=(
        ConnectionError,
        TimeoutError,
        OSError,  # Network-related errors
    ),
    name="Freesound",
)

pexels_circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    timeout=90,  # 1.5 minutes recovery time
    expected_exceptions=(
        ConnectionError,
        TimeoutError,
        OSError,  # Network-related errors
    ),
    name="Pexels",
)

openrouter_circuit_breaker = CircuitBreaker(
    failure_threshold=2,
    timeout=30,  # 30 seconds recovery time (faster for AI services)
    expected_exceptions=(
        ConnectionError,
        TimeoutError,
        OSError,  # Network-related errors
    ),
    name="OpenRouter",
)
