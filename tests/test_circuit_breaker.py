"""Tests for circuit breaker implementation."""

import asyncio
import contextlib

import pytest

from src.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    _load_circuit_breaker_config,
    freesound_circuit_breaker,
    google_stt_circuit_breaker,
    llm_circuit_breaker,
    pexels_circuit_breaker,
    scraper_circuit_breaker,
)


class TestCircuitBreaker:
    """Test cases for CircuitBreaker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout=1,  # Short timeout for testing
            expected_exceptions=(ConnectionError, TimeoutError),
            name="TestBreaker",
        )

    @pytest.mark.asyncio
    async def test_closed_state_allows_calls(self):
        """Test that CLOSED state allows function calls through."""

        @self.circuit_breaker
        async def test_function():
            return "success"

        result = await test_function()
        assert result == "success"
        assert self.circuit_breaker.is_closed

    @pytest.mark.asyncio
    async def test_successful_calls_reset_failure_count(self):
        """Test that successful calls reset failure tracking."""

        @self.circuit_breaker
        async def test_function(should_fail=False):
            if should_fail:
                raise ConnectionError("Test error")
            return "success"

        # Cause some failures
        for _ in range(2):
            with contextlib.suppress(ConnectionError):
                await test_function(should_fail=True)

        assert self.circuit_breaker.failure_count == 2

        # Successful call should reset count
        result = await test_function(should_fail=False)
        assert result == "success"
        assert self.circuit_breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold_failures(self):
        """Test that circuit opens after reaching failure threshold."""

        @self.circuit_breaker
        async def failing_function():
            raise ConnectionError("Simulated failure")

        # Trigger failures up to threshold
        for _i in range(3):
            try:
                await failing_function()
            except ConnectionError:
                pass
            except CircuitBreakerError:
                pytest.fail("Circuit should not be open yet")

        assert self.circuit_breaker.is_open

        # Next call should fail immediately with CircuitBreakerError
        with pytest.raises(CircuitBreakerError) as exc_info:
            await failing_function()

        assert "Circuit breaker TestBreaker is OPEN" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_circuit_transitions_to_half_open_after_timeout(self):
        """Test that circuit transitions to HALF_OPEN after timeout period."""

        @self.circuit_breaker
        async def failing_function():
            raise ConnectionError("Simulated failure")

        # Open the circuit
        for _ in range(3):
            with contextlib.suppress(ConnectionError):
                await failing_function()

        assert self.circuit_breaker.is_open

        # Wait for timeout to expire
        await asyncio.sleep(1.1)  # Slightly more than 1 second timeout

        # Next call should transition to HALF_OPEN
        with contextlib.suppress(ConnectionError):
            await failing_function()
            # Expected - function still fails, but circuit is now OPEN again

        # Circuit should have attempted HALF_OPEN transition
        assert self.circuit_breaker.total_calls == 4

    @pytest.mark.asyncio
    async def test_circuit_recovers_on_successful_half_open_call(self):
        """Test that circuit recovers to CLOSED on successful HALF_OPEN call."""
        call_count = 0

        @self.circuit_breaker
        async def test_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise ConnectionError("Simulated failure")
            return "success"

        # Open the circuit
        for _ in range(3):
            with contextlib.suppress(ConnectionError):
                await test_function()

        assert self.circuit_breaker.is_open

        # Wait for timeout
        await asyncio.sleep(1.1)

        # Next call should succeed and close circuit
        result = await test_function()
        assert result == "success"
        assert self.circuit_breaker.is_closed

    def test_sync_function_wrapper(self):
        """Test circuit breaker with synchronous functions."""

        @self.circuit_breaker
        def sync_function(should_fail=False):
            if should_fail:
                raise ConnectionError("Test error")
            return "success"

        # Should work normally
        result = sync_function(should_fail=False)
        assert result == "success"

        # Should handle failures
        for _ in range(3):
            with contextlib.suppress(ConnectionError):
                sync_function(should_fail=True)

        assert self.circuit_breaker.is_open

        # Should fail fast
        with pytest.raises(CircuitBreakerError):
            sync_function(should_fail=True)

    def test_unexpected_exceptions_pass_through(self):
        """Test that unexpected exceptions pass through without affecting circuit."""

        @self.circuit_breaker
        def test_function():
            raise ValueError("Unexpected error")

        # Unexpected exception should pass through
        with pytest.raises(ValueError):
            test_function()

        # Circuit should remain closed
        assert self.circuit_breaker.is_closed
        assert self.circuit_breaker.failure_count == 0

    def test_get_stats_returns_correct_metrics(self):
        """Test that statistics are correctly tracked and returned."""

        @self.circuit_breaker
        def test_function(should_fail=False):
            if should_fail:
                raise ConnectionError("Test error")
            return "success"

        # Make some calls
        test_function(should_fail=False)  # Success
        test_function(should_fail=False)  # Success
        with contextlib.suppress(ConnectionError):
            test_function(should_fail=True)  # Failure

        stats = self.circuit_breaker.get_stats()

        assert stats["name"] == "TestBreaker"
        assert stats["state"] == "CLOSED"
        assert stats["total_calls"] == 3
        assert stats["successful_calls"] == 2
        assert stats["failed_calls"] == 1
        assert stats["success_rate"] == 2 / 3
        assert stats["failure_count"] == 1
        assert stats["failure_threshold"] == 3

    def test_manual_reset(self):
        """Test manual circuit breaker reset."""

        @self.circuit_breaker
        def failing_function():
            raise ConnectionError("Test error")

        # Open the circuit
        for _ in range(3):
            with contextlib.suppress(ConnectionError):
                failing_function()

        assert self.circuit_breaker.is_open
        assert self.circuit_breaker.failure_count == 3

        # Manual reset
        self.circuit_breaker.reset()

        assert self.circuit_breaker.is_closed
        assert self.circuit_breaker.failure_count == 0


class TestPreconfiguredCircuitBreakers:
    """Test pre-configured circuit breakers for specific services."""

    def test_google_stt_circuit_breaker_configuration(self):
        """Test Google STT circuit breaker has appropriate configuration."""
        assert google_stt_circuit_breaker.name == "GoogleSTT"
        assert google_stt_circuit_breaker.failure_threshold == 3
        assert google_stt_circuit_breaker.timeout == 120
        assert ConnectionError in google_stt_circuit_breaker.expected_exceptions

    def test_freesound_circuit_breaker_configuration(self):
        """Test Freesound circuit breaker has appropriate configuration."""
        assert freesound_circuit_breaker.name == "Freesound"
        assert freesound_circuit_breaker.failure_threshold == 3
        assert freesound_circuit_breaker.timeout == 60
        assert ConnectionError in freesound_circuit_breaker.expected_exceptions

    def test_pexels_circuit_breaker_configuration(self):
        """Test Pexels circuit breaker has appropriate configuration."""
        assert pexels_circuit_breaker.name == "Pexels"
        assert pexels_circuit_breaker.failure_threshold == 3
        assert pexels_circuit_breaker.timeout == 90
        assert ConnectionError in pexels_circuit_breaker.expected_exceptions

    def test_llm_circuit_breaker_configuration(self):
        """Test LLM circuit breaker has appropriate configuration."""
        assert llm_circuit_breaker.name == "LLM"
        assert llm_circuit_breaker.failure_threshold == 2
        assert llm_circuit_breaker.timeout == 30
        assert ConnectionError in llm_circuit_breaker.expected_exceptions

    def test_scraper_circuit_breaker_configuration(self):
        """Test Scraper circuit breaker has appropriate configuration."""
        assert scraper_circuit_breaker.name == "Scraper"
        assert scraper_circuit_breaker.failure_threshold == 5
        assert scraper_circuit_breaker.timeout == 60
        assert ConnectionError in scraper_circuit_breaker.expected_exceptions


class TestCircuitBreakerConfigLoading:
    """Test configuration loading from performance.yaml."""

    def test_load_config_returns_dict(self):
        """Test that _load_circuit_breaker_config returns a dictionary."""
        config = _load_circuit_breaker_config()
        assert isinstance(config, dict)

    def test_load_config_contains_expected_services(self):
        """Test that loaded config contains expected service configurations."""
        config = _load_circuit_breaker_config()
        # Config should contain all services defined in performance.yaml
        expected_services = [
            "google_stt",
            "freesound",
            "pexels",
            "llm",
            "scraper",
        ]
        for service in expected_services:
            assert service in config, f"Missing config for {service}"
            assert "failure_threshold" in config[service]
            assert "timeout_sec" in config[service]

    def test_load_config_values_match_yaml(self):
        """Test that loaded values match config/performance.yaml settings."""
        config = _load_circuit_breaker_config()

        # Values should match what's in config/performance.yaml
        assert config["google_stt"]["failure_threshold"] == 3
        assert config["google_stt"]["timeout_sec"] == 120

        assert config["freesound"]["failure_threshold"] == 3
        assert config["freesound"]["timeout_sec"] == 60

        assert config["pexels"]["failure_threshold"] == 3
        assert config["pexels"]["timeout_sec"] == 90

        assert config["llm"]["failure_threshold"] == 2
        assert config["llm"]["timeout_sec"] == 30

        assert config["scraper"]["failure_threshold"] == 5
        assert config["scraper"]["timeout_sec"] == 60

    def test_preconfigured_breakers_use_config_values(self):
        """Test that pre-configured circuit breakers use values from config."""
        config = _load_circuit_breaker_config()

        # Verify each pre-configured breaker uses config values
        assert (
            google_stt_circuit_breaker.failure_threshold
            == config["google_stt"]["failure_threshold"]
        )
        assert google_stt_circuit_breaker.timeout == config["google_stt"]["timeout_sec"]

        assert (
            freesound_circuit_breaker.failure_threshold
            == config["freesound"]["failure_threshold"]
        )
        assert freesound_circuit_breaker.timeout == config["freesound"]["timeout_sec"]

        assert (
            pexels_circuit_breaker.failure_threshold
            == config["pexels"]["failure_threshold"]
        )
        assert pexels_circuit_breaker.timeout == config["pexels"]["timeout_sec"]

        assert (
            llm_circuit_breaker.failure_threshold
            == config["llm"]["failure_threshold"]
        )
        assert llm_circuit_breaker.timeout == config["llm"]["timeout_sec"]

        assert (
            scraper_circuit_breaker.failure_threshold
            == config["scraper"]["failure_threshold"]
        )
        assert scraper_circuit_breaker.timeout == config["scraper"]["timeout_sec"]


@pytest.mark.asyncio
async def test_circuit_breaker_integration_example():
    """Integration test demonstrating circuit breaker usage."""
    failure_count = 0

    # Create circuit breaker with low threshold for testing
    test_breaker = CircuitBreaker(
        failure_threshold=2,
        timeout=0.5,
        expected_exceptions=(ConnectionError,),
        name="IntegrationTest",
    )

    @test_breaker
    async def unstable_service_call():
        nonlocal failure_count
        failure_count += 1
        if failure_count <= 2:
            raise ConnectionError("Service unavailable")
        return "Service recovered"

    # First two calls should fail and open circuit
    for _i in range(2):
        try:
            await unstable_service_call()
            pytest.fail("Expected ConnectionError")
        except ConnectionError:
            pass

    assert test_breaker.is_open

    # Third call should fail fast
    with pytest.raises(CircuitBreakerError):
        await unstable_service_call()

    # Wait for recovery timeout
    await asyncio.sleep(0.6)

    # Service should now work and circuit should close
    result = await unstable_service_call()
    assert result == "Service recovered"
    assert test_breaker.is_closed

    # Verify statistics
    stats = test_breaker.get_stats()
    assert stats["total_calls"] == 4
    assert stats["successful_calls"] == 1
    assert (
        stats["failed_calls"] == 2
    )  # CircuitBreakerError doesn't count as failed call
