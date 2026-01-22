"""Unit tests for pipeline webhook module."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from src.pipeline.webhooks import (
    WebhookConfig,
    WebhookNotifier,
    load_webhook_config,
    validate_webhook_url,
)


class TestWebhookConfig:
    """Tests for WebhookConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WebhookConfig()

        assert config.url is None
        assert config.enabled is True
        assert config.timeout_sec == 5.0
        assert config.max_retries == 3
        assert config.retry_delay_sec == 1.0
        assert "phase.complete" in config.events
        assert "phase.failed" in config.events
        assert "pipeline.complete" in config.events
        assert "pipeline.failed" in config.events

    def test_custom_config(self):
        """Test custom configuration values."""
        config = WebhookConfig(
            url="https://example.com/webhook",
            enabled=False,
            timeout_sec=10.0,
            max_retries=5,
            retry_delay_sec=2.0,
            events=["phase.complete"],
        )

        assert config.url == "https://example.com/webhook"
        assert config.enabled is False
        assert config.timeout_sec == 10.0
        assert config.max_retries == 5
        assert config.retry_delay_sec == 2.0
        assert config.events == ["phase.complete"]

    def test_is_configured_true(self):
        """Test is_configured returns True when URL set and enabled."""
        config = WebhookConfig(url="https://example.com/webhook", enabled=True)
        assert config.is_configured() is True

    def test_is_configured_false_no_url(self):
        """Test is_configured returns False when no URL."""
        config = WebhookConfig(url=None, enabled=True)
        assert config.is_configured() is False

    def test_is_configured_false_disabled(self):
        """Test is_configured returns False when disabled."""
        config = WebhookConfig(url="https://example.com/webhook", enabled=False)
        assert config.is_configured() is False


class TestValidateWebhookUrl:
    """Tests for validate_webhook_url function."""

    def test_valid_https_url(self):
        """Test valid HTTPS URL passes validation."""
        is_valid, error = validate_webhook_url("https://example.com/webhook")
        assert is_valid is True
        assert error is None

    def test_valid_http_url(self):
        """Test valid HTTP URL passes validation."""
        is_valid, error = validate_webhook_url("http://localhost:8080/webhook")
        assert is_valid is True
        assert error is None

    def test_empty_url(self):
        """Test empty URL fails validation."""
        is_valid, error = validate_webhook_url("")
        assert is_valid is False
        assert error is not None
        assert "empty" in error.lower()

    def test_invalid_scheme(self):
        """Test invalid URL scheme fails validation."""
        is_valid, error = validate_webhook_url("ftp://example.com/webhook")
        assert is_valid is False
        assert error is not None
        assert "scheme" in error.lower()

    def test_missing_hostname(self):
        """Test URL missing hostname fails validation."""
        is_valid, error = validate_webhook_url("https:///webhook")
        assert is_valid is False
        assert error is not None
        assert "hostname" in error.lower()


class TestWebhookNotifier:
    """Tests for WebhookNotifier class."""

    def test_notifier_initialization_valid_url(self):
        """Test notifier initializes with valid URL."""
        config = WebhookConfig(url="https://example.com/webhook")
        notifier = WebhookNotifier(config)

        assert notifier.is_ready() is True
        assert notifier._validated is True
        assert notifier._validation_error is None

    def test_notifier_initialization_invalid_url(self):
        """Test notifier initializes with invalid URL."""
        config = WebhookConfig(url="ftp://invalid.com")
        notifier = WebhookNotifier(config)

        assert notifier.is_ready() is False
        assert notifier._validated is False
        assert notifier._validation_error is not None

    def test_notifier_initialization_no_url(self):
        """Test notifier initializes without URL."""
        config = WebhookConfig(url=None)
        notifier = WebhookNotifier(config)

        assert notifier.is_ready() is False

    def test_is_ready_disabled(self):
        """Test is_ready returns False when disabled."""
        config = WebhookConfig(url="https://example.com/webhook", enabled=False)
        notifier = WebhookNotifier(config)

        assert notifier.is_ready() is False

    @pytest.mark.asyncio
    async def test_notify_not_ready(self):
        """Test notify returns False when not ready."""
        config = WebhookConfig(url=None)
        notifier = WebhookNotifier(config)

        result = await notifier.notify("phase.complete", {"test": "data"})

        assert result is False

    @pytest.mark.asyncio
    async def test_notify_event_not_enabled(self):
        """Test notify returns False when event type not enabled."""
        config = WebhookConfig(
            url="https://example.com/webhook",
            events=["phase.complete"],  # Only phase.complete enabled
        )
        notifier = WebhookNotifier(config)

        result = await notifier.notify("phase.failed", {"test": "data"})

        assert result is False

    @pytest.mark.asyncio
    async def test_notify_success(self):
        """Test successful webhook notification."""
        config = WebhookConfig(url="https://example.com/webhook")
        notifier = WebhookNotifier(config)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session

            result = await notifier.notify("phase.complete", {"phase": "scraping"})

        assert result is True

    @pytest.mark.asyncio
    async def test_notify_with_provided_session(self):
        """Test notification with provided session."""
        config = WebhookConfig(url="https://example.com/webhook")
        notifier = WebhookNotifier(config)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)

        result = await notifier.notify(
            "phase.complete", {"phase": "scraping"}, session=mock_session
        )

        assert result is True
        mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_notify_phase_complete(self):
        """Test notify_phase_complete helper method."""
        config = WebhookConfig(url="https://example.com/webhook")
        notifier = WebhookNotifier(config)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session

            result = await notifier.notify_phase_complete(
                "scraping", {"products": 5, "success": 5}
            )

        assert result is True

    @pytest.mark.asyncio
    async def test_notify_phase_failed(self):
        """Test notify_phase_failed helper method."""
        config = WebhookConfig(url="https://example.com/webhook")
        notifier = WebhookNotifier(config)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session

            result = await notifier.notify_phase_failed(
                "production", "Video encoding failed", {"partial": "data"}
            )

        assert result is True

    @pytest.mark.asyncio
    async def test_notify_pipeline_complete(self):
        """Test notify_pipeline_complete helper method."""
        config = WebhookConfig(url="https://example.com/webhook")
        notifier = WebhookNotifier(config)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session

            result = await notifier.notify_pipeline_complete(
                {"total": 10, "success": 9}
            )

        assert result is True

    @pytest.mark.asyncio
    async def test_notify_pipeline_failed(self):
        """Test notify_pipeline_failed helper method."""
        config = WebhookConfig(url="https://example.com/webhook")
        notifier = WebhookNotifier(config)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session

            result = await notifier.notify_pipeline_failed(
                "Critical failure", {"partial": "summary"}
            )

        assert result is True

    @pytest.mark.asyncio
    async def test_send_with_retry_timeout(self):
        """Test retry logic on timeout."""
        config = WebhookConfig(
            url="https://example.com/webhook",
            max_retries=1,
            retry_delay_sec=0.01,  # Fast retry for test
            timeout_sec=0.1,
        )
        notifier = WebhookNotifier(config)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(side_effect=TimeoutError("Request timed out"))

        result = await notifier._send_with_retry({"test": "data"}, mock_session)

        assert result is False
        # Should have tried initial + max_retries times
        assert mock_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_send_with_retry_client_error(self):
        """Test retry logic on client error."""
        config = WebhookConfig(
            url="https://example.com/webhook",
            max_retries=1,
            retry_delay_sec=0.01,
        )
        notifier = WebhookNotifier(config)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(
            side_effect=aiohttp.ClientError("Connection failed")
        )

        result = await notifier._send_with_retry({"test": "data"}, mock_session)

        assert result is False

    @pytest.mark.asyncio
    async def test_send_with_retry_http_error_status(self):
        """Test retry on HTTP error status code."""
        config = WebhookConfig(
            url="https://example.com/webhook",
            max_retries=1,
            retry_delay_sec=0.01,
        )
        notifier = WebhookNotifier(config)

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)

        result = await notifier._send_with_retry({"test": "data"}, mock_session)

        assert result is False
        # Should retry on 500 error
        assert mock_session.post.call_count == 2


class TestLoadWebhookConfig:
    """Tests for load_webhook_config function."""

    def test_load_empty_config(self):
        """Test loading from empty config returns defaults."""
        config = load_webhook_config({})

        assert config.url is None
        assert config.enabled is True
        assert config.timeout_sec == 5.0

    def test_load_config_with_url(self):
        """Test loading config with webhook URL."""
        yaml_config = {
            "webhook": {
                "url": "https://example.com/webhook",
                "enabled": True,
            }
        }

        config = load_webhook_config(yaml_config)

        assert config.url == "https://example.com/webhook"
        assert config.enabled is True

    def test_load_config_all_options(self):
        """Test loading config with all options."""
        yaml_config = {
            "webhook": {
                "url": "https://example.com/webhook",
                "enabled": False,
                "timeout_sec": 10.0,
                "max_retries": 5,
                "retry_delay_sec": 2.0,
                "events": ["phase.complete", "pipeline.complete"],
            }
        }

        config = load_webhook_config(yaml_config)

        assert config.url == "https://example.com/webhook"
        assert config.enabled is False
        assert config.timeout_sec == 10.0
        assert config.max_retries == 5
        assert config.retry_delay_sec == 2.0
        assert config.events == ["phase.complete", "pipeline.complete"]

    def test_load_config_partial_options(self):
        """Test loading config with partial options uses defaults."""
        yaml_config = {
            "webhook": {
                "url": "https://example.com/webhook",
            }
        }

        config = load_webhook_config(yaml_config)

        assert config.url == "https://example.com/webhook"
        assert config.enabled is True  # Default
        assert config.timeout_sec == 5.0  # Default
        assert config.max_retries == 3  # Default
