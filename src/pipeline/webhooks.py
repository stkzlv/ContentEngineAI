"""Webhook notification support for pipeline events.

This module provides non-blocking webhook notifications for pipeline
phase completion and failure events.

Features:
    - Async POST requests with configurable timeout
    - Automatic retry with exponential backoff
    - URL validation before sending
    - Non-blocking (failures don't stop pipeline)

Usage:
    from src.pipeline.webhooks import WebhookNotifier, WebhookConfig

    config = WebhookConfig(url="https://example.com/webhook")
    notifier = WebhookNotifier(config)
    await notifier.notify_phase_complete("scraping", summary_data)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class WebhookConfig:
    """Webhook configuration settings.

    Attributes
    ----------
        url: Webhook endpoint URL (must be https:// or http://)
        enabled: Whether webhook notifications are enabled
        timeout_sec: Request timeout in seconds
        max_retries: Maximum retry attempts for failed requests
        retry_delay_sec: Initial delay between retries (doubles each attempt)

    """

    url: str | None = None
    enabled: bool = True
    timeout_sec: float = 5.0
    max_retries: int = 3
    retry_delay_sec: float = 1.0
    events: list[str] = field(
        default_factory=lambda: [
            "phase.complete",
            "phase.failed",
            "pipeline.complete",
            "pipeline.failed",
        ]
    )

    def is_configured(self) -> bool:
        """Check if webhook is properly configured and enabled.

        Returns
        -------
            True if webhook URL is set and notifications are enabled

        """
        return bool(self.url) and self.enabled


def validate_webhook_url(url: str) -> tuple[bool, str | None]:
    """Validate webhook URL format and scheme.

    Args:
    ----
        url: URL string to validate

    Returns:
    -------
        Tuple of (is_valid, error_message)

    """
    if not url:
        return False, "Webhook URL is empty"

    try:
        parsed = urlparse(url)

        # Check scheme
        if parsed.scheme not in ("http", "https"):
            return False, f"Invalid URL scheme: {parsed.scheme}. Use http:// or https://"

        # Check netloc (hostname)
        if not parsed.netloc:
            return False, "Invalid URL: missing hostname"

        return True, None

    except Exception as e:
        return False, f"Invalid URL format: {e}"


class WebhookNotifier:
    """Non-blocking webhook notifier for pipeline events.

    Sends POST requests to configured webhook URL on pipeline events.
    Failures are logged but never block the pipeline.

    Attributes
    ----------
        config: Webhook configuration

    """

    def __init__(self, config: WebhookConfig):
        """Initialize webhook notifier.

        Args:
        ----
            config: Webhook configuration settings

        """
        self.config = config
        self._validated = False
        self._validation_error: str | None = None

        # Validate URL on init
        if config.url:
            is_valid, error = validate_webhook_url(config.url)
            self._validated = is_valid
            self._validation_error = error
            if not is_valid:
                logger.warning(f"Webhook URL validation failed: {error}")

    def is_ready(self) -> bool:
        """Check if webhook notifier is ready to send notifications.

        Returns
        -------
            True if configured, enabled, and URL is valid

        """
        return self.config.is_configured() and self._validated

    async def _send_with_retry(
        self,
        payload: dict[str, Any],
        session: aiohttp.ClientSession,
    ) -> bool:
        """Send webhook with retry logic.

        Args:
        ----
            payload: JSON payload to send
            session: aiohttp client session

        Returns:
        -------
            True if webhook was sent successfully

        """
        url = self.config.url
        if not url:
            return False

        delay = self.config.retry_delay_sec

        timeout = aiohttp.ClientTimeout(total=self.config.timeout_sec)

        for attempt in range(self.config.max_retries + 1):
            try:
                async with session.post(
                    url,
                    json=payload,
                    timeout=timeout,  # type: ignore[arg-type]
                ) as response:
                    if response.status < 400:
                        logger.debug(
                            f"Webhook sent successfully: {response.status} "
                            f"(attempt {attempt + 1})"
                        )
                        return True

                    # Log non-success status but continue to retry
                    logger.warning(
                        f"Webhook returned status {response.status} "
                        f"(attempt {attempt + 1}/{self.config.max_retries + 1})"
                    )

            except TimeoutError:
                logger.warning(
                    f"Webhook timeout after {self.config.timeout_sec}s "
                    f"(attempt {attempt + 1}/{self.config.max_retries + 1})"
                )

            except aiohttp.ClientError as e:
                logger.warning(
                    f"Webhook request failed: {e} "
                    f"(attempt {attempt + 1}/{self.config.max_retries + 1})"
                )

            except Exception as e:
                logger.warning(
                    f"Unexpected webhook error: {e} "
                    f"(attempt {attempt + 1}/{self.config.max_retries + 1})"
                )

            # Wait before retry (except on last attempt)
            if attempt < self.config.max_retries:
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff

        logger.error(
            f"Webhook failed after {self.config.max_retries + 1} attempts: {url}"
        )
        return False

    async def notify(
        self,
        event: str,
        data: dict[str, Any],
        session: aiohttp.ClientSession | None = None,
    ) -> bool:
        """Send webhook notification for an event.

        Non-blocking: failures are logged but never raise exceptions.

        Args:
        ----
            event: Event type (e.g., "phase.complete", "pipeline.failed")
            data: Event data to include in payload
            session: Optional aiohttp session (creates one if not provided)

        Returns:
        -------
            True if webhook was sent successfully

        """
        if not self.is_ready():
            logger.debug(f"Webhook not ready, skipping event: {event}")
            return False

        # Check if event type is enabled
        if event not in self.config.events:
            logger.debug(f"Event type not enabled: {event}")
            return False

        payload = {
            "event": event,
            "data": data,
        }

        # Use provided session or create temporary one
        if session:
            return await self._send_with_retry(payload, session)
        else:
            async with aiohttp.ClientSession() as temp_session:
                return await self._send_with_retry(payload, temp_session)

    async def notify_phase_complete(
        self,
        phase: str,
        summary: dict[str, Any],
        session: aiohttp.ClientSession | None = None,
    ) -> bool:
        """Notify webhook of phase completion.

        Args:
        ----
            phase: Phase name (scraping, handoff, production, publishing)
            summary: Phase summary data
            session: Optional aiohttp session

        Returns:
        -------
            True if webhook was sent successfully

        """
        return await self.notify(
            event="phase.complete",
            data={"phase": phase, "summary": summary},
            session=session,
        )

    async def notify_phase_failed(
        self,
        phase: str,
        error: str,
        partial_summary: dict[str, Any] | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> bool:
        """Notify webhook of phase failure.

        Args:
        ----
            phase: Phase name that failed
            error: Error message
            partial_summary: Partial summary data if available
            session: Optional aiohttp session

        Returns:
        -------
            True if webhook was sent successfully

        """
        return await self.notify(
            event="phase.failed",
            data={"phase": phase, "error": error, "partial_summary": partial_summary},
            session=session,
        )

    async def notify_pipeline_complete(
        self,
        summary: dict[str, Any],
        session: aiohttp.ClientSession | None = None,
    ) -> bool:
        """Notify webhook of pipeline completion.

        Args:
        ----
            summary: Full pipeline summary data
            session: Optional aiohttp session

        Returns:
        -------
            True if webhook was sent successfully

        """
        return await self.notify(
            event="pipeline.complete",
            data={"summary": summary},
            session=session,
        )

    async def notify_pipeline_failed(
        self,
        error: str,
        partial_summary: dict[str, Any] | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> bool:
        """Notify webhook of pipeline failure.

        Args:
        ----
            error: Error message
            partial_summary: Partial summary data if available
            session: Optional aiohttp session

        Returns:
        -------
            True if webhook was sent successfully

        """
        return await self.notify(
            event="pipeline.failed",
            data={"error": error, "partial_summary": partial_summary},
            session=session,
        )


def load_webhook_config(yaml_config: dict[str, Any]) -> WebhookConfig:
    """Load webhook configuration from YAML config dict.

    Args:
    ----
        yaml_config: Parsed YAML configuration (global_batch section)

    Returns:
    -------
        WebhookConfig with settings from YAML or defaults

    """
    webhook_section = yaml_config.get("webhook", {})

    return WebhookConfig(
        url=webhook_section.get("url"),
        enabled=webhook_section.get("enabled", True),
        timeout_sec=webhook_section.get("timeout_sec", 5.0),
        max_retries=webhook_section.get("max_retries", 3),
        retry_delay_sec=webhook_section.get("retry_delay_sec", 1.0),
        events=webhook_section.get(
            "events",
            ["phase.complete", "phase.failed", "pipeline.complete", "pipeline.failed"],
        ),
    )
