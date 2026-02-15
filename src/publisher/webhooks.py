"""Webhook handler for Late.dev status updates.

This module provides webhook handling for receiving real-time status updates
from Late.dev, eliminating the need for polling. It supports:
- HMAC-SHA256 signature verification
- Idempotent event processing
- Status tracking updates

Events:
- post.scheduled: Post successfully scheduled
- post.published: Post successfully published
- post.failed: Post failed on all platforms
- post.partial: Post succeeded on some platforms, failed on others
- account.disconnected: Social account disconnected
"""

import hashlib
import hmac
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from src.publisher.constants import DEFAULT_OUTPUTS_DIR
from src.publisher.tracking import load_tracking, save_tracking
from src.video.config.constants import WEBHOOK_EVENT_HISTORY_LIMIT

logger = logging.getLogger(__name__)


class WebhookEventType(Enum):
    """Late.dev webhook event types."""

    POST_SCHEDULED = "post.scheduled"
    POST_PUBLISHED = "post.published"
    POST_FAILED = "post.failed"
    POST_PARTIAL = "post.partial"
    ACCOUNT_DISCONNECTED = "account.disconnected"


@dataclass
class WebhookEvent:
    """Parsed webhook event from Late.dev.

    Attributes
    ----------
        event_id: Unique event identifier for idempotency
        event_type: Type of webhook event
        post_id: Post identifier (None for account events)
        status: Current post status
        platforms: Platform-specific status data
        published_urls: URLs of published posts
        error_message: Error details for failed events
        account_id: Account ID (for account events)
        timestamp: Event timestamp
        raw_payload: Original webhook payload

    """

    event_id: str
    event_type: WebhookEventType
    post_id: str | None = None
    status: str | None = None
    platforms: list[dict[str, Any]] = field(default_factory=list)
    published_urls: list[str] = field(default_factory=list)
    error_message: str | None = None
    account_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    raw_payload: dict[str, Any] = field(default_factory=dict)


class WebhookVerificationError(Exception):
    """Raised when webhook signature verification fails."""

    pass


class WebhookProcessingError(Exception):
    """Raised when webhook event processing fails."""

    pass


class WebhookHandler:
    """Handler for Late.dev webhook events.

    Provides signature verification and event processing for Late.dev webhooks,
    updating local tracking state without polling.

    Attributes:
    ----------
        secret: Webhook secret for HMAC-SHA256 signature verification
        outputs_dir: Directory for tracking data persistence

    Example:
    -------
        >>> handler = WebhookHandler(
        ...     secret="your-webhook-secret",
        ...     outputs_dir=DEFAULT_OUTPUTS_DIR
        ... )
        >>> # In your web framework endpoint:
        >>> event = handler.process_webhook(
        ...     payload=request.body,
        ...     signature=request.headers.get("X-Late-Signature")
        ... )
        >>> print(f"Processed event: {event.event_type.value}")

    """

    # Signature header name used by Late.dev
    SIGNATURE_HEADER = "X-Late-Signature"

    def __init__(
        self,
        secret: str | None = None,
        outputs_dir: Path | str = DEFAULT_OUTPUTS_DIR,
    ):
        """Initialize webhook handler.

        Args:
        ----
            secret: Webhook secret for signature verification (optional but recommended)
            outputs_dir: Directory for tracking data (default: "outputs")

        """
        self.secret = secret
        self.outputs_dir = (
            Path(outputs_dir) if isinstance(outputs_dir, str) else outputs_dir
        )

        if not secret:
            logger.warning(
                "Webhook handler initialized without secret - "
                "signature verification disabled"
            )

        logger.info(
            "WebhookHandler initialized: outputs_dir=%s, signature_verification=%s",
            self.outputs_dir,
            "enabled" if secret else "disabled",
        )

    def verify_signature(self, payload: bytes, signature: str | None) -> bool:
        """Verify webhook signature using HMAC-SHA256.

        Args:
        ----
            payload: Raw webhook payload bytes
            signature: Signature from X-Late-Signature header

        Returns:
        -------
            True if signature is valid or verification is disabled

        Raises:
        ------
            WebhookVerificationError: If signature is invalid or missing when required

        """
        # If no secret configured, skip verification (with warning logged in __init__)
        if not self.secret:
            return True

        # If secret is configured but no signature provided, reject
        if not signature:
            raise WebhookVerificationError(
                "Missing X-Late-Signature header - webhook signature required"
            )

        # Compute expected signature
        expected_signature = hmac.new(
            key=self.secret.encode("utf-8"),
            msg=payload,
            digestmod=hashlib.sha256,
        ).hexdigest()

        # Compare signatures using constant-time comparison
        if not hmac.compare_digest(signature, expected_signature):
            logger.warning("Webhook signature verification failed")
            raise WebhookVerificationError(
                "Invalid webhook signature - request rejected"
            )

        logger.debug("Webhook signature verified successfully")
        return True

    def parse_event(self, payload: dict[str, Any]) -> WebhookEvent:
        """Parse webhook payload into WebhookEvent.

        Args:
        ----
            payload: Parsed JSON webhook payload

        Returns:
        -------
            WebhookEvent with extracted data

        Raises:
        ------
            WebhookProcessingError: If required fields are missing

        """
        # Extract event type
        event_type_str = payload.get("event") or payload.get("type")
        if not event_type_str:
            raise WebhookProcessingError("Missing event type in webhook payload")

        try:
            event_type = WebhookEventType(event_type_str)
        except ValueError as e:
            raise WebhookProcessingError(f"Unknown event type: {event_type_str}") from e

        # Generate event ID for idempotency (use provided or generate from payload)
        event_id = payload.get("eventId") or payload.get("id")
        if not event_id:
            # Generate deterministic ID from payload content
            post_id = payload.get("postId") or payload.get("post", {}).get("_id")
            timestamp = payload.get("timestamp", datetime.now(UTC).isoformat())
            event_id = f"{event_type_str}:{post_id}:{timestamp}"

        # Extract post data (may be nested or flat)
        post_data = payload.get("post") or payload
        post_id = post_data.get("postId") or post_data.get("_id")

        # Extract platforms data
        platforms = post_data.get("platforms", [])
        if isinstance(platforms, list):
            platforms = [
                p if isinstance(p, dict) else {"platform": p} for p in platforms
            ]

        # Extract published URLs
        published_urls = []
        if "platformPostUrl" in post_data:
            published_urls.append(post_data["platformPostUrl"])
        for platform in platforms:
            if isinstance(platform, dict) and "url" in platform:
                published_urls.append(platform["url"])

        # Extract error message for failed events
        error_message = None
        if event_type in (WebhookEventType.POST_FAILED, WebhookEventType.POST_PARTIAL):
            error_message = post_data.get("error") or post_data.get("errorMessage")
            if not error_message:
                # Try to extract from platform-level errors
                platform_errors = [
                    str(p.get("error"))
                    for p in platforms
                    if isinstance(p, dict) and p.get("error")
                ]
                if platform_errors:
                    error_message = "; ".join(platform_errors)

        # Extract account ID for account events
        account_id = None
        if event_type == WebhookEventType.ACCOUNT_DISCONNECTED:
            account_id = payload.get("accountId") or payload.get("account", {}).get(
                "_id"
            )

        # Parse timestamp
        timestamp_str = payload.get("timestamp") or payload.get("createdAt")
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                timestamp = datetime.now(UTC)
        else:
            timestamp = datetime.now(UTC)

        return WebhookEvent(
            event_id=event_id,
            event_type=event_type,
            post_id=post_id,
            status=post_data.get("status"),
            platforms=platforms,
            published_urls=published_urls,
            error_message=error_message,
            account_id=account_id,
            timestamp=timestamp,
            raw_payload=payload,
        )

    def is_event_processed(
        self,
        event_id: str,
        tracking: dict | None = None,
    ) -> bool:
        """Check if event was already processed (idempotency check).

        Args:
        ----
            event_id: Unique event identifier
            tracking: Pre-loaded tracking dict (avoids redundant file read)

        Returns:
        -------
            True if event was already processed

        """
        if tracking is None:
            tracking = load_tracking(self.outputs_dir)
        processed_events = tracking.get("webhook_events", {})
        return event_id in processed_events

    def mark_event_processed(
        self,
        event: WebhookEvent,
        tracking: dict | None = None,
    ) -> None:
        """Mark event as processed for idempotency.

        Args:
        ----
            event: Processed webhook event
            tracking: Pre-loaded tracking dict (caller saves)

        """
        save = tracking is None
        if tracking is None:
            tracking = load_tracking(self.outputs_dir)
        if "webhook_events" not in tracking:
            tracking["webhook_events"] = {}

        tracking["webhook_events"][event.event_id] = {
            "event_type": event.event_type.value,
            "post_id": event.post_id,
            "processed_at": datetime.now(UTC).isoformat(),
        }

        # Keep only last N events to prevent unbounded growth
        if len(tracking["webhook_events"]) > WEBHOOK_EVENT_HISTORY_LIMIT:
            # Sort by processed_at and keep newest entries
            sorted_events = sorted(
                tracking["webhook_events"].items(),
                key=lambda x: x[1].get("processed_at", ""),
                reverse=True,
            )
            tracking["webhook_events"] = dict(
                sorted_events[:WEBHOOK_EVENT_HISTORY_LIMIT]
            )

        if save:
            save_tracking(tracking, self.outputs_dir)

    def update_post_status(
        self,
        event: WebhookEvent,
        tracking: dict | None = None,
    ) -> None:
        """Update post status in tracking based on webhook event.

        Args:
        ----
            event: Parsed webhook event
            tracking: Pre-loaded tracking dict (caller saves)

        """
        if not event.post_id:
            logger.debug("No post_id in event %s, skipping", event.event_type.value)
            return

        save = tracking is None
        if tracking is None:
            tracking = load_tracking(self.outputs_dir)

        # Initialize post_status section if needed
        if "post_status" not in tracking:
            tracking["post_status"] = {}

        # Map event type to status
        status_map = {
            WebhookEventType.POST_SCHEDULED: "scheduled",
            WebhookEventType.POST_PUBLISHED: "published",
            WebhookEventType.POST_FAILED: "failed",
            WebhookEventType.POST_PARTIAL: "partial",
        }

        status = status_map.get(event.event_type, event.status)

        # Update or create post status entry
        tracking["post_status"][event.post_id] = {
            "post_id": event.post_id,
            "status": status,
            "platforms": event.platforms,
            "published_urls": event.published_urls,
            "error_message": event.error_message,
            "updated_at": datetime.now(UTC).isoformat(),
            "event_type": event.event_type.value,
        }

        if save:
            save_tracking(tracking, self.outputs_dir)
        logger.info("Updated post status: %s -> %s", event.post_id, status)

    def handle_account_disconnected(
        self,
        event: WebhookEvent,
        tracking: dict | None = None,
    ) -> None:
        """Handle account disconnection event.

        Args:
        ----
            event: Account disconnected event
            tracking: Pre-loaded tracking dict (caller saves)

        """
        if not event.account_id:
            logger.warning("Account disconnected event without account_id")
            return

        save = tracking is None
        if tracking is None:
            tracking = load_tracking(self.outputs_dir)

        # Initialize disconnected_accounts section if needed
        if "disconnected_accounts" not in tracking:
            tracking["disconnected_accounts"] = {}

        tracking["disconnected_accounts"][event.account_id] = {
            "account_id": event.account_id,
            "disconnected_at": event.timestamp.isoformat(),
            "raw_event": event.raw_payload,
        }

        if save:
            save_tracking(tracking, self.outputs_dir)
        logger.warning("Account disconnected: %s", event.account_id)

    def process_webhook(
        self,
        payload: bytes | str | dict,
        signature: str | None = None,
    ) -> WebhookEvent:
        """Process incoming webhook request.

        Full webhook processing pipeline:
        1. Verify signature (if secret configured)
        2. Parse payload
        3. Check idempotency (skip if already processed)
        4. Update tracking state
        5. Mark event as processed

        Args:
        ----
            payload: Raw webhook payload (bytes, JSON string, or dict)
            signature: Value of X-Late-Signature header (optional if no secret)

        Returns:
        -------
            Parsed WebhookEvent

        Raises:
        ------
            WebhookVerificationError: If signature verification fails
            WebhookProcessingError: If payload parsing fails

        Example:
        -------
            >>> # Flask example
            >>> @app.route("/webhooks/late", methods=["POST"])
            >>> def handle_late_webhook():
            ...     event = handler.process_webhook(
            ...         payload=request.data,
            ...         signature=request.headers.get("X-Late-Signature")
            ...     )
            ...     return {"status": "ok", "event_id": event.event_id}

        """
        # Convert payload to bytes for signature verification
        if isinstance(payload, dict):
            payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            payload_dict = payload
        elif isinstance(payload, str):
            payload_bytes = payload.encode("utf-8")
            payload_dict = json.loads(payload)
        else:
            payload_bytes = payload
            payload_dict = json.loads(payload.decode("utf-8"))

        # Step 1: Verify signature
        self.verify_signature(payload_bytes, signature)

        # Step 2: Parse event
        event = self.parse_event(payload_dict)
        logger.info(
            "Received webhook: %s (post_id=%s, event_id=%s)",
            event.event_type.value,
            event.post_id,
            event.event_id,
        )

        # Load tracking once for all operations
        tracking = load_tracking(self.outputs_dir)

        # Step 3: Idempotency check
        if self.is_event_processed(event.event_id, tracking=tracking):
            logger.info("Event already processed, skipping: %s", event.event_id)
            return event

        # Step 4: Update tracking based on event type
        if event.event_type == WebhookEventType.ACCOUNT_DISCONNECTED:
            self.handle_account_disconnected(event, tracking=tracking)
        else:
            self.update_post_status(event, tracking=tracking)

        # Step 5: Mark as processed and save once
        self.mark_event_processed(event, tracking=tracking)
        save_tracking(tracking, self.outputs_dir)

        logger.info("Webhook processed successfully: %s", event.event_id)
        return event


# =============================================================================
# TRACKING HELPER FUNCTIONS
# =============================================================================


def get_post_status(
    post_id: str,
    outputs_dir: Path = DEFAULT_OUTPUTS_DIR,
) -> dict[str, Any] | None:
    """Get current status for a post from webhook updates.

    Args:
    ----
        post_id: Post identifier
        outputs_dir: Outputs directory path

    Returns:
    -------
        Post status dict or None if not found

    """
    tracking = load_tracking(outputs_dir)
    return tracking.get("post_status", {}).get(post_id)


def get_disconnected_accounts(
    outputs_dir: Path = DEFAULT_OUTPUTS_DIR,
) -> list[dict[str, Any]]:
    """Get list of disconnected accounts from webhook events.

    Returns
    -------
        List of disconnected account records

    """
    tracking = load_tracking(outputs_dir)
    return list(tracking.get("disconnected_accounts", {}).values())


def clear_webhook_events(
    outputs_dir: Path = DEFAULT_OUTPUTS_DIR,
) -> int:
    """Clear processed webhook events history.

    Returns
    -------
        Number of events cleared

    """
    tracking = load_tracking(outputs_dir)
    count = len(tracking.get("webhook_events", {}))

    if count > 0:
        tracking["webhook_events"] = {}
        save_tracking(tracking, outputs_dir)
        logger.info("Cleared webhook events: %d event(s)", count)

    return count
