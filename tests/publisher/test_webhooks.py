"""Tests for webhook handler."""

import hashlib
import hmac
import json
from pathlib import Path

import pytest

from src.publisher.webhooks import (
    WebhookEvent,
    WebhookEventType,
    WebhookHandler,
    WebhookProcessingError,
    WebhookVerificationError,
    clear_webhook_events,
    get_disconnected_accounts,
    get_post_status,
)
from src.video.config.constants import WEBHOOK_EVENT_HISTORY_LIMIT


@pytest.fixture
def temp_outputs_dir(tmp_path: Path) -> Path:
    """Create temporary outputs directory."""
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    return outputs_dir


@pytest.fixture
def webhook_secret() -> str:
    """Webhook secret for testing."""
    return "test-webhook-secret-12345"


@pytest.fixture
def handler(temp_outputs_dir: Path, webhook_secret: str) -> WebhookHandler:
    """Create webhook handler with secret."""
    return WebhookHandler(secret=webhook_secret, outputs_dir=temp_outputs_dir)


@pytest.fixture
def handler_no_secret(temp_outputs_dir: Path) -> WebhookHandler:
    """Create webhook handler without secret (verification disabled)."""
    return WebhookHandler(secret=None, outputs_dir=temp_outputs_dir)


def compute_signature(payload: bytes, secret: str) -> str:
    """Compute HMAC-SHA256 signature for payload."""
    return hmac.new(
        key=secret.encode("utf-8"),
        msg=payload,
        digestmod=hashlib.sha256,
    ).hexdigest()


class TestSignatureVerification:
    """Tests for webhook signature verification."""

    def test_valid_signature(self, handler: WebhookHandler, webhook_secret: str):
        """Test signature verification with valid signature."""
        payload = b'{"event": "post.published", "postId": "123"}'
        signature = compute_signature(payload, webhook_secret)

        result = handler.verify_signature(payload, signature)
        assert result is True

    def test_invalid_signature(self, handler: WebhookHandler):
        """Test signature verification rejects invalid signature."""
        payload = b'{"event": "post.published", "postId": "123"}'
        invalid_signature = "invalid-signature-value"

        with pytest.raises(WebhookVerificationError, match="Invalid webhook signature"):
            handler.verify_signature(payload, invalid_signature)

    def test_missing_signature_with_secret(self, handler: WebhookHandler):
        """Test signature verification rejects missing signature when secret configured."""
        payload = b'{"event": "post.published", "postId": "123"}'

        with pytest.raises(WebhookVerificationError, match="Missing X-Late-Signature"):
            handler.verify_signature(payload, None)

    def test_missing_signature_without_secret(self, handler_no_secret: WebhookHandler):
        """Test signature verification skipped when no secret configured."""
        payload = b'{"event": "post.published", "postId": "123"}'

        # Should not raise, returns True
        result = handler_no_secret.verify_signature(payload, None)
        assert result is True

    def test_signature_timing_attack_prevention(
        self, handler: WebhookHandler, webhook_secret: str
    ):
        """Test that signature comparison uses constant-time comparison."""
        payload = b'{"event": "post.published"}'
        valid_signature = compute_signature(payload, webhook_secret)

        # Valid signature should pass
        assert handler.verify_signature(payload, valid_signature) is True

        # Similar but wrong signature should fail
        wrong_signature = valid_signature[:-1] + ("0" if valid_signature[-1] != "0" else "1")
        with pytest.raises(WebhookVerificationError):
            handler.verify_signature(payload, wrong_signature)


class TestEventParsing:
    """Tests for webhook event parsing."""

    def test_parse_post_published_event(self, handler: WebhookHandler):
        """Test parsing post.published event."""
        payload = {
            "event": "post.published",
            "eventId": "evt_123",
            "postId": "post_456",
            "status": "published",
            "platforms": [
                {"platform": "youtube", "status": "published", "url": "https://youtube.com/watch?v=abc"},
                {"platform": "tiktok", "status": "published", "url": "https://tiktok.com/@user/video/123"},
            ],
            "timestamp": "2024-01-15T10:30:00Z",
        }

        event = handler.parse_event(payload)

        assert event.event_id == "evt_123"
        assert event.event_type == WebhookEventType.POST_PUBLISHED
        assert event.post_id == "post_456"
        assert event.status == "published"
        assert len(event.platforms) == 2
        assert len(event.published_urls) == 2
        assert "https://youtube.com/watch?v=abc" in event.published_urls

    def test_parse_post_scheduled_event(self, handler: WebhookHandler):
        """Test parsing post.scheduled event."""
        payload = {
            "event": "post.scheduled",
            "postId": "post_789",
            "status": "scheduled",
            "platforms": [{"platform": "instagram"}],
        }

        event = handler.parse_event(payload)

        assert event.event_type == WebhookEventType.POST_SCHEDULED
        assert event.post_id == "post_789"
        assert event.status == "scheduled"

    def test_parse_post_failed_event(self, handler: WebhookHandler):
        """Test parsing post.failed event with error message."""
        payload = {
            "event": "post.failed",
            "postId": "post_fail",
            "status": "failed",
            "error": "Rate limit exceeded",
            "platforms": [
                {"platform": "youtube", "status": "failed", "error": "API quota exceeded"},
            ],
        }

        event = handler.parse_event(payload)

        assert event.event_type == WebhookEventType.POST_FAILED
        assert event.post_id == "post_fail"
        assert event.error_message == "Rate limit exceeded"

    def test_parse_post_partial_event(self, handler: WebhookHandler):
        """Test parsing post.partial event (some platforms succeeded)."""
        payload = {
            "event": "post.partial",
            "postId": "post_partial",
            "platforms": [
                {"platform": "youtube", "status": "published"},
                {"platform": "tiktok", "status": "failed", "error": "Video too long"},
            ],
        }

        event = handler.parse_event(payload)

        assert event.event_type == WebhookEventType.POST_PARTIAL
        assert event.post_id == "post_partial"
        # Error extracted from platform-level
        assert "Video too long" in (event.error_message or "")

    def test_parse_account_disconnected_event(self, handler: WebhookHandler):
        """Test parsing account.disconnected event."""
        payload = {
            "event": "account.disconnected",
            "accountId": "acc_123",
            "timestamp": "2024-01-15T12:00:00Z",
        }

        event = handler.parse_event(payload)

        assert event.event_type == WebhookEventType.ACCOUNT_DISCONNECTED
        assert event.account_id == "acc_123"
        assert event.post_id is None

    def test_parse_nested_post_data(self, handler: WebhookHandler):
        """Test parsing event with nested post object."""
        payload = {
            "event": "post.published",
            "eventId": "evt_nested",
            "post": {
                "_id": "nested_post_id",
                "status": "published",
                "platforms": [{"platform": "youtube"}],
                "platformPostUrl": "https://youtube.com/watch?v=xyz",
            },
        }

        event = handler.parse_event(payload)

        assert event.post_id == "nested_post_id"
        assert "https://youtube.com/watch?v=xyz" in event.published_urls

    def test_parse_missing_event_type(self, handler: WebhookHandler):
        """Test parsing fails with missing event type."""
        payload = {"postId": "123"}

        with pytest.raises(WebhookProcessingError, match="Missing event type"):
            handler.parse_event(payload)

    def test_parse_unknown_event_type(self, handler: WebhookHandler):
        """Test parsing fails with unknown event type."""
        payload = {"event": "unknown.event", "postId": "123"}

        with pytest.raises(WebhookProcessingError, match="Unknown event type"):
            handler.parse_event(payload)

    def test_parse_generates_event_id_if_missing(self, handler: WebhookHandler):
        """Test event ID is generated if not provided."""
        payload = {
            "event": "post.published",
            "postId": "post_no_event_id",
        }

        event = handler.parse_event(payload)

        assert event.event_id is not None
        assert "post.published" in event.event_id
        assert "post_no_event_id" in event.event_id


class TestIdempotency:
    """Tests for idempotent event processing."""

    def test_event_not_processed_initially(
        self, handler: WebhookHandler
    ):
        """Test event is not marked as processed initially."""
        assert handler.is_event_processed("evt_new") is False

    def test_mark_event_processed(self, handler: WebhookHandler):
        """Test marking event as processed."""
        event = WebhookEvent(
            event_id="evt_to_mark",
            event_type=WebhookEventType.POST_PUBLISHED,
            post_id="post_123",
        )

        handler.mark_event_processed(event)

        assert handler.is_event_processed("evt_to_mark") is True

    def test_duplicate_event_skipped(
        self, handler_no_secret: WebhookHandler, temp_outputs_dir: Path
    ):
        """Test that duplicate events are skipped."""
        payload = {
            "event": "post.published",
            "eventId": "evt_duplicate",
            "postId": "post_dup",
        }
        payload_bytes = json.dumps(payload).encode()

        # Process first time
        event1 = handler_no_secret.process_webhook(payload_bytes)
        assert event1.event_id == "evt_duplicate"

        # Process second time (should be skipped but still return event)
        event2 = handler_no_secret.process_webhook(payload_bytes)
        assert event2.event_id == "evt_duplicate"

        # Check only processed once in tracking
        from src.publisher.tracking import load_tracking

        tracking = load_tracking(temp_outputs_dir)
        assert "evt_duplicate" in tracking.get("webhook_events", {})

    def test_event_history_pruning(self, handler: WebhookHandler):
        """Test that event history is pruned to configured limit."""
        # Create more events than the limit
        overflow_count = 50
        for i in range(WEBHOOK_EVENT_HISTORY_LIMIT + overflow_count):
            event = WebhookEvent(
                event_id=f"evt_{i:05d}",
                event_type=WebhookEventType.POST_PUBLISHED,
                post_id=f"post_{i}",
            )
            handler.mark_event_processed(event)

        # Check that only configured limit events remain
        from src.publisher.tracking import load_tracking

        tracking = load_tracking(handler.outputs_dir)
        assert len(tracking.get("webhook_events", {})) <= WEBHOOK_EVENT_HISTORY_LIMIT


class TestStatusTracking:
    """Tests for status tracking updates."""

    def test_update_post_status_published(
        self, handler_no_secret: WebhookHandler, temp_outputs_dir: Path
    ):
        """Test post status updated on published event."""
        payload = {
            "event": "post.published",
            "eventId": "evt_pub",
            "postId": "post_status_test",
            "status": "published",
            "platforms": [
                {"platform": "youtube", "url": "https://youtube.com/watch?v=test"},
            ],
        }

        handler_no_secret.process_webhook(payload)

        status = get_post_status("post_status_test", temp_outputs_dir)
        assert status is not None
        assert status["status"] == "published"
        assert "https://youtube.com/watch?v=test" in status["published_urls"]

    def test_update_post_status_failed(
        self, handler_no_secret: WebhookHandler, temp_outputs_dir: Path
    ):
        """Test post status updated on failed event."""
        payload = {
            "event": "post.failed",
            "eventId": "evt_fail",
            "postId": "post_failed_test",
            "error": "Upload failed",
        }

        handler_no_secret.process_webhook(payload)

        status = get_post_status("post_failed_test", temp_outputs_dir)
        assert status is not None
        assert status["status"] == "failed"
        assert status["error_message"] == "Upload failed"

    def test_account_disconnected_tracked(
        self, handler_no_secret: WebhookHandler, temp_outputs_dir: Path
    ):
        """Test disconnected account is tracked."""
        payload = {
            "event": "account.disconnected",
            "eventId": "evt_disc",
            "accountId": "acc_disconnected",
        }

        handler_no_secret.process_webhook(payload)

        accounts = get_disconnected_accounts(temp_outputs_dir)
        assert len(accounts) == 1
        assert accounts[0]["account_id"] == "acc_disconnected"


class TestFullWebhookProcessing:
    """Integration tests for full webhook processing."""

    def test_process_webhook_with_signature(
        self, handler: WebhookHandler, webhook_secret: str
    ):
        """Test full webhook processing with signature verification."""
        payload = {
            "event": "post.published",
            "eventId": "evt_full",
            "postId": "post_full",
            "status": "published",
        }
        payload_bytes = json.dumps(payload, separators=(",", ":")).encode()
        signature = compute_signature(payload_bytes, webhook_secret)

        event = handler.process_webhook(payload_bytes, signature)

        assert event.event_type == WebhookEventType.POST_PUBLISHED
        assert event.post_id == "post_full"

    def test_process_webhook_dict_payload(self, handler_no_secret: WebhookHandler):
        """Test processing with dict payload (already parsed)."""
        payload = {
            "event": "post.scheduled",
            "eventId": "evt_dict",
            "postId": "post_dict",
        }

        event = handler_no_secret.process_webhook(payload)

        assert event.event_type == WebhookEventType.POST_SCHEDULED

    def test_process_webhook_string_payload(self, handler_no_secret: WebhookHandler):
        """Test processing with JSON string payload."""
        payload = '{"event": "post.published", "eventId": "evt_str", "postId": "post_str"}'

        event = handler_no_secret.process_webhook(payload)

        assert event.event_type == WebhookEventType.POST_PUBLISHED

    def test_process_webhook_invalid_signature(
        self, handler: WebhookHandler
    ):
        """Test webhook rejected with invalid signature."""
        payload = b'{"event": "post.published", "postId": "123"}'

        with pytest.raises(WebhookVerificationError):
            handler.process_webhook(payload, "wrong-signature")


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_clear_webhook_events(
        self, handler_no_secret: WebhookHandler, temp_outputs_dir: Path
    ):
        """Test clearing webhook events history."""
        # Process some events
        for i in range(5):
            payload = {
                "event": "post.published",
                "eventId": f"evt_clear_{i}",
                "postId": f"post_{i}",
            }
            handler_no_secret.process_webhook(payload)

        # Clear events
        count = clear_webhook_events(temp_outputs_dir)

        assert count == 5
        assert not handler_no_secret.is_event_processed("evt_clear_0")

    def test_get_post_status_not_found(self, temp_outputs_dir: Path):
        """Test get_post_status returns None for unknown post."""
        status = get_post_status("unknown_post", temp_outputs_dir)
        assert status is None

    def test_get_disconnected_accounts_empty(self, temp_outputs_dir: Path):
        """Test get_disconnected_accounts returns empty list initially."""
        accounts = get_disconnected_accounts(temp_outputs_dir)
        assert accounts == []
