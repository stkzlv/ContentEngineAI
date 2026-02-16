"""Extended tests for tracking module — covers atomic writes, retry queue, edge cases."""

import json

import pytest

from src.publisher.tracking import (
    add_to_retry_queue,
    clear_retry_queue,
    get_publish_record,
    get_retry_queue,
    get_retry_queue_count,
    get_retry_queue_item,
    load_tracking,
    record_publish,
    remove_from_retry_queue,
    save_tracking,
)


@pytest.fixture
def outputs_dir(tmp_path):
    """Temporary outputs directory."""
    return tmp_path


class TestAtomicSaveTracking:
    """Test save_tracking atomic write pattern."""

    def test_creates_parent_dirs(self, tmp_path):
        """Creates parent directories if they don't exist."""
        deep_dir = tmp_path / "a" / "b" / "c"
        save_tracking({"posts": {}}, deep_dir)

        tracking_path = deep_dir / "publish_history.json"
        assert tracking_path.exists()

    def test_writes_valid_json(self, outputs_dir):
        """Saved data is valid JSON."""
        data = {"posts": {"key": {"value": "test"}}}
        save_tracking(data, outputs_dir)

        path = outputs_dir / "publish_history.json"
        loaded = json.loads(path.read_text())
        assert loaded == data

    def test_no_temp_file_after_success(self, outputs_dir):
        """Temp file is cleaned up after successful write."""
        save_tracking({"posts": {}}, outputs_dir)

        temp_path = outputs_dir / "publish_history.tmp"
        assert not temp_path.exists()

    def test_overwrites_existing_file(self, outputs_dir):
        """Overwrites existing tracking file."""
        save_tracking({"posts": {"old": {}}}, outputs_dir)
        save_tracking({"posts": {"new": {}}}, outputs_dir)

        loaded = load_tracking(outputs_dir)
        assert "new" in loaded["posts"]
        assert "old" not in loaded["posts"]


class TestLoadTracking:
    """Test load_tracking edge cases."""

    def test_returns_default_when_no_file(self, outputs_dir):
        """Returns default structure when file doesn't exist."""
        result = load_tracking(outputs_dir)
        assert result == {"posts": {}}

    def test_returns_default_on_corrupt_json(self, outputs_dir):
        """Returns default when JSON is corrupted."""
        path = outputs_dir / "publish_history.json"
        path.write_text("{invalid json content")

        result = load_tracking(outputs_dir)
        assert result == {"posts": {}}

    def test_returns_default_on_non_dict(self, outputs_dir):
        """Returns default when JSON is not a dict."""
        path = outputs_dir / "publish_history.json"
        path.write_text(json.dumps([1, 2, 3]))

        result = load_tracking(outputs_dir)
        assert result == {"posts": {}}

    def test_loads_valid_data(self, outputs_dir):
        """Loads valid tracking data."""
        data = {"posts": {"B0TEST001:youtube": {"post_id": "p123"}}}
        save_tracking(data, outputs_dir)

        result = load_tracking(outputs_dir)
        assert result == data


class TestRecordPublish:
    """Test record_publish function."""

    def test_records_new_publish(self, outputs_dir):
        """Records a new publish entry."""
        record_publish("B0TEST001", "youtube", "post_123", outputs_dir)

        record = get_publish_record("B0TEST001", "youtube", outputs_dir)
        assert record is not None
        assert record["post_id"] == "post_123"
        assert record["platform"] == "youtube"
        assert "published_at" in record

    def test_multiple_platforms(self, outputs_dir):
        """Records different platforms separately."""
        record_publish("B0TEST001", "youtube", "yt_123", outputs_dir)
        record_publish("B0TEST001", "tiktok", "tt_456", outputs_dir)

        yt = get_publish_record("B0TEST001", "youtube", outputs_dir)
        tt = get_publish_record("B0TEST001", "tiktok", outputs_dir)
        assert yt["post_id"] == "yt_123"
        assert tt["post_id"] == "tt_456"

    def test_get_nonexistent_record_returns_none(self, outputs_dir):
        """Returns None for nonexistent product/platform."""
        result = get_publish_record("NONEXISTENT", "youtube", outputs_dir)
        assert result is None


class TestRetryQueue:
    """Test retry queue operations."""

    def test_add_to_retry_queue(self, outputs_dir):
        """Adds item to retry queue."""
        add_to_retry_queue("B0TEST001", ["youtube"], "Upload failed", None, outputs_dir)

        queue = get_retry_queue(outputs_dir)
        assert len(queue) == 1
        assert queue[0]["product_id"] == "B0TEST001"
        assert queue[0]["error"] == "Upload failed"
        assert queue[0]["retry_count"] == 1

    def test_retry_count_increments(self, outputs_dir):
        """Retry count increments on subsequent adds."""
        add_to_retry_queue("B0TEST001", ["youtube"], "Error 1", None, outputs_dir)
        add_to_retry_queue("B0TEST001", ["youtube"], "Error 2", None, outputs_dir)

        item = get_retry_queue_item("B0TEST001", outputs_dir)
        assert item["retry_count"] == 2
        assert item["error"] == "Error 2"

    def test_remove_from_retry_queue(self, outputs_dir):
        """Removes item from retry queue."""
        add_to_retry_queue("B0TEST001", ["youtube"], "Error", None, outputs_dir)

        result = remove_from_retry_queue("B0TEST001", outputs_dir)
        assert result is True
        assert get_retry_queue_count(outputs_dir) == 0

    def test_remove_nonexistent_returns_false(self, outputs_dir):
        """Returns False when item not in queue."""
        result = remove_from_retry_queue("NONEXISTENT", outputs_dir)
        assert result is False

    def test_clear_retry_queue(self, outputs_dir):
        """Clears all items from retry queue."""
        add_to_retry_queue("B0TEST001", ["youtube"], "E1", None, outputs_dir)
        add_to_retry_queue("B0TEST002", ["tiktok"], "E2", None, outputs_dir)

        count = clear_retry_queue(outputs_dir)
        assert count == 2
        assert get_retry_queue_count(outputs_dir) == 0

    def test_clear_empty_queue_returns_zero(self, outputs_dir):
        """Clearing empty queue returns 0."""
        count = clear_retry_queue(outputs_dir)
        assert count == 0

    def test_get_retry_queue_item_not_found(self, outputs_dir):
        """Returns None for nonexistent queue item."""
        result = get_retry_queue_item("NONEXISTENT", outputs_dir)
        assert result is None

    def test_retry_queue_preserves_scheduled_time(self, outputs_dir):
        """Scheduled time is preserved in retry queue entry."""
        add_to_retry_queue(
            "B0TEST001",
            ["youtube", "tiktok"],
            "Error",
            "2026-03-01T10:00:00",
            outputs_dir,
        )

        item = get_retry_queue_item("B0TEST001", outputs_dir)
        assert item["scheduled_time"] == "2026-03-01T10:00:00"
        assert item["platforms"] == ["youtube", "tiktok"]
