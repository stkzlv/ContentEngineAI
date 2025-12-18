"""Tests for cleanup management functionality."""

import json
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.publisher.cleanup import CleanupManager
from src.publisher.models import CleanupConfig, Platform


@pytest.fixture
def temp_outputs_dir(tmp_path):
    """Create temporary outputs directory."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    return outputs


@pytest.fixture
def cleanup_config():
    """Basic cleanup configuration."""
    return CleanupConfig(
        enabled=True,
        verify_before_delete=True,
        require_all_platforms=True,
        archive_before_delete=False,
    )


@pytest.fixture
def mock_publisher():
    """Mock publisher instance."""
    publisher = AsyncMock()
    publisher.get_status = AsyncMock(return_value={"status": "published"})
    return publisher


@pytest.fixture
def product_directory(temp_outputs_dir):
    """Create mock product directory with files."""
    product_id = "B0TEST001"
    product_dir = temp_outputs_dir / product_id
    product_dir.mkdir()

    # Create video file
    video = product_dir / f"video_{product_id}.mp4"
    video.write_text("mock video data" * 1000)

    # Create metadata
    metadata = product_dir / "data.json"
    metadata.write_text(json.dumps({"title": "Test Product"}))

    # Create images subdirectory
    images_dir = product_dir / "images"
    images_dir.mkdir()
    for i in range(3):
        image = images_dir / f"image_{i}.jpg"
        image.write_text("mock image data" * 500)

    return product_dir


def create_tracking_file(outputs_dir: Path, product_id: str, platform: str, post_id: str):
    """Helper to create publish tracking file."""
    tracking_path = outputs_dir / "publish_history.json"
    if tracking_path.exists():
        data = json.loads(tracking_path.read_text())
    else:
        data = {"posts": {}}

    key = f"{product_id}:{platform}"
    data["posts"][key] = {
        "product_id": product_id,
        "platform": platform,
        "post_id": post_id,
        "published_at": datetime.now(UTC).isoformat(),
        "post_url": f"https://{platform}.com/{post_id}",
    }
    tracking_path.write_text(json.dumps(data, indent=2))


class TestCleanupManagerInit:
    """Tests for CleanupManager initialization."""

    def test_initialization(self, temp_outputs_dir, cleanup_config, mock_publisher):
        """Test CleanupManager initialization."""
        manager = CleanupManager(temp_outputs_dir, cleanup_config, mock_publisher)

        assert manager.outputs_dir == temp_outputs_dir
        assert manager.config == cleanup_config
        assert manager.publisher == mock_publisher
        assert manager.audit_log_path == temp_outputs_dir / "cleanup_audit.json"

    def test_audit_log_path_created(
        self, temp_outputs_dir, cleanup_config, mock_publisher
    ):
        """Test that audit log path is correctly set."""
        manager = CleanupManager(temp_outputs_dir, cleanup_config, mock_publisher)

        assert manager.audit_log_path.parent == temp_outputs_dir
        assert manager.audit_log_path.name == "cleanup_audit.json"


class TestCalculateDirSize:
    """Tests for _calculate_dir_size() method."""

    def test_empty_directory(self, temp_outputs_dir, cleanup_config, mock_publisher):
        """Test size calculation for empty directory."""
        manager = CleanupManager(temp_outputs_dir, cleanup_config, mock_publisher)

        empty_dir = temp_outputs_dir / "empty"
        empty_dir.mkdir()

        size = manager._calculate_dir_size(empty_dir)
        assert size == 0

    def test_directory_with_files(
        self, temp_outputs_dir, cleanup_config, mock_publisher, product_directory
    ):
        """Test size calculation for directory with files."""
        manager = CleanupManager(temp_outputs_dir, cleanup_config, mock_publisher)

        size = manager._calculate_dir_size(product_directory)
        assert size > 0
        assert 30000 < size < 50000

    def test_directory_with_subdirectories(
        self, temp_outputs_dir, cleanup_config, mock_publisher
    ):
        """Test size calculation includes subdirectories."""
        manager = CleanupManager(temp_outputs_dir, cleanup_config, mock_publisher)

        test_dir = temp_outputs_dir / "test"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_text("test" * 100)

        subdir = test_dir / "subdir"
        subdir.mkdir()
        (subdir / "file2.txt").write_text("test" * 100)

        nested = subdir / "nested"
        nested.mkdir()
        (nested / "file3.txt").write_text("test" * 100)

        size = manager._calculate_dir_size(test_dir)
        assert 1000 < size < 1500

    def test_nonexistent_directory(
        self, temp_outputs_dir, cleanup_config, mock_publisher
    ):
        """Test size calculation for nonexistent directory."""
        manager = CleanupManager(temp_outputs_dir, cleanup_config, mock_publisher)

        nonexistent = temp_outputs_dir / "does_not_exist"
        size = manager._calculate_dir_size(nonexistent)
        assert size == 0


class TestLogCleanup:
    """Tests for _log_cleanup() method."""

    def test_creates_new_audit_log(
        self, temp_outputs_dir, cleanup_config, mock_publisher
    ):
        """Test creating new audit log when file doesn't exist."""
        manager = CleanupManager(temp_outputs_dir, cleanup_config, mock_publisher)

        manager._log_cleanup(
            product_id="B0TEST001",
            platforms=[Platform.YOUTUBE, Platform.TIKTOK],
            post_urls=["https://youtube.com/post123", "https://tiktok.com/post456"],
            disk_freed_bytes=1024000,
            archive_path=None,
        )

        assert manager.audit_log_path.exists()
        data = json.loads(manager.audit_log_path.read_text())
        assert len(data["cleanups"]) == 1

        record = data["cleanups"][0]
        assert record["product_id"] == "B0TEST001"
        assert record["platforms"] == ["youtube", "tiktok"]
        assert record["disk_freed_bytes"] == 1024000
        assert record["archive_path"] is None

    def test_appends_to_existing_audit_log(
        self, temp_outputs_dir, cleanup_config, mock_publisher
    ):
        """Test appending to existing audit log."""
        manager = CleanupManager(temp_outputs_dir, cleanup_config, mock_publisher)

        manager._log_cleanup(
            "B0TEST001", [Platform.YOUTUBE], ["https://youtube.com/post1"], 500000
        )
        manager._log_cleanup(
            "B0TEST002", [Platform.TIKTOK], ["https://tiktok.com/post2"], 750000
        )

        data = json.loads(manager.audit_log_path.read_text())
        assert len(data["cleanups"]) == 2

    def test_handles_corrupted_audit_log(
        self, temp_outputs_dir, cleanup_config, mock_publisher
    ):
        """Test handling of corrupted audit log file."""
        manager = CleanupManager(temp_outputs_dir, cleanup_config, mock_publisher)
        manager.audit_log_path.write_text("invalid json {{{")

        manager._log_cleanup(
            "B0TEST001", [Platform.YOUTUBE], ["https://youtube.com/post1"], 500000
        )

        data = json.loads(manager.audit_log_path.read_text())
        assert len(data["cleanups"]) == 1


class TestVerifyPublication:
    """Tests for verify_publication() async method."""

    @pytest.mark.asyncio
    async def test_all_platforms_published(
        self, temp_outputs_dir, cleanup_config, mock_publisher
    ):
        """Test verification when all platforms are published."""
        create_tracking_file(temp_outputs_dir, "B0TEST001", "youtube", "post123")
        create_tracking_file(temp_outputs_dir, "B0TEST001", "tiktok", "post456")

        mock_publisher.get_status = AsyncMock(return_value={"status": "published"})
        manager = CleanupManager(temp_outputs_dir, cleanup_config, mock_publisher)

        success, statuses = await manager.verify_publication(
            "B0TEST001", [Platform.YOUTUBE, Platform.TIKTOK]
        )

        assert success is True
        assert statuses["youtube"] == "published"
        assert statuses["tiktok"] == "published"
        assert mock_publisher.get_status.call_count == 2

    @pytest.mark.asyncio
    async def test_one_platform_not_published(
        self, temp_outputs_dir, cleanup_config, mock_publisher
    ):
        """Test verification when one platform is not published."""
        create_tracking_file(temp_outputs_dir, "B0TEST001", "youtube", "post123")
        create_tracking_file(temp_outputs_dir, "B0TEST001", "tiktok", "post456")

        async def mock_status(post_id):
            if post_id == "post123":
                return {"status": "published"}
            return {"status": "scheduled"}

        mock_publisher.get_status = AsyncMock(side_effect=mock_status)
        manager = CleanupManager(temp_outputs_dir, cleanup_config, mock_publisher)

        success, statuses = await manager.verify_publication(
            "B0TEST001", [Platform.YOUTUBE, Platform.TIKTOK]
        )

        assert success is False
        assert statuses["youtube"] == "published"
        assert statuses["tiktok"] == "scheduled"

    @pytest.mark.asyncio
    async def test_require_all_platforms_false(
        self, temp_outputs_dir, mock_publisher
    ):
        """Test verification with require_all_platforms=False."""
        config = CleanupConfig(
            enabled=True,
            verify_before_delete=True,
            require_all_platforms=False,
        )
        create_tracking_file(temp_outputs_dir, "B0TEST001", "youtube", "post123")
        create_tracking_file(temp_outputs_dir, "B0TEST001", "tiktok", "post456")

        async def mock_status(post_id):
            if post_id == "post123":
                return {"status": "published"}
            return {"status": "failed"}

        mock_publisher.get_status = AsyncMock(side_effect=mock_status)
        manager = CleanupManager(temp_outputs_dir, config, mock_publisher)

        success, statuses = await manager.verify_publication(
            "B0TEST001", [Platform.YOUTUBE, Platform.TIKTOK]
        )

        # Should succeed because at least one platform is published
        assert success is True

    @pytest.mark.asyncio
    async def test_missing_publish_record(
        self, temp_outputs_dir, cleanup_config, mock_publisher
    ):
        """Test verification with missing publish record."""
        manager = CleanupManager(temp_outputs_dir, cleanup_config, mock_publisher)

        success, statuses = await manager.verify_publication(
            "B0TEST001", [Platform.YOUTUBE]
        )

        assert success is False
        assert statuses["youtube"] == "not_published"
        mock_publisher.get_status.assert_not_called()

    @pytest.mark.asyncio
    async def test_api_error_handling(
        self, temp_outputs_dir, cleanup_config, mock_publisher
    ):
        """Test verification handles API errors gracefully."""
        create_tracking_file(temp_outputs_dir, "B0TEST001", "youtube", "post123")

        mock_publisher.get_status = AsyncMock(side_effect=Exception("API unavailable"))
        manager = CleanupManager(temp_outputs_dir, cleanup_config, mock_publisher)

        success, statuses = await manager.verify_publication(
            "B0TEST001", [Platform.YOUTUBE]
        )

        assert success is False
        assert statuses["youtube"] == "api_error"


class TestArchiveDirectory:
    """Tests for archive_directory() method."""

    def test_creates_archive(self, temp_outputs_dir, mock_publisher, product_directory):
        """Test archive creation."""
        config = CleanupConfig(
            enabled=True,
            archive_before_delete=True,
            archive_dir=temp_outputs_dir / "archive",
        )
        manager = CleanupManager(temp_outputs_dir, config, mock_publisher)

        archive_path = manager.archive_directory(product_directory)

        assert archive_path.exists()
        assert archive_path.suffix == ".zip"
        assert "B0TEST001" in archive_path.name

        # Verify ZIP contents
        with zipfile.ZipFile(archive_path, "r") as zf:
            names = zf.namelist()
            assert any("video_" in n for n in names)
            assert any("data.json" in n for n in names)

    def test_creates_archive_directory(
        self, temp_outputs_dir, mock_publisher, product_directory
    ):
        """Test that archive directory is created if it doesn't exist."""
        archive_dir = temp_outputs_dir / "deep" / "nested" / "archive"
        config = CleanupConfig(
            enabled=True,
            archive_before_delete=True,
            archive_dir=archive_dir,
        )
        manager = CleanupManager(temp_outputs_dir, config, mock_publisher)

        archive_path = manager.archive_directory(product_directory)

        assert archive_dir.exists()
        assert archive_path.exists()

    def test_archive_nonexistent_directory(
        self, temp_outputs_dir, cleanup_config, mock_publisher
    ):
        """Test archiving nonexistent directory raises error."""
        manager = CleanupManager(temp_outputs_dir, cleanup_config, mock_publisher)

        with pytest.raises(ValueError, match="does not exist"):
            manager.archive_directory(temp_outputs_dir / "nonexistent")

    def test_archive_file_not_directory(
        self, temp_outputs_dir, cleanup_config, mock_publisher
    ):
        """Test archiving a file raises error."""
        file_path = temp_outputs_dir / "file.txt"
        file_path.write_text("test")

        manager = CleanupManager(temp_outputs_dir, cleanup_config, mock_publisher)

        with pytest.raises(ValueError, match="not a directory"):
            manager.archive_directory(file_path)


class TestShouldCleanup:
    """Tests for _should_cleanup() method."""

    def test_immediate_cleanup(self, temp_outputs_dir, mock_publisher):
        """Test cleanup allowed when keep_published_days=0."""
        config = CleanupConfig(enabled=True, keep_published_days=0)
        manager = CleanupManager(temp_outputs_dir, config, mock_publisher)

        assert manager._should_cleanup("B0TEST001", None) is True

    def test_old_enough_for_cleanup(self, temp_outputs_dir, mock_publisher):
        """Test cleanup allowed when product is old enough."""
        config = CleanupConfig(enabled=True, keep_published_days=7)
        manager = CleanupManager(temp_outputs_dir, config, mock_publisher)

        old_date = datetime.now(UTC) - timedelta(days=10)
        assert manager._should_cleanup("B0TEST001", old_date) is True

    def test_too_new_for_cleanup(self, temp_outputs_dir, mock_publisher):
        """Test cleanup blocked when product is too new."""
        config = CleanupConfig(enabled=True, keep_published_days=7)
        manager = CleanupManager(temp_outputs_dir, config, mock_publisher)

        recent_date = datetime.now(UTC) - timedelta(days=3)
        assert manager._should_cleanup("B0TEST001", recent_date) is False

    def test_none_published_at(self, temp_outputs_dir, mock_publisher):
        """Test cleanup allowed when published_at is None."""
        config = CleanupConfig(enabled=True, keep_published_days=7)
        manager = CleanupManager(temp_outputs_dir, config, mock_publisher)

        assert manager._should_cleanup("B0TEST001", None) is True


class TestCleanup:
    """Tests for cleanup() async method."""

    @pytest.mark.asyncio
    async def test_cleanup_disabled(
        self, temp_outputs_dir, mock_publisher, product_directory
    ):
        """Test cleanup returns early when disabled."""
        config = CleanupConfig(enabled=False)
        manager = CleanupManager(temp_outputs_dir, config, mock_publisher)

        result = await manager.cleanup("B0TEST001", [Platform.YOUTUBE])

        assert result["success"] is False
        assert "disabled" in result["message"]
        assert product_directory.exists()

    @pytest.mark.asyncio
    async def test_cleanup_missing_directory(
        self, temp_outputs_dir, cleanup_config, mock_publisher
    ):
        """Test cleanup handles missing product directory."""
        manager = CleanupManager(temp_outputs_dir, cleanup_config, mock_publisher)

        result = await manager.cleanup("B0MISSING", [Platform.YOUTUBE])

        assert result["success"] is False
        assert "not found" in result["message"]

    @pytest.mark.asyncio
    async def test_cleanup_dry_run(
        self, temp_outputs_dir, mock_publisher, product_directory
    ):
        """Test dry_run mode doesn't delete files."""
        config = CleanupConfig(enabled=True, verify_before_delete=False)
        manager = CleanupManager(temp_outputs_dir, config, mock_publisher)

        result = await manager.cleanup("B0TEST001", [Platform.YOUTUBE], dry_run=True)

        assert result["success"] is True
        assert "[DRY RUN]" in result["message"]
        assert result["disk_freed"] == 0
        assert product_directory.exists()

    @pytest.mark.asyncio
    async def test_cleanup_verification_failure(
        self, temp_outputs_dir, cleanup_config, mock_publisher, product_directory
    ):
        """Test cleanup blocked when verification fails."""
        create_tracking_file(temp_outputs_dir, "B0TEST001", "youtube", "post123")
        mock_publisher.get_status = AsyncMock(return_value={"status": "scheduled"})

        manager = CleanupManager(temp_outputs_dir, cleanup_config, mock_publisher)

        result = await manager.cleanup("B0TEST001", [Platform.YOUTUBE])

        assert result["success"] is False
        assert "not published" in result["message"]
        assert product_directory.exists()

    @pytest.mark.asyncio
    async def test_cleanup_success_without_archive(
        self, temp_outputs_dir, mock_publisher, product_directory
    ):
        """Test successful cleanup without archiving."""
        config = CleanupConfig(
            enabled=True,
            verify_before_delete=False,
            archive_before_delete=False,
        )
        create_tracking_file(temp_outputs_dir, "B0TEST001", "youtube", "post123")
        manager = CleanupManager(temp_outputs_dir, config, mock_publisher)

        result = await manager.cleanup("B0TEST001", [Platform.YOUTUBE])

        assert result["success"] is True
        assert result["disk_freed"] > 0
        assert not product_directory.exists()
        assert manager.audit_log_path.exists()

    @pytest.mark.asyncio
    async def test_cleanup_success_with_archive(
        self, temp_outputs_dir, mock_publisher, product_directory
    ):
        """Test successful cleanup with archiving."""
        config = CleanupConfig(
            enabled=True,
            verify_before_delete=False,
            archive_before_delete=True,
            archive_dir=temp_outputs_dir / "archive",
        )
        create_tracking_file(temp_outputs_dir, "B0TEST001", "youtube", "post123")
        manager = CleanupManager(temp_outputs_dir, config, mock_publisher)

        result = await manager.cleanup("B0TEST001", [Platform.YOUTUBE])

        assert result["success"] is True
        assert not product_directory.exists()

        # Archive should exist
        archive_dir = temp_outputs_dir / "archive"
        archives = list(archive_dir.glob("*.zip"))
        assert len(archives) == 1

    @pytest.mark.asyncio
    async def test_cleanup_logs_audit(
        self, temp_outputs_dir, mock_publisher, product_directory
    ):
        """Test cleanup creates audit log entry."""
        config = CleanupConfig(enabled=True, verify_before_delete=False)
        create_tracking_file(temp_outputs_dir, "B0TEST001", "youtube", "post123")
        manager = CleanupManager(temp_outputs_dir, config, mock_publisher)

        await manager.cleanup("B0TEST001", [Platform.YOUTUBE])

        data = json.loads(manager.audit_log_path.read_text())
        assert len(data["cleanups"]) == 1
        assert data["cleanups"][0]["product_id"] == "B0TEST001"


class TestCleanupAll:
    """Tests for cleanup_all() batch method."""

    @pytest.mark.asyncio
    async def test_cleanup_all_disabled(
        self, temp_outputs_dir, mock_publisher
    ):
        """Test cleanup_all returns early when disabled."""
        config = CleanupConfig(enabled=False)
        manager = CleanupManager(temp_outputs_dir, config, mock_publisher)

        result = await manager.cleanup_all([Platform.YOUTUBE])

        assert result["cleaned"] == 0
        assert result["skipped"] == 0

    @pytest.mark.asyncio
    async def test_cleanup_all_multiple_products(
        self, temp_outputs_dir, mock_publisher
    ):
        """Test cleanup_all processes multiple products."""
        config = CleanupConfig(enabled=True, verify_before_delete=False)

        # Create multiple product directories
        for i in range(3):
            product_id = f"B0TEST00{i}"
            product_dir = temp_outputs_dir / product_id
            product_dir.mkdir()
            (product_dir / "video.mp4").write_text("video" * 100)
            create_tracking_file(temp_outputs_dir, product_id, "youtube", f"post{i}")

        manager = CleanupManager(temp_outputs_dir, config, mock_publisher)

        result = await manager.cleanup_all([Platform.YOUTUBE])

        assert result["cleaned"] == 3
        assert result["disk_freed"] > 0

    @pytest.mark.asyncio
    async def test_cleanup_all_skips_special_dirs(
        self, temp_outputs_dir, mock_publisher
    ):
        """Test cleanup_all skips archive and hidden directories."""
        config = CleanupConfig(enabled=True, verify_before_delete=False)

        # Create product directory
        product_dir = temp_outputs_dir / "B0TEST001"
        product_dir.mkdir()
        (product_dir / "video.mp4").write_text("video")
        create_tracking_file(temp_outputs_dir, "B0TEST001", "youtube", "post1")

        # Create directories that should be skipped
        (temp_outputs_dir / "archive").mkdir()
        (temp_outputs_dir / ".hidden").mkdir()
        (temp_outputs_dir / "__pycache__").mkdir()

        manager = CleanupManager(temp_outputs_dir, config, mock_publisher)

        result = await manager.cleanup_all([Platform.YOUTUBE])

        assert result["cleaned"] == 1
        assert (temp_outputs_dir / "archive").exists()
        assert (temp_outputs_dir / ".hidden").exists()

    @pytest.mark.asyncio
    async def test_cleanup_all_dry_run(
        self, temp_outputs_dir, mock_publisher
    ):
        """Test cleanup_all dry_run mode."""
        config = CleanupConfig(enabled=True, verify_before_delete=False)

        product_dir = temp_outputs_dir / "B0TEST001"
        product_dir.mkdir()
        (product_dir / "video.mp4").write_text("video")
        create_tracking_file(temp_outputs_dir, "B0TEST001", "youtube", "post1")

        manager = CleanupManager(temp_outputs_dir, config, mock_publisher)

        result = await manager.cleanup_all([Platform.YOUTUBE], dry_run=True)

        assert result["cleaned"] == 1
        assert result["disk_freed"] == 0
        assert product_dir.exists()

    @pytest.mark.asyncio
    async def test_cleanup_all_handles_errors(
        self, temp_outputs_dir, mock_publisher
    ):
        """Test cleanup_all handles individual errors gracefully."""
        config = CleanupConfig(enabled=True, verify_before_delete=True)

        # Create products - one will fail verification
        for i in range(2):
            product_id = f"B0TEST00{i}"
            product_dir = temp_outputs_dir / product_id
            product_dir.mkdir()
            (product_dir / "video.mp4").write_text("video")

        # Only create tracking for first product
        create_tracking_file(temp_outputs_dir, "B0TEST000", "youtube", "post0")
        mock_publisher.get_status = AsyncMock(return_value={"status": "published"})

        manager = CleanupManager(temp_outputs_dir, config, mock_publisher)

        result = await manager.cleanup_all([Platform.YOUTUBE])

        # First succeeds, second fails (no tracking)
        assert result["cleaned"] == 1
        assert result["skipped"] == 1
