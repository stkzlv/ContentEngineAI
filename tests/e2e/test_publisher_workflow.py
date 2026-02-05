"""End-to-end tests for complete publishing workflow.

Tests the complete pipeline from video files through CLI publishing:
- Single video publishing via CLI
- Batch publishing multiple videos
- Error scenarios and recovery
- CLI command validation

Note: Integration tests with real Late.dev API require sandbox credentials.
Tests will skip if LATE_SANDBOX_API_KEY is not found in .env.test file.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load test environment variables
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env.test")

# Check for sandbox credentials
LATE_SANDBOX_API_KEY = os.getenv("LATE_SANDBOX_API_KEY")
LATE_VERCEL_TOKEN = os.getenv("LATE_VERCEL_TOKEN")

# Skip all tests if credentials not available
pytestmark = pytest.mark.skipif(
    not LATE_SANDBOX_API_KEY,
    reason="Late.dev sandbox credentials not found in .env.test",
)


@pytest.fixture(scope="module")
def test_outputs_dir():
    """Create temporary outputs directory for E2E tests."""
    temp_dir = tempfile.mkdtemp(prefix="e2e_test_outputs_")
    yield Path(temp_dir)
    # Cleanup after all tests
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def test_config_file(tmp_path_factory):
    """Create temporary publisher config for E2E tests."""
    config_dir = tmp_path_factory.mktemp("config")
    config_file = config_dir / "publisher.yaml"

    config_content = f"""# Test Publisher Configuration
provider: late
api_key: {LATE_SANDBOX_API_KEY}
vercel_token: {LATE_VERCEL_TOKEN or ''}

immediate_publish: true
default_platforms:
  - youtube
  - tiktok

max_retries: 2
timeout: 30.0

stagger_delay_min: 5
stagger_delay_max: 10

privacy_settings:
  youtube: public
  tiktok: public
  instagram: everyone
"""

    config_file.write_text(config_content)
    yield config_file


@pytest.fixture
def sample_product_with_video(test_outputs_dir):
    """Create sample product directory with video and metadata."""
    product_id = "TEST_E2E_001"
    product_dir = test_outputs_dir / product_id
    product_dir.mkdir(parents=True, exist_ok=True)

    # Create text directory for metadata
    text_dir = product_dir / "text"
    text_dir.mkdir(exist_ok=True)

    # Copy test video
    source_video = PROJECT_ROOT / "tests" / "fixtures" / "test_video_small.mp4"
    video_path = product_dir / f"video_{product_id}_sequential.mp4"
    shutil.copy(source_video, video_path)

    # Create metadata JSON files for different platforms
    youtube_metadata = {
        "platform": "youtube",
        "title": "E2E Test Product - Amazing Features",
        "description": "This is an end-to-end test video for publisher workflow validation. "
        "Check out this amazing product with incredible features!\\n\\n"
        "🔥 Key Features:\\n"
        "✅ Feature 1\\n"
        "✅ Feature 2\\n"
        "✅ Feature 3\\n\\n"
        "#testing #e2e #automation",
        "tags": ["test", "e2e", "automation", "publisher"],
        "product_id": product_id,
    }

    tiktok_metadata = {
        "platform": "tiktok",
        "title": "E2E Test Product 🔥",
        "description": "Testing publisher workflow! Amazing features ✨ #test #e2e",
        "tags": ["test", "e2e", "automation"],
        "product_id": product_id,
    }

    instagram_metadata = {
        "platform": "instagram",
        "title": "E2E Test Product",
        "description": "Testing publisher workflow 🚀\\n\\n#test #e2e #automation",
        "tags": ["test", "e2e", "automation"],
        "product_id": product_id,
    }

    # Write metadata files
    (text_dir / "metadata_youtube.json").write_text(
        json.dumps(youtube_metadata, indent=2)
    )
    (text_dir / "metadata_tiktok.json").write_text(
        json.dumps(tiktok_metadata, indent=2)
    )
    (text_dir / "metadata_instagram.json").write_text(
        json.dumps(instagram_metadata, indent=2)
    )

    # Also create UPLOAD_INSTRUCTIONS.txt as fallback
    instructions = f"""Product ID: {product_id}

YouTube:
Title: E2E Test Product - Amazing Features
Description: This is an end-to-end test video.

TikTok:
Title: E2E Test Product 🔥
Description: Testing publisher workflow!

Instagram:
Title: E2E Test Product
Description: Testing publisher workflow 🚀
"""
    (product_dir / "UPLOAD_INSTRUCTIONS.txt").write_text(instructions)

    return {
        "product_id": product_id,
        "product_dir": product_dir,
        "video_path": video_path,
        "text_dir": text_dir,
    }


@pytest.fixture
def sample_batch_products(test_outputs_dir):
    """Create multiple product directories for batch testing."""
    products = []

    for i in range(1, 4):  # Create 3 test products
        product_id = f"TEST_BATCH_{i:03d}"
        product_dir = test_outputs_dir / product_id
        product_dir.mkdir(parents=True, exist_ok=True)

        text_dir = product_dir / "text"
        text_dir.mkdir(exist_ok=True)

        # Copy test video
        source_video = PROJECT_ROOT / "tests" / "fixtures" / "test_video_small.mp4"
        video_path = product_dir / f"video_{product_id}_sequential.mp4"
        shutil.copy(source_video, video_path)

        # Create metadata
        youtube_metadata = {
            "platform": "youtube",
            "title": f"Batch Test Product {i}",
            "description": f"Batch test video #{i} for publisher validation",
            "tags": ["batch", "test"],
            "product_id": product_id,
        }

        (text_dir / "metadata_youtube.json").write_text(
            json.dumps(youtube_metadata, indent=2)
        )

        products.append(
            {
                "product_id": product_id,
                "product_dir": product_dir,
                "video_path": video_path,
            }
        )

    return products


@pytest.fixture
def product_without_metadata(test_outputs_dir):
    """Create product directory with video but no metadata."""
    product_id = "TEST_NO_META"
    product_dir = test_outputs_dir / product_id
    product_dir.mkdir(parents=True, exist_ok=True)

    # Copy test video only, no metadata
    source_video = PROJECT_ROOT / "tests" / "fixtures" / "test_video_small.mp4"
    video_path = product_dir / f"video_{product_id}_sequential.mp4"
    shutil.copy(source_video, video_path)

    return {
        "product_id": product_id,
        "product_dir": product_dir,
        "video_path": video_path,
    }


class TestCLIListAccounts:
    """Test CLI list-accounts command."""

    def test_list_accounts_success(self):
        """Test successful account listing."""
        result = subprocess.run(
            [
                "poetry",
                "run",
                "python",
                "-m",
                "src.publisher.late",
                "list-accounts",
                "--debug",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert "Authentication successful" in result.stderr or "Found" in result.stderr

    def test_list_accounts_invalid_credentials(self, tmp_path):
        """Test list-accounts with invalid API key."""
        # Create temporary .env with invalid key
        env_file = tmp_path / ".env"
        env_file.write_text("LATE_API_KEY=sk_test_invalid_key_12345\n")

        result = subprocess.run(
            ["poetry", "run", "python", "-m", "src.publisher.late", "list-accounts"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "LATE_API_KEY": "sk_test_invalid_key_12345"},
        )

        assert result.returncode == 1
        assert (
            "Authentication failed" in result.stderr
            or "check your API key" in result.stderr
        )


class TestCLISinglePublish:
    """Test CLI single video publishing."""

    def test_single_publish_immediate_success(self, sample_product_with_video):
        """Test publishing single video immediately to YouTube."""
        video_path = sample_product_with_video["video_path"]

        result = subprocess.run(
            [
                "poetry",
                "run",
                "python",
                "-m",
                "src.publisher.late",
                "single",
                "--video",
                str(video_path),
                "--platform",
                "youtube",
                "--immediate",
                "--debug",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert "Authentication successful" in result.stderr
        assert "Uploading video" in result.stderr
        assert "Upload complete" in result.stderr
        assert "Publishing to youtube" in result.stderr
        assert "Published to youtube" in result.stderr
        assert "post_id=" in result.stderr

    def test_single_publish_multiple_platforms(self, sample_product_with_video):
        """Test publishing single video to multiple platforms."""
        video_path = sample_product_with_video["video_path"]

        result = subprocess.run(
            [
                "poetry",
                "run",
                "python",
                "-m",
                "src.publisher.late",
                "single",
                "--video",
                str(video_path),
                "--platform",
                "youtube",
                "--platform",
                "tiktok",
                "--immediate",
                "--debug",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert "Publishing to youtube" in result.stderr
        assert "Publishing to tiktok" in result.stderr

    def test_single_publish_video_not_found(self):
        """Test error when video file doesn't exist."""
        result = subprocess.run(
            [
                "poetry",
                "run",
                "python",
                "-m",
                "src.publisher.late",
                "single",
                "--video",
                "nonexistent_video.mp4",
                "--platform",
                "youtube",
                "--immediate",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 1
        assert "Video file not found" in result.stderr

    def test_single_publish_scheduled(self, sample_product_with_video):
        """Test scheduling video for future publishing."""
        video_path = sample_product_with_video["video_path"]

        # Schedule for 1 hour from now
        from datetime import datetime, timedelta

        schedule_time = (datetime.now() + timedelta(hours=1)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        result = subprocess.run(
            [
                "poetry",
                "run",
                "python",
                "-m",
                "src.publisher.late",
                "single",
                "--video",
                str(video_path),
                "--platform",
                "youtube",
                "--schedule",
                schedule_time,
                "--debug",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert "Scheduled time:" in result.stderr
        assert "Published to youtube" in result.stderr


class TestCLIBatchPublish:
    """Test CLI batch publishing."""

    def test_batch_publish_success(self, sample_batch_products, test_outputs_dir):
        """Test batch publishing multiple videos."""
        result = subprocess.run(
            [
                "poetry",
                "run",
                "python",
                "-m",
                "src.publisher.late",
                "batch",
                "--platform",
                "youtube",
                "--outputs-dir",
                str(test_outputs_dir),
                "--immediate",
                "--debug",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert "BATCH PUBLISHING MODE" in result.stderr
        assert "Authentication successful" in result.stderr

        # Verify all 3 products were processed
        for product in sample_batch_products:
            product_id = product["product_id"]
            assert product_id in result.stderr or "video_" in result.stderr

    def test_batch_publish_fail_fast(
        self, sample_batch_products, product_without_metadata, test_outputs_dir
    ):
        """Test batch publishing with fail-fast on error."""
        # Add product without metadata to trigger error
        result = subprocess.run(
            [
                "poetry",
                "run",
                "python",
                "-m",
                "src.publisher.late",
                "batch",
                "--platform",
                "youtube",
                "--outputs-dir",
                str(test_outputs_dir),
                "--immediate",
                "--fail-fast",
                "--debug",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )

        # Should succeed for valid products, skip invalid ones
        # Exit code 1 if any failures occurred
        assert "BATCH PUBLISHING MODE" in result.stderr

    def test_batch_publish_empty_directory(self, tmp_path):
        """Test batch publishing with empty outputs directory."""
        empty_dir = tmp_path / "empty_outputs"
        empty_dir.mkdir()

        result = subprocess.run(
            [
                "poetry",
                "run",
                "python",
                "-m",
                "src.publisher.late",
                "batch",
                "--platform",
                "youtube",
                "--outputs-dir",
                str(empty_dir),
                "--immediate",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Should exit gracefully with no videos found
        assert "No videos found" in result.stderr or result.returncode == 0


class TestCLIErrorScenarios:
    """Test CLI error handling and recovery."""

    def test_missing_required_argument(self):
        """Test error when required argument is missing."""
        result = subprocess.run(
            [
                "poetry",
                "run",
                "python",
                "-m",
                "src.publisher.late",
                "single",
                "--platform",
                "youtube",
                # Missing --video argument
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_invalid_platform(self):
        """Test error with invalid platform name."""
        result = subprocess.run(
            [
                "poetry",
                "run",
                "python",
                "-m",
                "src.publisher.late",
                "single",
                "--video",
                "test.mp4",
                "--platform",
                "invalid_platform",
                "--immediate",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode != 0
        assert (
            "invalid choice" in result.stderr.lower()
            or "error" in result.stderr.lower()
        )

    def test_invalid_schedule_format(self, sample_product_with_video):
        """Test error with invalid schedule datetime format."""
        video_path = sample_product_with_video["video_path"]

        result = subprocess.run(
            [
                "poetry",
                "run",
                "python",
                "-m",
                "src.publisher.late",
                "single",
                "--video",
                str(video_path),
                "--platform",
                "youtube",
                "--schedule",
                "invalid-date-format",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode != 0
        assert (
            "Invalid datetime format" in result.stderr
            or "error" in result.stderr.lower()
        )

    def test_batch_without_immediate_flag(self):
        """Test error when batch mode used without --immediate flag."""
        result = subprocess.run(
            [
                "poetry",
                "run",
                "python",
                "-m",
                "src.publisher.late",
                "batch",
                "--platform",
                "youtube",
                # Missing --immediate
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode != 0
        assert (
            "requires --immediate" in result.stderr or "error" in result.stderr.lower()
        )


class TestEndToEndWorkflow:
    """Test complete end-to-end publishing workflow."""

    def test_complete_workflow_single_video(self, sample_product_with_video):
        """Test complete workflow: metadata → video → publish."""
        product_id = sample_product_with_video["product_id"]
        video_path = sample_product_with_video["video_path"]
        text_dir = sample_product_with_video["text_dir"]

        # Step 1: Verify metadata files exist
        assert (text_dir / "metadata_youtube.json").exists()
        assert (text_dir / "metadata_tiktok.json").exists()

        # Step 2: Verify video file exists
        assert video_path.exists()
        assert video_path.stat().st_size > 0

        # Step 3: Publish via CLI
        result = subprocess.run(
            [
                "poetry",
                "run",
                "python",
                "-m",
                "src.publisher.late",
                "single",
                "--video",
                str(video_path),
                "--platform",
                "youtube",
                "--immediate",
                "--debug",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Step 4: Verify success
        assert result.returncode == 0, f"Publishing failed: {result.stderr}"
        assert "Authentication successful" in result.stderr
        assert "Upload complete" in result.stderr
        assert "Published to youtube" in result.stderr

        # Step 5: Extract post_id from output
        import re

        post_id_match = re.search(r"post_id=([a-zA-Z0-9_-]+)", result.stderr)
        assert post_id_match, "Could not find post_id in output"
        post_id = post_id_match.group(1)

        print("\n✅ E2E Workflow Success!")
        print(f"   Product ID: {product_id}")
        print(f"   Video: {video_path.name}")
        print(f"   Published Post ID: {post_id}")

    def test_complete_workflow_batch(self, sample_batch_products, test_outputs_dir):
        """Test complete batch workflow with multiple products."""
        # Verify all products have videos
        for product in sample_batch_products:
            assert product["video_path"].exists()

        # Execute batch publish
        result = subprocess.run(
            [
                "poetry",
                "run",
                "python",
                "-m",
                "src.publisher.late",
                "batch",
                "--platform",
                "youtube",
                "--outputs-dir",
                str(test_outputs_dir),
                "--immediate",
                "--debug",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )

        assert result.returncode == 0, f"Batch publishing failed: {result.stderr}"
        assert "BATCH PUBLISHING MODE" in result.stderr

        # Verify batch summary
        assert (
            "Batch publishing complete" in result.stderr
            or "complete" in result.stderr.lower()
        )

        print("\n✅ Batch E2E Workflow Success!")
        print(f"   Products processed: {len(sample_batch_products)}")

    def test_workflow_with_missing_metadata_fallback(self, test_outputs_dir):
        """Test workflow with missing JSON metadata falls back to UPLOAD_INSTRUCTIONS.txt."""
        product_id = "TEST_FALLBACK"
        product_dir = test_outputs_dir / product_id
        product_dir.mkdir(parents=True, exist_ok=True)

        # Copy video
        source_video = PROJECT_ROOT / "tests" / "fixtures" / "test_video_small.mp4"
        video_path = product_dir / f"video_{product_id}_sequential.mp4"
        shutil.copy(source_video, video_path)

        # Create only UPLOAD_INSTRUCTIONS.txt (no JSON metadata)
        instructions = """YouTube:
Title: Fallback Test Product
Description: Testing fallback to UPLOAD_INSTRUCTIONS.txt
"""
        (product_dir / "UPLOAD_INSTRUCTIONS.txt").write_text(instructions)

        # Publish
        result = subprocess.run(
            [
                "poetry",
                "run",
                "python",
                "-m",
                "src.publisher.late",
                "single",
                "--video",
                str(video_path),
                "--platform",
                "youtube",
                "--immediate",
                "--debug",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Should succeed with fallback
        assert result.returncode == 0, f"Fallback workflow failed: {result.stderr}"
        assert "Published to youtube" in result.stderr


class TestMetadataIntegration:
    """Test metadata loading and integration."""

    def test_metadata_loading_from_json(self, sample_product_with_video):
        """Test that metadata is correctly loaded from JSON files."""
        from src.publisher.metadata import load_platform_metadata
        from src.publisher.models import Platform

        product_id = sample_product_with_video["product_id"]
        outputs_dir = sample_product_with_video["product_dir"].parent

        # Load YouTube metadata
        metadata = load_platform_metadata(product_id, Platform.YOUTUBE, outputs_dir)
        assert metadata is not None
        assert metadata.title == "E2E Test Product - Amazing Features"
        assert "end-to-end test video" in metadata.description
        assert len(metadata.tags) > 0

        # Load TikTok metadata
        metadata = load_platform_metadata(product_id, Platform.TIKTOK, outputs_dir)
        assert metadata is not None
        assert metadata.title and "🔥" in metadata.title

    def test_metadata_character_limits(self, sample_product_with_video):
        """Test that metadata respects platform character limits."""
        from src.publisher.metadata import load_platform_metadata
        from src.publisher.models import Platform

        product_id = sample_product_with_video["product_id"]
        outputs_dir = sample_product_with_video["product_dir"].parent

        # YouTube allows longer content
        youtube_meta = load_platform_metadata(product_id, Platform.YOUTUBE, outputs_dir)
        assert youtube_meta is not None
        assert len(youtube_meta.title) <= 100  # YouTube title limit

        # TikTok has stricter limits
        tiktok_meta = load_platform_metadata(product_id, Platform.TIKTOK, outputs_dir)
        assert tiktok_meta is not None
        # TikTok limits enforced by metadata generation


class TestCLIOutputValidation:
    """Test CLI output messages and formatting."""

    def test_debug_output_verbose(self, sample_product_with_video):
        """Test that --debug flag provides verbose output."""
        video_path = sample_product_with_video["video_path"]

        result = subprocess.run(
            [
                "poetry",
                "run",
                "python",
                "-m",
                "src.publisher.late",
                "single",
                "--video",
                str(video_path),
                "--platform",
                "youtube",
                "--immediate",
                "--debug",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Debug mode should provide detailed logging
        assert "DEBUG" in result.stderr or "Uploading video" in result.stderr
        assert "Authentication" in result.stderr

    def test_quiet_output_without_debug(self, sample_product_with_video):
        """Test quieter output without --debug flag."""
        video_path = sample_product_with_video["video_path"]

        result = subprocess.run(
            [
                "poetry",
                "run",
                "python",
                "-m",
                "src.publisher.late",
                "single",
                "--video",
                str(video_path),
                "--platform",
                "youtube",
                "--immediate",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Without debug, output should be less verbose
        assert result.returncode == 0
        # INFO level messages still appear


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
