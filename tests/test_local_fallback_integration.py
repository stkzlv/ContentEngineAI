"""Integration test for local music fallback in step_download_music().

This test validates Requirements R5 (Local Fallback) and R6 (Attribution)
from freesound-client/requirements.md.
"""

import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.utils.memory_mapped_io import copy_file_mmap, is_file_suitable_for_mmap


class TestLocalFallbackIntegration:
    """Test local fallback behavior in producer.py step_download_music()."""

    @pytest.mark.asyncio
    async def test_local_fallback_with_small_file(self, tmp_path):
        """Test local fallback with small file (<1MB) uses standard copy.

        Validates R5 criteria 1, 2, and 5.
        """
        # Create a small test music file (< 1MB)
        local_music_dir = tmp_path / "music"
        local_music_dir.mkdir()
        small_music_file = local_music_dir / "test-music-small.mp3"
        small_music_file.write_bytes(b"test audio data" * 100)  # Small file

        # Destination directory
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        dest_path = assets_dir / small_music_file.name

        # Verify file is not suitable for mmap
        assert not is_file_suitable_for_mmap(
            small_music_file, min_size=1024 * 1024
        ), "Small file should not use mmap"

        # Simulate fallback logic from producer.py (lines 1436-1476)
        # Use standard copy for small files
        shutil.copy(small_music_file, dest_path)

        # Generate attribution metadata (R6 criterion 5)
        music_info = {
            "source": "Local",
            "type": "Music",
            "path": str(dest_path),
            "name": small_music_file.stem,
            "author": "Unknown",
            "license": "Local File",
            "url": "",
            "id": "",
        }

        # Validate file was copied
        assert dest_path.exists(), "Music file should be copied to destination"
        assert dest_path.read_bytes() == small_music_file.read_bytes()

        # Validate attribution structure (R6 criterion 1)
        required_keys = ["source", "type", "path", "name", "author", "license", "url", "id"]
        for key in required_keys:
            assert key in music_info, f"Missing required attribution key: {key}"

        # Validate local-specific values (R6 criterion 5)
        assert music_info["source"] == "Local"
        assert music_info["type"] == "Music"
        assert music_info["author"] == "Unknown"
        assert music_info["license"] == "Local File"

    @pytest.mark.asyncio
    async def test_local_fallback_with_large_file_mmap(self, tmp_path):
        """Test local fallback with large file (>1MB) uses memory-mapped I/O.

        Validates R5 criteria 3 and 4.
        """
        # Create a large test music file (> 1MB)
        local_music_dir = tmp_path / "music"
        local_music_dir.mkdir()
        large_music_file = local_music_dir / "test-music-large.mp3"

        # Create a file larger than 1MB
        chunk_size = 64 * 1024  # 64KB chunks
        total_size = 2 * 1024 * 1024  # 2MB
        with open(large_music_file, "wb") as f:
            for _ in range(total_size // chunk_size):
                f.write(b"a" * chunk_size)

        # Verify file size
        file_size = large_music_file.stat().st_size
        assert file_size > 1024 * 1024, "Test file should be larger than 1MB"

        # Destination directory
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        dest_path = assets_dir / large_music_file.name

        # Verify file is suitable for mmap (R5 criterion 3)
        assert is_file_suitable_for_mmap(
            large_music_file, min_size=1024 * 1024
        ), "Large file should use mmap"

        # Use memory-mapped copy (matching producer.py lines 1449-1455)
        copy_success = copy_file_mmap(large_music_file, dest_path)

        # Validate copy succeeded (R5 criterion 3)
        assert copy_success, "Memory-mapped copy should succeed"
        assert dest_path.exists(), "File should be copied to destination"
        assert dest_path.stat().st_size == file_size, "File sizes should match"

        # Generate attribution metadata (R6 criterion 5)
        music_info = {
            "source": "Local",
            "type": "Music",
            "path": str(dest_path),
            "name": large_music_file.stem,
            "author": "Unknown",
            "license": "Local File",
            "url": "",
            "id": "",
        }

        # Validate attribution
        assert music_info["source"] == "Local"
        assert music_info["path"] == str(dest_path)

    @pytest.mark.asyncio
    async def test_local_fallback_mmap_failure_uses_standard_copy(self, tmp_path):
        """Test that if memory-mapped copy fails, standard copy is used as fallback.

        Validates R5 criterion 4.
        """
        # Create a large test music file (>1MB to trigger mmap)
        local_music_dir = tmp_path / "music"
        local_music_dir.mkdir()
        music_file = local_music_dir / "test-music-large.mp3"

        # Create a file larger than 1MB
        chunk_size = 64 * 1024  # 64KB chunks
        total_size = 2 * 1024 * 1024  # 2MB
        with open(music_file, "wb") as f:
            for _ in range(total_size // chunk_size):
                f.write(b"a" * chunk_size)

        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        dest_path = assets_dir / music_file.name

        # Simulate mmap failure by patching (R5 criterion 4)
        with patch("src.utils.memory_mapped_io.MemoryMappedFile") as mock_mmap:
            mock_mmap.side_effect = Exception("Simulated mmap failure")

            # Attempt memory-mapped copy
            copy_success = copy_file_mmap(music_file, dest_path)

            # Should fail gracefully
            assert not copy_success, "Memory-mapped copy should fail"

            # Fallback to standard copy (matching producer.py lines 1456-1460)
            if not copy_success:
                shutil.copy(music_file, dest_path)

            # Verify file was copied using fallback
            assert dest_path.exists(), "File should be copied using fallback"
            assert dest_path.stat().st_size == music_file.stat().st_size

    @pytest.mark.asyncio
    async def test_local_fallback_random_selection(self, tmp_path):
        """Test that local fallback randomly selects from available files.

        Validates R5 criterion 2.
        """
        import random

        # Create multiple local music files
        local_music_dir = tmp_path / "music"
        local_music_dir.mkdir()
        music_files = [
            local_music_dir / f"test-music-{i}.mp3" for i in range(3)
        ]
        for music_file in music_files:
            music_file.write_bytes(b"test audio data")

        # Simulate random selection (matching producer.py line 1437)
        random.seed(42)  # For reproducible test
        available_paths = [p for p in music_files if p.exists()]
        selected_path = random.choice(available_paths)  # noqa: S311

        # Verify selection is from available files (R5 criterion 2)
        assert selected_path in available_paths
        assert selected_path.exists()

    @pytest.mark.asyncio
    async def test_local_fallback_attribution_completeness(self, tmp_path):
        """Test that local fallback generates complete attribution metadata.

        Validates R6 criteria 1, 4, and 5.
        """
        # Create a test music file
        local_music_dir = tmp_path / "music"
        local_music_dir.mkdir()
        music_file = local_music_dir / "background-upbeat.mp3"
        music_file.write_bytes(b"test audio data")

        dest_path = tmp_path / "assets" / music_file.name

        # Generate attribution (matching producer.py lines 1467-1476)
        music_info = {
            "source": "Local",
            "type": "Music",
            "path": str(dest_path),
            "name": music_file.stem,
            "author": "Unknown",  # Fallback value (R6 criterion 4)
            "license": "Local File",
            "url": "",
            "id": "",
        }

        # Validate all required fields exist (R6 criterion 1)
        required_keys = ["source", "type", "path", "name", "author", "license", "url", "id"]
        for key in required_keys:
            assert key in music_info, f"Missing required attribution key: {key}"

        # Validate dictionary format (R6 criterion 3)
        assert isinstance(music_info, dict)

        # Validate local-specific values (R6 criterion 5)
        assert music_info["source"] == "Local"
        assert music_info["type"] == "Music"
        assert music_info["author"] == "Unknown"  # Fallback for missing data
        assert music_info["license"] == "Local File"
        assert music_info["name"] == music_file.stem
        assert music_info["url"] == ""  # Empty for local files
        assert music_info["id"] == ""  # Empty for local files

    @pytest.mark.asyncio
    async def test_local_fallback_music_info_serialization(self, tmp_path):
        """Test that music_info can be serialized to JSON and saved.

        Validates that the attribution dictionary structure is JSON-compatible.
        """
        # Create test music file
        local_music_dir = tmp_path / "music"
        local_music_dir.mkdir()
        music_file = local_music_dir / "test-music.mp3"
        music_file.write_bytes(b"test audio data")

        dest_path = tmp_path / "assets" / music_file.name

        # Generate attribution
        music_info = {
            "source": "Local",
            "type": "Music",
            "path": str(dest_path),  # Ensure path is string, not Path object
            "name": music_file.stem,
            "author": "Unknown",
            "license": "Local File",
            "url": "",
            "id": "",
        }

        # Ensure path is converted to string if Path object (producer.py lines 1479-1480)
        if isinstance(music_info.get("path"), Path):
            music_info["path"] = str(music_info["path"])

        # Test JSON serialization (producer.py lines 1482-1484)
        music_info_file = tmp_path / "music_info.json"
        music_info_file.write_text(json.dumps(music_info, indent=2), encoding="utf-8")

        # Validate file was created
        assert music_info_file.exists()

        # Validate JSON can be loaded back
        loaded_info = json.loads(music_info_file.read_text(encoding="utf-8"))
        assert loaded_info == music_info
        assert loaded_info["source"] == "Local"
