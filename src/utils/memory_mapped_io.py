"""Memory-mapped file operations for efficient handling of large media files."""

import logging
import mmap
from pathlib import Path
from typing import BinaryIO

logger = logging.getLogger(__name__)


class MemoryMappedFile:
    """Context manager for memory-mapped file operations."""

    def __init__(self, file_path: Path, mode: str = "r"):
        self.file_path = file_path
        self.mode = mode
        self.file_obj: BinaryIO | None = None
        self.mmap_obj: mmap.mmap | None = None

    def __enter__(self) -> mmap.mmap:
        """Open file and create memory map."""
        try:
            if self.mode == "r":
                self.file_obj = open(self.file_path, "rb")
                self.mmap_obj = mmap.mmap(
                    self.file_obj.fileno(), 0, access=mmap.ACCESS_READ
                )
            elif self.mode == "r+":
                self.file_obj = open(self.file_path, "r+b")
                self.mmap_obj = mmap.mmap(
                    self.file_obj.fileno(), 0, access=mmap.ACCESS_WRITE
                )
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")

            logger.debug(
                f"Memory mapped {self.file_path} "
                f"({self.file_path.stat().st_size} bytes)"
            )
            if self.mmap_obj is None:
                raise RuntimeError("Failed to create memory map")
            return self.mmap_obj

        except Exception as e:
            logger.error(f"Failed to memory map {self.file_path}: {e}")
            self._cleanup()
            raise

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Clean up memory map and file handle."""
        self._cleanup()

    def _cleanup(self):
        """Clean up resources."""
        if self.mmap_obj:
            try:
                self.mmap_obj.close()
            except Exception as e:
                logger.warning(f"Error closing memory map: {e}")
            self.mmap_obj = None

        if self.file_obj:
            try:
                self.file_obj.close()
            except Exception as e:
                logger.warning(f"Error closing file: {e}")
            self.file_obj = None


def copy_file_mmap(
    src_path: Path, dst_path: Path, chunk_size: int = 64 * 1024 * 1024
) -> bool:
    """Copy file using memory mapping for efficiency with large files.

    Args:
    ----
        src_path: Source file path
        dst_path: Destination file path
        chunk_size: Size of chunks to copy (64MB default)

    Returns:
    -------
        True if successful, False otherwise

    """
    try:
        if not src_path.exists():
            logger.error(f"Source file does not exist: {src_path}")
            return False

        file_size = src_path.stat().st_size
        logger.debug(f"Copying {src_path} to {dst_path} ({file_size} bytes)")

        # For small files, use regular copy
        if file_size < 1024 * 1024:  # 1MB threshold
            import shutil

            shutil.copy2(src_path, dst_path)
            return True

        # Ensure destination directory exists
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # Use memory mapping for large files
        with (
            MemoryMappedFile(src_path, "r") as src_mmap,
            open(dst_path, "wb") as dst_file,
        ):
            bytes_copied = 0
            while bytes_copied < file_size:
                chunk_end = min(bytes_copied + chunk_size, file_size)
                chunk = src_mmap[bytes_copied:chunk_end]
                dst_file.write(chunk)
                bytes_copied = chunk_end

        logger.debug(f"Successfully copied {file_size} bytes using memory mapping")
        return True

    except Exception as e:
        logger.error(f"Failed to copy {src_path} to {dst_path}: {e}")
        # Clean up partial file
        if dst_path.exists():
            from contextlib import suppress

            with suppress(Exception):
                dst_path.unlink()
        return False


def read_file_chunk_mmap(file_path: Path, offset: int, size: int) -> bytes | None:
    """Read a specific chunk of a file using memory mapping.

    Args:
    ----
        file_path: Path to file
        offset: Byte offset to start reading
        size: Number of bytes to read

    Returns:
    -------
        Bytes data or None on error

    """
    try:
        with MemoryMappedFile(file_path, "r") as mmap_obj:
            if offset >= len(mmap_obj):
                return b""

            end_offset = min(offset + size, len(mmap_obj))
            chunk_data: bytes = mmap_obj[offset:end_offset]
            return chunk_data

    except Exception as e:
        logger.error(f"Failed to read chunk from {file_path}: {e}")
        return None


def get_file_hash_mmap(file_path: Path, algorithm: str = "sha256") -> str | None:
    """Calculate file hash using memory mapping for large files.

    Args:
    ----
        file_path: Path to file
        algorithm: Hash algorithm (sha256, md5, etc.)

    Returns:
    -------
        Hex digest string or None on error

    """
    try:
        import hashlib

        hash_obj = hashlib.new(algorithm)
        file_size = file_path.stat().st_size

        # For small files, read directly
        if file_size < 1024 * 1024:  # 1MB threshold
            with open(file_path, "rb") as f:
                hash_obj.update(f.read())
            return hash_obj.hexdigest()

        # Use memory mapping for large files
        with MemoryMappedFile(file_path, "r") as mmap_obj:
            # Process in chunks to avoid memory issues
            chunk_size = 64 * 1024 * 1024  # 64MB chunks
            offset = 0

            while offset < len(mmap_obj):
                chunk_end = min(offset + chunk_size, len(mmap_obj))
                chunk = mmap_obj[offset:chunk_end]
                hash_obj.update(chunk)
                offset = chunk_end

        return hash_obj.hexdigest()

    except Exception as e:
        logger.error(f"Failed to calculate hash for {file_path}: {e}")
        return None


def is_file_suitable_for_mmap(file_path: Path, min_size: int = 1024 * 1024) -> bool:
    """Check if a file is suitable for memory mapping.

    Args:
    ----
        file_path: Path to file
        min_size: Minimum size for memory mapping (1MB default)

    Returns:
    -------
        True if suitable for memory mapping

    """
    try:
        if not file_path.exists():
            return False

        file_size = file_path.stat().st_size

        # Only use memory mapping for files above threshold
        if file_size < min_size:
            return False

        # Check if we have enough virtual memory
        # This is a rough heuristic
        try:
            import psutil

            available_memory: int = psutil.virtual_memory().available
            result: bool = file_size < available_memory * 0.8
            return result  # Use max 80% of available memory
        except ImportError:
            # If psutil not available, assume it's suitable if file is not huge
            return file_size < 1024 * 1024 * 1024  # 1GB threshold

    except Exception as e:
        logger.warning(f"Error checking file suitability for mmap: {e}")
        return False
