"""Utility Functions Module for ContentEngineAI

This module provides a collection of utility functions used throughout the
ContentEngineAI project for common operations such as file management, text
processing, network operations, and debugging support.
"""

import asyncio
import json
import logging
import mimetypes
import os
import re
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp
from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Import the script sanitizer
from .script_sanitizer import sanitize_script

try:
    from tenacity import AsyncRetrying  # type: ignore[attr-defined]
except ImportError:
    AsyncRetrying = Retrying  # Fallback

try:
    from playwright.async_api import Page
except ImportError:

    class Page:  # type: ignore
        pass


# Constants for file handling
MAX_FILENAME_LENGTH = 200  # Maximum safe filename length

logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)

    Args:
    ----
        seconds: Time in seconds
    Returns:
        Formatted timestamp string in SRT format

    """
    ms = int((seconds % 1) * 1000)
    time_obj = timedelta(seconds=int(seconds))
    hours = int(time_obj.total_seconds() // 3600)
    minutes = int((time_obj.total_seconds() % 3600) // 60)
    secs = int(time_obj.total_seconds() % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def ensure_dirs_exist(path: Path) -> None:
    """Ensure that the parent directories for the given path exist.
    If path is a directory, ensure the path itself exists.
    Logs an error but does not re-raise exceptions during directory creation.
    """
    try:
        if path.suffix:  # If path includes a filename, make parent dirs
            path.parent.mkdir(parents=True, exist_ok=True)
        else:  # If path is a directory path, make the path itself
            path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create directories for {path}: {e}")


def cleanup_temp_dirs(*temp_dirs: Path, verify: bool = False) -> bool:
    """Remove temporary directories and their contents safely.

    Args:
    ----
        *temp_dirs: Paths to temporary directories to clean up
        verify: If True, verify that all directories were successfully removed

    Returns:
    -------
        bool: True if all directories were successfully cleaned up (or if verify=False)

    """
    all_cleaned = True
    for temp_dir in temp_dirs:
        if not temp_dir or not isinstance(temp_dir, Path):
            continue

        if not temp_dir.exists():
            logger.debug(f"Temporary directory does not exist: {temp_dir}")
            continue

        try:
            # First remove all empty files (0 bytes) which might cause issues
            for file_path in temp_dir.glob("**/*"):
                if file_path.is_file() and file_path.stat().st_size == 0:
                    try:
                        file_path.unlink()
                        logger.debug(f"Removed empty file: {file_path}")
                    except Exception as e:
                        logger.debug(f"Failed to remove empty file {file_path}: {e}")

            # Then remove the entire directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")

            # Verify removal if requested
            if verify and temp_dir.exists():
                logger.warning(f"Directory still exists after cleanup: {temp_dir}")
                all_cleaned = False

        except Exception as e:
            logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")
            all_cleaned = False

    return all_cleaned if verify else True


async def download_file(
    url: str,
    output_path: Path,
    session: aiohttp.ClientSession,
    timeout_sec: int = 30,
    retry_attempts: int = 3,
    retry_min_wait_sec: int = 2,
    retry_max_wait_sec: int = 10,
) -> bool:
    """Download a file from a URL and save it to the specified path with retries.

    Args:
    ----
        url: The URL of the file to download.
        output_path: The path where the file should be saved.
        session: The aiohttp.ClientSession to use for the download.
        timeout_sec: Request timeout in seconds.
        retry_attempts: Number of retry attempts.
        retry_min_wait_sec: Minimum wait time between retries.
        retry_max_wait_sec: Maximum wait time between retries.

    Returns:
    -------
        True if the download was successful, False otherwise.

    """
    # Using a shared retryer pattern could be better than creating one per call
    # but for now, keeping it here for clarity and independence.
    download_retryer = AsyncRetrying(
        stop=stop_after_attempt(retry_attempts),
        wait=wait_exponential(
            multiplier=1, min=retry_min_wait_sec, max=retry_max_wait_sec
        ),
        retry=retry_if_exception_type(
            (aiohttp.ClientError, asyncio.TimeoutError, IOError)
        ),  # Also retry on IOError if file is empty
        reraise=False,  # Don't re-raise, let us handle the RetryError
    )

    try:
        async for attempt in download_retryer:
            with attempt:
                # Ensure directory exists before writing
                ensure_dirs_exist(output_path)

                try:
                    # Use a temporary file for download to avoid partial files
                    temp_file = output_path.with_suffix(output_path.suffix + ".tmp")

                    # Perform the download with a timeout
                    async with session.get(url, timeout=timeout_sec) as response:  # type: ignore[attr-defined]
                        response.raise_for_status()  # Raise exception for non-200
                        # status

                        # Stream the response to the temporary file
                        with open(temp_file, "wb") as f:
                            async for chunk in response.content.iter_any():
                                f.write(chunk)

                            # Flush the file to ensure all data is written
                            f.flush()

                        # Verify the file is not empty (after file is closed)
                        if temp_file.stat().st_size == 0:
                            raise OSError(f"Downloaded file is empty: {url}")

                        # Rename temporary file to final name
                        temp_file.replace(output_path)

                        logger.info(f"Successfully downloaded {url} to {output_path}")
                        return True

                except Exception as e:
                    # Clean up temporary file if it exists
                    if "temp_file" in locals() and temp_file.exists():
                        temp_file.unlink(missing_ok=True)

                    # Re-raise the exception to trigger retry
                    logger.warning(
                        f"Download attempt {attempt.retry_state.attempt_number} "
                        f"failed for {url}: {e}"
                    )
                    raise

    except RetryError as e:
        # All retries failed
        logger.error(f"Failed to download {url} after multiple attempts: {e.__cause__}")
        return False

    # This should never be reached due to the retry logic, but satisfies MyPy
    return False


def sanitize_filename(filename: str) -> str:
    """Sanitizes a string to be safe for use as a filename or directory name,
    removing or replacing invalid characters and limiting length.

    Args:
    ----
        filename: The input string.

    Returns:
    -------
        A sanitized string suitable for filesystem use.

    """
    if not filename or not filename.strip():
        return "file"

    # Start with the filename
    name = filename

    # Remove path components and get just the filename
    # Only use basename if it looks like a proper path (starts with / or contains
    # path separators)
    # Don't apply basename if \ is part of the filename content
    if name.startswith("/") or ("/" in name and not name.startswith("\\")):
        # Only use basename if it's actually a path, not just a filename with slashes
        # Check if it contains any invalid filename characters - if so, don't
        # treat as path
        invalid_chars = ["<", ">", ":", '"', "\\", "|", "?", "*"]
        if "/" in name and not any(char in name for char in invalid_chars):
            name = os.path.basename(name)

    # Replace invalid characters with underscores
    # Windows has more restrictions than Unix, so we handle both
    # Replace invalid characters with underscores
    name = re.sub(r'[<>:"/\\\\|?*]', "_", name)

    # Replace multiple spaces/underscores with a single one
    # But preserve consecutive underscores that were created by invalid char replacement
    name = re.sub(r"\s+", "_", name)  # Replace spaces with underscores
    name = re.sub(r"_{3,}", "_", name)  # Collapse 3+ underscores to 1 underscore

    # Remove leading/trailing spaces, dots, and underscores
    name = name.strip(". ")
    # Remove leading and trailing underscores
    name = name.strip("_")

    # Additional cleanup: remove any remaining leading/trailing underscores
    name = name.lstrip("_").rstrip("_")

    # Remove underscores that are adjacent to dots (before extensions)
    # But only if there are multiple underscores
    name = re.sub(r"_{2,}\.", ".", name)

    # Remove trailing underscores that are immediately before dots (extensions)
    # This handles cases like "file_name_.txt" -> "file_name.txt"
    # but preserves cases like "file_.txt" -> "file_.txt"
    # We only remove the underscore if there are multiple underscores in the name
    if name.count("_") > 1 and re.search(r"_.\w+$", name):
        # Find the last underscore before the extension and remove it
        name = re.sub(r"_\.([^.]+)$", r".\1", name)

    # If the name starts with a dot after sanitization, prepend 'file'
    if name.startswith("."):
        name = "file" + name

    # Ensure name is not empty after sanitization
    if not name:
        return "file"

    # Truncate length if necessary, preserving extension
    max_len = MAX_FILENAME_LENGTH  # Maximum length for a safe filename part
    if len(name) > max_len:
        name_part, ext_part = os.path.splitext(name)
        # Ensure name_part is not empty after truncation
        name_part = name_part[: max_len - len(ext_part)].strip("._ ") or "part"
        name = name_part + ext_part

    return name


def get_filename_from_url(
    url: str, identifier: Any, content_type: str | None = None
) -> str:
    """Generates a sanitized filename from a URL, an identifier, and optionally a
    content type.
    Helps create unique and safe filenames for downloaded media.

    Args:
    ----
        url: The source URL (used for path hints).
        identifier: A unique identifier (e.g., product ASIN, item index, Pexels ID)
                    to ensure filename uniqueness.
        content_type: Optional MIME type (e.g., 'image/jpeg', 'video/mp4') to help
        guess extension.

    Returns:
    -------
        A sanitized filename string including the identifier.

    """
    try:
        parsed = urlparse(url)
        # Get potential base name from URL path, remove query/fragment
        base_name_from_path = os.path.basename(parsed.path)

        # Try to get extension from URL path or content type
        name_part, ext = os.path.splitext(base_name_from_path)
        if not ext and content_type:
            ext = mimetypes.guess_extension(content_type) or ""  # Use "" if guess fails
        elif not ext:
            ext = ""  # No extension found

        # Sanitize the name part from the URL
        safe_name_part = sanitize_filename(name_part)

        # Use identifier and maybe a type hint if the URL path name is too general
        # or empty
        # Also check if the original name_part was empty (which gets converted to
        # 'file')
        if not safe_name_part or safe_name_part in [
            "index",
            "default",
            "file",
        ]:  # Add common generic names
            # Use content type prefix for generic names, but "item" for
            # empty/invalid URLs
            if not name_part:  # Original name_part was empty (invalid URL)
                safe_name_part = f"item_{identifier}"
            else:
                type_hint = content_type.split("/")[0] if content_type else "item"
                safe_name_part = f"{type_hint}_{identifier}"
        else:
            # Prepend identifier for uniqueness
            safe_name_part = f"{identifier}_{safe_name_part}"

        # Combine sanitized name part and extension
        # Ensure extension starts with a dot if not empty
        full_ext = ext if ext.startswith(".") else (f".{ext}" if ext else "")
        filename = f"{safe_name_part}{full_ext}"

        # Final sanitization in case the combination created issues
        return sanitize_filename(filename)

    except Exception as e:
        logger.warning(f"Filename generation error for '{url}': {e}")
        # Fallback to a safe default name using just the identifier and type hint
        ext = (
            mimetypes.guess_extension(content_type) if content_type else None
        ) or ".dat"
        return sanitize_filename(f"item_{identifier}{ext}")


async def take_screenshot(page: Page, debug_dir: Path, name: str):
    """Takes a screenshot of the current Playwright page and saves it to a debug
    directory.

    Args:
    ----
        page: The Playwright Page object.
        debug_dir: The base directory for saving debug output.
        name: A descriptive name for the screenshot file (will be sanitized).

    """
    try:
        safe_name = sanitize_filename(name)
        screenshot_path = debug_dir / "screenshots" / f"{safe_name}.png"
        ensure_dirs_exist(screenshot_path)  # Ensure parent directory exists
        await page.screenshot(path=screenshot_path)
        logger.debug(f"Screenshot saved: {screenshot_path}")
    except Exception as e:
        logger.warning(f"Failed to take screenshot '{name}': {e}")


def remove_duplicates(paths: list[Path]) -> list[Path]:
    """Removes duplicate Path objects from a list while preserving order.

    Args:
    ----
        paths: A list of Path objects.

    Returns:
    -------
        A new list containing unique Path objects in their original order.

    """
    seen = set()
    unique_paths = []
    for path in paths:
        # Use the string representation for comparison in the set
        path_str = str(path)
        if path_str not in seen:
            seen.add(path_str)
            unique_paths.append(path)
    return unique_paths


# Export all the utility functions
__all__ = [
    "sanitize_script",
    "ensure_dirs_exist",
    "cleanup_temp_dirs",
    "download_file",
    "sanitize_filename",
    "get_filename_from_url",
    "take_screenshot",
    "remove_duplicates",
    "format_timestamp",
]
