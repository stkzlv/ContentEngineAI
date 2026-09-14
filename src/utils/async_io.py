"""Async I/O optimization utilities for non-blocking file operations."""

import asyncio
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


async def async_run_ffmpeg(
    cmd: list[str],
    timeout_sec: float = 300.0,
    log_path: Path | None = None,
) -> tuple[bool, str, str]:
    """Run FFmpeg command asynchronously with timeout and logging.

    Args:
    ----
        cmd: FFmpeg command as list of strings
        timeout_sec: Timeout in seconds
        log_path: Optional path to save command log

    Returns:
    -------
        Tuple of (success, stdout, stderr)

    """
    try:
        if log_path:
            log_path.write_text(
                " ".join(f"'{part}'" if " " in part else part for part in cmd)
            )

        # Use asyncio.create_subprocess_exec for true async execution
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Wait for completion with timeout
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout_sec,
        )

        success = process.returncode == 0
        stdout_str = stdout.decode(errors="ignore") if stdout else ""
        stderr_str = stderr.decode(errors="ignore") if stderr else ""

        if not success:
            logger.error("FFmpeg failed with return code %s", process.returncode)
            if stderr_str:
                logger.error("FFmpeg stderr: %s", stderr_str)

        return success, stdout_str, stderr_str

    except TimeoutError:
        logger.error("FFmpeg process timed out after %s seconds", timeout_sec)
        if "process" in locals():
            try:
                process.kill()
                await process.wait()
            except Exception as e:
                logger.debug("Error terminating process: %s", e)
        return False, "", "Process timed out"

    except Exception as e:
        logger.exception("Error running FFmpeg command: %s", e)
        return False, "", str(e)


async def async_probe_media(
    file_path: Path,
    ffprobe_path: str = "ffprobe",
    timeout_sec: float = 30.0,
) -> dict[str, Any] | None:
    """Probe media file asynchronously to get metadata.

    Args:
    ----
        file_path: Path to media file
        ffprobe_path: Path to ffprobe binary
        timeout_sec: Timeout in seconds

    Returns:
    -------
        Media metadata dict or None on error

    """
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration,size:stream=codec_type,width,height",
        "-of",
        "json",
        str(file_path),
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout_sec,
        )

        if process.returncode != 0:
            logger.warning("ffprobe failed for %s: %s", file_path.name, stderr.decode())
            return None

        import json

        result: dict[str, Any] = json.loads(stdout.decode())
        return result

    except TimeoutError:
        logger.error("ffprobe timed out for %s", file_path.name)
        if "process" in locals():
            try:
                process.kill()
                await process.wait()
            except Exception as e:
                logger.debug("Error terminating process: %s", e)
        return None

    except Exception as e:
        logger.error("Error probing %s: %s", file_path.name, e)
        return None


async def async_get_media_duration(
    file_path: Path,
    ffprobe_path: str = "ffprobe",
    timeout_sec: float = 30.0,
) -> float:
    """Get media duration asynchronously.

    Args:
    ----
        file_path: Path to media file
        ffprobe_path: Path to ffprobe binary
        timeout_sec: Timeout in seconds

    Returns:
    -------
        Duration in seconds, 0.0 on error

    """
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout_sec,
        )

        if process.returncode != 0:
            logger.warning("ffprobe failed for %s: %s", file_path.name, stderr.decode())
            return 0.0

        return float(stdout.decode().strip())

    except TimeoutError:
        logger.error("ffprobe timed out for %s", file_path.name)
        if "process" in locals():
            try:
                process.kill()
                await process.wait()
            except Exception as e:
                logger.debug("Error terminating process: %s", e)
        return 0.0

    except Exception as e:
        logger.error("Error getting duration for %s: %s", file_path.name, e)
        return 0.0


class AsyncSemaphore:
    """Manages concurrent async operations with rate limiting."""

    def __init__(self, max_concurrent: int = 4):
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_limit(self, coro):
        """Run coroutine with concurrency limit."""
        async with self.semaphore:
            return await coro


# Global semaphore instances for different operation types
# Performance: Increased from 2→4 to reduce bottleneck (v0.12.0)
ffmpeg_semaphore = AsyncSemaphore(max_concurrent=4)  # FFmpeg is CPU intensive
io_semaphore = AsyncSemaphore(max_concurrent=8)  # I/O operations
network_semaphore = AsyncSemaphore(max_concurrent=4)  # Network operations
