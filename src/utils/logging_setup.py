"""Centralized logging configuration for ContentEngineAI.

This module provides standardized logging setup to avoid duplication across
producer, scraper, and other components. Includes automatic secret masking
to prevent accidental credential exposure in logs.
"""

import logging
import re
import sys
from collections.abc import Mapping
from pathlib import Path

from .secrets import SECRET_KEY_PATTERNS, mask_secret

# Pre-compiled patterns for detecting secrets in log messages
# These match common secret formats (API keys, tokens, etc.)
_SECRET_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Generic API key patterns (sk-xxx, pk-xxx, api-xxx)
    re.compile(r"\b(sk|pk|api|key)[-_]?[a-zA-Z0-9]{16,}\b", re.IGNORECASE),
    # Bearer tokens
    re.compile(r"\bBearer\s+[a-zA-Z0-9\-_.]+\b", re.IGNORECASE),
    # Base64-like tokens (32+ chars)
    re.compile(r"\b[a-zA-Z0-9+/]{32,}={0,2}\b"),
    # UUID-like tokens
    re.compile(r"\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b"),
)


class SecretMaskingFilter(logging.Filter):
    """Logging filter that masks secrets in log records.

    This filter scans log messages for patterns that look like secrets
    (API keys, tokens, passwords) and masks them before output.

    The filter is thread-safe as it only modifies the current log record
    and uses immutable pattern matching.

    Examples
    --------
        >>> filter = SecretMaskingFilter()
        >>> # Applied automatically via setup_debug_logging()

    """

    __slots__ = ("_patterns", "_key_patterns")

    def __init__(self, name: str = "") -> None:
        """Initialize the secret masking filter.

        Parameters
        ----------
        name : str, optional
            Filter name (default: "")

        """
        super().__init__(name)
        self._patterns = _SECRET_VALUE_PATTERNS
        self._key_patterns = SECRET_KEY_PATTERNS

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter and mask secrets in the log record.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to process

        Returns
        -------
        bool
            Always True (record is never filtered out, only modified)

        """
        # Format the message first (msg % args), then mask the result.
        # This avoids destroying %-format specifiers like %s/%d in the
        # format string (e.g., "keyword: %s" contains "KEY" which would
        # be falsely matched as a secret key pattern).
        try:
            formatted_msg = record.getMessage()
            record.msg = self._mask_string(formatted_msg)
            record.args = None
        except (TypeError, ValueError):
            # If formatting fails, fall back to masking raw parts
            if record.msg and isinstance(record.msg, str):
                record.msg = self._mask_string(record.msg)
            if record.args:
                record.args = self._mask_args(record.args)

        return True

    def _mask_string(self, text: str) -> str:
        """Mask secret patterns in a string.

        Parameters
        ----------
        text : str
            Text to scan for secrets

        Returns
        -------
        str
            Text with secrets masked

        """
        if not text:
            return text

        result = text

        # Check for key=value patterns first (e.g., API_KEY=xxx)
        result = re.sub(
            r"(\b\w*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)\w*\s*[=:]\s*)(\S+)",
            lambda m: m.group(1) + mask_secret(m.group(2)),
            result,
            flags=re.IGNORECASE,
        )

        # Mask standalone secret-like values
        for pattern in self._patterns:
            result = pattern.sub(lambda m: mask_secret(m.group(0)), result)

        return result

    def _mask_args(
        self, args: tuple[object, ...] | Mapping[str, object]
    ) -> tuple[object, ...] | dict[str, object]:
        """Mask secrets in log record args.

        Parameters
        ----------
        args : tuple or dict
            Log record args (for % formatting)

        Returns
        -------
        tuple or dict
            Args with secrets masked

        """
        if isinstance(args, dict):
            return {
                k: self._mask_string(str(v)) if isinstance(v, str) else v
                for k, v in args.items()
            }
        # Default to tuple handling
        return tuple(
            self._mask_string(str(arg)) if isinstance(arg, str) else arg for arg in args
        )


def setup_debug_logging(
    log_file: Path,
    debug_mode: bool = False,
    verbose: bool = False,
    component_name: str = "ContentEngineAI",
) -> None:
    """Configure standardized logging with console and file handlers.

    Parameters
    ----------
    log_file : Path
        Path to the log file for persistent logging
    debug_mode : bool, optional
        Enable DEBUG level logging (default: False = INFO level)
    verbose : bool, optional
        Enable verbose console formatting (default: False)
    component_name : str, optional
        Name of the component for logging messages (default: "ContentEngineAI")

    Notes
    -----
    - Console output uses simplified format by default, detailed format when verbose
    - File output always uses detailed format with function names and line numbers
    - Log file is overwritten on each run (mode='w')
    - Third-party loggers (numba, httpx, google, etc.) are suppressed to WARNING

    """
    log_level = logging.DEBUG if debug_mode else logging.INFO

    # Clear any existing handlers to avoid duplication
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler configuration
    console_handler = logging.StreamHandler(sys.stdout)
    if verbose:
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    else:
        console_formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)

    # File handler configuration (overwrite mode)
    file_handler = logging.FileHandler(
        log_file,
        mode="w",  # Overwrite file on each run
        encoding="utf-8",
    )
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(log_level)

    # Create shared secret masking filter
    secret_filter = SecretMaskingFilter()

    # Apply filter to both handlers
    console_handler.addFilter(secret_filter)
    file_handler.addFilter(secret_filter)

    # Configure root logger
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("numba").setLevel(logging.WARNING)
    # Suppress websocket cleanup messages (harmless "goodbye" errors)
    logging.getLogger("websocket").setLevel(logging.CRITICAL)
    if not debug_mode:
        for lib in ["httpx", "google", "aiohttp", "urllib3", "asyncio", "hpack"]:
            logging.getLogger(lib).setLevel(logging.WARNING)
    else:
        # In debug mode, still suppress urllib3 DEBUG spam but allow INFO
        logging.getLogger("urllib3").setLevel(logging.INFO)
        # Keep websocket quiet even in debug mode (cleanup messages are noise)
        logging.getLogger("websocket").setLevel(logging.CRITICAL)

    # Log initialization message
    logger = logging.getLogger(component_name)
    logger.debug(
        f"Logging initialized: level={logging.getLevelName(log_level)}, "
        f"log_file={log_file}, verbose={verbose}"
    )
