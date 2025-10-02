"""Centralized logging configuration for ContentEngineAI.

This module provides standardized logging setup to avoid duplication across
producer, scraper, and other components.
"""

import logging
import sys
from pathlib import Path


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
