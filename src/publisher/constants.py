"""Shared constants for the publisher module."""

from pathlib import Path

# Default outputs directory used across tracking, webhooks, cleanup, batch, etc.
DEFAULT_OUTPUTS_DIR = Path("outputs")

# Late SDK pagination page size
SDK_LIST_PAGE_SIZE = 100

# Maximum concurrent cleanup operations
MAX_CONCURRENT_CLEANUPS = 3
