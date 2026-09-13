"""Shared constants for the publisher module."""

from pathlib import Path

from src.utils.outputs_paths import get_project_root

# Default outputs directory used across tracking, webhooks, cleanup, batch, etc.
# Anchored on the project root rather than the working directory: a bare
# Path("outputs") planted stray trees wherever a command happened to run
# from, and the unanchored gitignore hid every one of them.
DEFAULT_OUTPUTS_DIR = get_project_root() / "outputs"

# Late SDK pagination page size
SDK_LIST_PAGE_SIZE = 100

# Maximum concurrent cleanup operations
MAX_CONCURRENT_CLEANUPS = 3
