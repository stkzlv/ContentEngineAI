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

# =============================================================================
# PROVIDER API UPLOAD LIMITS (platform constraints, not user preferences)
# =============================================================================
# Maximum file size for direct upload (larger files require Vercel Blob token)
LATE_DIRECT_UPLOAD_MAX_BYTES = 4 * 1024 * 1024  # 4 MB
# Maximum file size the provider accepts for any upload
LATE_MAX_UPLOAD_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB
# Default retry-after delay when the rate-limit header is missing
LATE_DEFAULT_RETRY_AFTER_SEC = 60
# Minimum API key length for validation (security best practice)
LATE_API_KEY_MIN_LENGTH = 10
# Standard exponential backoff multiplier for the provider's retries
DEFAULT_EXPONENTIAL_BACKOFF_BASE = 2

# Maximum webhook events to retain for idempotency tracking
WEBHOOK_EVENT_HISTORY_LIMIT = 1000

# =============================================================================
# SCHEDULE CONFLICT RESOLUTION (safety limits)
# =============================================================================
# Maximum attempts to find an available slot before giving up
SCHEDULE_MAX_SLOT_SEARCH_ATTEMPTS = 100
# Multiplier for max attempts when finding alternatives (count * multiplier)
SCHEDULE_ALTERNATIVE_SEARCH_MULTIPLIER = 10
