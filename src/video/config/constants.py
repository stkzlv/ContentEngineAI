# src/video/config/constants.py
"""Technical constants shared across video configuration modules.

This file contains ONLY technical/format constants that should NOT be user-configurable:
- File format patterns and specifications
- Mathematical conversion factors
- Regex patterns
- Encoding standards

User-configurable values (timeouts, limits, preferences) belong in YAML config files.
"""

# =============================================================================
# TTS VALIDATION LIMITS (API constraints, not user preferences)
# =============================================================================
TTS_SPEAKING_RATE_MIN = 0.25  # Google TTS API minimum
TTS_SPEAKING_RATE_MAX = 4.0  # Google TTS API maximum
TTS_PITCH_MIN = -20.0  # Google TTS API minimum
TTS_PITCH_MAX = 20.0  # Google TTS API maximum

# =============================================================================
# FONT TECHNICAL CONSTANTS (file format standards)
# =============================================================================
FONT_FILE_EXTENSIONS = [".ttf", ".otf"]
FONT_REGULAR_SUFFIXES = ["-regular", "-r"]
DEFAULT_FALLBACK_FONT = "Arial"  # System font guaranteed to exist
FALLBACK_FONT_ALTERNATIVES = ["Montserrat", "Rubik", "Poppins", "Gabarito"]

# SRT Subtitle Format Constants
SRT_TIME_SEPARATOR = " --> "
SRT_BLOCK_SEPARATOR = "\n\n"
SRT_MIN_BLOCK_LINES = 3
SRT_TIME_HOUR_SEPARATOR = ":"
SRT_TIME_MINUTE_SEPARATOR = ":"
SRT_TIME_SECOND_SEPARATOR = ","
SRT_HOURS_IN_SECONDS = 3600
SRT_MINUTES_IN_SECONDS = 60
SRT_MILLISECONDS_DIVISOR = 1000
SRT_LINE_IDENTIFIER = "-->"
SRT_ENCODING = "utf-8"

# Text Processing Patterns
TEXT_NORMALIZATION_PATTERN = r"[^\w\s]"
TEXT_WHITESPACE_PATTERN = r"\s+"
TEXT_WHITESPACE_REPLACEMENT = " "

# ASS Subtitle Format Constants
ASS_COLOR_PATTERN = r"&H(?:(\w{2}))?(\w{2})(\w{2})(\w{2})"
ASS_DEFAULT_ALPHA = "00"
RGB_HEX_FORMAT = "0x{red}{green}{blue}"
RGB_OPACITY_FORMAT = "{rgb_hex}@{opacity:.2f}"
FULL_OPACITY_THRESHOLD = 0.99

# =============================================================================
# WHISPER STT TECHNICAL DEFAULTS (reasonable defaults for subtitle readability)
# =============================================================================
DEFAULT_WHISPER_MODEL_DIR = "~/.cache/whisper_models"

# =============================================================================
# VIDEO ASSEMBLER TECHNICAL CONSTANTS
# =============================================================================
ASSEMBLER_IMAGE_LOOP = 1  # FFmpeg loop setting for static images
ASSEMBLER_PAD_COLOR = "black"  # Standard padding color

# =============================================================================
# FREESOUND API TECHNICAL CONSTANTS
# =============================================================================
FREESOUND_TOKEN_EXPIRY_SEC = 3600  # OAuth2 standard token lifetime
FREESOUND_TOKEN_REFRESH_BUFFER_SEC = 60  # Buffer before expiry to refresh
FREESOUND_DOWNLOAD_CHUNK_SIZE = 8192 * 4  # Network buffer size

# =============================================================================
# PLATFORM SAFE ZONE BOUNDARIES (cross-platform worst case on 1080x1920)
# See docs/platform-safe-zones.md for per-platform breakdown
# =============================================================================
SAFE_ZONE_MIN_X = 0.046  # Left: 50px on 1080w (all platforms similar)
SAFE_ZONE_MAX_X = 0.778  # Right: 840px on 1080w (TikTok engagement buttons)
SAFE_ZONE_MIN_Y = 0.104  # Top: 200px on 1920h (YouTube Shorts header)
SAFE_ZONE_MAX_Y = 0.75  # Bottom: 1440px on 1920h (TikTok overlay)

# =============================================================================
# SUBTITLE POSITIONING TECHNICAL CONSTANTS (coordinate system standards)
# =============================================================================
SUBTITLE_FALLBACK_SPACING_PERCENT = 0.02
SUBTITLE_MAX_SAFE_Y_POSITION = SAFE_ZONE_MAX_Y
SUBTITLE_CENTER_POSITION_FRACTION = 0.5
SUBTITLE_LEFT_POSITION_FRACTION = SAFE_ZONE_MIN_X
SUBTITLE_RIGHT_POSITION_FRACTION = SAFE_ZONE_MAX_X
SUBTITLE_BASE_FONT_SIZE_PERCENT = 0.04
SUBTITLE_MIN_FONT_SIZE = 16  # Minimum readable size
SUBTITLE_MAX_FONT_SIZE = 100  # Maximum practical size

# =============================================================================
# MATHEMATICAL CONSTANTS
# =============================================================================
DEFAULT_EXPONENTIAL_BACKOFF_BASE = 2  # Standard exponential backoff multiplier

# =============================================================================
# LATE.DEV API UPLOAD LIMITS (platform constraints, not user preferences)
# =============================================================================
# Maximum file size for direct upload (larger files require Vercel Blob token)
LATE_DIRECT_UPLOAD_MAX_BYTES = 4 * 1024 * 1024  # 4 MB
# Maximum file size Late.dev accepts for any upload
LATE_MAX_UPLOAD_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB
# Default retry-after delay when rate limit header is missing
LATE_DEFAULT_RETRY_AFTER_SEC = 60
# Maximum webhook events to retain for idempotency tracking
WEBHOOK_EVENT_HISTORY_LIMIT = 1000
# Minimum API key length for validation (security best practice)
LATE_API_KEY_MIN_LENGTH = 10

# =============================================================================
# SCHEDULE CONFLICT RESOLUTION (safety limits)
# =============================================================================
# Maximum attempts to find available slot before giving up
SCHEDULE_MAX_SLOT_SEARCH_ATTEMPTS = 100
# Multiplier for max attempts when finding alternatives (count * multiplier)
SCHEDULE_ALTERNATIVE_SEARCH_MULTIPLIER = 10
