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
# Freesound grants 24-hour access tokens; used only when a token response
# does not state its own lifetime.
FREESOUND_TOKEN_EXPIRY_SEC = 86400
FREESOUND_TOKEN_REFRESH_BUFFER_SEC = 60  # Buffer before expiry to refresh
FREESOUND_DOWNLOAD_CHUNK_SIZE = 8192 * 4  # Network buffer size

# =============================================================================
# PLATFORM SAFE ZONE BOUNDARIES (2026 cross-platform union on 1080x1920)
# Worst-case of TikTok, YouTube Shorts, and Instagram Reels. Instagram drives
# both top and bottom after Meta's March 2026 Reels unification (14% top, 35%
# bottom interactive zone). See docs/platform-safe-zones.md for the breakdown.
# =============================================================================
SAFE_ZONE_MIN_X = 0.056  # Left: 60px on 1080w (all platforms ~60px)
SAFE_ZONE_MAX_X = 0.833  # Right: 900px on 1080w (TikTok buttons + Jan 2026 playlist)
SAFE_ZONE_MIN_Y = 0.141  # Top: 270px on 1920h (Instagram Reels 14% header)
SAFE_ZONE_MAX_Y = 0.651  # Bottom: 1250px on 1920h (Instagram Reels 35% zone)

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

# The publisher's API limits, webhook and scheduling constants live in
# `src/publisher/constants.py`, and the LLM settings with the stock-phrase
# bound in `src/ai/llm_settings.py`. Nothing outside the video pipeline reads
# this module any more: a retry delay for the publishing provider is not a
# video setting, and editing one here dragged the whole video config package
# into the importer.
