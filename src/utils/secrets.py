# src/utils/secrets.py
"""Secrets utility module for ContentEngineAI.

Provides functions for masking secrets in logs and detecting secret-like
environment variable names to prevent accidental credential exposure.
"""

import re

# Patterns that indicate a key name contains a secret
SECRET_KEY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"API[_-]?KEY", re.IGNORECASE),
    re.compile(r"SECRET[_-]?KEY", re.IGNORECASE),
    re.compile(r"ACCESS[_-]?KEY", re.IGNORECASE),
    re.compile(r"PRIVATE[_-]?KEY", re.IGNORECASE),
    re.compile(r"AUTH[_-]?TOKEN", re.IGNORECASE),
    re.compile(r"ACCESS[_-]?TOKEN", re.IGNORECASE),
    re.compile(r"REFRESH[_-]?TOKEN", re.IGNORECASE),
    re.compile(r"BEARER[_-]?TOKEN", re.IGNORECASE),
    re.compile(r"PASSWORD", re.IGNORECASE),
    re.compile(r"PASSWD", re.IGNORECASE),
    re.compile(r"CREDENTIAL", re.IGNORECASE),
    re.compile(r"_SECRET$", re.IGNORECASE),
    re.compile(r"_TOKEN$", re.IGNORECASE),
    re.compile(r"_KEY$", re.IGNORECASE),
)


def mask_secret(value: str | None, visible_chars: int = 4) -> str:
    """Mask a secret value, showing only first and last N characters.

    Args:
    ----
        value: The secret value to mask. Can be None or empty.
        visible_chars: Number of characters to show at start and end.
            Defaults to 4.

    Returns:
    -------
        Masked string with format "xxxx...xxxx" or "****" for short/empty values.

    Examples:
    --------
        >>> mask_secret("sk-1234567890abcdef")
        'sk-1...cdef'
        >>> mask_secret("short")
        '****'
        >>> mask_secret(None)
        '****'
        >>> mask_secret("")
        '****'

    """
    if not value:
        return "****"

    # For short strings, mask entirely
    min_length = visible_chars * 2 + 1
    if len(value) <= min_length:
        return "****"

    return f"{value[:visible_chars]}...{value[-visible_chars:]}"


def is_secret_key(key_name: str | None) -> bool:
    """Check if a key name indicates it contains a secret value.

    Detects common patterns for API keys, tokens, passwords, and credentials.

    Args:
    ----
        key_name: The environment variable or config key name to check.
            Can be None or empty.

    Returns:
    -------
        True if the key name matches a secret pattern, False otherwise.

    Examples:
    --------
        >>> is_secret_key("OPENROUTER_API_KEY")
        True
        >>> is_secret_key("LATE_API_KEY")
        True
        >>> is_secret_key("AUTH_TOKEN")
        True
        >>> is_secret_key("DATABASE_PASSWORD")
        True
        >>> is_secret_key("DEBUG_MODE")
        False
        >>> is_secret_key("VIDEO_PROFILE")
        False
        >>> is_secret_key(None)
        False

    """
    if not key_name:
        return False

    return any(pattern.search(key_name) for pattern in SECRET_KEY_PATTERNS)


def mask_secrets_in_dict(
    data: dict[str, str | None],
    additional_keys: set[str] | None = None,
) -> dict[str, str]:
    """Mask all secret values in a dictionary based on key names.

    Args:
    ----
        data: Dictionary with string keys and string/None values.
        additional_keys: Optional set of additional key names to treat as secrets,
            regardless of pattern matching.

    Returns:
    -------
        New dictionary with secret values masked.

    Examples:
    --------
        >>> mask_secrets_in_dict({"API_KEY": "secret123", "DEBUG": "true"})
        {'API_KEY': '****', 'DEBUG': 'true'}

    """
    additional = additional_keys or set()
    result: dict[str, str] = {}

    for key, value in data.items():
        if is_secret_key(key) or key in additional:
            result[key] = mask_secret(value)
        else:
            result[key] = value if value is not None else ""

    return result


__all__ = [
    "mask_secret",
    "is_secret_key",
    "mask_secrets_in_dict",
    "SECRET_KEY_PATTERNS",
]
