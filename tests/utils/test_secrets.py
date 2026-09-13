"""Unit tests for secrets utility module."""

import os
from unittest.mock import patch

import pytest

from src.utils.secrets import (
    SECRET_KEY_PATTERNS,
    is_secret_key,
    mask_secret,
    mask_secrets_in_dict,
)


class TestMaskSecret:
    """Test secret value masking."""

    @pytest.mark.unit
    def test_mask_secret_basic(self):
        """Test basic secret masking shows first/last 4 chars."""
        result = mask_secret("sk-1234567890abcdef")
        assert result == "sk-1...cdef"

    @pytest.mark.unit
    def test_mask_secret_long_value(self):
        """Test masking of long secret values."""
        secret = "abcdefghijklmnopqrstuvwxyz123456"  # noqa: S105
        result = mask_secret(secret)
        assert result == "abcd...3456"
        assert len(result) < len(secret)

    @pytest.mark.unit
    def test_mask_secret_short_value(self):
        """Test masking of short values returns full mask."""
        result = mask_secret("short")
        assert result == "****"

    @pytest.mark.unit
    def test_mask_secret_exact_threshold(self):
        """Test value at exact threshold (9 chars = 4*2+1)."""
        result = mask_secret("123456789")
        assert result == "****"

    @pytest.mark.unit
    def test_mask_secret_above_threshold(self):
        """Test value just above threshold (10 chars)."""
        result = mask_secret("1234567890")
        assert result == "1234...7890"

    @pytest.mark.unit
    def test_mask_secret_none(self):
        """Test masking None returns full mask."""
        result = mask_secret(None)
        assert result == "****"

    @pytest.mark.unit
    def test_mask_secret_empty(self):
        """Test masking empty string returns full mask."""
        result = mask_secret("")
        assert result == "****"

    @pytest.mark.unit
    def test_mask_secret_custom_visible_chars(self):
        """Test custom visible character count."""
        result = mask_secret("abcdefghijklmnopqrst", visible_chars=6)
        assert result == "abcdef...opqrst"

    @pytest.mark.unit
    def test_mask_secret_preserves_prefix(self):
        """Test that common prefixes are preserved."""
        # OpenRouter key
        result = mask_secret("sk-or-v1-abcdefghijklmnop")
        assert result.startswith("sk-o")
        # Pexels key
        result = mask_secret("pexels_key_abcdefghijklmnop")
        assert result.startswith("pexe")


class TestIsSecretKey:
    """Test secret key name detection."""

    @pytest.mark.unit
    def test_is_secret_key_api_key(self):
        """Test detection of API_KEY patterns."""
        assert is_secret_key("OPENROUTER_API_KEY") is True
        assert is_secret_key("PEXELS_API_KEY") is True
        assert is_secret_key("api_key") is True
        assert is_secret_key("API-KEY") is True

    @pytest.mark.unit
    def test_is_secret_key_token(self):
        """Test detection of TOKEN patterns."""
        assert is_secret_key("AUTH_TOKEN") is True
        assert is_secret_key("ACCESS_TOKEN") is True
        assert is_secret_key("REFRESH_TOKEN") is True
        assert is_secret_key("BEARER_TOKEN") is True

    @pytest.mark.unit
    def test_is_secret_key_password(self):
        """Test detection of PASSWORD patterns."""
        assert is_secret_key("DATABASE_PASSWORD") is True
        assert is_secret_key("PASSWORD") is True
        assert is_secret_key("PASSWD") is True

    @pytest.mark.unit
    def test_is_secret_key_secret(self):
        """Test detection of SECRET patterns."""
        assert is_secret_key("CLIENT_SECRET") is True
        assert is_secret_key("SECRET_KEY") is True
        assert is_secret_key("APP_SECRET") is True

    @pytest.mark.unit
    def test_is_secret_key_credential(self):
        """Test detection of CREDENTIAL patterns."""
        assert is_secret_key("GOOGLE_APPLICATION_CREDENTIALS") is True
        assert is_secret_key("AWS_CREDENTIALS") is True

    @pytest.mark.unit
    def test_is_secret_key_suffix_patterns(self):
        """Test detection of suffix patterns (_SECRET, _TOKEN, _KEY)."""
        assert is_secret_key("MY_SECRET") is True
        assert is_secret_key("MY_TOKEN") is True
        assert is_secret_key("MY_KEY") is True

    @pytest.mark.unit
    def test_is_secret_key_non_secret(self):
        """Test that non-secret keys are not detected."""
        assert is_secret_key("DEBUG_MODE") is False
        assert is_secret_key("VIDEO_PROFILE") is False
        assert is_secret_key("OUTPUT_DIR") is False
        assert is_secret_key("MAX_RETRIES") is False

    @pytest.mark.unit
    def test_is_secret_key_case_insensitive(self):
        """Test case-insensitive detection."""
        assert is_secret_key("api_key") is True
        assert is_secret_key("API_KEY") is True
        assert is_secret_key("Api_Key") is True

    @pytest.mark.unit
    def test_is_secret_key_none(self):
        """Test None key name."""
        assert is_secret_key(None) is False

    @pytest.mark.unit
    def test_is_secret_key_empty(self):
        """Test empty key name."""
        assert is_secret_key("") is False


class TestMaskSecretsInDict:
    """Test dictionary secret masking."""

    @pytest.mark.unit
    def test_mask_secrets_in_dict_basic(self):
        """Test basic dictionary masking."""
        data = {
            "API_KEY": "sk-1234567890abcdef",
            "DEBUG": "true",
        }
        result = mask_secrets_in_dict(data)
        assert result["API_KEY"] == "sk-1...cdef"
        assert result["DEBUG"] == "true"

    @pytest.mark.unit
    def test_mask_secrets_in_dict_mixed(self):
        """Test masking with mixed secret and non-secret keys."""
        data = {
            "OPENROUTER_API_KEY": "sk-or-v1-abcdefgh",
            "PASSWORD": "supersecret123",
            "OUTPUT_DIR": "outputs",
            "MAX_RETRIES": "3",
        }
        result = mask_secrets_in_dict(data)
        assert (
            result["OPENROUTER_API_KEY"] == "sk-o...efgh"
        )  # Long enough for partial mask
        assert result["PASSWORD"] == "supe...t123"  # noqa: S105
        assert result["OUTPUT_DIR"] == "outputs"
        assert result["MAX_RETRIES"] == "3"

    @pytest.mark.unit
    def test_mask_secrets_in_dict_none_values(self):
        """Test handling of None values."""
        data = {
            "API_KEY": None,
            "DEBUG": "true",
        }
        result = mask_secrets_in_dict(data)
        assert result["API_KEY"] == "****"
        assert result["DEBUG"] == "true"

    @pytest.mark.unit
    def test_mask_secrets_in_dict_additional_keys(self):
        """Test masking with additional custom keys."""
        data = {
            "MY_CUSTOM_SECRET": "secret_value_here",
            "DEBUG": "true",
        }
        result = mask_secrets_in_dict(data, additional_keys={"MY_CUSTOM_SECRET"})
        assert result["MY_CUSTOM_SECRET"] == "secr...here"  # noqa: S105
        assert result["DEBUG"] == "true"

    @pytest.mark.unit
    def test_mask_secrets_in_dict_empty(self):
        """Test masking empty dictionary."""
        result = mask_secrets_in_dict({})
        assert result == {}


class TestSecretKeyPatterns:
    """Test SECRET_KEY_PATTERNS tuple."""

    @pytest.mark.unit
    def test_patterns_are_compiled(self):
        """Test that patterns are pre-compiled regex objects."""
        import re

        for pattern in SECRET_KEY_PATTERNS:
            assert hasattr(pattern, "search")
            assert hasattr(pattern, "match")

    @pytest.mark.unit
    def test_patterns_cover_common_secrets(self):
        """Test patterns cover common secret naming conventions."""
        common_secrets = [
            "API_KEY",
            "SECRET_KEY",
            "ACCESS_KEY",
            "PRIVATE_KEY",
            "AUTH_TOKEN",
            "ACCESS_TOKEN",
            "REFRESH_TOKEN",
            "BEARER_TOKEN",
            "PASSWORD",
            "PASSWD",
            "CREDENTIAL",
            "MY_SECRET",
            "MY_TOKEN",
            "MY_KEY",
        ]
        for secret_name in common_secrets:
            assert is_secret_key(secret_name), f"Failed to detect: {secret_name}"
