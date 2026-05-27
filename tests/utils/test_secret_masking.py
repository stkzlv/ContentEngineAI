"""Tests for the secret-masking filter (#127)."""

from __future__ import annotations

import logging

from src.utils.logging_setup import SecretMaskingFilter


def _mask(text: str) -> str:
    """Run the masking filter on a log record and return the masked message."""
    filt = SecretMaskingFilter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg=text,
        args=None,
        exc_info=None,
    )
    filt.filter(record)
    return record.msg


class TestSecretMaskingNoFalsePositives:
    def test_product_keyword_passes_through(self) -> None:
        assert _mask("keywords: wireless earbuds") == "keywords: wireless earbuds"

    def test_lowercase_key_passes_through(self) -> None:
        assert _mask("keyword: smart plug") == "keyword: smart plug"

    def test_word_containing_key_passes_through(self) -> None:
        assert _mask("monkey wrench") == "monkey wrench"

    def test_word_containing_token_passes_through(self) -> None:
        assert _mask("tokenizer loaded") == "tokenizer loaded"

    def test_word_containing_secret_passes_through(self) -> None:
        assert _mask("secretion rate: 0.5") == "secretion rate: 0.5"

    def test_word_containing_password_passes_through(self) -> None:
        assert _mask("passwords: disabled") == "passwords: disabled"


class TestSecretMaskingRealSecrets:
    def test_api_key_value_masked(self) -> None:
        result = _mask("API_KEY=sk-1234567890abcdef")
        assert "sk-1234567890abcdef" not in result
        assert "API_KEY=" in result

    def test_late_api_key_masked(self) -> None:
        result = _mask("LATE_API_KEY: abc123defgh456789")
        assert "abc123defgh456789" not in result
        assert "LATE_API_KEY:" in result

    def test_auth_token_masked(self) -> None:
        result = _mask("AUTH_TOKEN=eyJhbGciOiJIUzI1NiJ9")
        assert "eyJhbGciOiJIUzI1NiJ9" not in result
        assert "AUTH_TOKEN=" in result

    def test_openrouter_api_key_masked(self) -> None:
        result = _mask("OPENROUTER_API_KEY=sk-or-v1-abc123xyz789012")
        assert "sk-or-v1-abc123xyz789012" not in result
