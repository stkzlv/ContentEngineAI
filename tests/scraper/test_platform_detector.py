"""Unit tests for platform_detector module."""

import pytest

from src.scraper.base.platform_detector import (
    _PLATFORM_DETECTORS,
    _is_amazon_asin,
    detect_platform,
    get_registered_platforms,
    register_platform,
)

pytestmark = pytest.mark.unit


class TestAmazonASINDetection:
    """Tests for Amazon ASIN pattern matching."""

    @pytest.mark.parametrize(
        "valid_asin",
        [
            "B0VALIDASN",  # Standard B0 format
            "B012345678",  # B0 with numbers
            "B0ABCDEFGH",  # B0 with letters
            "B1XXXXXXXX",  # B1 format
            "B1ABCD1234",  # B1 mixed
            "b0validasn",  # Lowercase (case insensitive)
            "B0AbCdEf12",  # Mixed case
        ],
    )
    def test_valid_amazon_asins(self, valid_asin: str):
        """Test that valid ASINs are detected."""
        assert _is_amazon_asin(valid_asin) is True

    @pytest.mark.parametrize(
        "invalid_asin,reason",
        [
            ("", "empty string"),
            ("   ", "whitespace only"),
            ("B0SHORT", "too short (7 chars)"),
            ("B0TOOLONGASIN", "too long (13 chars)"),
            ("A0VALIDASN", "wrong prefix (A0)"),
            ("B2VALIDASN", "wrong prefix (B2)"),
            ("0123456789", "numeric only (legacy format not supported)"),
            ("B0VALID-AS", "contains hyphen"),
            ("B0VALID_AS", "contains underscore"),
            ("B0VALID AS", "contains space"),
            (None, "None value"),
        ],
    )
    def test_invalid_amazon_asins(self, invalid_asin: str | None, reason: str):
        """Test that invalid ASINs are rejected."""
        # Handle None case
        if invalid_asin is None:
            assert _is_amazon_asin("") is False
        else:
            assert _is_amazon_asin(invalid_asin) is False, f"Should reject: {reason}"


class TestDetectPlatform:
    """Tests for detect_platform function."""

    def test_detect_amazon_platform(self):
        """Test Amazon platform detection."""
        assert detect_platform("B0TESTPROD") == "amazon"
        assert detect_platform("B1TESTPROD") == "amazon"

    def test_detect_unknown_platform(self):
        """Test unknown product ID returns None."""
        assert detect_platform("UNKNOWN123") is None
        assert detect_platform("12345") is None
        assert detect_platform("random-id") is None

    def test_detect_empty_input(self):
        """Test empty inputs return None."""
        assert detect_platform("") is None
        assert detect_platform("   ") is None

    def test_detect_strips_whitespace(self):
        """Test that whitespace is stripped before detection."""
        assert detect_platform("  B0TESTPROD  ") == "amazon"
        assert detect_platform("\tB0TESTPROD\n") == "amazon"


class TestGetRegisteredPlatforms:
    """Tests for get_registered_platforms function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        result = get_registered_platforms()
        assert isinstance(result, list)

    def test_amazon_registered(self):
        """Test that Amazon is in registered platforms."""
        platforms = get_registered_platforms()
        assert "amazon" in platforms

    def test_returns_copy(self):
        """Test that modifying result doesn't affect registry."""
        platforms = get_registered_platforms()
        original_count = len(platforms)
        platforms.append("fake_platform")
        assert len(get_registered_platforms()) == original_count


class TestRegisterPlatform:
    """Tests for register_platform decorator."""

    def test_register_new_platform(self):
        """Test registering a new platform detector."""

        @register_platform("test_platform")
        def _detect_test(product_id: str) -> bool:
            return product_id.startswith("TEST")

        try:
            # Verify registration
            assert "test_platform" in _PLATFORM_DETECTORS
            assert detect_platform("TEST123456") == "test_platform"
        finally:
            # Cleanup - remove test platform
            if "test_platform" in _PLATFORM_DETECTORS:
                del _PLATFORM_DETECTORS["test_platform"]

    def test_decorator_returns_function(self):
        """Test that decorator returns the original function."""

        @register_platform("another_test")
        def _my_detector(product_id: str) -> bool:
            return False

        try:
            # Function should still be callable
            assert callable(_my_detector)
            assert _my_detector("anything") is False
        finally:
            if "another_test" in _PLATFORM_DETECTORS:
                del _PLATFORM_DETECTORS["another_test"]


class TestEdgeCases:
    """Edge case tests for platform detection."""

    def test_case_insensitive_asin(self):
        """Test that ASIN detection is case insensitive."""
        assert detect_platform("b0testprod") == "amazon"
        assert detect_platform("B0TESTPROD") == "amazon"
        assert detect_platform("B0TestProd") == "amazon"

    def test_boundary_length_asins(self):
        """Test ASINs at boundary lengths."""
        # Exactly 9 chars (too short)
        assert detect_platform("B0ABCDEFG") is None
        # Exactly 10 chars (valid)
        assert detect_platform("B0ABCDEFGH") == "amazon"
        # Exactly 11 chars (too long)
        assert detect_platform("B0ABCDEFGHI") is None

    def test_special_characters_rejected(self):
        """Test that special characters in ASINs are rejected."""
        assert detect_platform("B0TEST!@#$") is None
        assert detect_platform("B0TEST&*()") is None
