"""Tests for src/scraper/amazon/utils.py.

Currently covers build_affiliate_url: tag application, ASIN extraction, and
the WARN-on-missing-tag behaviour that guards against silent affiliate-revenue
loss when the calling CLI hasn't loaded .env before scraper utilities run.
"""

import logging

import pytest

from src.scraper.amazon.utils import build_affiliate_url


class TestBuildAffiliateUrl:
    def test_canonical_dp_url_gets_tag(self):
        result = build_affiliate_url(
            "https://www.amazon.com/dp/B0ABCDEFGH", associate_tag="mytag-20"
        )
        assert result == "https://www.amazon.com/dp/B0ABCDEFGH?tag=mytag-20"

    def test_search_result_url_is_canonicalized(self):
        """Deep search URLs with /dp/<ASIN>/ embedded reduce to canonical form."""
        url = (
            "https://www.amazon.com/Product-Name-Long/dp/B0ABCDEFGH/"
            "ref=sr_1_1?dib=xxx&keywords=foo&qid=123&sr=8-1"
        )
        result = build_affiliate_url(url, associate_tag="mytag-20")
        assert result == "https://www.amazon.com/dp/B0ABCDEFGH?tag=mytag-20"

    def test_existing_tag_is_replaced(self):
        url = "https://www.amazon.com/dp/B0ABCDEFGH?tag=other-99"
        result = build_affiliate_url(url, associate_tag="mytag-20")
        assert result == "https://www.amazon.com/dp/B0ABCDEFGH?tag=mytag-20"

    def test_empty_url_returns_empty(self):
        assert build_affiliate_url("", associate_tag="mytag-20") == ""

    def test_missing_tag_returns_input_unchanged(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ):
        """No tag in env or config means the URL ships untagged."""
        monkeypatch.delenv("AMAZON_ASSOCIATE_TAG", raising=False)
        url = "https://www.amazon.com/dp/B0ABCDEFGH"
        with caplog.at_level(logging.WARNING, logger="src.scraper.amazon.utils"):
            result = build_affiliate_url(url, associate_tag="")
        assert result == url

    def test_missing_tag_logs_warning(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ):
        """Silent fallback is a bug magnet; require a WARN so it's grep-able."""
        monkeypatch.delenv("AMAZON_ASSOCIATE_TAG", raising=False)
        with caplog.at_level(logging.WARNING, logger="src.scraper.amazon.utils"):
            build_affiliate_url(
                "https://www.amazon.com/dp/B0ABCDEFGH", associate_tag=""
            )
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "no associate tag configured" in r.getMessage().lower() for r in warnings
        ), f"expected WARN about missing tag, got {[r.getMessage() for r in warnings]}"

    def test_tag_set_no_warning(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ):
        """No warning when the tag IS configured."""
        monkeypatch.setenv("AMAZON_ASSOCIATE_TAG", "mytag-20")
        with caplog.at_level(logging.WARNING, logger="src.scraper.amazon.utils"):
            build_affiliate_url("https://www.amazon.com/dp/B0ABCDEFGH")
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert (
            not warnings
        ), f"expected no warnings, got {[r.getMessage() for r in warnings]}"
