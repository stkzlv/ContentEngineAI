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


class TestAffiliateLinksDisabled:
    """`affiliate_links.enabled: false` says "no program", not "misconfigured".

    Treating those two as the same thing meant an install with no affiliate
    account logged a revenue-loss warning on every single product.
    """

    def test_disabled_returns_clean_url_without_tag(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("AMAZON_ASSOCIATE_TAG", raising=False)
        monkeypatch.setattr(
            "src.scraper.amazon.utils._affiliate_links_enabled", lambda: False
        )
        result = build_affiliate_url(
            "https://www.amazon.com/gp/product/dp/B0ABCDEFGH?ref=sr_1&dib=xyz",
            associate_tag="",
        )
        assert result == "https://www.amazon.com/dp/B0ABCDEFGH"
        assert "tag=" not in result

    def test_disabled_does_not_warn(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ):
        """The warning exists to catch a mistake. This is not a mistake."""
        monkeypatch.delenv("AMAZON_ASSOCIATE_TAG", raising=False)
        monkeypatch.setattr(
            "src.scraper.amazon.utils._affiliate_links_enabled", lambda: False
        )
        with caplog.at_level(logging.DEBUG, logger="src.scraper.amazon.utils"):
            build_affiliate_url(
                "https://www.amazon.com/dp/B0ABCDEFGH", associate_tag=""
            )
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_disabled_still_honours_an_explicit_tag(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The flag governs the missing-tag path only.

        A caller passing a tag explicitly, or an env var being set, means a
        program does exist; the flag must not silently discard it.
        """
        monkeypatch.setattr(
            "src.scraper.amazon.utils._affiliate_links_enabled", lambda: False
        )
        result = build_affiliate_url(
            "https://www.amazon.com/dp/B0ABCDEFGH", associate_tag="mytag-20"
        )
        assert result == "https://www.amazon.com/dp/B0ABCDEFGH?tag=mytag-20"

    def test_disabled_passes_through_url_without_asin(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("AMAZON_ASSOCIATE_TAG", raising=False)
        monkeypatch.setattr(
            "src.scraper.amazon.utils._affiliate_links_enabled", lambda: False
        )
        url = "https://www.amazon.com/some/other/path"
        assert build_affiliate_url(url, associate_tag="") == url

    def test_enabled_by_default_when_config_absent(self):
        """Missing config must not silently disable the warning."""
        from src.scraper.amazon.utils import _affiliate_links_enabled

        assert _affiliate_links_enabled() is True


class TestAffiliateLinksEnvOverride:
    """`AMAZON_AFFILIATE_LINKS_ENABLED` mirrors how the tag prefers env."""

    @pytest.mark.parametrize("value", ["false", "FALSE", "0", "no", "off"])
    def test_falsey_values_disable(self, value: str, monkeypatch):
        from src.scraper.amazon.utils import _affiliate_links_enabled

        monkeypatch.setenv("AMAZON_AFFILIATE_LINKS_ENABLED", value)
        assert _affiliate_links_enabled() is False

    @pytest.mark.parametrize("value", ["true", "1", "yes"])
    def test_truthy_values_enable(self, value: str, monkeypatch):
        from src.scraper.amazon.utils import _affiliate_links_enabled

        monkeypatch.setenv("AMAZON_AFFILIATE_LINKS_ENABLED", value)
        assert _affiliate_links_enabled() is True

    def test_blank_env_falls_through_to_config(self, monkeypatch):
        """An empty var must not read as false, or `FOO=` would disable it."""
        from src.scraper.amazon.utils import _affiliate_links_enabled

        monkeypatch.setenv("AMAZON_AFFILIATE_LINKS_ENABLED", "  ")
        assert _affiliate_links_enabled() is True

    def test_env_disables_the_tag_warning_end_to_end(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("AMAZON_ASSOCIATE_TAG", raising=False)
        monkeypatch.setenv("AMAZON_AFFILIATE_LINKS_ENABLED", "false")
        with caplog.at_level(logging.DEBUG, logger="src.scraper.amazon.utils"):
            result = build_affiliate_url(
                "https://www.amazon.com/gp/product/dp/B0ABCDEFGH?ref=sr_1",
                associate_tag="",
            )
        assert result == "https://www.amazon.com/dp/B0ABCDEFGH"
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
