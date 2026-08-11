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
        # Pin the flag rather than inheriting whatever the bundled YAML ships:
        # this test is about the warning, and shouldn't start failing because
        # someone flipped affiliate_links.enabled in config/scraper.yaml.
        monkeypatch.setattr(
            "src.scraper.amazon.utils._affiliate_links_enabled", lambda: True
        )
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


class TestAffiliateLinksConfigRead:
    """Cover the YAML read itself, not just a patched-out helper.

    Every other test here either replaces `_affiliate_links_enabled` wholesale
    or drives the env var, which leaves the config lookup with no coverage at
    all. Because the helper falls back to True on any failure, a wrong key path
    is indistinguishable from "enabled": misspelling `affiliate_links` in the
    source kept the whole suite green. These tests pin the exact key path, so
    that mutation fails.
    """

    @staticmethod
    def _config(monkeypatch: pytest.MonkeyPatch, payload: dict):
        monkeypatch.delenv("AMAZON_AFFILIATE_LINKS_ENABLED", raising=False)
        monkeypatch.setattr("src.scraper.amazon.config.CONFIG", payload)

    def test_reads_enabled_false_from_config(self, monkeypatch: pytest.MonkeyPatch):
        from src.scraper.amazon.utils import _affiliate_links_enabled

        self._config(
            monkeypatch,
            {"scrapers": {"amazon": {"affiliate_links": {"enabled": False}}}},
        )
        assert _affiliate_links_enabled() is False

    def test_reads_enabled_true_from_config(self, monkeypatch: pytest.MonkeyPatch):
        from src.scraper.amazon.utils import _affiliate_links_enabled

        self._config(
            monkeypatch,
            {"scrapers": {"amazon": {"affiliate_links": {"enabled": True}}}},
        )
        assert _affiliate_links_enabled() is True

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"scrapers": {}},
            {"scrapers": {"amazon": {}}},
            {"scrapers": {"amazon": {"affiliate_links": {}}}},
        ],
        ids=["empty", "no-amazon", "no-block", "empty-block"],
    )
    def test_missing_config_defaults_to_enabled(
        self, payload: dict, monkeypatch: pytest.MonkeyPatch
    ):
        """A missing or unreadable config must not silence the warning.

        Defaulting the other way would turn a broken config into a silent
        revenue loss, which is the exact failure the warning exists to catch.
        """
        from src.scraper.amazon.utils import _affiliate_links_enabled

        self._config(monkeypatch, payload)
        assert _affiliate_links_enabled() is True

    def test_config_read_failure_defaults_to_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        class Exploding(dict):
            def get(self, *a, **kw):
                raise RuntimeError("config blew up")

        self._config(monkeypatch, Exploding())
        from src.scraper.amazon.utils import _affiliate_links_enabled

        assert _affiliate_links_enabled() is True

    def test_disabled_config_reaches_build_affiliate_url(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ):
        """End to end through the real helper: config -> clean untagged URL."""
        monkeypatch.delenv("AMAZON_ASSOCIATE_TAG", raising=False)
        self._config(
            monkeypatch,
            {"scrapers": {"amazon": {"affiliate_links": {"enabled": False}}}},
        )
        with caplog.at_level(logging.DEBUG, logger="src.scraper.amazon.utils"):
            result = build_affiliate_url(
                "https://www.amazon.com/gp/product/dp/B0ABCDEFGH?ref=sr_1",
                associate_tag="",
            )
        assert result == "https://www.amazon.com/dp/B0ABCDEFGH"
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


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
        """An empty var must not read as false, or `FOO=` would disable it.

        Asserted against a config pinned to False, so a blank env var falling
        through is what the result proves. Asserting True against the bundled
        YAML would pass even if the blank value short-circuited to enabled.
        """
        from src.scraper.amazon.utils import _affiliate_links_enabled

        monkeypatch.setenv("AMAZON_AFFILIATE_LINKS_ENABLED", "  ")
        monkeypatch.setattr(
            "src.scraper.amazon.config.CONFIG",
            {"scrapers": {"amazon": {"affiliate_links": {"enabled": False}}}},
        )
        assert _affiliate_links_enabled() is False

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


class TestCleanProductUrl:
    """`_clean_product_url` must preserve the domain it was given."""

    def test_preserves_non_com_amazon_domain(self):
        from src.scraper.amazon.utils import _clean_product_url

        assert (
            _clean_product_url("https://www.amazon.co.uk/dp/B0ABCDEFGH?tag=x-21")
            == "https://www.amazon.co.uk/dp/B0ABCDEFGH"
        )

    def test_preserves_scheme(self):
        from src.scraper.amazon.utils import _clean_product_url

        assert (
            _clean_product_url("http://www.amazon.de/Some-Title/dp/B0ABCDEFGH/ref=x")
            == "http://www.amazon.de/dp/B0ABCDEFGH"
        )

    def test_returns_input_when_no_asin(self):
        from src.scraper.amazon.utils import _clean_product_url

        url = "https://www.amazon.com/gp/bestsellers?ref=nav"
        assert _clean_product_url(url) == url
