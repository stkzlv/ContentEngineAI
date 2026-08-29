"""The URL-shortener config had a typed model that nothing populated or read.

`URLShortenerSettings` declared the fields and hung off the root config, while
the only consumer -- the scraper's affiliate-link step -- opened
`config/url_shortener.yaml` itself and walked the dict with its own defaults.
Two consequences: the two default sets drifted (the model still said `picsee`
after the file was flipped to `bare`), and a typo'd key fell through to a
default instead of being reported.

The model was also a paraphrase rather than a mirror: `api_timeout_sec` for the
file's `api.timeout_sec`, `picsee_custom_domain` for `picsee.custom_domain`,
and no field at all for `api_base_url` or `bulk_timeout_multiplier`, both of
which the consumer read. It could not have been populated from the file as it
stood.
"""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from src.utils.url_shortener import (
    URLShortenerSettings,
    load_url_shortener_settings,
)


def write(tmp_path, block: dict) -> str:
    path = tmp_path / "url_shortener.yaml"
    path.write_text(yaml.safe_dump({"url_shortener": block}), encoding="utf-8")
    return str(path)


class TestTheModelMirrorsTheFile:
    def test_the_nested_blocks_load(self, tmp_path):
        settings = load_url_shortener_settings(
            write(
                tmp_path,
                {
                    "enabled": True,
                    "provider": "picsee",
                    "api": {"timeout_sec": 45, "max_retries": 5},
                    "picsee": {
                        "api_key_env_var": "PICSEE_API_KEY",
                        "api_base_url": "https://api.pics.ee",
                        "custom_domain": "example.test",
                        "max_bulk_size": 50,
                    },
                    "integration": {"shorten_on_scrape": False},
                },
            )
        )

        assert settings.api.timeout_sec == 45
        assert settings.api.max_retries == 5
        assert settings.picsee.custom_domain == "example.test"
        assert settings.picsee.max_bulk_size == 50
        assert settings.integration.shorten_on_scrape is False

    def test_a_typo_is_reported_rather_than_swallowed(self, tmp_path):
        """The dict walk fell through to a magic-number default instead."""
        with pytest.raises(ValidationError):
            load_url_shortener_settings(write(tmp_path, {"api": {"timeout_secs": 45}}))

    def test_a_typo_at_the_top_level_is_reported(self, tmp_path):
        with pytest.raises(ValidationError):
            load_url_shortener_settings(write(tmp_path, {"enabld": True}))

    def test_an_undeclared_provider_is_refused(self, tmp_path):
        """It would otherwise resolve to an empty block.

        A run would then report shortening as enabled and shorten nothing,
        which is the silent shape this change exists to remove.
        """
        with pytest.raises(ValidationError, match="Unknown url_shortener.provider"):
            load_url_shortener_settings(write(tmp_path, {"provider": "bitly"}))

    def test_a_missing_file_loads_the_defaults(self, tmp_path):
        """A fork without the file gets the no-op provider, not a crash."""
        settings = load_url_shortener_settings(tmp_path / "absent.yaml")

        assert settings.provider == "bare"

    def test_an_empty_section_loads_the_defaults(self, tmp_path):
        path = tmp_path / "url_shortener.yaml"
        path.write_text("url_shortener:\n", encoding="utf-8")

        assert load_url_shortener_settings(path).provider == "bare"


class TestTheDefaultsAgree:
    """The drift this replaces: two default sets, independently maintained."""

    def test_the_model_default_matches_the_shipped_file(self):
        shipped = load_url_shortener_settings()

        assert shipped.provider == URLShortenerSettings().provider, (
            "the model default and the bundled file disagree about the "
            "provider; that is the drift the typed load exists to prevent"
        )

    def test_the_bundled_file_loads_at_all(self):
        """Strict now, so a stale key in the shipped file fails the suite."""
        settings = load_url_shortener_settings()

        assert settings.provider in URLShortenerSettings.provider_names()


class TestTheScraperReadsTheTypedObject:
    def test_the_step_no_longer_opens_the_file_itself(self):
        """A second reader of the same file is what let the defaults drift.

        Reads the method's own source rather than the module's, so a comment
        naming the file elsewhere does not satisfy or fail this.
        """
        import inspect

        from src.scraper.amazon.scraper import BotasaurusAmazonScraper

        source = inspect.getsource(BotasaurusAmazonScraper._shorten_affiliate_links)
        # The docstring explains why the read is gone, so strip it before
        # asserting the read itself is absent.
        body = source.split('"""', 2)[-1]

        assert "yaml" not in body
        assert "open(" not in body

    def test_the_configured_provider_reaches_the_shortener(self, tmp_path, monkeypatch):
        """Driven through the real method, not through the loader.

        The loader having tests proves the file parses. What was missing was a
        path from the parsed object to the code that shortens.
        """
        from src.scraper.amazon.models import ProductData
        from src.scraper.base.models import Platform

        created: dict = {}

        def fake_create(**kwargs):
            created.update(kwargs)

            class _Shortener:
                async def shorten(self, url):
                    raise RuntimeError("not reached")

            return _Shortener()

        import src.utils.url_shortener as shortener_mod

        monkeypatch.setattr(shortener_mod, "create_url_shortener", fake_create)
        monkeypatch.setenv("PICSEE_API_KEY", "test-key")

        from src.scraper.amazon.scraper import BotasaurusAmazonScraper

        scraper = BotasaurusAmazonScraper.__new__(BotasaurusAmazonScraper)
        scraper.debug_mode = False
        scraper.logger = __import__("logging").getLogger("test")
        scraper.url_shortener_settings = load_url_shortener_settings(
            write(
                tmp_path,
                {
                    "enabled": True,
                    "provider": "picsee",
                    "api": {"timeout_sec": 45, "max_retries": 7},
                    "picsee": {
                        "api_key_env_var": "PICSEE_API_KEY",
                        "custom_domain": "example.test",
                        "max_bulk_size": 25,
                    },
                    "integration": {"shorten_on_scrape": True},
                },
            )
        )

        product = ProductData(
            title="A product",
            price="1.00",
            url="https://www.amazon.com/dp/B0TEST1234",
            platform=Platform.AMAZON,
            asin="B0TEST1234",
            affiliate_link="https://www.amazon.com/dp/B0TEST1234",
        )
        scraper._shorten_affiliate_links([product])

        assert created["provider"] == "picsee"
        assert created["timeout"] == 45
        assert created["max_retries"] == 7
        assert created["custom_domain"] == "example.test"
        assert created["max_bulk_size"] == 25

    def test_shortening_is_skipped_when_the_file_disables_it(
        self, tmp_path, monkeypatch
    ):
        from src.scraper.amazon.scraper import BotasaurusAmazonScraper

        called = {"n": 0}

        def fake_create(**kwargs):
            called["n"] += 1

        import src.utils.url_shortener as shortener_mod

        monkeypatch.setattr(shortener_mod, "create_url_shortener", fake_create)

        scraper = BotasaurusAmazonScraper.__new__(BotasaurusAmazonScraper)
        scraper.debug_mode = False
        scraper.logger = __import__("logging").getLogger("test")
        scraper.url_shortener_settings = load_url_shortener_settings(
            write(tmp_path, {"enabled": False})
        )

        scraper._shorten_affiliate_links([])

        assert called["n"] == 0


class TestTheDefaultsPreserveTheOldBehaviour:
    """The consumer read an absent key as off; the model said on.

    Swapping one for the other would make a partial override file start
    calling a third-party API on every scrape and change what lands in
    `data.json`, without anyone editing that file.
    """

    def test_an_absent_enabled_key_is_off(self, tmp_path):
        settings = load_url_shortener_settings(write(tmp_path, {"provider": "picsee"}))

        assert settings.enabled is False

    def test_an_absent_integration_block_does_not_shorten_on_scrape(self, tmp_path):
        settings = load_url_shortener_settings(write(tmp_path, {"enabled": True}))

        assert settings.integration.shorten_on_scrape is False

    def test_a_picsee_block_without_an_env_var_name_still_finds_the_key(
        self, tmp_path, monkeypatch
    ):
        """The old consumer defaulted the *name*, not the value.

        Leaving it unset made such a config skip shortening entirely, with only
        a debug-gated warning to say so.
        """
        settings = load_url_shortener_settings(
            write(
                tmp_path,
                {
                    "enabled": True,
                    "provider": "picsee",
                    "picsee": {"custom_domain": "example.test"},
                    "integration": {"shorten_on_scrape": True},
                },
            )
        )

        assert settings.active_provider().api_key_env_var == "PICSEE_API_KEY"

    def test_the_shipped_config_still_shortens(self):
        """The bundled file sets both keys, so the run is unchanged."""
        settings = load_url_shortener_settings()

        assert settings.enabled is True
        assert settings.integration.shorten_on_scrape is True


class TestTheLoaderDoesNotDependOnTheWorkingDirectory:
    def test_the_default_path_is_anchored_on_the_repo(self, tmp_path, monkeypatch):
        """The scraper runs from anywhere; its own config reads are anchored.

        A cwd-relative default would silently load the `bare` no-op instead of
        the operator's provider.
        """
        monkeypatch.chdir(tmp_path)

        assert load_url_shortener_settings().enabled is True


class TestTheConstructorWiresTheLoader:
    """Both scraper tests above assign the settings by hand.

    Deleting the load from `__init__` left the whole suite green: at runtime
    `_shorten_affiliate_links` then raises `AttributeError`, its broad handler
    catches it, and every link falls back to the long URL.
    """

    def test_a_normally_constructed_scraper_carries_the_settings(self):
        from src.scraper.amazon.scraper import BotasaurusAmazonScraper

        scraper = BotasaurusAmazonScraper()

        assert scraper.url_shortener_settings.provider in (
            URLShortenerSettings.provider_names()
        )

    def test_construction_does_not_need_the_repo_as_the_cwd(
        self, tmp_path, monkeypatch
    ):
        """It used to import the video config package, which loads five
        cwd-relative YAML files eagerly, so a scraper built from elsewhere
        failed to construct at all.
        """
        from src.scraper.amazon.scraper import BotasaurusAmazonScraper

        monkeypatch.chdir(tmp_path)

        assert BotasaurusAmazonScraper().url_shortener_settings is not None
