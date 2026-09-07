"""The scraper's configuration is typed end to end (#125).

Three loaders read `config/scraper.yaml` and each handed consumers a raw
dict that sixty-odd sites walked with `dict.get(key, MAGIC)`. A misspelled
key, in the file or in a reader, was read back as that reader's own number,
silently. All three loaders now validate through `ScraperConfig`, whose
submodels refuse unknown keys, and hand out either the typed object or a
defaults-filled dict in the file's shape. Defaults live in the field
declarations and nowhere else.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.scraper.config_models import ScraperConfig

REPO = Path(__file__).resolve().parents[2]
BUNDLED = REPO / "config" / "scraper.yaml"


def _bundled() -> dict:
    return dict(yaml.safe_load(BUNDLED.read_text()))


@pytest.mark.unit
class TestTheBundledFileLoads:
    def test_it_validates_under_the_strict_models(self) -> None:
        cfg = ScraperConfig.from_legacy_dict(_bundled())
        assert cfg.amazon.max_products == 1
        assert isinstance(cfg.batch.keywords, dict)
        assert set(cfg.batch.keywords) == {"value", "novelty", "utility"}

    def test_the_runtime_dict_keeps_the_files_shape_and_fills_every_key(self) -> None:
        runtime = ScraperConfig.from_legacy_dict(_bundled()).to_runtime_dict()
        assert set(runtime) == {"global_settings", "batch", "scrapers"}
        download = runtime["global_settings"]["download_config"]
        # Absent from the file, present from the model.
        assert download["min_image_file_size"] == 10000
        assert download["validation_timeout"] is None
        assert runtime["global_settings"]["batch_processing"]["max_pages"] == 7
        assert runtime["scrapers"]["amazon"]["max_products"] == 1


@pytest.mark.unit
class TestAMisspelledKeyFailsAtLoad:
    """The whole point: a typo is an error, not a default."""

    @pytest.mark.parametrize(
        "path",
        [
            ("global_settings", "download_config", "downlaod_timeout"),
            ("global_settings", "sytem_timeouts"),
            ("batch", "products_per_keywrod"),
            ("scrapers", "amazon", "max_prodcuts"),
            # The top level and the `scrapers:` block are sections too; picking
            # the known ones by name let these load as defaults (review finding).
            ("glboal_settings",),
            ("scrapers", "amazno"),
        ],
    )
    def test_the_model_refuses_it(self, path: tuple[str, ...]) -> None:
        data = _bundled()
        node = data
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = 1
        with pytest.raises(ValidationError, match=path[-1]):
            ScraperConfig.from_legacy_dict(data)

    def test_the_adapter_raises_rather_than_falling_back(self, tmp_path) -> None:
        from src.scraper.config_adapter import ScraperConfigAdapter

        data = _bundled()
        data["global_settings"]["retry_config"]["base_dely"] = 1.0
        (tmp_path / "scraper.yaml").write_text(yaml.safe_dump(data))
        with pytest.raises(ValidationError, match="base_dely"):
            ScraperConfigAdapter(config_root=str(tmp_path)).get_merged_config_dict()

    def test_the_platform_manager_raises(self, tmp_path, monkeypatch) -> None:
        from src.scraper.base.config import PlatformConfigManager

        data = _bundled()
        data["global_settings"]["browser_config"]["page_load_timeout_ms_"] = 5
        f = tmp_path / "scraper.yaml"
        f.write_text(yaml.safe_dump(data))
        # The manager resolves the path against the project root.
        with pytest.raises(ValidationError, match="page_load_timeout_ms_"):
            PlatformConfigManager(
                config_path=str(f.relative_to(REPO))
                if f.is_relative_to(REPO)
                else str(f)
            )

    def test_the_scraper_loader_raises(self, tmp_path) -> None:
        from src.scraper.amazon.scraper import BotasaurusAmazonScraper

        data = _bundled()
        data["global_settings"]["video_config"]["min_dimensoin"] = 1
        f = tmp_path / "scraper.yaml"
        f.write_text(yaml.safe_dump(data))
        scraper = BotasaurusAmazonScraper.__new__(BotasaurusAmazonScraper)
        with pytest.raises(ValidationError, match="min_dimensoin"):
            BotasaurusAmazonScraper._load_config(scraper, str(f))


@pytest.mark.unit
class TestASectionLeftOutTakesTheDefaults:
    def test_no_batch_block_reads_as_the_model(self) -> None:
        runtime = ScraperConfig.from_legacy_dict(
            {"global_settings": {}, "scrapers": {"amazon": {}}}
        ).to_runtime_dict()
        assert runtime["batch"]["products_per_keyword"] == 2
        assert runtime["batch"]["keywords"] == []

    def test_no_scrapers_block_loads_through_the_adapter(self, tmp_path) -> None:
        """The adapter synthesises the amazon block; it used to inject a
        `platform` key the strict model refuses (review finding).
        """
        from src.scraper.config_adapter import ScraperConfigAdapter

        (tmp_path / "scraper.yaml").write_text(
            yaml.safe_dump({"global_settings": {"retry_config": {}}})
        )
        runtime = ScraperConfigAdapter(
            config_root=str(tmp_path)
        ).get_merged_config_dict()
        assert runtime["scrapers"]["amazon"]["enabled"] is True
        # The model's default, not an injected one (the adapter used to write
        # `debug_mode: True` before validating).
        assert runtime["global_settings"]["debug_mode"] is False

    def test_a_missing_file_reads_as_the_models_defaults(self, tmp_path) -> None:
        from src.scraper.config_adapter import ScraperConfigAdapter

        runtime = ScraperConfigAdapter(
            config_root=str(tmp_path)
        ).get_merged_config_dict()
        assert runtime["scrapers"]["amazon"]["max_products"] == 2
        assert runtime["global_settings"]["debug_mode"] is False

    def test_the_consolidated_shape_keeps_its_values(self) -> None:
        """A top-level `amazon` block is this model's own field; it used to
        be overwritten by the defaults of the absent `scrapers` block.
        """
        cfg = ScraperConfig.from_legacy_dict({"amazon": {"max_products": 42}})
        assert cfg.amazon.max_products == 42

    def test_both_shapes_at_once_is_refused_as_a_validation_error(self) -> None:
        """A `ValidationError`, so the loaders' re-raise sees it; a plain
        `ValueError` was swallowed into the defaults (review finding).
        """
        with pytest.raises(ValidationError, match="both"):
            ScraperConfig.from_legacy_dict(
                {"amazon": {"max_products": 1}, "scrapers": {"amazon": {}}}
            )

    def test_both_shapes_reach_the_manager_and_the_import_loader(
        self, tmp_path, monkeypatch
    ) -> None:
        from src.config_manager import UnifiedConfigManager
        from src.scraper.amazon import config as scraper_config

        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "scraper.yaml").write_text(
            yaml.safe_dump({"amazon": {"max_products": 1}, "scrapers": {"amazon": {}}})
        )
        with pytest.raises(ValidationError, match="both"):
            UnifiedConfigManager(
                config_root=str(tmp_path / "config")
            ).get_scraper_config()
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValidationError, match="both"):
            scraper_config.load_browser_config_from_yaml()

    @pytest.mark.parametrize("content", ["", "- a list\n", "just a string\n"])
    def test_an_empty_or_non_mapping_file_is_refused(self, tmp_path, content) -> None:
        """An empty file is a truncated write, not a request for defaults; it
        used to raise and the first cut of this branch loaded it (review).
        """
        from src.scraper.amazon.scraper import BotasaurusAmazonScraper
        from src.scraper.config_adapter import ScraperConfigAdapter

        f = tmp_path / "scraper.yaml"
        f.write_text(content)
        with pytest.raises(ValidationError):
            ScraperConfigAdapter(config_root=str(tmp_path)).get_merged_config_dict()
        scraper = BotasaurusAmazonScraper.__new__(BotasaurusAmazonScraper)
        with pytest.raises(ValidationError):
            BotasaurusAmazonScraper._load_config(scraper, str(f))

    def test_malformed_yaml_is_refused_by_every_loader(self, tmp_path) -> None:
        from src.scraper.amazon.scraper import BotasaurusAmazonScraper
        from src.scraper.config_adapter import ScraperConfigAdapter

        f = tmp_path / "scraper.yaml"
        f.write_text("invalid: yaml: content: [")
        with pytest.raises(yaml.YAMLError):
            ScraperConfigAdapter(config_root=str(tmp_path)).get_merged_config_dict()
        scraper = BotasaurusAmazonScraper.__new__(BotasaurusAmazonScraper)
        with pytest.raises(yaml.YAMLError):
            BotasaurusAmazonScraper._load_config(scraper, str(f))


@pytest.mark.unit
class TestTheBatchsPerRunLimitReachesTheBrowser:
    """The batch sets a per-keyword share on the scraper. It wrote it into
    the dict the browser phase used to read; the typed read ignored it
    (review finding), so the share is an attribute the typed read honours.
    """

    @staticmethod
    def _scraper():
        import logging

        from src.scraper.amazon.scraper import BotasaurusAmazonScraper

        scraper = BotasaurusAmazonScraper.__new__(BotasaurusAmazonScraper)
        scraper.settings = ScraperConfig()
        scraper.debug_mode = False
        scraper.debug_options = {}
        scraper.logger = logging.getLogger("test")
        setattr(scraper, "throttle", object())  # noqa: B010 -- typed attribute
        scraper.run_max_products = None
        return scraper

    @staticmethod
    def _items_sent(scraper) -> list[dict]:
        from unittest.mock import patch

        sent: list[dict] = []

        def fake_factory(debug_mode, throttle):
            def run(payload):
                sent.extend(payload["items"])
                return []

            return run

        with patch(
            "src.scraper.amazon.scraper.create_batch_browser_function", fake_factory
        ):
            scraper.scrape_batch_browser(["usb hub"])
        return sent

    def test_the_configured_limit_by_default(self) -> None:
        scraper = self._scraper()
        assert self._items_sent(scraper)[0]["max_products"] == 2

    def test_the_batchs_share_when_set(self) -> None:
        scraper = self._scraper()
        scraper.run_max_products = 3
        assert self._items_sent(scraper)[0]["max_products"] == 3

    def test_the_batch_sets_the_attribute_not_the_dict(self) -> None:
        source = (REPO / "src" / "pipeline" / "global_batch.py").read_text()
        assert 'amazon_config["max_products"]' not in source
        assert source.count("scraper.run_max_products = ") == 2


@pytest.mark.unit
class TestDefaultsLiveInTheModel:
    def test_the_model_defaults_are_the_numbers_the_readers_used_to_carry(self) -> None:
        cfg = ScraperConfig()
        gs = cfg.global_settings
        assert gs.download_config.download_timeout == 30
        assert gs.download_config.download_chunk_size == 8192
        assert gs.download_config.concurrent_image_downloads == 5
        assert gs.download_config.concurrent_video_downloads == 3
        assert gs.download_config.min_image_file_size == 10000
        assert gs.system_timeouts.head_request_timeout == 10
        assert gs.system_timeouts.system_command_timeout == 5
        assert gs.system_timeouts.system_profiler_timeout == 10
        assert gs.media_config.ffprobe_timeout_sec == 30
        assert gs.batch_processing.max_pages == 7
        assert gs.retries == 3
        assert gs.retry_config.default_max_retries == 3
        assert gs.rate_limiting.debug_pause_duration == 5
        assert gs.browser_config.max_products_per_search == 5
        assert cfg.batch.products_per_keyword == 2
        assert cfg.batch.logging.separator_width == 60
        assert cfg.amazon.filter_parameters.price_to_cents_multiplier == 100

    def test_the_validation_timeout_follows_the_head_timeout_when_unset(self) -> None:
        """What the dict-walk did when the key was absent, kept as a rule."""
        from unittest.mock import patch

        from src.scraper.amazon.image_utils import _validation_timeout

        data = _bundled()
        data["global_settings"]["system_timeouts"]["head_request_timeout"] = 21
        cfg = ScraperConfig.from_legacy_dict(data)
        with patch("src.scraper.amazon.config.SETTINGS", cfg):
            assert _validation_timeout() == 21
        data["global_settings"]["download_config"]["validation_timeout"] = 4
        cfg = ScraperConfig.from_legacy_dict(data)
        with patch("src.scraper.amazon.config.SETTINGS", cfg):
            assert _validation_timeout() == 4

    def test_the_settings_accessor_reads_the_loaded_file(self) -> None:
        from src.scraper.amazon.config import get_settings

        assert get_settings().amazon.max_products == 1  # the bundled value, not 2


@pytest.mark.unit
class TestNoReaderCarriesItsOwnNumber:
    """No config reader falls back to a literal of its own.

    The receivers are what the scraper's configuration is walked through;
    reads of result data (`stats.get("total_images", 0)`) are not this bug
    and are left alone.
    """

    RECEIVER = r"(?:CONFIG|[a-z_]*(?:config|settings|timeouts|cfg))"
    PATTERN = re.compile(
        RECEIVER + r"\.get\([\"'][a-z_]+[\"']\s*,\s*[0-9]+(?:\.[0-9]+)?\)"
    )

    def test_no_numeric_fallback_on_a_config_receiver(self) -> None:
        hits = []
        for path in (REPO / "src" / "scraper").rglob("*.py"):
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if self.PATTERN.search(line):
                    hits.append(f"{path.relative_to(REPO)}:{i}: {line.strip()}")
        assert not hits, "\n".join(hits)
