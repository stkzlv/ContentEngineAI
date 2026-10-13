"""The batch scrapes into its own outputs root (#740, REQ-OPS-031).

The scraping phase built the scraper without an output directory, so the
scrape wrote under the repository's `outputs/` while the handoff, production
and publishing phases read `--outputs-dir`: a run pointed elsewhere found
none of the products it had just scraped.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.pipeline.phases import scraping as scraping_phase
from src.scraper.amazon import botasaurus_output


class _RecordingScraper:
    built: list[dict] = []

    def __init__(self, **kwargs):
        _RecordingScraper.built.append(kwargs)
        self.output_dir = kwargs.get("output_dir")
        self.run_max_products = 1
        self.throttle = SimpleNamespace(
            summary_lines=lambda: [], dead_queries=[], throttled_inputs=[]
        )

    def scrape_batch_browser(self, inputs, **_kwargs):
        return [{"input": i, "products": [], "skipped_asins": []} for i in inputs]


@pytest.mark.unit
@pytest.mark.req("REQ-OPS-031")
class TestTheBatchScrapesIntoItsOutputsRoot:
    @pytest.mark.asyncio
    async def test_the_scraper_is_given_the_batch_root(self, tmp_path, monkeypatch):
        _RecordingScraper.built = []
        monkeypatch.setattr(
            "src.scraper.amazon.scraper.BotasaurusAmazonScraper", _RecordingScraper
        )
        config = MagicMock()
        config.outputs_dir = tmp_path / "elsewhere"
        config.product_ids = []
        config.keywords = ["phone stand"]
        config.max_products = 1
        config.products_per_keyword = 1
        config.fail_fast = False

        await scraping_phase.run_scraping_phase(config, lambda: None)

        assert _RecordingScraper.built[0]["output_dir"] == str(tmp_path / "elsewhere")

    def test_the_real_scraper_writes_products_under_that_root(
        self, tmp_path, monkeypatch
    ):
        from src.scraper.amazon.scraper import BotasaurusAmazonScraper

        # The override is module-level: restore it so no later test inherits it.
        monkeypatch.setattr(botasaurus_output, "_output_dir_override", None)
        root = tmp_path / "elsewhere"
        root.mkdir()

        BotasaurusAmazonScraper(output_dir=str(root))

        assert botasaurus_output.get_product_output_dir("B0TEST") == root / "B0TEST"
