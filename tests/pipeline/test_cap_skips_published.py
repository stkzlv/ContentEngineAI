"""An already-published product does not count toward `max_products` (#737).

The scraping phase stopped at `max_products` products and the batch dropped
the published ones only afterwards, before render. A run capped at one
product could scrape one it had already published, render nothing and exit
0, leaving its slot empty. The scrape now skips such search results before
media download and takes the next one.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.pipeline.config import PipelineState, ScrapingPhaseSummary
from src.pipeline.phases import scraping as scraping_phase


class _FakeScraper:
    """Search pages keyed by (input, page); `process_raw_products` keeps order."""

    pages: dict[tuple[str, int], list[dict]] = {}

    def __init__(self, **_kwargs):
        self.output_dir = "unused"
        self.run_max_products = 1
        self.throttle = SimpleNamespace(
            summary_lines=lambda: [], dead_queries=[], throttled_inputs=[]
        )
        self.processed: list[str] = []

    def scrape_batch_browser(self, inputs, search_params=None, start_page=1):
        return [
            {"input": i, "products": list(self.pages.get((i, start_page), []))}
            for i in inputs
        ]

    def process_raw_products(self, raw, target_download_count=None, pillar=None):
        chosen = raw[:target_download_count]
        self.processed.extend(r["asin"] for r in chosen)
        return [SimpleNamespace(asin=r["asin"]) for r in chosen]

    @staticmethod
    def _is_asin(value):
        return value.startswith("B0")

    @staticmethod
    def _is_url(value):
        return value.startswith("http")


def _config(**overrides):
    config = MagicMock()
    config.product_ids = []
    config.keywords = ["phone stand"]
    config.max_products = 1
    config.products_per_keyword = 1
    config.scraper_filters = None
    config.debug = False
    config.fail_fast = False
    config.keyword_pillar_map = {}
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


@pytest.fixture
def fake_scraper(monkeypatch):
    _FakeScraper.pages = {}
    monkeypatch.setattr(
        "src.scraper.amazon.scraper.BotasaurusAmazonScraper", _FakeScraper
    )
    monkeypatch.setattr(scraping_phase, "media_on_disk", lambda *_a: (0, 0))
    monkeypatch.setattr(
        "src.scraper.amazon.config.CONFIG",
        {"global_settings": {"batch_processing": {"max_retry_pages": 3}}},
    )
    return _FakeScraper


async def _scrape(config, published=()):
    return await scraping_phase.run_scraping_phase(
        config,
        lambda: None,
        is_already_published=lambda asin: asin in published,
    )


@pytest.mark.unit
@pytest.mark.req("REQ-BAT-088")
class TestTheScrapeSkipsPublishedResults:
    @pytest.mark.asyncio
    async def test_the_next_result_fills_the_cap(self, fake_scraper):
        fake_scraper.pages = {
            ("phone stand", 1): [{"asin": "B0PUB"}, {"asin": "B0NEW"}]
        }

        summary = await _scrape(_config(), published={"B0PUB"})

        assert summary.successful_products == ["B0NEW"]
        assert summary.already_published == ["B0PUB"]
        assert summary.failed == 0

    @pytest.mark.asyncio
    async def test_without_the_check_the_published_one_takes_the_slot(
        self, fake_scraper
    ):
        """The defect: the cap is spent on a product the run then drops."""
        fake_scraper.pages = {
            ("phone stand", 1): [{"asin": "B0PUB"}, {"asin": "B0NEW"}]
        }

        summary = await scraping_phase.run_scraping_phase(_config(), lambda: None)

        assert summary.successful_products == ["B0PUB"]

    @pytest.mark.asyncio
    async def test_a_later_page_is_searched_when_page_one_is_all_published(
        self, fake_scraper
    ):
        fake_scraper.pages = {
            ("phone stand", 1): [{"asin": "B0PUB1"}, {"asin": "B0PUB2"}],
            ("phone stand", 2): [{"asin": "B0PUB3"}],
            ("phone stand", 3): [{"asin": "B0NEW"}],
        }

        summary = await _scrape(_config(), published={"B0PUB1", "B0PUB2", "B0PUB3"})

        assert summary.successful_products == ["B0NEW"]
        assert summary.already_published == ["B0PUB1", "B0PUB2", "B0PUB3"]

    @pytest.mark.asyncio
    async def test_an_input_with_nothing_new_is_not_a_failed_input(self, fake_scraper):
        fake_scraper.pages = {("phone stand", 1): [{"asin": "B0PUB"}]}

        summary = await _scrape(_config(), published={"B0PUB"})

        assert summary.successful_products == []
        assert summary.failed == 0
        assert summary.failed_products == []
        assert summary.already_published == ["B0PUB"]

    @pytest.mark.asyncio
    async def test_an_input_with_no_results_is_still_a_failure(self, fake_scraper):
        summary = await _scrape(_config(), published={"B0PUB"})

        assert summary.failed == 1
        assert summary.failed_products == ["phone stand"]

    @pytest.mark.asyncio
    async def test_published_results_get_no_media_download(self, fake_scraper):
        fake_scraper.pages = {
            ("phone stand", 1): [{"asin": "B0PUB"}, {"asin": "B0NEW"}]
        }
        seen: list[str] = []
        original = _FakeScraper.process_raw_products

        def spy(self, raw, target_download_count=None, pillar=None):
            seen.extend(r["asin"] for r in raw)
            return original(self, raw, target_download_count, pillar)

        fake_scraper.process_raw_products = spy
        try:
            await _scrape(_config(), published={"B0PUB"})
        finally:
            fake_scraper.process_raw_products = original

        assert "B0PUB" not in seen

    @pytest.mark.asyncio
    async def test_the_cap_counts_only_new_products_across_keywords(self, fake_scraper):
        fake_scraper.pages = {
            ("kw1", 1): [{"asin": "B0PUB"}],
            ("kw2", 1): [{"asin": "B0NEW2"}],
            ("kw3", 1): [{"asin": "B0NEW3"}],
        }

        summary = await _scrape(
            _config(keywords=["kw1", "kw2", "kw3"], max_products=2),
            published={"B0PUB"},
        )

        assert summary.successful_products == ["B0NEW2", "B0NEW3"]


def _history(tmp_path, asin):
    (tmp_path / "publish_history.json").write_text(
        json.dumps(
            {
                "posts": {
                    f"{asin}:{p}": {"post_id": "1"}
                    for p in ("youtube", "tiktok", "instagram")
                }
            }
        ),
        encoding="utf-8",
    )


def _orchestrator(tmp_path, **flags):
    from src.pipeline.global_batch import GlobalPipelineOrchestrator

    pipeline = GlobalPipelineOrchestrator.__new__(GlobalPipelineOrchestrator)
    pipeline.config = MagicMock()
    pipeline.config.outputs_dir = tmp_path
    pipeline.config.platforms = None
    # An unset MagicMock attribute is truthy, which would turn the guard off.
    pipeline.config.force = flags.get("force", False)
    pipeline.config.skip_publish = flags.get("skip_publish", False)
    return pipeline


@pytest.mark.unit
@pytest.mark.req("REQ-BAT-088")
class TestTheBatchHandsTheScrapeItsRule:
    @pytest.mark.asyncio
    async def test_the_batch_scrape_skips_a_product_published_everywhere(
        self, tmp_path, fake_scraper
    ):
        _history(tmp_path, "B0PUB")
        fake_scraper.pages = {
            ("phone stand", 1): [{"asin": "B0PUB"}, {"asin": "B0NEW"}]
        }
        pipeline = _orchestrator(tmp_path)
        pipeline.config.product_ids = []
        pipeline.config.keywords = ["phone stand"]
        pipeline.config.max_products = 1
        pipeline.config.products_per_keyword = 1
        pipeline.config.scraper_filters = None
        pipeline.config.debug = False
        pipeline.config.fail_fast = False
        pipeline.config.keyword_pillar_map = {}
        pipeline._resolve_profile_uses_videos = lambda: None

        summary = await pipeline._execute_scraping_phase()

        assert summary.successful_products == ["B0NEW"]
        assert summary.already_published == ["B0PUB"]

    @pytest.mark.parametrize("flag", ["force", "skip_publish"])
    def test_force_and_skip_publish_skip_nothing(self, tmp_path, flag):
        _history(tmp_path, "B0PUB")

        assert (
            _orchestrator(tmp_path, **{flag: True})._published_everywhere("B0PUB")
            is False
        )

    def test_a_topic_is_never_skipped(self, tmp_path):
        from src.pipeline.global_batch import TOPIC_ID_PREFIX

        topic = f"{TOPIC_ID_PREFIX}abc"
        _history(tmp_path, topic)

        assert _orchestrator(tmp_path)._published_everywhere(topic) is False


@pytest.mark.unit
@pytest.mark.req("REQ-BAT-049")
@pytest.mark.req("REQ-BAT-030")
class TestARunThatFoundNothingNewStillEndsCleanly:
    @pytest.mark.asyncio
    async def test_scrape_skips_reach_the_verdict_and_the_summary(
        self, tmp_path, monkeypatch
    ):
        pipeline = _orchestrator(tmp_path)
        pipeline.config.process_all_products = False
        pipeline.config.topics = []
        pipeline.config.keywords = ["phone stand"]
        pipeline.config.strict = False
        pipeline.config.resume = False
        pipeline.state = PipelineState.create_new(pipeline.config)
        pipeline._save_state = MagicMock()
        monkeypatch.setattr(
            "src.video.producer.cli.discover_products_for_batch",
            lambda *a, **k: [],
        )
        pipeline._execute_scraping_phase = AsyncMock(
            return_value=ScrapingPhaseSummary(
                1, 0, 0, [], [], {}, 1.0, already_published=["B0PUB"]
            )
        )
        pipeline._notify_webhook = AsyncMock()
        pipeline._execute_publishing_phase = AsyncMock(return_value=None)

        summary = await pipeline.run_pipeline()

        assert summary.production.already_published_products == ["B0PUB"]
        assert summary.outcome() == "nothing new"
        assert summary.exit_code() == 0
