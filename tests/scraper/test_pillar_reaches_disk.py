"""Tests that a keyword's pillar reaches `data.json`.

The three scraper arms failed differently. The global batch assigned the
pillar to the in-memory record *after* the file had been written, so its file
said `pillar: null`. The standalone multi-keyword arm assigned it but never
wrote through the record's serialiser at all, so its file had no `pillar` key.
The standalone single-keyword arm never assigned one.
Nothing failed: the record the caller held was correct, and the caller is what
every existing test looked at. Only the file was wrong, and the producer reads
the file.

So these assert on the written bytes. On the two arms that assigned at all,
a test checking the returned record would have passed against the broken
version -- and the record is what every existing test looked at.

The pillar shapes the script, the prompt preamble and the audience. It is no
longer recorded in the published-products registry, so nothing here asserts
about that file.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.scraper.amazon.models import ProductData


def _product(asin: str = "B0TEST0001") -> ProductData:
    return ProductData(
        title="A product",
        price="$10",
        url=f"https://www.amazon.com/dp/{asin}",
        platform=None,
        asin=asin,
    )


def _scraper(tmp_path: Path, keywords_block):
    """A real scraper with its config stubbed and its output redirected."""
    from src.scraper.amazon.scraper import BotasaurusAmazonScraper

    scraper = BotasaurusAmazonScraper.__new__(BotasaurusAmazonScraper)
    scraper.config = {"batch": {"keywords": keywords_block}}
    scraper._keyword_pillars = None
    scraper.output_dir = str(tmp_path)
    scraper.debug_mode = False
    scraper.logger = MagicMock()
    return scraper


def _written(tmp_path: Path, asin: str) -> dict:
    path = tmp_path / asin / "data.json"
    assert path.exists(), f"no data.json written at {path}"
    data: Any = json.loads(path.read_text())
    if isinstance(data, list):
        data = data[0]
    return dict(data)


@pytest.mark.unit
class TestPillarResolution:
    def test_a_configured_keyword_resolves_to_its_pillar(self, tmp_path):
        scraper = _scraper(tmp_path, {"value": ["smart plug"], "utility": ["ssd"]})
        assert scraper.pillar_for_keyword("smart plug") == "value"
        assert scraper.pillar_for_keyword("ssd") == "utility"

    def test_an_unconfigured_keyword_resolves_to_nothing(self, tmp_path):
        scraper = _scraper(tmp_path, {"value": ["smart plug"]})
        assert scraper.pillar_for_keyword("something else") is None

    def test_a_flat_keyword_list_maps_nothing(self, tmp_path):
        """The pre-pillar config shape. It must load, not raise."""
        scraper = _scraper(tmp_path, ["smart plug"])
        assert scraper.pillar_for_keyword("smart plug") is None

    def test_a_missing_batch_block_maps_nothing(self, tmp_path):
        from src.scraper.amazon.scraper import BotasaurusAmazonScraper

        scraper = BotasaurusAmazonScraper.__new__(BotasaurusAmazonScraper)
        scraper.config = {}
        scraper._keyword_pillars = None
        assert scraper.pillar_for_keyword("smart plug") is None


@pytest.mark.unit
class TestPillarOnDisk:
    """The three paths that write `data.json`, each asserted from the file."""

    def test_process_raw_products_writes_the_pillar(self, tmp_path):
        """The global batch's path."""
        scraper = _scraper(tmp_path, {"value": ["smart plug"]})
        product = _product()

        with (
            patch.object(scraper, "_orchestrate_media_downloads"),
            patch.object(
                scraper, "_validate_and_convert_products", return_value=[product]
            ),
            patch.object(scraper, "_shorten_affiliate_links"),
        ):
            scraper.process_raw_products([{"asin": "B0TEST0001"}], pillar="value")

        assert _written(tmp_path, "B0TEST0001")["pillar"] == "value"

    def test_process_raw_products_without_a_pillar_writes_null(self, tmp_path):
        """An unconfigured keyword must not invent one."""
        scraper = _scraper(tmp_path, {"value": ["smart plug"]})
        product = _product()

        with (
            patch.object(scraper, "_orchestrate_media_downloads"),
            patch.object(
                scraper, "_validate_and_convert_products", return_value=[product]
            ),
            patch.object(scraper, "_shorten_affiliate_links"),
        ):
            scraper.process_raw_products([{"asin": "B0TEST0001"}])

        assert _written(tmp_path, "B0TEST0001")["pillar"] is None

    def test_scrape_products_writes_the_pillar(self, tmp_path):
        """The standalone single-keyword path, which is the issue's repro."""
        scraper = _scraper(tmp_path, {"value": ["smart plug"]})
        product = _product()

        with (
            patch.object(scraper, "scrape_products_unified", return_value=[product]),
            patch.object(scraper, "_shorten_affiliate_links"),
        ):
            scraper.scrape_products(["smart plug"], None)

        assert _written(tmp_path, "B0TEST0001")["pillar"] == "value"

    def test_the_batch_controller_writes_the_pillar(self, tmp_path):
        """The standalone multi-keyword path.

        It never called `_save_products` at all, so the file was whatever the
        browser callback wrote mid-scrape -- before the pillar existed and
        before the media downloads finished.
        """
        from src.scraper.amazon.batch_controller import BatchController
        from src.scraper.amazon.models import BatchConfig

        scraper = _scraper(tmp_path, {"value": ["smart plug"]})
        product = _product()

        from src.scraper.amazon.models import SearchParameters

        config = BatchConfig(
            product_ids=[],
            keywords=["smart plug"],
            fail_fast=False,
            search_params=SearchParameters(),
            max_products=1,
            products_per_keyword=1,
            keyword_pillar_map={"smart plug": "value"},
        )
        controller = BatchController(scraper, config)

        with (
            patch.object(scraper, "scrape_products_unified", return_value=[product]),
            patch.object(scraper, "_shorten_affiliate_links"),
        ):
            controller._process_keywords()

        assert _written(tmp_path, "B0TEST0001")["pillar"] == "value"


@pytest.mark.unit
class TestPillarSurvivesATruncatedResume:
    """Recording the pillar inside the script step is not enough.

    A resume that finds a completed step's artifact missing truncates the
    state to step keys only, dropping every top-level scalar, and then skips
    the steps it kept. So the step that would re-record the pillar never runs,
    and a repeat render then draws from a different template pool, preamble and
    audience than the script already on disk was written for.

    These drive the real `create_video_for_product` with the state load and
    the pipeline execution stubbed, so they assert what the steps are handed
    rather than where the code sits. A source-position check passes on a
    resolution moved below the pipeline run, which disables it entirely.
    """

    async def _resolved_pillar(
        self, tmp_path, product_pillar, cli_overrides=None, loaded_state=None
    ):
        import warnings

        from src.video.producer import orchestration

        seen: dict = {}

        async def _fake_load(ctx):
            # A truncated resume leaves step keys only; an ordinary one keeps
            # whatever the previous run recorded.
            ctx.state = (
                {"gather_visuals": {"status": "done"}}
                if loaded_state is None
                else dict(loaded_state)
            )

        async def _fake_execute(ctx):
            seen["pillar"] = ctx.state.get("pillar")
            return True, None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from src.video.config import load_video_config_modular

            config = load_video_config_modular()
        config.global_output_root_path = tmp_path

        product = ProductData(
            title="A product",
            price="$10",
            url="https://www.amazon.com/dp/B0TEST0001",
            platform=None,
            asin="B0TEST0001",
            pillar=product_pillar,
        )

        with (
            patch.object(orchestration, "_load_pipeline_state", _fake_load),
            patch.object(orchestration, "execute_pipeline_parallel", _fake_execute),
        ):
            await orchestration.create_video_for_product(
                config,
                product,
                "slideshow_images1",
                {},
                None,
                False,
                False,
                None,
                cli_overrides=cli_overrides,
            )
        return seen.get("pillar")

    async def test_a_product_pillar_survives_a_truncated_state(self, tmp_path):
        """The state load wiped it; the resolution after it puts it back."""
        assert await self._resolved_pillar(tmp_path, "value") == "value"

    async def test_a_cli_override_wins_over_the_product(self, tmp_path):
        assert (
            await self._resolved_pillar(
                tmp_path, "value", cli_overrides={"pillar": "utility"}
            )
            == "utility"
        )

    async def test_no_pillar_anywhere_records_nothing(self, tmp_path):
        assert await self._resolved_pillar(tmp_path, None) is None

    async def test_a_recorded_pillar_survives_a_resume_without_the_flag(self, tmp_path):
        """An untruncated resume keeps the previous run's state.

        The flag is not repeated on a rerun, and the script on disk was
        already written under it, so letting the product record win here files
        the row under an arm the shipped script was not written for.
        """
        assert (
            await self._resolved_pillar(
                tmp_path, "value", loaded_state={"pillar": "utility"}
            )
            == "utility"
        )
