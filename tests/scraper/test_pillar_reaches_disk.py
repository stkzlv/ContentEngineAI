"""Tests that a keyword's pillar reaches `data.json`.

The pillar was assigned to the in-memory record *after* the file had been
written, so the arms that write through the record's serialiser wrote
`pillar: null`, and the arm that did not write through it at all had no
`pillar` key.
Nothing failed: the record the caller held was correct, and the caller is what
every existing test looked at. Only the file was wrong, and the producer reads
the file.

So these assert on the written bytes. A test that checked the returned records
would have passed against the broken version on all three paths.
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
class TestPillarReachesTheRegistry:
    """`data.json` carrying the pillar is only half the journey.

    The registry reads `pipeline_state.json`, and that key was written from
    `cli_overrides` alone. So a product-level pillar shaped the script and
    then vanished, filing the row as unlabelled for a video that was
    genuinely rendered under a pillar. Making the file carry the value is what
    exposed this: before, no scraped render used one.
    """

    async def _run_script_step(self, tmp_path, product_pillar, state=None):
        """Drive the real `step_generate_script` past its resume shortcut.

        With the script already on disk the step loads it and returns, which
        is enough: the pillar is read before that branch.
        """
        from src.video.producer import steps as steps_mod

        script = tmp_path / "script.txt"
        script.write_text("A script.", encoding="utf-8")

        ctx = MagicMock()
        # Keep the hook-headline path out of this test. It is unrelated, and
        # with a MagicMock config it only stays harmless because the prompt
        # join happens to reject a MagicMock before any HTTP call.
        ctx.config.video_settings.hook_overlay.enabled = False
        ctx.state = {} if state is None else state
        ctx.product = ProductData(
            title="A product",
            price="$10",
            url="https://www.amazon.com/dp/B0TEST0001",
            platform=None,
            asin="B0TEST0001",
            pillar=product_pillar,
        )
        ctx.run_paths = {"script_file": script, "script_prompt": tmp_path / "p.txt"}
        ctx.debug_mode = False
        await steps_mod.step_generate_script(ctx)
        return ctx.state

    async def test_a_product_level_pillar_is_recorded_in_state(self, tmp_path):
        state = await self._run_script_step(tmp_path, "value")
        assert state["pillar"] == "value"

    async def test_a_cli_override_still_wins(self, tmp_path):
        """The override is the reason the key existed; it must keep winning."""
        state = await self._run_script_step(
            tmp_path, "value", state={"pillar": "novelty"}
        )
        assert state["pillar"] == "novelty"

    async def test_no_pillar_records_nothing(self, tmp_path):
        """An unconfigured keyword must not write an empty label."""
        state = await self._run_script_step(tmp_path, None)
        assert "pillar" not in state

    def test_the_registry_reads_the_key_the_producer_writes(self, tmp_path):
        """Pins the two halves against each other by name.

        The producer writes `pillar` at the top level of the state file and
        the registry reads exactly that; a rename on either side is silent.
        """
        from src.publisher.product_registry import _read_pillar_from_state

        state_dir = tmp_path / "B0TEST0001" / "temp"
        state_dir.mkdir(parents=True)
        (state_dir / "pipeline_state.json").write_text(
            json.dumps({"pillar": "value", "script_template": "x"})
        )

        assert _read_pillar_from_state("B0TEST0001", tmp_path) == "value"


@pytest.mark.unit
class TestPillarSurvivesATruncatedResume:
    """Recording the pillar inside the script step is not enough.

    A resume that finds a completed step's artifact missing truncates the
    state to step keys only, dropping every top-level scalar, and then skips
    the steps it kept. So the step that would re-record the pillar never runs,
    and the registry reads a state file without one -- for a video whose
    script was written under it.

    Resolving it after the state load, where the CLI override is applied,
    happens on every run including that one.
    """

    def test_the_resolution_runs_after_the_state_load(self):
        """Pins the ordering by reading the source, because the alternative is
        to stand up the whole parallel orchestrator.

        The assertion is about position: a resolution placed before the load
        is overwritten by it, and one placed inside a step is skipped by it.
        """
        import inspect

        from src.video.producer import orchestration

        src = inspect.getsource(orchestration.create_video_for_product)
        load_at = src.index("_load_pipeline_state(ctx)")
        resolve_at = src.index('ctx.state["pillar"] = resolved_pillar')
        assert load_at < resolve_at
