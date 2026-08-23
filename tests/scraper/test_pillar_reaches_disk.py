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


@pytest.mark.unit
class TestRegistryFindsThePillarAfterCleanup:
    """`pipeline_state.json` does not survive a normal run.

    A successful non-debug render deletes the `temp/` directory it lives in,
    and the registry is written after that. So reading only the state file
    filed every non-debug render unlabelled.

    The fallback is `metadata.json`, not `data.json`. Both survive, but
    `data.json` records what the product was *scraped* under, so on a run with
    `--pillar` it would file the row under an arm the render never used --
    turning an empty cell into a confidently wrong one, which for an
    arm-comparison registry is the worse failure.
    """

    def _write(self, tmp_path, *, state=None, metadata=None, data=None):
        root = tmp_path / "B0TEST0001"
        root.mkdir(parents=True, exist_ok=True)
        if state is not None:
            (root / "temp").mkdir(exist_ok=True)
            (root / "temp" / "pipeline_state.json").write_text(json.dumps(state))
        if metadata is not None:
            (root / "metadata.json").write_text(json.dumps(metadata))
        if data is not None:
            (root / "data.json").write_text(json.dumps(data))

    def test_the_state_file_wins_when_it_is_there(self, tmp_path):
        """It records the run's own decision, so a --pillar override wins."""
        from src.publisher.product_registry import _read_pillar_from_state

        self._write(
            tmp_path,
            state={"pillar": "utility"},
            metadata={"pillar": "utility"},
            data={"pillar": "value"},
        )
        assert _read_pillar_from_state("B0TEST0001", tmp_path) == "utility"

    def test_metadata_answers_once_temp_is_gone(self, tmp_path):
        """The normal case: cleanup ran, and metadata.json is what is left."""
        from src.publisher.product_registry import _read_pillar_from_state

        self._write(tmp_path, metadata={"pillar": "utility"}, data={"pillar": "value"})
        assert _read_pillar_from_state("B0TEST0001", tmp_path) == "utility"

    def test_the_scraped_pillar_is_never_reported_as_the_rendered_one(self, tmp_path):
        """A run under `--pillar utility` on a product scraped as `value`.

        With temp gone and no metadata, the honest answer is that we do not
        know -- not `value`, which the render did not use.
        """
        from src.publisher.product_registry import _read_pillar_from_state

        self._write(tmp_path, data={"pillar": "value"})
        assert _read_pillar_from_state("B0TEST0001", tmp_path) == ""

    def test_nothing_at_all_gives_nothing(self, tmp_path):
        from src.publisher.product_registry import _read_pillar_from_state

        assert _read_pillar_from_state("B0TEST0001", tmp_path) == ""

    def test_a_null_pillar_is_not_a_value(self, tmp_path):
        """A render with no pillar writes null; that is not a label."""
        from src.publisher.product_registry import _read_pillar_from_state

        self._write(tmp_path, metadata={"pillar": None})
        assert _read_pillar_from_state("B0TEST0001", tmp_path) == ""


@pytest.mark.unit
class TestTheRenderedPillarIsRecordedWhereItSurvives:
    """The producer has to write it somewhere outside `temp/`."""

    async def _write_metadata(self, monkeypatch, tmp_path, state):
        from src.video.producer import steps as steps_mod

        async def _fake_description(*a, **kw):
            return "A description."

        monkeypatch.setattr(steps_mod, "generate_ai_description", _fake_description)

        ctx = MagicMock()
        ctx.state = state
        ctx.product = ProductData(
            title="A product",
            price="$10",
            url="https://www.amazon.com/dp/B0TEST0001",
            platform=None,
            asin="B0TEST0001",
            pillar="value",
        )
        ctx.run_paths = {
            "run_root": tmp_path,
            "description_file": tmp_path / "description.txt",
        }
        ctx.debug_mode = False
        await steps_mod._generate_unified_metadata(ctx)
        return json.loads((tmp_path / "metadata.json").read_text())

    async def test_metadata_json_carries_the_rendered_pillar(
        self, monkeypatch, tmp_path
    ):
        """The run's resolved value, which an override may have set."""
        written = await self._write_metadata(
            monkeypatch, tmp_path, {"pillar": "utility"}
        )
        assert written["pillar"] == "utility"

    async def test_no_pillar_writes_null_not_the_scraped_one(
        self, monkeypatch, tmp_path
    ):
        """The product record says `value`; the run resolved nothing.

        Writing the product's own value here would reintroduce the confusion
        this file exists to remove.
        """
        written = await self._write_metadata(monkeypatch, tmp_path, {})
        assert written["pillar"] is None


@pytest.mark.unit
class TestTheRegistryReadsOptimizedMetadataToo:
    """Optimized metadata mode writes no `metadata.json` at all.

    It writes one `metadata_<platform>.json` per platform instead, so a
    registry that looked only for the unified file filed every render in that
    mode unlabelled -- on the same non-debug path where the state file is
    already gone.
    """

    def test_a_platform_metadata_file_answers(self, tmp_path):
        from src.publisher.product_registry import _read_pillar_from_state

        root = tmp_path / "B0TEST0001"
        root.mkdir(parents=True)
        (root / "metadata_youtube.json").write_text(json.dumps({"pillar": "utility"}))

        assert _read_pillar_from_state("B0TEST0001", tmp_path) == "utility"

    def test_the_unified_file_still_wins(self, tmp_path):
        from src.publisher.product_registry import _read_pillar_from_state

        root = tmp_path / "B0TEST0001"
        root.mkdir(parents=True)
        (root / "metadata.json").write_text(json.dumps({"pillar": "value"}))
        (root / "metadata_youtube.json").write_text(json.dumps({"pillar": "utility"}))

        assert _read_pillar_from_state("B0TEST0001", tmp_path) == "value"


@pytest.mark.unit
class TestAReRenderRefreshesTheRecordedPillar:
    """`metadata.json` is reused when it exists, and the registry reads it.

    So a re-render under a different `--pillar` would otherwise leave the
    previous run's arm on a video whose script was written for another one --
    the confidently-wrong label this design refuses.
    """

    async def _reuse(self, tmp_path, existing: dict, state: dict):
        from src.video.producer import steps as steps_mod

        (tmp_path / "metadata.json").write_text(json.dumps(existing))
        ctx = MagicMock()
        ctx.state = state
        ctx.product = ProductData(
            title="A product",
            price="$10",
            url="https://www.amazon.com/dp/B0TEST0001",
            platform=None,
            asin="B0TEST0001",
            pillar="value",
        )
        ctx.run_paths = {
            "run_root": tmp_path,
            "description_file": tmp_path / "description.txt",
        }
        ctx.debug_mode = False
        steps_mod._check_existing_metadata(ctx)
        return json.loads((tmp_path / "metadata.json").read_text())

    async def test_a_stale_pillar_is_replaced(self, tmp_path):
        written = await self._reuse(
            tmp_path,
            {"description": "old", "pillar": "value"},
            {"pillar": "utility"},
        )
        assert written["pillar"] == "utility"

    async def test_a_file_predating_the_key_is_backfilled(self, tmp_path):
        written = await self._reuse(
            tmp_path, {"description": "old"}, {"pillar": "utility"}
        )
        assert written["pillar"] == "utility"

    def test_the_optimized_writer_records_the_pillar(self, tmp_path):
        """The registry reading these files is only half of it.

        In optimized mode these are the only files written outside `temp/`,
        so if the writer omits the key there is nothing for the registry to
        find.
        """
        from src.ai.platform_metadata.utilities import save_metadata_to_file
        from src.publisher.models import Platform

        class _Meta:
            platform = Platform.YOUTUBE.value

            def to_dict(self):
                return {"title": "T", "description": "D", "hashtags": ["tech"]}

        out = tmp_path / "B0TEST0001" / "metadata_youtube.json"
        save_metadata_to_file(_Meta(), out, pillar="utility")

        assert json.loads(out.read_text())["pillar"] == "utility"
