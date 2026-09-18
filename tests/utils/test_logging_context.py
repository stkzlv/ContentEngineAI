"""Every file record carries the run and the product it belongs to.

The batch log interleaves products, so following one ASIN through a run
meant reading around it. The ids are context variables stamped on by a
filter: an awaited step logs under the product that awaited it, and a value
bound inside `log_context` is gone when the block ends. Timestamps carry the
local UTC offset, since the schedule is in one zone and the provider reports
another.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.utils import logging_setup
from src.utils.logging_setup import (
    PRODUCT_ID,
    RUN_ID,
    UNBOUND,
    ContextFilter,
    IsoFormatter,
    current_run_id,
    log_context,
    setup_debug_logging,
)
from src.utils.outputs_paths import get_project_root

REPO = get_project_root()
LINE = re.compile(
    r"^(?P<time>\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}[+-]\d\d:\d\d) - "
    r"(?P<run>\S+) - (?P<product>\S+) - (?P<logger>\S+) - (?P<level>[A-Z]+) - "
    r"(?P<where>\S+:\d+) - (?P<message>.*)$"
)


def _fields(line: str) -> dict[str, str]:
    m = LINE.match(line)
    assert m, line
    return m.groupdict()


def _record(message: str = "m") -> logging.LogRecord:
    record = logging.LogRecord("t", logging.INFO, __file__, 1, message, None, None)
    ContextFilter().filter(record)
    return record


def _fresh(fn, *args):
    """Run with both ids unbound, in a copied context so nothing leaks out.

    Copying alone is not enough: an entry-point setup in an earlier test of
    the same worker leaves a run id bound, and a copy inherits it.
    """

    def run():
        RUN_ID.set(UNBOUND)
        PRODUCT_ID.set(UNBOUND)
        return fn(*args)

    saved = logging_setup._process_run_id
    logging_setup._process_run_id = None
    try:
        return contextvars.copy_context().run(run)
    finally:
        logging_setup._process_run_id = saved


@pytest.fixture
def clean_root() -> Iterator[None]:
    yield
    root = logging.getLogger()
    for handler in root.handlers[:]:
        handler.close()
        root.removeHandler(handler)


class TestTheFilterStampsTheIds:
    def test_unbound_is_a_dash(self):
        def body():
            record = _record()
            assert record.run_id == UNBOUND == "-"
            assert record.product_id == UNBOUND

        _fresh(body)

    def test_bound_values_land_on_the_record(self):
        with log_context(run_id="run00001", product_id="B0TEST0001"):
            record = _record()
        assert record.run_id == "run00001"
        assert record.product_id == "B0TEST0001"


class TestLogContext:
    def test_the_previous_value_returns_when_the_block_ends(self):
        def body():
            with log_context(product_id="outer"):
                with log_context(product_id="inner"):
                    assert PRODUCT_ID.get() == "inner"
                assert PRODUCT_ID.get() == "outer"
            assert PRODUCT_ID.get() == UNBOUND

        _fresh(body)

    def test_it_restores_after_an_exception(self):
        def body():
            with pytest.raises(RuntimeError), log_context(product_id="B0X"):
                raise RuntimeError("boom")
            assert PRODUCT_ID.get() == UNBOUND

        _fresh(body)

    def test_binding_only_the_product_leaves_the_run_alone(self):
        with log_context(run_id="run00002"):
            with log_context(product_id="B0Y"):
                assert RUN_ID.get() == "run00002"
            assert RUN_ID.get() == "run00002"

    def test_an_awaited_step_logs_under_the_product_that_awaited_it(self):
        seen: dict[str, str] = {}

        async def step(name: str) -> None:
            await asyncio.sleep(0)
            seen[name] = PRODUCT_ID.get()

        async def product(name: str) -> None:
            with log_context(product_id=name):
                await step(name)

        async def main() -> None:
            await asyncio.gather(product("B0AAA00001"), product("B0BBB00002"))

        asyncio.run(main())
        assert seen == {"B0AAA00001": "B0AAA00001", "B0BBB00002": "B0BBB00002"}


class TestAWorkerThreadStillNamesTheRun:
    """An executor thread starts with an empty context."""

    def test_the_run_id_reaches_a_thread_and_the_product_needs_a_copied_context(
        self, tmp_path: Path, clean_root: None
    ):
        from concurrent.futures import ThreadPoolExecutor

        def body():
            setup_debug_logging(tmp_path / "t.log", mark_run=True, run_id="runthread")
            with log_context(product_id="B0THREAD01"), ThreadPoolExecutor() as pool:
                bare = pool.submit(_record, "bare").result()
                carried = pool.submit(contextvars.copy_context().run, _record).result()
            return bare, carried

        bare, carried = _fresh(body)
        assert (bare.run_id, bare.product_id) == ("runthread", UNBOUND)
        assert (carried.run_id, carried.product_id) == ("runthread", "B0THREAD01")


class TestCurrentRunId:
    def test_none_before_an_entry_point_binds_one(self):
        assert _fresh(current_run_id) is None

    def test_the_bound_id_otherwise(self):
        with log_context(run_id="run00003"):
            assert current_run_id() == "run00003"


class TestTheFileLine:
    def test_it_has_the_fields_in_order_with_an_offset_timestamp(
        self, tmp_path: Path, clean_root: None
    ):
        log_file = tmp_path / "t.log"

        def body():
            setup_debug_logging(log_file, component_name="Probe")
            with log_context(product_id="B0TEST0001"):
                logging.getLogger("src.probe").info("hello")
            for handler in logging.getLogger().handlers:
                handler.flush()

        _fresh(body)
        lines = log_file.read_text(encoding="utf-8").splitlines()
        marker, hello = lines[0], lines[-1]
        m = LINE.match(hello)
        assert m, hello
        assert m["product"] == "B0TEST0001"
        assert m["logger"] == "src.probe"
        assert m["message"] == "hello"
        assert re.fullmatch(r"[0-9a-f]{8}", m["run"])
        assert m["run"] in marker and "Probe run starting" in marker
        assert _fields(marker)["product"] == UNBOUND

    def test_the_marker_and_the_records_share_the_run_id(
        self, tmp_path: Path, clean_root: None
    ):
        log_file = tmp_path / "t.log"

        def body():
            setup_debug_logging(log_file, component_name="Probe", run_id="fixedrun")
            logging.getLogger("src.probe").info("after")
            for handler in logging.getLogger().handlers:
                handler.flush()
            return current_run_id()

        assert _fresh(body) == "fixedrun"
        text = log_file.read_text(encoding="utf-8")
        assert "(run fixedrun)" in text
        assert all(_fields(ln)["run"] == "fixedrun" for ln in text.splitlines())

    def test_no_marker_means_no_run_id_bound(self, tmp_path: Path, clean_root: None):
        def body():
            setup_debug_logging(tmp_path / "t.log", mark_run=False)
            return current_run_id()

        assert _fresh(body) is None

    def test_the_timestamp_is_local_with_an_offset(self):
        record = logging.LogRecord("t", logging.INFO, __file__, 1, "m", None, None)
        record.created = 1_700_000_000.5
        stamp = IsoFormatter().formatTime(record)
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.500[+-]\d\d:\d\d", stamp)


class TestEveryPerProductPathBindsTheProduct:
    """One id per product, bound where the product loop is."""

    @pytest.mark.parametrize(
        "rel",
        [
            "src/scraper/amazon/batch_controller.py",
            "src/pipeline/phases/production.py",
            "src/pipeline/phases/publishing.py",
            "src/publisher/batch.py",
            "src/publisher/late/cli.py",
            "src/publisher/cleanup.py",
            "src/video/producer/orchestration.py",
        ],
    )
    def test_the_loop_binds(self, rel: str):
        source = (REPO / rel).read_text(encoding="utf-8")
        assert "with log_context(product_id=" in source, rel

    def test_the_render_uses_the_entry_points_run_id(self):
        source = (REPO / "src/video/producer/orchestration.py").read_text()
        assert "run_id = current_run_id() or " in source
