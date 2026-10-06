"""Renders wait for free memory and report their peak.

A render started with 1.5 GB available and swap full tipped the machine into a
kernel OOM that killed the user's browser. Each render now waits for its
budget and gives up after a bounded wait; the lowpri scope makes the render
the OOM killer's first choice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.utils import memory_guard
from src.utils.memory_guard import (
    InsufficientMemoryError,
    MemoryState,
    log_peak,
    read_state,
    shortfall,
    wait_for_memory,
)
from src.video.config.core_models import MemoryGuardSettings

GB_KB = 1024 * 1024
SETTINGS = MemoryGuardSettings(
    min_available_gb=3, min_swap_free_gb=1, wait_sec=60, poll_sec=5
)
OK = MemoryState(8.0, 11.0, 6.0, 0.0)
SHORT = MemoryState(1.5, 11.0, 0.0, 30.0)


@pytest.fixture(autouse=True)
def _guard_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(memory_guard.SKIP_ENV, raising=False)


def _meminfo(tmp_path: Path, available_gb: float, swap_free_gb: float) -> Path:
    path = tmp_path / "meminfo"
    path.write_text(
        f"MemTotal:       {14 * GB_KB} kB\n"
        f"MemAvailable:   {int(available_gb * GB_KB)} kB\n"
        f"SwapTotal:      {11 * GB_KB} kB\n"
        f"SwapFree:       {int(swap_free_gb * GB_KB)} kB\n"
    )
    return path


def test_the_state_comes_from_meminfo_and_pressure(tmp_path: Path) -> None:
    pressure = tmp_path / "pressure"
    pressure.write_text(
        "some avg10=1.00 avg60=0.5 avg300=0.1 total=1\n"
        "full avg10=12.50 avg60=0.5 avg300=0.1 total=1\n"
    )
    state = read_state(_meminfo(tmp_path, 1.5, 0.25), pressure)

    assert state == MemoryState(1.5, 11.0, 0.25, 12.5)


def test_missing_kernel_files_read_as_unknown(tmp_path: Path) -> None:
    assert read_state(tmp_path / "none", tmp_path / "none") is None
    state = read_state(_meminfo(tmp_path, 4, 4), tmp_path / "none")
    assert state is not None and state.pressure_full_avg10 is None


@pytest.mark.req("REQ-OPS-105")
@pytest.mark.parametrize(
    ("state", "reason"),
    [
        (OK, None),
        (MemoryState(2.0, 11.0, 6.0, 0.0), "2.0 GB available, under 3 GB"),
        # Swap counts only when RAM headroom is thin: pages stay swapped
        # after RAM frees, and full swap beside plenty available is no risk.
        (MemoryState(8.0, 11.0, 0.0, 0.0), None),
        (
            MemoryState(3.2, 11.0, 0.5, 0.0),
            "3.2 GB available and 0.5 GB swap free, under 4 GB together",
        ),
        # A machine with no swap is not short of swap.
        (MemoryState(8.0, 0.0, 0.0, 0.0), None),
    ],
)
def test_shortfall(state: MemoryState, reason: str | None) -> None:
    assert shortfall(state, SETTINGS) == reason


@pytest.mark.req("REQ-OPS-105")
@pytest.mark.asyncio
async def test_a_short_machine_is_waited_on_then_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter(range(0, 1000, 10))
    monkeypatch.setattr(memory_guard.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(memory_guard, "read_state", lambda: SHORT)
    sleep = AsyncMock()
    monkeypatch.setattr(memory_guard.asyncio, "sleep", sleep)

    with pytest.raises(InsufficientMemoryError, match="1.5 GB available"):
        await wait_for_memory(SETTINGS, "B0X")

    # The clock reads 0 (the deadline, 60), then 10 to 50 sleep, 60 refuses.
    assert sleep.await_count == 5


@pytest.mark.req("REQ-OPS-105")
@pytest.mark.asyncio
async def test_the_render_starts_once_memory_frees(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    states = iter([SHORT, SHORT, OK])
    monkeypatch.setattr(memory_guard, "read_state", lambda: next(states))
    monkeypatch.setattr(memory_guard.asyncio, "sleep", AsyncMock())
    caplog.set_level("INFO", logger=memory_guard.logger.name)

    await wait_for_memory(SETTINGS, "B0X")

    assert "Waiting up to 60 s for memory before B0X" in caplog.text
    assert "Memory freed for B0X" in caplog.text


@pytest.mark.req("REQ-OPS-105")
@pytest.mark.asyncio
@pytest.mark.parametrize("skip", ["disabled", "env", "no proc"])
async def test_the_wait_is_skipped(monkeypatch: pytest.MonkeyPatch, skip: str) -> None:
    settings = SETTINGS.model_copy(update={"enabled": skip != "disabled"})
    if skip == "env":
        monkeypatch.setenv(memory_guard.SKIP_ENV, "1")
    monkeypatch.setattr(
        memory_guard, "read_state", lambda: None if skip == "no proc" else SHORT
    )

    await wait_for_memory(settings, "B0X")


@pytest.mark.req("REQ-OPS-106")
@pytest.mark.parametrize("cap", ["max", str(6 * 1024**3)])
def test_the_peak_is_logged_inside_a_capped_scope_only(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, cap: str
) -> None:
    """Outside a capped scope the cgroup is the terminal's, not the render's."""
    (tmp_path / "memory.peak").write_text(f"{int(1.5 * 1024**3)}\n")
    (tmp_path / "memory.max").write_text(f"{cap}\n")
    caplog.set_level("INFO", logger=memory_guard.logger.name)

    log_peak("B0X", tmp_path)

    logged = "Memory peak after B0X: 1.50 GB of a 6.0 GB cap" in caplog.text
    assert logged is (cap != "max")


def test_no_peak_file_logs_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log_peak("B0X", tmp_path)
    assert "Memory peak" not in caplog.text


@pytest.mark.req("REQ-OPS-105")
def test_the_bundled_config_turns_the_guard_on() -> None:
    from src.video.config import load_video_config_modular

    guard = load_video_config_modular().memory_guard
    # A stock render's tree peaks at 4.1-4.3 GB (docs/testing.md).
    assert guard.enabled and guard.min_available_gb >= 4.3


def _products(tmp_path: Path) -> Path:
    for asin in ("B0MEM00001", "B0MEM00002"):
        (tmp_path / asin).mkdir()
        (tmp_path / asin / "data.json").write_text(
            json.dumps(
                {
                    "asin": asin,
                    "title": asin,
                    "price": "$10",
                    "url": "https://example.com",
                    "platform": "amazon",
                }
            )
        )
    return tmp_path


def _refuse():
    return AsyncMock(side_effect=InsufficientMemoryError("Not starting: short"))


@pytest.mark.req("REQ-OPS-105")
@pytest.mark.asyncio
async def test_the_producer_stops_its_batch_when_memory_stays_short(
    tmp_path: Path,
) -> None:
    from src.video.config import load_video_config_modular

    config = load_video_config_modular()
    argv = [
        "producer",
        "--batch",
        "--batch-profile",
        "slideshow_images1",
        "--outputs-dir",
        str(_products(tmp_path)),
    ]
    with (
        patch.object(sys, "argv", argv),
        patch("src.video.producer.cli.load_video_config_modular", return_value=config),
        patch("src.video.producer.cli.setup_logging", return_value=Path("test.log")),
        patch("src.video.producer.cli.validate_config_and_exit_on_error"),
        patch("src.video.producer.cli.load_dotenv"),
        patch("os.getenv", return_value="dummy_key"),
        patch("shutil.which", return_value="/usr/bin/ffmpeg"),
        patch("src.video.producer.cli.wait_for_memory", _refuse()) as wait,
        patch(
            "src.video.producer.cli.create_video_for_product", new_callable=AsyncMock
        ) as create,
        patch("asyncio.sleep", return_value=None),
        patch("src.utils.connection_pool.close_global_pool", new_callable=AsyncMock),
    ):
        from src.video.producer.cli import main

        with pytest.raises(SystemExit) as stopped:
            await main()

    # Its own exit code, which the topics batch stops on.
    assert stopped.value.code == memory_guard.EXIT_NO_MEMORY
    assert wait.await_count == 1  # stopped after the first refusal
    create.assert_not_called()


@pytest.mark.req("REQ-OPS-105")
@pytest.mark.asyncio
async def test_the_global_batch_stops_production_when_memory_stays_short(
    tmp_path: Path,
) -> None:
    from src.pipeline.phases import production
    from src.scraper.amazon.models import ProductData

    products = [
        (tmp_path / a, ProductData(asin=a, title=a, price="", url="", platform="t"))
        for a in ("B0MEM00001", "B0MEM00002")
    ]
    batch = SimpleNamespace(
        outputs_dir=tmp_path,
        random_profile=False,
        profile="slideshow_images1",
        profile_pool=None,
        topic_profile_pool=None,
        debug=False,
        fail_fast=False,
    )
    with (
        patch.object(production, "wait_for_memory", _refuse()) as wait,
        patch(
            "src.video.producer.orchestration.create_video_for_product",
            new_callable=AsyncMock,
        ) as create,
    ):
        summary, videos = await production.run_production_phase(
            batch, products, lambda: None, []
        )

    assert wait.await_count == 1
    create.assert_not_called()
    assert videos == [] and summary.failed == 1
