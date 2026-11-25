"""The publisher keeps a series claim queue in step (REQ-PUB-168)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.publisher import series_queue
from src.utils.claim_queue import QUEUE_COLUMNS, set_published

pytestmark = pytest.mark.req("REQ-PUB-168")
CLAIM = "A magnet wipes a phone"


def _queue(path: Path, status: str = "rendered") -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=QUEUE_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {k: "" for k in QUEUE_COLUMNS} | {"claim": CLAIM, "status": status}
        )
    return path


def _row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        return next(csv.DictReader(f))


@pytest.fixture
def outputs(tmp_path):
    queue = _queue(tmp_path / "q.csv")
    (tmp_path / "B0TEST0168").mkdir()
    (tmp_path / "B0TEST0168" / "series.json").write_text(
        json.dumps({"claim": CLAIM, "queue_file": str(queue)})
    )
    return tmp_path, queue


def test_scheduled_records_the_post_id_and_live_is_never_moved_back(tmp_path):
    queue = _queue(tmp_path / "q.csv")
    set_published(queue, CLAIM, "scheduled", "post-1")
    assert (_row(queue)["status"], _row(queue)["post_id"]) == ("scheduled", "post-1")
    set_published(queue, CLAIM, "live")
    set_published(queue, CLAIM, "scheduled", "post-2")
    assert _row(queue)["status"] == "live"


def test_recording_a_publish_marks_the_series_row_scheduled(outputs):
    from src.publisher.tracking import record_publish

    root, queue = outputs
    record_publish("B0TEST0168", "youtube", "post-9", root)
    assert (_row(queue)["status"], _row(queue)["post_id"]) == ("scheduled", "post-9")


def test_a_record_without_a_series_changes_nothing(outputs):
    from src.publisher.tracking import record_publish

    root, queue = outputs
    (root / "B0TEST0999").mkdir()
    record_publish("B0TEST0999", "youtube", "post-x", root)
    assert _row(queue)["status"] == "rendered"


def test_a_published_post_marks_its_row_live(outputs):
    from src.publisher.tracking import record_publish

    root, queue = outputs
    record_publish("B0TEST0168", "youtube", "post-9", root)
    marked = series_queue.mark_live(
        [
            {"id": "post-9", "status": "published"},
            {"id": "other", "status": "scheduled"},
        ],
        root,
    )
    assert marked == 1 and _row(queue)["status"] == "live"


@pytest.mark.asyncio
async def test_the_sweep_makes_no_extra_call_without_a_series(tmp_path):
    from src.publisher import partial_post_sweep

    publisher = MagicMock()
    publisher.list_posts = AsyncMock(return_value=[])
    await partial_post_sweep._mark_live_series(publisher, tmp_path)
    publisher.list_posts.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_sweep_marks_live_when_a_series_is_configured(outputs):
    from src.publisher import partial_post_sweep
    from src.publisher.tracking import record_publish

    root, queue = outputs
    record_publish("B0TEST0168", "youtube", "post-9", root)
    publisher = MagicMock()
    publisher.list_posts = AsyncMock(
        return_value=[{"id": "post-9", "status": "published"}]
    )
    await partial_post_sweep._mark_live_series(publisher, root)
    assert _row(queue)["status"] == "live"


@pytest.mark.asyncio
async def test_the_delivery_sweep_runs_the_live_marks(tmp_path):
    from src.publisher import partial_post_sweep
    from src.publisher.models import DeliverySweepConfig

    publisher = MagicMock()
    with (
        patch.object(
            partial_post_sweep, "sweep_partial_posts", AsyncMock(return_value=[])
        ),
        patch.object(partial_post_sweep, "_mark_live_series", AsyncMock()) as live,
    ):
        await partial_post_sweep.run_delivery_sweep(
            publisher, DeliverySweepConfig(enabled=True, limit=10), tmp_path
        )
    live.assert_awaited_once_with(publisher, tmp_path)


def test_a_bad_series_record_changes_nothing(outputs):
    from src.publisher.tracking import record_publish

    root, queue = outputs
    (root / "B0TEST0168" / "series.json").write_text(json.dumps(["not a record"]))
    record_publish("B0TEST0168", "youtube", "post-8", root)
    assert _row(queue)["post_id"] == ""


def test_the_producer_leaves_the_series_record(tmp_path):
    from types import SimpleNamespace

    from src.video.config.series_models import SeriesConfig
    from src.video.series import write_series_record

    queue = tmp_path / "q.csv"
    write_series_record(
        SeriesConfig(queue_file=queue), SimpleNamespace(claim=CLAIM), tmp_path
    )
    assert series_queue.series_record(".", tmp_path) == (queue, CLAIM)


def test_live_survives_cleanup_and_counts_a_partial_post(outputs):
    """Cleanup removes the render directory after publishing; the live mark
    reads publish history, and a partial post is live somewhere.
    """
    import shutil

    from src.publisher.tracking import record_publish

    root, queue = outputs
    record_publish("B0TEST0168", "youtube", "post-9", root)
    shutil.rmtree(root / "B0TEST0168")
    assert series_queue.has_series_posts(root)
    assert series_queue.mark_live([{"id": "post-9", "status": "partial"}], root) == 1
    assert _row(queue)["status"] == "live"


def test_the_series_record_stores_an_absolute_queue_path(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from src.video.config.series_models import SeriesConfig
    from src.video.series import write_series_record

    monkeypatch.chdir(tmp_path)
    write_series_record(
        SeriesConfig(queue_file="q.csv"), SimpleNamespace(claim=CLAIM), tmp_path
    )
    found = series_queue.series_record(".", tmp_path)
    assert found is not None and found[0].is_absolute()
