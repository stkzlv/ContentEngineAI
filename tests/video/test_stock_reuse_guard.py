"""Stock clips used in recent renders stay out of the next (design 0013)."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.utils.render_choices_store import recent_stock_ids
from src.video.config import config
from src.video.config.visual_models import StockMediaSettings, StockReuseGuard
from src.video.render_choices import choices_from_context, record_render_choices
from src.video.stock_media import StockMediaFetcher


def _fetcher(enabled: bool, recent: dict[str, int] | None) -> StockMediaFetcher:
    settings = StockMediaSettings(
        pexels_api_key_env_var="NO_SUCH_KEY",
        stock_reuse_guard=StockReuseGuard(enabled=enabled),
    )
    fetcher = StockMediaFetcher(
        settings, {}, config.media_settings, recent_stock_ids=recent
    )
    fetcher.pexels_client = MagicMock()
    fetcher.pexels_client.search_photos.return_value = {
        "photos": [
            {"id": n, "src": {"original": f"https://x/{n}.jpg"}, "photographer": "a"}
            for n in range(1, 7)
        ]
    }
    return fetcher


async def _pick(fetcher: StockMediaFetcher, count: int) -> list[int]:
    items = await fetcher._search_and_select_pexels("desk lamp", "photos", count)
    return sorted(item["id"] for item in items)


@pytest.mark.req("REQ-VID-110")
@pytest.mark.asyncio
async def test_a_recently_used_candidate_is_excluded() -> None:
    recent = {"pexels:1": 0, "pexels:2": 3, "pexels:3": 9}

    assert await _pick(_fetcher(True, recent), 3) == [4, 5, 6]


@pytest.mark.req("REQ-VID-110")
@pytest.mark.asyncio
async def test_too_few_fresh_fill_from_the_least_recently_used(caplog) -> None:
    recent = {
        f"pexels:{n}": age for n, age in zip(range(1, 6), [0, 7, 2, 9, 4], strict=True)
    }

    caplog.set_level(logging.INFO, logger="src.video.stock_media")
    picked = await _pick(_fetcher(True, recent), 3)

    # 6 is fresh; then 4 (9 renders ago) and 2 (7 ago), not the newer ones.
    assert picked == [2, 4, 6]
    assert "reusing 2 least recently used" in caplog.text


@pytest.mark.asyncio
async def test_off_leaves_the_pool_as_today() -> None:
    recent = {f"pexels:{n}": 0 for n in range(1, 7)}

    assert len(await _pick(_fetcher(False, recent), 6)) == 6
    assert await _pick(_fetcher(True, None), 6) == [1, 2, 3, 4, 5, 6]


def _row(product: str, ids: list[str] | None) -> dict:
    row: dict = {"product_id": product}
    if ids is not None:
        row["stock_ids"] = ids
    return row


@pytest.mark.req("REQ-VID-110")
def test_recent_ids_carry_their_age_and_respect_the_window(tmp_path: Path) -> None:
    record_render_choices(tmp_path, _row("A", ["pexels:1", "pexels:2"]))
    record_render_choices(tmp_path, _row("B", ["pexels:2"]))
    record_render_choices(tmp_path, _row("C", None))
    # pexels:1 again, newer than A: its age is the newer use.
    record_render_choices(tmp_path, _row("D", ["pexels:3", "pexels:1"]))
    # A rerun of B moves it to the newest render and replaces its ids.
    record_render_choices(tmp_path, _row("B", ["pexels:4"]))

    assert recent_stock_ids(tmp_path, 10) == {
        "pexels:4": 0,
        "pexels:3": 1,
        "pexels:1": 1,
        "pexels:2": 3,
    }
    assert recent_stock_ids(tmp_path, 2) == {
        "pexels:4": 0,
        "pexels:3": 1,
        "pexels:1": 1,
    }


@pytest.mark.req("REQ-VID-110")
def test_the_record_survives_a_product_directory_cleanup(tmp_path: Path) -> None:
    product_dir = tmp_path / "B0X"
    product_dir.mkdir()
    record_render_choices(tmp_path, _row("B0X", ["pexels:7"]))

    shutil.rmtree(product_dir)

    assert recent_stock_ids(tmp_path, 30) == {"pexels:7": 0}


def _ctx(tmp_path: Path, visuals: str | None) -> SimpleNamespace:
    path = tmp_path / "gathered_visuals.json"
    if visuals is not None:
        path.write_text(visuals)
    return SimpleNamespace(
        state={},
        config=config,
        profile_name="p",
        profile=SimpleNamespace(video_assembly_mode="sequential"),
        run_paths={
            "music_info_file": None,
            "run_root": tmp_path / "B0X",
            "gathered_visuals_file": path,
        },
        script="x",
    )


@pytest.mark.req("REQ-VID-110")
def test_each_render_records_the_stock_ids_it_gathered(tmp_path: Path) -> None:
    visuals = json.dumps(
        {
            "stock_media": [
                {"path": "a.jpg", "stock_id": "pexels:9"},
                {"path": "b.mp4", "stock_id": "pexels:4"},
                {"path": "c.jpg"},
            ]
        }
    )

    assert choices_from_context(_ctx(tmp_path, visuals))["stock_ids"] == [
        "pexels:4",
        "pexels:9",
    ]


@pytest.mark.parametrize("visuals", [None, "{not json", "[1, 2]"])
def test_a_missing_or_unreadable_visuals_file_records_no_ids(
    tmp_path: Path, visuals: str | None
) -> None:
    assert choices_from_context(_ctx(tmp_path, visuals))["stock_ids"] == []


@pytest.mark.req("REQ-VID-110")
def test_the_shipped_config_keeps_the_guard_off() -> None:
    guard = config.stock_media_settings.stock_reuse_guard

    assert guard.enabled is False
    assert guard.window == 30


@pytest.mark.req("REQ-VID-110")
def test_the_producer_reads_recent_ids_only_when_the_guard_is_on(
    tmp_path: Path,
) -> None:
    from src.video.producer import steps

    record_render_choices(tmp_path, _row("A", ["pexels:5"]))
    cfg = config.model_copy(deep=True)
    ctx = SimpleNamespace(config=cfg, run_paths={"run_root": tmp_path / "B0NEW"})

    assert steps._recent_stock_ids(ctx) is None
    cfg.stock_media_settings.stock_reuse_guard.enabled = True
    assert steps._recent_stock_ids(ctx) == {"pexels:5": 0}


@pytest.mark.asyncio
async def test_downloaded_items_carry_their_stock_id(tmp_path: Path, monkeypatch):
    from src.video import stock_media

    async def fake_download(url, path, *args, **kwargs):
        Path(path).write_bytes(b"x")
        return True

    monkeypatch.setattr(stock_media, "download_file", fake_download)
    fetcher = _fetcher(False, None)

    items = await fetcher.fetch_and_download_stock(
        ["desk"], 2, 0, tmp_path, AsyncMock()
    )

    assert {item.stock_id for item in items} <= {f"pexels:{n}" for n in range(1, 7)}
    assert len(items) == 2 and all(item.stock_id for item in items)


@pytest.mark.req("REQ-VID-110")
@pytest.mark.asyncio
async def test_with_the_judge_a_fresh_weak_clip_never_beats_a_relevant_one(
    monkeypatch,
) -> None:
    """The judge scores the whole page; the guard only breaks ties (REQ-VID-108)."""
    from src.video import stock_relevance

    llm = config.llm_settings.model_copy(deep=True)
    llm.stock_relevance.enabled = True
    llm.stock_relevance.min_score = 2
    # ids 1-6 score 3, 3, 0, 0, 3, 2; 1 and 2 were used, 1 the more recently.
    monkeypatch.setattr(
        stock_relevance,
        "score_candidates",
        AsyncMock(return_value=[3, 3, 0, 0, 3, 2]),
    )
    fetcher = _fetcher(True, {"pexels:1": 0, "pexels:2": 5})
    fetcher.llm_settings = llm
    fetcher.secrets = {llm.api_key_env_var: "key"}

    items = await fetcher._search_and_select_pexels(
        "desk lamp", "photos", 3, script="A desk lamp.", session=AsyncMock()
    )

    # Fresh above the floor (5, then 6), then the older reused one (2);
    # never the fresh clips that scored 0.
    assert [item["id"] for item in items] == [5, 6, 2]
    assert [item["score"] for item in items] == [3, 2, 3]
