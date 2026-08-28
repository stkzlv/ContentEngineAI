"""A run directory should say what it searched for, not only what it got.

Footage that does not match the narration has two causes needing opposite
fixes: the phrase named the wrong subject (a prompt problem, which #299 was),
or the library answered a good phrase loosely (a retrieval problem, where a
relevance filter or a second provider is the lever). Without attribution the
two are indistinguishable from the output, and diagnosing #299 meant
re-running the phrase generator to infer what a past render *would* have done.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.video.producer.state import load_visuals_info, save_visuals_info
from src.video.producer.steps import _fetch_stock_across_queries
from src.video.stock_media import StockMediaInfo


def _item(name: str) -> StockMediaInfo:
    return StockMediaInfo(
        source="Pexels",
        type="image",
        url=f"https://example.test/{name}",
        author="Someone",
        path=Path(f"stock/{name}.jpg"),
    )


class TestEachItemCarriesTheQueryItCameFrom:
    @pytest.mark.asyncio
    async def test_the_query_is_stamped_on_the_pooled_items(self, tmp_path):
        fetcher = AsyncMock()
        fetcher.fetch_and_download_stock = AsyncMock(
            side_effect=[[_item("a")], [_item("b")]]
        )

        pooled = await _fetch_stock_across_queries(
            fetcher,
            [["wifi", "router", "on", "a", "desk"], ["hand", "typing", "password"]],
            image_count=2,
            video_count=0,
            assets_dir=tmp_path,
            session=None,
        )

        assert [i.query for i in pooled] == [
            "wifi router on a desk",
            "hand typing password",
        ]

    @pytest.mark.asyncio
    async def test_a_deduped_item_keeps_the_query_that_found_it_first(self, tmp_path):
        """Two searches can return the same file; it is downloaded once."""
        shared = _item("same")
        fetcher = AsyncMock()
        fetcher.fetch_and_download_stock = AsyncMock(
            side_effect=[[shared], [_item("same")]]
        )

        pooled = await _fetch_stock_across_queries(
            fetcher,
            [["first", "query", "here"], ["second", "query", "here"]],
            image_count=2,
            video_count=0,
            assets_dir=tmp_path,
            session=None,
        )

        assert len(pooled) == 1
        assert pooled[0].query == "first query here"


class TestTheQueriesThemselvesArePersisted:
    """Item attribution cannot express a search that found nothing.

    From the item list, a phrase that returned no footage looks identical to
    a phrase that was never issued -- and the empty one is the more
    diagnostic of the two.
    """

    def test_the_run_directory_records_every_query(self, tmp_path):
        run_paths = {"gathered_visuals_file": tmp_path / "gathered_visuals.json"}

        save_visuals_info(
            [],
            [],
            [_item("a")],
            run_paths,
            search_queries=["wifi router on a desk", "a query that found nothing"],
        )

        written = json.loads(run_paths["gathered_visuals_file"].read_text())
        assert written["search_queries"] == [
            "wifi router on a desk",
            "a query that found nothing",
        ]

    def test_an_older_file_without_the_key_still_loads(self, tmp_path):
        """Rendered before this shipped; a strict read would break resume."""
        path = tmp_path / "gathered_visuals.json"
        path.write_text(
            json.dumps(
                {
                    "scraped_images": [],
                    "scraped_videos": [],
                    "stock_media": [
                        {
                            "source": "Pexels",
                            "type": "image",
                            "url": "u",
                            "author": "a",
                            "path": "stock/x.jpg",
                            "duration": None,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        _, _, stock = load_visuals_info(path)

        assert len(stock) == 1
        assert stock[0].query is None

    def test_a_file_written_now_round_trips_the_query(self, tmp_path):
        run_paths = {"gathered_visuals_file": tmp_path / "gathered_visuals.json"}
        item = _item("a")
        item.query = "wifi router on a desk"

        save_visuals_info([], [], [item], run_paths, search_queries=[item.query])
        _, _, stock = load_visuals_info(run_paths["gathered_visuals_file"])

        assert stock[0].query == "wifi router on a desk"


def test_the_gather_step_passes_the_queries_it_issued():
    """The serialiser accepting them is not the guard; the caller is.

    Dropping `search_queries=` from the call site leaves every test above
    green, because they call `save_visuals_info` directly.
    """
    import ast

    tree = ast.parse(Path("src/video/producer/steps.py").read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "save_visuals_info"
    ]

    assert calls, "no save_visuals_info call site found"
    assert all(
        any(kw.arg == "search_queries" for kw in call.keywords) for call in calls
    ), (
        "a save_visuals_info call omits search_queries, so the run directory "
        "records what was gathered but not what was asked for"
    )
