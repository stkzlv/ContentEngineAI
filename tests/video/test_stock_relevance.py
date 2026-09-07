# ruff: noqa: S311
"""The stock relevance judge (#341).

Every text signal for choosing stock footage was measured and refuted on
#307; the thumbnail judged by the multimodal model is what separates a
decorative fan from a cooling one. Measured on #341, fits are common but
provider rank is noise for a concrete subject and scarce for an abstract
one, so the whole pool is scored and the best taken, with random choice
inside a score so renders still vary.

The invariant that matters most: nothing here can lose a render. A failed
judgement is an unknown score, a pool with no known scores keeps the random
sample, and a floor short of the count is filled from below it.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.video.config import config
from src.video.config.llm_settings import StockRelevanceConfig
from src.video.stock_media import StockMediaFetcher, StockMediaInfo
from src.video.stock_relevance import (
    UNKNOWN,
    parse_score,
    score_candidates,
    select_by_relevance,
)


def _cands(n: int) -> list[dict]:
    return [
        {"id": i, "url": f"http://x/{i}.jpg", "thumbnail": f"http://x/t{i}.jpg"}
        for i in range(n)
    ]


@pytest.mark.unit
class TestParsingTheAnswer:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ('{"score": 2}', 2),
            ('```json\n{"score": 3}\n```', 3),
            ("score: 1", 1),
            ('{"score": 7}', None),
            ("", None),
            (None, None),
            ("no idea", None),
        ],
    )
    def test_reads_the_digit_or_gives_up(self, text, expected) -> None:
        assert parse_score(text) == expected


@pytest.mark.unit
class TestSelection:
    def test_takes_the_best_scores_first(self) -> None:
        cands = _cands(6)
        chosen = select_by_relevance(cands, [0, 3, 1, 2, 3, 0], 3, 2, random.Random(1))
        assert chosen is not None
        assert sorted(c["score"] for c in chosen) == [2, 3, 3]

    def test_random_within_a_score_so_renders_vary(self) -> None:
        cands = _cands(8)
        scores = [3] * 8
        picks = {
            tuple(
                c["id"]
                for c in (
                    select_by_relevance(cands, scores, 2, 2, random.Random(seed)) or []
                )
            )
            for seed in range(20)
        }
        assert len(picks) > 1

    def test_fills_from_below_the_floor_rather_than_shortening(self, caplog) -> None:
        """A stock shortfall skips the render; a loosely related shot does not."""
        cands = _cands(5)
        with caplog.at_level("WARNING"):
            chosen = select_by_relevance(cands, [1, 0, 2, 1, 0], 4, 2, random.Random(3))
        assert chosen is not None
        assert len(chosen) == 4
        assert [c["score"] for c in chosen] == [2, 1, 1, 0]
        assert "filling 3 from below the floor" in caplog.text

    def test_an_unknown_score_is_the_last_resort(self) -> None:
        cands = _cands(3)
        chosen = select_by_relevance(cands, [None, 0, None], 2, 2, random.Random(0))
        assert chosen is not None
        assert chosen[0]["score"] == 0
        assert chosen[1]["score"] is None
        assert UNKNOWN < 0

    def test_no_known_score_means_keep_the_callers_sample(self) -> None:
        assert select_by_relevance(_cands(3), [None, None, None], 2, 2) is None


class _FakeResponse:
    def __init__(self, status=200, data=b"jpg"):
        self.status = status
        self.content_type = "image/jpeg"
        self._data = data

    async def read(self):
        return self._data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FakeSession:
    def __init__(self, failing: set[str] | None = None):
        self.failing = failing or set()

    def get(self, url):
        if url in self.failing:
            return _FakeResponse(status=404)
        return _FakeResponse()


def _fake_client(answers):
    """A `genai.Client` whose model answers in order; an Exception raises."""
    answers = list(answers)

    async def generate_content(**kwargs):
        text = answers.pop(0)
        if isinstance(text, Exception):
            raise text
        return MagicMock(text=text)

    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(side_effect=generate_content)
    client.aio.aclose = AsyncMock()
    return client


@pytest.mark.unit
class TestScoringTheThumbnails:
    @pytest.mark.asyncio
    async def test_one_score_per_candidate_in_order(self) -> None:
        client = _fake_client(['{"score": 3}', '{"score": 0}', "2"])
        with patch("google.genai.Client", return_value=client):
            scores = await score_candidates(
                _cands(3),
                "laptop fan clogged dust",
                "dust in the fins",
                api_key="k",
                settings=StockRelevanceConfig(enabled=True, concurrency=1),
                session=_FakeSession(),
            )
        assert scores == [3, 0, 2]
        assert client.aio.models.generate_content.await_count == 3

    @pytest.mark.asyncio
    async def test_a_failed_judgement_is_unknown_not_an_error(self) -> None:
        from google.genai import errors

        client = _fake_client(
            ['{"score": 2}', errors.APIError(429, {"message": "quota"}), '{"score": 1}']
        )
        cands = _cands(4)  # the fourth has no reachable thumbnail
        with patch("google.genai.Client", return_value=client):
            scores = await score_candidates(
                cands,
                "q",
                "s",
                api_key="k",
                settings=StockRelevanceConfig(enabled=True, concurrency=1),
                session=_FakeSession(failing={cands[3]["thumbnail"]}),
            )
        assert scores == [2, None, 1, None]

    @pytest.mark.asyncio
    async def test_a_client_that_cannot_be_built_is_unknown_scores(self) -> None:
        """The constructor alone raises under a SOCKS proxy without socksio;
        the judge is then unavailable and the caller keeps its sample.
        """
        with patch("google.genai.Client", side_effect=ImportError("socksio")):
            scores = await score_candidates(
                _cands(3),
                "q",
                "s",
                api_key="k",
                settings=StockRelevanceConfig(enabled=True),
                session=_FakeSession(),
            )
        assert scores == [None, None, None]

    @pytest.mark.asyncio
    async def test_past_max_candidates_is_unknown_not_fetched(self) -> None:
        client = _fake_client(['{"score": 2}', '{"score": 2}'])
        with patch("google.genai.Client", return_value=client):
            scores = await score_candidates(
                _cands(5),
                "q",
                "s",
                api_key="k",
                settings=StockRelevanceConfig(enabled=True, max_candidates=2),
                session=_FakeSession(),
            )
        assert scores == [2, 2, None, None, None]
        assert client.aio.models.generate_content.await_count == 2


def _fetcher(llm=None, secrets=None) -> StockMediaFetcher:
    fetcher = StockMediaFetcher(
        settings=config.stock_media_settings,
        secrets=secrets
        if secrets is not None
        else {"PEXELS_API_KEY": "p", "GEMINI_API_KEY": "g"},
        media_settings=config.media_settings,
        api_settings=config.api_settings,
        llm_settings=llm,
    )
    fetcher.pexels_client = MagicMock()
    return fetcher


def _stub_photos(fetcher: StockMediaFetcher, n: int) -> None:
    client = MagicMock()
    client.search_photos.return_value = _photos(n)
    fetcher.pexels_client = client


def _photos(n: int) -> dict:
    return {
        "photos": [
            {
                "id": i,
                "src": {"original": f"http://x/{i}.jpg", "tiny": f"http://x/t{i}.jpg"},
                "photographer": "a",
            }
            for i in range(n)
        ]
    }


@pytest.mark.unit
class TestTheFetcherUsesIt:
    @pytest.mark.asyncio
    async def test_the_thumbnail_is_carried_from_the_search_result(self) -> None:
        fetcher = _fetcher()
        _stub_photos(fetcher, 3)
        items = await fetcher._search_and_select_pexels("q", "photos", 3)
        assert {i["thumbnail"] for i in items} == {
            f"http://x/t{i}.jpg" for i in range(3)
        }

    @pytest.mark.asyncio
    async def test_off_means_the_random_sample_and_no_judge(self) -> None:
        llm = config.llm_settings.model_copy(
            update={"stock_relevance": StockRelevanceConfig(enabled=False)}
        )
        fetcher = _fetcher(llm)
        _stub_photos(fetcher, 10)
        with patch(
            "src.video.stock_relevance.score_candidates", new=AsyncMock()
        ) as scorer:
            items = await fetcher._search_and_select_pexels(
                "q", "photos", 2, script="s", session=MagicMock()
            )
        assert len(items) == 2
        scorer.assert_not_awaited()
        assert all("score" not in i for i in items)

    @pytest.mark.asyncio
    async def test_no_script_means_the_random_sample(self) -> None:
        llm = config.llm_settings.model_copy(
            update={"stock_relevance": StockRelevanceConfig(enabled=True)}
        )
        fetcher = _fetcher(llm)
        _stub_photos(fetcher, 10)
        with patch(
            "src.video.stock_relevance.score_candidates", new=AsyncMock()
        ) as scorer:
            items = await fetcher._search_and_select_pexels(
                "q", "photos", 2, script=None, session=MagicMock()
            )
        assert len(items) == 2
        scorer.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_api_key_means_the_random_sample(self) -> None:
        llm = config.llm_settings.model_copy(
            update={"stock_relevance": StockRelevanceConfig(enabled=True)}
        )
        fetcher = _fetcher(llm, secrets={"PEXELS_API_KEY": "p"})
        _stub_photos(fetcher, 10)
        with patch(
            "src.video.stock_relevance.score_candidates", new=AsyncMock()
        ) as scorer:
            items = await fetcher._search_and_select_pexels(
                "q", "photos", 2, script="s", session=MagicMock()
            )
        assert len(items) == 2
        scorer.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_judged_search_asks_for_the_whole_page(self) -> None:
        """The first cut left `per_page` at twice the count, so the judge saw
        six thumbnails per phrase and chose the best three of them -- a slice
        that by measurement holds about none of the fits (review finding).
        """
        llm = config.llm_settings.model_copy(
            update={"stock_relevance": StockRelevanceConfig(enabled=True)}
        )
        fetcher = _fetcher(llm)
        _stub_photos(fetcher, 3)
        with patch(
            "src.video.stock_relevance.score_candidates",
            new=AsyncMock(return_value=[1, 2, 3]),
        ):
            await fetcher._search_and_select_pexels(
                "q", "photos", 2, script="s", session=MagicMock()
            )
        client = fetcher.pexels_client
        assert client is not None
        assert client.search_photos.call_args.kwargs["per_page"] == 80

    @pytest.mark.asyncio
    async def test_a_low_cap_never_shrinks_the_request_below_the_unjudged_one(
        self,
    ) -> None:
        """`max_candidates: 4` with eight images asked for returned four and
        skipped the render on the media floor (review finding). The judged
        request is at least the unjudged one.
        """
        llm = config.llm_settings.model_copy(
            update={
                "stock_relevance": StockRelevanceConfig(enabled=True, max_candidates=4)
            }
        )
        fetcher = _fetcher(llm)
        _stub_photos(fetcher, 16)
        with patch(
            "src.video.stock_relevance.score_candidates",
            new=AsyncMock(return_value=[2] * 16),
        ):
            items = await fetcher._search_and_select_pexels(
                "q", "photos", 8, script="s", session=MagicMock()
            )
        client = fetcher.pexels_client
        assert client is not None
        assert client.search_photos.call_args.kwargs["per_page"] == 16
        assert len(items) == 8

    @pytest.mark.asyncio
    async def test_an_unjudged_search_keeps_its_page_size(self) -> None:
        fetcher = _fetcher()
        _stub_photos(fetcher, 3)
        await fetcher._search_and_select_pexels("q", "photos", 2)
        client = fetcher.pexels_client
        assert client is not None
        assert client.search_photos.call_args.kwargs["per_page"] == 4

    @pytest.mark.asyncio
    async def test_on_it_takes_the_best_and_stamps_the_score(self) -> None:
        llm = config.llm_settings.model_copy(
            update={"stock_relevance": StockRelevanceConfig(enabled=True)}
        )
        fetcher = _fetcher(llm)
        _stub_photos(fetcher, 6)
        scores = [0, 0, 3, 1, 2, 0]
        with patch(
            "src.video.stock_relevance.score_candidates",
            new=AsyncMock(return_value=scores),
        ):
            items = await fetcher._search_and_select_pexels(
                "q", "photos", 2, script="s", session=MagicMock()
            )
        assert sorted(i["score"] for i in items) == [2, 3]
        assert {i["id"] for i in items} == {2, 4}

    @pytest.mark.asyncio
    async def test_a_dead_judge_keeps_the_random_sample(self) -> None:
        llm = config.llm_settings.model_copy(
            update={"stock_relevance": StockRelevanceConfig(enabled=True)}
        )
        fetcher = _fetcher(llm)
        _stub_photos(fetcher, 6)
        with patch(
            "src.video.stock_relevance.score_candidates",
            new=AsyncMock(return_value=[None] * 6),
        ):
            items = await fetcher._search_and_select_pexels(
                "q", "photos", 2, script="s", session=MagicMock()
            )
        assert len(items) == 2

    @pytest.mark.asyncio
    async def test_the_score_reaches_the_downloaded_item(self, tmp_path) -> None:
        fetcher = _fetcher()
        selected = [
            {
                "id": 1,
                "url": "http://x/1.jpg",
                "photographer": "a",
                "type": "image",
                "duration": None,
                "source": "Pexels",
                "score": 3,
            }
        ]
        with (
            patch.object(
                fetcher,
                "_search_and_select_pexels",
                new=AsyncMock(return_value=selected),
            ),
            patch(
                "src.video.stock_media.download_file", new=AsyncMock(return_value=True)
            ),
        ):
            (tmp_path / "1.jpg").write_bytes(b"x")
            with patch(
                "src.video.stock_media.get_filename_from_url", return_value="1.jpg"
            ):
                items = await fetcher.fetch_and_download_stock(
                    ["q"], 1, 0, tmp_path, MagicMock(), script="s"
                )
        assert [i.relevance_score for i in items] == [3]


@pytest.mark.unit
class TestTheRunDirectoryRecordsIt:
    def test_the_score_round_trips_through_gathered_visuals(self, tmp_path) -> None:
        from src.video.producer.state import load_visuals_info, save_visuals_info

        item = StockMediaInfo(
            "Pexels",
            "image",
            "http://x/1.jpg",
            "a",
            tmp_path / "1.jpg",
            None,
            "laptop fan",
            3,
        )
        paths = {"gathered_visuals_file": tmp_path / "gathered_visuals.json"}
        save_visuals_info([], [], [item], paths, ["laptop fan"])
        data = json.loads(paths["gathered_visuals_file"].read_text())
        assert data["stock_media"][0]["relevance_score"] == 3
        _, _, loaded = load_visuals_info(paths["gathered_visuals_file"])
        assert loaded[0].relevance_score == 3


@pytest.mark.unit
class TestTheGatherStepPassesTheScript:
    def test_the_fetch_is_handed_the_script(self) -> None:
        """A judge nothing hands a script to is a judge that never runs."""
        import ast

        source = Path("src/video/producer/steps.py").read_text()
        tree = ast.parse(source)
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_fetch_stock_across_queries"
        ]
        assert calls, "the gather step no longer calls _fetch_stock_across_queries"
        for call in calls:
            assert any(k.arg == "script" for k in call.keywords), ast.unparse(call)
        constructs = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "StockMediaFetcher"
        ]
        assert constructs
        for call in constructs:
            assert any(k.arg == "llm_settings" for k in call.keywords), ast.unparse(
                call
            )


@pytest.mark.unit
class TestTheShippedConfig:
    def test_the_bundled_config_turns_it_on(self) -> None:
        cfg = config.llm_settings.stock_relevance
        assert cfg.enabled is True
        assert cfg.min_score == 2
        assert cfg.max_candidates == 80

    def test_the_model_default_is_off(self) -> None:
        assert StockRelevanceConfig().enabled is False
