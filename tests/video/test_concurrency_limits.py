"""Fan-outs run no more at once than their limit (REQ-OPS-038).

Each test makes every call wait until the fan-out has had the chance to
start all of them, and records the most that were in flight together.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ai.llm_settings import StockRelevanceConfig
from src.video.config import config
from src.video.config.visual_models import StockMediaSettings
from src.video.stock_media import StockMediaFetcher


class Gauge:
    def __init__(self) -> None:
        self.now = self.peak = 0

    async def hold(self) -> None:
        self.now += 1
        self.peak = max(self.peak, self.now)
        await asyncio.sleep(0.01)
        self.now -= 1


def _client(gauge: Gauge, answer: str) -> MagicMock:
    async def generate_content(**kwargs):
        await gauge.hold()
        return MagicMock(text=answer)

    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(side_effect=generate_content)
    client.aio.aclose = AsyncMock()
    return client


class _Response:
    status = 200
    content_type = "image/jpeg"

    async def read(self) -> bytes:
        return b"jpeg"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        return None


class _Session:
    def get(self, url):
        return _Response()


@pytest.mark.req("REQ-OPS-038")
@pytest.mark.asyncio
async def test_the_stock_judge_holds_its_concurrency() -> None:
    from src.video.stock_relevance import score_candidates

    gauge = Gauge()
    candidates = [{"id": i, "thumbnail": f"http://x/{i}.jpg"} for i in range(9)]
    with patch("google.genai.Client", return_value=_client(gauge, '{"score": 2}')):
        scores = await score_candidates(
            candidates,
            "q",
            "s",
            api_key="k",
            settings=StockRelevanceConfig(enabled=True, concurrency=3),
            session=_Session(),
        )

    assert scores == [2] * 9
    assert gauge.peak == 3


@pytest.mark.req("REQ-OPS-038")
@pytest.mark.asyncio
async def test_the_image_judge_holds_its_concurrency(tmp_path: Path) -> None:
    from PIL import Image

    from src.video.image_curation import score_images

    images = []
    for i in range(8):
        path = tmp_path / f"{i}.jpg"
        Image.new("RGB", (64, 64), (i * 20, 0, 0)).save(path)
        images.append(path)
    gauge = Gauge()
    answer = '{"usable": true, "kind": "product", "score": 2}'
    with patch("google.genai.Client", return_value=_client(gauge, answer)):
        await score_images(
            images, api_key="k", model="m", concurrency=2, timeout_seconds=5
        )

    assert gauge.peak == 2


@pytest.mark.req("REQ-OPS-038")
@pytest.mark.asyncio
async def test_stock_downloads_hold_their_concurrency(
    tmp_path: Path, monkeypatch
) -> None:
    from src.video import stock_media

    gauge = Gauge()

    async def fake_download(url, path, *args, **kwargs):
        await gauge.hold()
        Path(path).write_bytes(b"x")
        return True

    monkeypatch.setattr(stock_media, "download_file", fake_download)
    assert config.api_settings is not None
    fetcher = StockMediaFetcher(
        StockMediaSettings(pexels_api_key_env_var="NO_SUCH_KEY"),
        {},
        config.media_settings,
        api_settings=config.api_settings.model_copy(
            update={"stock_media_concurrent_downloads": 2}
        ),
    )
    fetcher.pexels_client = MagicMock()
    fetcher.pexels_client.search_photos.return_value = {
        "photos": [
            {"id": n, "src": {"original": f"https://x/{n}.jpg"}, "photographer": "a"}
            for n in range(1, 9)
        ]
    }

    items = await fetcher.fetch_and_download_stock(
        ["desk"], 6, 0, tmp_path, AsyncMock()
    )

    assert len(items) == 6
    assert gauge.peak == 2
