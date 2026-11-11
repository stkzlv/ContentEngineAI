"""A timed-out upload is retried; a timed-out post creation is not.

From SDK 1.4 the client raises on the first timeout of a POST instead of
retrying it, because replaying a post creation could publish the post twice.
A repeated upload only stores another copy of the file, so the publisher
retries uploads itself.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from late.client.exceptions import LateTimeoutError

from src.publisher.late.client import LatePublisher

pytestmark = pytest.mark.req("REQ-PUB-003", "REQ-PUB-011")


def _publisher() -> LatePublisher:
    publisher = LatePublisher(api_key="sk_test_abc123", max_retries=3)
    publisher.client = MagicMock()
    return publisher


@pytest.mark.asyncio
async def test_a_timed_out_upload_is_retried(tmp_path: Path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"0" * 1024)
    publisher = _publisher()
    publisher.client.media.upload.side_effect = [
        LateTimeoutError("POST /v1/media timed out and was NOT retried"),
        {"files": [{"url": "https://blob.test/v.mp4"}]},
    ]

    with patch("src.publisher.late.client.asyncio.sleep", new_callable=AsyncMock):
        url = await publisher.upload_media(video)

    assert url == "https://blob.test/v.mp4"
    assert publisher.client.media.upload.call_count == 2


@pytest.mark.asyncio
async def test_a_timed_out_post_creation_is_not_retried():
    publisher = _publisher()
    publisher.client.posts.create.side_effect = LateTimeoutError("timed out")

    async def create():
        return await publisher._call_sdk(publisher.client.posts.create, content="c")

    with (
        patch("src.publisher.late.client.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(LateTimeoutError),
    ):
        await publisher._retry_with_backoff(create, "Create post")

    assert publisher.client.posts.create.call_count == 1
