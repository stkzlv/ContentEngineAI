"""A render's cover reaches each platform that takes one (design 0011).

Driven through the real `upload_media` and `publish`, with only the SDK call
stubbed, so the assertions read the post the provider would be sent.
"""

import json
import logging
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import aiohttp
import httpx
import pytest
from late import LateError

from src.publisher.late.client import LatePublisher
from src.utils.cover_record import cover_paths

VIDEO_NAME = "video_B0TESTCOV1_main.mp4"
CONTENTS = {
    "youtube": {"content": "A caption", "title": "A title"},
    "instagram": {"content": "A caption", "first_comment": "Link in bio"},
    "tiktok": {"content": "A caption"},
}
COVER_WORDS = ("thumb", "cover")


def _render(tmp_path: Path, *, record: bool = True, image: bool = True) -> Path:
    video = tmp_path / VIDEO_NAME
    video.write_bytes(b"\x00" * 2048)
    jpg, rec = cover_paths(video)
    if image:
        jpg.write_bytes(b"\xff\xd8\xff" + b"\x00" * 1024)
    if record:
        rec.write_text(
            json.dumps({"file": jpg.name, "video": video.name, "at_ms": 750})
        )
    return video


class _Provider:
    """The SDK seen by the client: uploads answer with a URL, posts are kept."""

    def __init__(self, enabled: bool = True, cover_error: Exception | None = None):
        self.pub = LatePublisher(
            api_key="sk_test_abc123", max_retries=1, covers_enabled=enabled
        )
        self.pub.client = MagicMock()
        self.uploads: list[str] = []
        self.posts: list[dict] = []
        self.cover_error = cover_error

    async def _call_sdk(self, method, *args, **kwargs):
        if method is self.pub.client.media.upload:
            name = Path(args[0]).name
            self.uploads.append(name)
            if name.endswith(".jpg") and self.cover_error is not None:
                raise self.cover_error
            return {"url": f"https://cdn.test/{name}"}
        self.posts.append(kwargs)
        return MagicMock()

    def __enter__(self):
        self._stack = ExitStack()
        self._stack.enter_context(
            patch.object(self.pub, "_call_sdk", side_effect=self._call_sdk)
        )
        self._stack.enter_context(
            patch.object(
                self.pub,
                "_parse_publish_response",
                return_value={
                    "post_id": "p1",
                    "status": "scheduled",
                    "published_urls": [],
                },
            )
        )
        self._stack.enter_context(
            patch.object(self.pub, "_extract_platform_failures", return_value=[])
        )
        return self

    def __exit__(self, *exc):
        self._stack.close()

    async def post(self, media_id: str, names: list[str]) -> dict[str, dict]:
        await self.pub.publish(
            media_id=media_id,
            platforms=[{"platform": n, "account_id": f"acc_{n}"} for n in names],
            platform_contents={n: CONTENTS[n] for n in names},
        )
        return {p["platform"]: p for p in self.posts[-1]["platforms"]}


def _cover_keys(entry: dict) -> list[str]:
    keys = {k.lower() for k in entry} | {
        k.lower() for k in entry.get("platformSpecificData", {})
    }
    return [k for k in keys if any(w in k for w in COVER_WORDS)]


@pytest.mark.asyncio
class TestEachPlatformGetsItsCover:
    @pytest.mark.req("REQ-PUB-154")
    @pytest.mark.req("REQ-PUB-155")
    @pytest.mark.req("REQ-PUB-156")
    @pytest.mark.req("REQ-PUB-157")
    async def test_one_post_to_all_three(self, tmp_path):
        video = _render(tmp_path)
        with _Provider() as sdk:
            media = await sdk.pub.upload_media(video)
            entries = await sdk.post(media, ["youtube", "instagram", "tiktok"])

        instagram = entries["instagram"]["platformSpecificData"]
        assert instagram["instagramThumbnail"] == (
            f"https://cdn.test/{cover_paths(video)[0].name}"
        )
        assert instagram["firstComment"] == "Link in bio", "the cover replaced it"
        tiktok = entries["tiktok"]["platformSpecificData"]
        assert tiktok["videoCoverTimestampMs"] == 750
        assert "tiktokSettings" in tiktok, "the cover replaced the settings"
        assert not [k for k in tiktok if k.lower() == "videocoverimageurl"]
        assert not _cover_keys(entries["youtube"])
        media_keys = {k.lower() for i in sdk.posts[-1]["media_items"] for k in i}
        assert not [k for k in media_keys if any(w in k for w in COVER_WORDS)]

    @pytest.mark.req("REQ-PUB-154")
    async def test_off_sends_no_cover_and_uploads_none(self, tmp_path):
        video = _render(tmp_path)
        with _Provider(enabled=False) as sdk:
            media = await sdk.pub.upload_media(video)
            entries = await sdk.post(media, ["instagram", "tiktok"])

        assert not _cover_keys(entries["instagram"])
        assert not _cover_keys(entries["tiktok"])
        assert sdk.uploads == [VIDEO_NAME]

    @pytest.mark.req("REQ-PUB-154")
    async def test_a_youtube_only_post_uploads_no_image(self, tmp_path):
        video = _render(tmp_path)
        with _Provider() as sdk:
            media = await sdk.pub.upload_media(video)
            await sdk.post(media, ["youtube"])

        assert sdk.uploads == [VIDEO_NAME]

    @pytest.mark.req("REQ-PUB-155")
    async def test_instagram_without_a_first_comment(self, tmp_path):
        video = _render(tmp_path)
        with _Provider() as sdk:
            media = await sdk.pub.upload_media(video)
            await sdk.pub.publish(
                media_id=media,
                platforms=[{"platform": "instagram", "account_id": "acc_ig"}],
                content="A caption",
            )

        (entry,) = sdk.posts[-1]["platforms"]
        assert entry["platformSpecificData"]["instagramThumbnail"].endswith(".jpg")

    @pytest.mark.req("REQ-PUB-156")
    async def test_a_tiktok_only_post_uploads_no_image(self, tmp_path):
        video = _render(tmp_path)
        with _Provider() as sdk:
            media = await sdk.pub.upload_media(video)
            await sdk.post(media, ["tiktok"])

        assert sdk.uploads == [VIDEO_NAME]


@pytest.mark.asyncio
@pytest.mark.req("REQ-PUB-158")
class TestOnePostPerPlatform:
    """The per-platform mode posts one video several times."""

    async def test_each_post_carries_the_same_cover_uploaded_once(self, tmp_path):
        video = _render(tmp_path)
        with _Provider() as sdk:
            media = await sdk.pub.upload_media(video)
            instagram = await sdk.post(media, ["instagram"])
            tiktok = await sdk.post(media, ["tiktok"])
            again = await sdk.post(media, ["instagram"])

        url = instagram["instagram"]["platformSpecificData"]["instagramThumbnail"]
        assert again["instagram"]["platformSpecificData"]["instagramThumbnail"] == url
        assert tiktok["tiktok"]["platformSpecificData"]["videoCoverTimestampMs"] == 750
        assert sdk.uploads.count(cover_paths(video)[0].name) == 1

    async def test_two_videos_get_their_own_covers(self, tmp_path):
        first = _render(tmp_path)
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        second = other_dir / "video_B0TESTCOV2_main.mp4"
        second.write_bytes(b"\x00" * 2048)
        jpg, rec = cover_paths(second)
        jpg.write_bytes(b"\xff\xd8\xff" + b"\x00" * 512)
        rec.write_text(json.dumps({"at_ms": 1200}))
        with _Provider() as sdk:
            media_one = await sdk.pub.upload_media(first)
            media_two = await sdk.pub.upload_media(second)
            two = await sdk.post(media_two, ["instagram", "tiktok"])
            one = await sdk.post(media_one, ["instagram", "tiktok"])

        assert two["tiktok"]["platformSpecificData"]["videoCoverTimestampMs"] == 1200
        assert one["tiktok"]["platformSpecificData"]["videoCoverTimestampMs"] == 750
        assert two["instagram"]["platformSpecificData"]["instagramThumbnail"].endswith(
            jpg.name
        )


@pytest.mark.asyncio
@pytest.mark.req("REQ-PUB-159")
class TestNoCoverStillPublishes:
    async def _publish(self, sdk, video, caplog):
        caplog.set_level(logging.WARNING, logger="src.publisher.late.client")
        media = await sdk.pub.upload_media(video)
        return await sdk.post(media, ["instagram", "tiktok"])

    @pytest.mark.parametrize(
        ("record", "image"), [(False, True), (True, False)], ids=["record", "image"]
    )
    async def test_missing_file(self, tmp_path, caplog, record, image):
        video = _render(tmp_path, record=record, image=image)
        with _Provider() as sdk:
            entries = await self._publish(sdk, video, caplog)

        assert len(sdk.posts) == 1
        assert not _cover_keys(entries["instagram"])
        assert not _cover_keys(entries["tiktok"])
        assert VIDEO_NAME in caplog.text

    @pytest.mark.parametrize(
        "record_text", ["{not json", '{"at_ms": "750"}', '{"at_ms": true}', "[]"]
    )
    async def test_unusable_record(self, tmp_path, caplog, record_text):
        video = _render(tmp_path)
        cover_paths(video)[1].write_text(record_text)
        with _Provider() as sdk:
            entries = await self._publish(sdk, video, caplog)

        assert not _cover_keys(entries["tiktok"])
        assert VIDEO_NAME in caplog.text

    @pytest.mark.parametrize(
        "error",
        [
            LateError("refused"),
            httpx.ConnectError("dropped"),
            aiohttp.ClientConnectionError("dropped"),
            json.JSONDecodeError("html page", "<html>", 0),
            OSError("unreadable"),
        ],
        ids=["sdk", "httpx", "aiohttp", "html", "os"],
    )
    async def test_failed_upload(self, tmp_path, caplog, error):
        video = _render(tmp_path)
        with _Provider(cover_error=error) as sdk:
            entries = await self._publish(sdk, video, caplog)

        assert len(sdk.posts) == 1
        assert not _cover_keys(entries["instagram"])
        assert entries["tiktok"]["platformSpecificData"]["videoCoverTimestampMs"] == 750
        assert VIDEO_NAME in caplog.text

    async def test_a_media_id_this_publisher_never_uploaded(self, tmp_path, caplog):
        caplog.set_level(logging.WARNING, logger="src.publisher.late.client")
        with _Provider() as sdk:
            entries = await sdk.post("https://cdn.test/elsewhere.mp4", ["instagram"])

        assert not _cover_keys(entries["instagram"])
        assert "elsewhere.mp4" in caplog.text


@pytest.mark.req("REQ-PUB-154")
def test_the_bundled_config_ships_covers_off(monkeypatch):
    from src.publisher.config import load_publisher_config
    from src.publisher.registry import create_publisher_from_config

    monkeypatch.delenv("LATE_API_KEY", raising=False)
    config = load_publisher_config(cli_overrides={"api_key": "sk_test_" + "x" * 40})

    assert config.covers_config.enabled is False
    assert create_publisher_from_config(config).covers_enabled is False


@pytest.mark.req("REQ-PUB-154")
def test_the_setting_reaches_the_publisher(tmp_path, monkeypatch):
    import yaml

    from src.publisher.config import load_publisher_config
    from src.publisher.registry import create_publisher_from_config

    shipped = yaml.safe_load(Path("config/publisher.yaml").read_text())
    shipped["covers"] = {"enabled": True}
    path = tmp_path / "publisher.yaml"
    path.write_text(yaml.safe_dump(shipped))
    monkeypatch.delenv("LATE_API_KEY", raising=False)
    config = load_publisher_config(
        config_path=path, cli_overrides={"api_key": "sk_test_" + "x" * 40}
    )

    assert create_publisher_from_config(config).covers_enabled is True


def _metadata():
    from src.publisher.models import Platform, PublishMetadata

    return PublishMetadata(
        platform=Platform.TIKTOK,
        title="A title",
        description="A caption",
        hashtags=["tech"],
        keywords=[],
        product_id="B0TESTCOV1",
        carries_affiliate_content=False,
    )


def _assert_covered(posts: list[dict], video: Path) -> None:
    entries = {p["platform"]: p for post in posts for p in post["platforms"]}
    assert entries["instagram"]["platformSpecificData"]["instagramThumbnail"] == (
        f"https://cdn.test/{cover_paths(video)[0].name}"
    )
    assert entries["tiktok"]["platformSpecificData"]["videoCoverTimestampMs"] == 750


@pytest.mark.asyncio
@pytest.mark.req("REQ-PUB-158")
class TestEveryPublishPath:
    """`single`, the batch and the pipeline phase post through
    `publish_product`; `schedule` through its own two builders. Each uploads
    the video on the publisher it then posts with.
    """

    TARGETS = [
        {"platform": "instagram", "account_id": "acc_ig"},
        {"platform": "tiktok", "account_id": "acc_tt"},
    ]

    @pytest.mark.parametrize("platform_specific", [False, True])
    async def test_publish_product(self, tmp_path, platform_specific):
        from src.publisher.publish_modes import publish_product

        video = _render(tmp_path)
        with (
            _Provider() as sdk,
            patch(
                "src.publisher.publish_modes.load_platform_metadata",
                return_value=_metadata(),
            ),
        ):
            media = await sdk.pub.upload_media(video)
            await publish_product(
                publisher=sdk.pub,
                media_id=media,
                product_id="B0TESTCOV1",
                platforms=self.TARGETS,
                outputs_dir=tmp_path,
                platform_specific=platform_specific,
            )

        assert len(sdk.posts) == (2 if platform_specific else 1)
        _assert_covered(sdk.posts, video)

    @pytest.mark.parametrize("builder", ["_post_unified", "_post_per_platform"])
    async def test_schedule(self, tmp_path, builder):
        from datetime import UTC, datetime, timedelta

        from src.publisher.schedule import ScheduleManager

        video = _render(tmp_path)
        manager = ScheduleManager(schedule_path=tmp_path / "schedule.json")
        with _Provider() as sdk:
            media = await sdk.pub.upload_media(video)
            await getattr(manager, builder)(
                publisher=sdk.pub,
                platform_dicts=self.TARGETS,
                platform_metas={"instagram": _metadata(), "tiktok": _metadata()},
                titles={},
                carries_affiliate={"instagram": False, "tiktok": False},
                first_comments={},
                media_id=media,
                product_id="B0TESTCOV1",
                next_time=datetime.now(UTC) + timedelta(days=1),
                next_idx=0,
            )

        _assert_covered(sdk.posts, video)
