"""Tests for the video-script mirror plumbing in platform metadata.

Covers `_read_video_script` (helper that reads the script.txt artifact and
threads it into the per-platform LLM prompts so caption templates can mirror
the script's closing engagement-bait line) plus the propagation guarantee in
`generate_multi_platform`: the script is read once and passed as `video_script`
to every per-platform generator.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ai.platform_metadata import PlatformMetadataFactory, _read_video_script
from src.scraper.amazon.models import ProductData
from src.scraper.base.models import Platform


class TestReadVideoScript:
    def test_returns_none_when_intermediate_paths_is_none(self):
        assert _read_video_script(None) is None

    def test_returns_none_when_intermediate_paths_is_empty(self):
        assert _read_video_script({}) is None

    def test_returns_none_when_script_key_missing(self, tmp_path: Path):
        paths = {"description": tmp_path / "description.txt"}
        assert _read_video_script(paths) is None

    def test_returns_none_when_script_file_does_not_exist(self, tmp_path: Path):
        paths = {"script": tmp_path / "missing-script.txt"}
        assert _read_video_script(paths) is None

    def test_returns_script_text_when_file_exists(self, tmp_path: Path):
        script_path = tmp_path / "script.txt"
        script = (
            "Best 65W chargers under $50 for tech-savvy young adults. "
            "This is a great charger. "
            "Most people only need two ports, but three is usually better. "
            "Link in bio if you want one."
        )
        script_path.write_text(script, encoding="utf-8")
        assert _read_video_script({"script": script_path}) == script

    def test_strips_surrounding_whitespace(self, tmp_path: Path):
        script_path = tmp_path / "script.txt"
        script_path.write_text("\n\n  hello  \n\n", encoding="utf-8")
        assert _read_video_script({"script": script_path}) == "hello"

    def test_empty_script_file_returns_none(self, tmp_path: Path):
        script_path = tmp_path / "script.txt"
        script_path.write_text("   \n\n  ", encoding="utf-8")
        assert _read_video_script({"script": script_path}) is None

    def test_unreadable_file_returns_none(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """File present but unreadable: helper returns None and logs WARN."""
        script_path = tmp_path / "script.txt"
        script_path.write_text("ok", encoding="utf-8")

        def fake_read_text(*args, **kwargs):
            raise OSError("permission denied")

        monkeypatch.setattr(Path, "read_text", fake_read_text)

        import logging

        with caplog.at_level(logging.WARNING, logger="src.ai.platform_metadata"):
            result = _read_video_script({"script": script_path})

        assert result is None
        assert any("Failed to read script" in r.getMessage() for r in caplog.records)


class TestGenerateMultiPlatformScriptPropagation:
    """Assert `generate_multi_platform` reads the script once and passes it
    as the `video_script` kwarg to every per-platform generator.

    This is the integration guarantee of the caption-side mirror: if a future
    refactor of generate_multi_platform drops the script propagation, every
    caption silently loses the closing-line mirror. Test fails loudly on that.
    """

    @pytest.fixture
    def product(self) -> ProductData:
        return ProductData(
            title="Test Product",
            price="$19.99",
            url="https://example.com/test",
            platform=Platform.AMAZON,
            description="Test description",
            asin="B0TEST0001",
        )

    @pytest.fixture
    def script_text(self) -> str:
        return (
            "First sentence with audio keyword. Middle sentences. "
            "USB-C or Lightning - which still annoys you more? "
            "Link in bio if you want one."
        )

    @pytest.fixture
    def intermediate_paths(self, tmp_path: Path, script_text: str) -> dict[str, Path]:
        script_path = tmp_path / "script.txt"
        script_path.write_text(script_text, encoding="utf-8")
        return {"script": script_path}

    @pytest.mark.asyncio
    async def test_script_propagates_to_every_generator(
        self,
        product: ProductData,
        script_text: str,
        intermediate_paths: dict[str, Path],
    ):
        """Every generator's generate() receives video_script=<script content>."""
        mock_generators = {
            platform: MagicMock(generate=AsyncMock(return_value=None))
            for platform in ("youtube", "tiktok", "instagram")
        }

        def fake_create(platform: str, _settings: dict):
            return mock_generators[platform]

        platform_settings = {p: {"enabled": True} for p in mock_generators}
        settings = MagicMock()

        with patch.object(PlatformMetadataFactory, "create", side_effect=fake_create):
            await PlatformMetadataFactory.generate_multi_platform(
                product=product,
                settings=settings,
                secrets={},
                session=MagicMock(),
                platform_settings=platform_settings,
                intermediate_paths=intermediate_paths,
            )

        for platform, mock in mock_generators.items():
            assert (
                mock.generate.await_count == 1
            ), f"{platform} generator was not called"
            kwargs = mock.generate.await_args.kwargs
            assert kwargs.get("video_script") == script_text, (
                f"{platform} generator did not receive video_script="
                f"<script>; got {kwargs.get('video_script')!r}"
            )

    @pytest.mark.asyncio
    async def test_missing_script_passes_none_to_every_generator(
        self,
        product: ProductData,
        tmp_path: Path,
    ):
        """No script.txt: every generator receives video_script=None and runs."""
        mock_generators = {
            platform: MagicMock(generate=AsyncMock(return_value=None))
            for platform in ("youtube", "tiktok", "instagram")
        }

        def fake_create(platform: str, _settings: dict):
            return mock_generators[platform]

        platform_settings = {p: {"enabled": True} for p in mock_generators}

        # intermediate_paths with no 'script' key
        intermediate_paths = {"description": tmp_path / "description.txt"}

        with patch.object(PlatformMetadataFactory, "create", side_effect=fake_create):
            await PlatformMetadataFactory.generate_multi_platform(
                product=product,
                settings=MagicMock(),
                secrets={},
                session=MagicMock(),
                platform_settings=platform_settings,
                intermediate_paths=intermediate_paths,
            )

        for platform, mock in mock_generators.items():
            kwargs = mock.generate.await_args.kwargs
            assert kwargs.get("video_script") is None, (
                f"{platform} generator received non-None script when path missing: "
                f"{kwargs.get('video_script')!r}"
            )
