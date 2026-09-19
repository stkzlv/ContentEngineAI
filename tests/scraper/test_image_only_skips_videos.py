"""An image-only profile fetches no video and a rejected video says why.

`profile_uses_videos=False` only changed how media counts were validated:
the three video-extraction methods still ran on the product page and the
downloader fetched the file (126 MB, 7 s on 2026-09-19), validation
rejected it, and it was unlinked with no log line.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.scraper.amazon import download_async, product_extractor
from src.scraper.amazon.scraper import BotasaurusAmazonScraper


class FakeDriver:
    def __init__(self, texts: dict[str, str], current_url: str):
        self.texts = texts
        self.current_url = current_url

    def select(self, selector: str, wait=None):
        text = self.texts.get(selector)
        return (
            SimpleNamespace(text=text, get_attribute=lambda _n: None) if text else None
        )

    def select_all(self, selector: str, wait=None):
        return []

    def get_text(self, _selector: str) -> str:
        # The extractor scans the page body for availability warnings first.
        return ""


PAGE = {
    "#productTitle": "A product",
    ".a-price .a-offscreen": "$19.99",
    "#feature-bullets ul": "A description long enough to be real.",
}


def _extract(monkeypatch, extract_videos: bool):
    calls: list[str] = []
    monkeypatch.setattr(
        product_extractor, "extract_high_res_images_botasaurus", lambda *a, **k: ["img"]
    )

    def fake_videos(*a, **k):
        calls.append("videos")
        return ["vid"]

    monkeypatch.setattr(
        product_extractor, "extract_functional_videos_with_validation", fake_videos
    )
    driver = FakeDriver(PAGE, "https://www.amazon.com/dp/B0TEST1234")
    data = product_extractor.extract_product_data_from_page(
        driver, "B0TEST1234", "k", extract_videos=extract_videos
    )
    return data, calls


class TestTheExtractorSkipsVideosForAnImageOnlyProfile:
    def test_no_video_method_runs_and_the_skip_is_logged(self, monkeypatch, caplog):
        with caplog.at_level(logging.INFO):
            data, calls = _extract(monkeypatch, extract_videos=False)

        assert data is not None and data["videos"] == []
        assert calls == []
        assert any("image-only" in r.message for r in caplog.records)

    def test_videos_are_extracted_by_default(self, monkeypatch):
        data, calls = _extract(monkeypatch, extract_videos=True)

        assert data is not None and data["videos"] == ["vid"]
        assert calls == ["videos"]


class TestTheBrowserTaskCarriesTheFlag:
    """Both scrape paths build the task dict; both must pass it on."""

    def test_every_extractor_call_site_passes_extract_videos(self):
        from src.utils.outputs_paths import get_project_root

        source = (
            get_project_root() / "src/scraper/amazon/browser_functions.py"
        ).read_text()
        calls = source.count("extract_product_data_from_page(")
        assert calls == source.count('extract_videos=data.get("extract_videos", True)')
        assert calls >= 3

    def test_both_task_dicts_carry_the_flag(self):
        from src.utils.outputs_paths import get_project_root

        source = (get_project_root() / "src/scraper/amazon/scraper.py").read_text()
        assert (
            source.count('"extract_videos": self.profile_uses_videos is not False') == 2
        )


class TestTheDownloadQueueCarriesNoVideos:
    def _scraper(self, profile_uses_videos: bool | None) -> BotasaurusAmazonScraper:
        scraper = BotasaurusAmazonScraper.__new__(BotasaurusAmazonScraper)
        scraper.debug_mode = False
        scraper.logger = logging.getLogger("test")
        scraper.output_dir = None
        scraper.profile_uses_videos = profile_uses_videos
        return scraper

    @pytest.mark.parametrize(
        "profile_uses_videos, expected_videos",
        [(False, []), (True, ["v.mp4"]), (None, ["v.mp4"])],
    )
    def test_videos_follow_the_flag(self, profile_uses_videos, expected_videos):
        scraper = self._scraper(profile_uses_videos)
        results = [{"asin": "B0TEST1234", "images": ["i.jpg"], "videos": ["v.mp4"]}]
        seen: list[dict] = []

        def fake_download(tasks):
            seen.extend(tasks)
            return [
                {
                    "asin": t["asin"],
                    "downloaded_images": [],
                    "downloaded_videos": [],
                    "total_images": 0,
                    "total_videos": 0,
                }
                for t in tasks
            ]

        with patch("src.scraper.amazon.scraper.download_media_files", fake_download):
            scraper._orchestrate_media_downloads(results, None)

        assert [t["videos"] for t in seen] == [expected_videos]


class TestARejectedVideoSaysWhy:
    def test_the_issues_are_logged_and_the_file_goes(self, tmp_path: Path, caplog):
        video = tmp_path / "B0TEST1234_video_0.mp4"
        video.write_bytes(b"x")
        result = SimpleNamespace(issues=["duration 3s below minimum 5s", "no audio"])

        with caplog.at_level(logging.INFO):
            download_async._discard_rejected_media("VIDEO", video, result)

        assert not video.exists()
        line = next(r.message for r in caplog.records if "Rejected" in r.message)
        assert "B0TEST1234_video_0.mp4" in line and "duration 3s" in line

    def test_no_issues_still_leaves_a_reason(self, tmp_path: Path, caplog):
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")

        with caplog.at_level(logging.INFO):
            download_async._discard_rejected_media(
                "IMAGE", video, SimpleNamespace(issues=[])
            )

        assert any(
            "[IMAGE] Rejected" in r.message and "failed validation" in r.message
            for r in caplog.records
        )
