"""The gallery and listing videos come from the page's image data (REQ-SCR-086).

`tests/fixtures/scraper/image_block.js` is a made-up copy of the inline
script's shape: the landing colour's gallery in `'initial'` as a JSON array
inside a JavaScript string literal (one entry with a null `hiRes`), the
listing videos in `'videos'`, and a title whose quote and apostrophe are
escaped the way the literal escapes them.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.scraper.amazon import media_extractor, video_extractor
from src.scraper.amazon.image_data import (
    ImageBlock,
    ListingVideo,
    parse_image_block,
    read_image_block,
)

pytestmark = [pytest.mark.unit, pytest.mark.req("REQ-SCR-086")]

BLOCK = (Path(__file__).parents[1] / "fixtures/scraper/image_block.js").read_text()
IMG = "https://m.media-amazon.com/images/I/"

# Another colour's gallery, as the page carries it in a second script.
OTHER_COLOUR = (
    '"colorImages":{"white":[{"large":"' + IMG + 'WHITE._AC_.jpg",'
    '"hiRes":"' + IMG + 'WHITE._AC_SL1500_.jpg"}]}'
)


def test_the_gallery_comes_in_order_with_large_where_hires_is_null():
    assert parse_image_block(BLOCK).images == [
        IMG + "TEST1._AC_SL1500_.jpg",
        IMG + "TEST2._AC_.jpg",
        IMG + "TEST3._AC_SL1500_.jpg",
    ]


def test_the_listing_video_comes_with_its_title_and_duration():
    assert parse_image_block(BLOCK).videos == [
        ListingVideo(
            url="https://m.media-amazon.com/images/S/test-video/"
            "default.jobtemplate.mp4.480.mp4",
            title='A Hub\'s "4K" output',
            duration_sec=22.0,
        )
    ]


def test_another_colours_gallery_is_not_read():
    assert parse_image_block(OTHER_COLOUR) == ImageBlock()


@pytest.mark.parametrize("key", ["initial", "videos"])
def test_a_later_propertys_parse_call_is_not_this_keys(key):
    text = (
        f"'{key}': [], 'other': A.$.parseJSON('"
        '[{"hiRes":"' + IMG + 'X.jpg","url":"https://example.test/v.mp4"}]'
        "')"
    )

    assert parse_image_block(text) == ImageBlock()


def test_a_gallery_without_hires_falls_back_to_the_scans():
    text = (
        "'colorImages': { 'initial': A.$.parseJSON('"
        '[{"hiRes":null,"large":"' + IMG + 'SMALL._AC_.jpg"}]'
        "')}"
    )
    driver = MagicMock()
    driver.run_js.side_effect = lambda script, *_a: (
        text if "colorImages" in script and "return t;" in script else None
    )
    driver.select_all.return_value = []
    driver.get_all_image_links.return_value = []

    assert parse_image_block(text).hires_count == 0
    media_extractor.extract_high_res_images_botasaurus(driver, max_images=2)

    driver.select_all.assert_called()


def test_a_page_without_the_block_reads_empty():
    driver = MagicMock()
    driver.run_js.return_value = None

    assert read_image_block(driver) == ImageBlock()


def test_a_broken_literal_reads_empty():
    broken = "'colorImages': { 'initial': A.$.parseJSON('[{\"hiRes\": ]')}"

    assert parse_image_block(broken).images == []


def _driver_serving_the_block() -> MagicMock:
    driver = MagicMock()
    driver.run_js.side_effect = lambda script, *_a: (
        BLOCK if "colorImages" in script else None
    )
    return driver


def test_the_extractor_returns_the_gallery_without_scanning(monkeypatch):
    driver = _driver_serving_the_block()

    images = media_extractor.extract_high_res_images_botasaurus(driver, max_images=2)

    assert images == [IMG + "TEST1._AC_SL1500_.jpg", IMG + "TEST2._AC_.jpg"]
    driver.select_all.assert_not_called()


def test_off_scans_the_page_as_before(monkeypatch):
    real = media_extractor.get_config_value
    monkeypatch.setattr(
        media_extractor,
        "get_config_value",
        lambda *keys, default=None: (
            False if keys[-1] == "use_image_data" else real(*keys, default=default)
        ),
    )
    driver = _driver_serving_the_block()
    driver.select_all.return_value = []
    driver.get_all_image_links.return_value = []

    media_extractor.extract_high_res_images_botasaurus(driver, max_images=2)

    scripts = [call.args[0] for call in driver.run_js.call_args_list]
    assert not any(
        "'colorImages'" in script and "return t;" in script for script in scripts
    )


def test_the_listing_video_leads_the_video_candidates(monkeypatch):
    driver = MagicMock()

    def run_js(script, *_a):
        if "colorImages" in script and "return t;" in script:
            return BLOCK
        if "asinMatch" in script:
            return {"asin": "B0TEST0001", "title": "x", "keywords": []}
        return None

    driver.run_js.side_effect = run_js
    driver.select_all.return_value = []

    videos = video_extractor.extract_functional_videos_with_validation(driver)

    assert videos[0].endswith("default.jobtemplate.mp4.480.mp4")
