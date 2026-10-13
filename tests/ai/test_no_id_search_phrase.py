"""A product id or URL is no search phrase (#748, REQ-CNT-054).

A product scraped from its id records the id as its keyword. With the hook
rules on, the caption prompt asked the model to open with that "phrase", and
unified descriptions opened with the product id on every platform.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.ai.description_generator import search_phrase_rule
from src.video.search_phrase import search_phrase


def _settings():
    settings = MagicMock()
    settings.script_templates.hook_rules.enabled = True
    return settings


@pytest.mark.unit
@pytest.mark.req("REQ-CNT-054")
@pytest.mark.parametrize(
    "keyword",
    [
        "B0TESTID01",
        "B0ABCDEFGH",
        "https://amzn.to/abc",
        "https://www.amazon.com/dp/B0X",
    ],
)
def test_an_id_or_url_keyword_gives_no_phrase_and_no_rule(keyword):
    product = SimpleNamespace(topic=None, keyword=keyword, title="Smart watch")

    assert search_phrase(product) is None
    assert search_phrase_rule(product, _settings()) == ""


@pytest.mark.unit
@pytest.mark.req("REQ-CNT-054")
@pytest.mark.parametrize("keyword", ["smart watch", "smartwatch", "usb c hub"])
def test_a_real_keyword_still_leads(keyword):
    product = SimpleNamespace(topic=None, keyword=keyword, title="X")

    assert search_phrase(product) == keyword
    assert f'"{keyword}"' in search_phrase_rule(product, _settings())


@pytest.mark.unit
@pytest.mark.req("REQ-PUB-008")
def test_the_short_youtube_title_does_not_open_with_the_id():
    from src.video.config import config
    from src.video.producer import steps

    cfg = config.model_copy(deep=True)
    cfg.description_settings.short_product_titles = True
    ctx = SimpleNamespace(
        config=cfg,
        product=SimpleNamespace(
            title="Smart Watch for Men Women, 1.83 HD Touch Screen",
            keyword="B0TESTID01",
            topic=None,
        ),
        state={
            "generate_script": {"hook_headline": "Track your sleep without a phone"}
        },
    )

    assert steps._unified_title(ctx) == "Track your sleep without a phone"
