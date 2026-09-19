"""Inspecting a search-result card is one round trip and never waits.

The driver's `select` and `select_all` default to a four-second wait, and
the card inspection tried nine link selectors before giving up on a card,
so a sponsored or placeholder card (no `/dp/` link) cost about forty seconds
to skip: 85 s of a 218 s scrape on 2026-09-19 went to its first two cards.
With the waits gone each lookup was still a DOM round trip, seventeen of
them, and a skipped card still cost eight seconds. One script call now
returns the text, the product link, the rating and the review counts; the
selector chains stay as the fallback for an element that cannot run a
script, and none of them waits.
"""

from __future__ import annotations

import ast
from types import SimpleNamespace

from src.scraper.amazon.product_extractor import (
    CARD_FACTS_JS,
    CARD_LOOKUP_WAIT,
    extract_serp_product_info,
)
from src.utils.outputs_paths import get_project_root

_DEFAULT = object()
PRODUCT_URL = "https://www.amazon.com/dp/B0TEST00001?ref=x"


class ScriptCard:
    """An element that answers the one script call and counts its lookups."""

    def __init__(self, facts: dict):
        self.facts = facts
        self.scripts: list[str] = []
        self.lookups = 0

    def run_js(self, script: str, args=None):
        self.scripts.append(script)
        return self.facts

    def select(self, selector: str, wait=_DEFAULT):
        self.lookups += 1
        return None

    def select_all(self, selector: str, wait=_DEFAULT):
        self.lookups += 1
        return []


class SelectorCard:
    """An element with no script call; every lookup records the wait it carried."""

    def __init__(self, links: dict[str, str] | None = None, text: str = ""):
        self.links = links or {}
        self.text = text
        self.waits: list[object] = []

    def select(self, selector: str, wait=_DEFAULT):
        self.waits.append(wait)
        href = self.links.get(selector)
        if href is None:
            return None
        return SimpleNamespace(
            get_attribute=lambda name: href if name == "href" else None, text=""
        )

    def select_all(self, selector: str, wait=_DEFAULT):
        self.waits.append(wait)
        return []


class TestOneScriptCallPerCard:
    def test_a_card_with_no_product_link_costs_one_call_and_no_lookup(self):
        card = ScriptCard(
            {"text": "Sponsored", "link": None, "rating": "", "reviews": []}
        )

        assert extract_serp_product_info(card, "smart watch") is None
        assert card.scripts == [CARD_FACTS_JS]
        assert card.lookups == 0

    def test_a_product_card_yields_its_facts_from_the_one_call(self):
        card = ScriptCard(
            {
                "text": "Smart Watch for Men",
                "link": PRODUCT_URL,
                "rating": "4.5 out of 5 stars",
                "reviews": ["Sponsored", "(1,234)"],
            }
        )

        info = extract_serp_product_info(card, "smart watch")

        assert info is not None
        assert (info.asin, info.rating, info.reviews_count) == (
            "B0TEST00001",
            "4.5",
            "(1,234)",
        )
        assert card.lookups == 0

    def test_a_relative_link_is_anchored_on_the_site(self):
        card = ScriptCard(
            {"text": "", "link": "/dp/B0TEST00002/ref=sr", "rating": "", "reviews": []}
        )

        info = extract_serp_product_info(card, "k")

        assert (
            info is not None
            and info.url.startswith("https://")
            and info.asin == "B0TEST00002"
        )

    def test_a_skip_indicator_in_the_text_skips_the_card(self):
        card = ScriptCard(
            {
                "text": "People also search for",
                "link": PRODUCT_URL,
                "rating": "",
                "reviews": [],
            }
        )

        assert extract_serp_product_info(card, "k") is None


class TestTheSelectorFallbackNeverWaits:
    def test_a_card_with_no_product_link_is_skipped_without_a_wait(self):
        card = SelectorCard()

        assert extract_serp_product_info(card, "smart watch") is None
        assert card.waits, "the card was never inspected"
        assert all(w is CARD_LOOKUP_WAIT for w in card.waits), card.waits

    def test_a_product_card_still_yields_the_asin_without_a_wait(self):
        card = SelectorCard({"h2 a[href*='/dp/']": PRODUCT_URL})

        info = extract_serp_product_info(card, "smart watch")

        assert info is not None and info.asin == "B0TEST00001"
        assert all(w is CARD_LOOKUP_WAIT for w in card.waits), card.waits

    def test_the_wait_is_none_which_the_driver_reads_as_one_lookup(self):
        assert CARD_LOOKUP_WAIT is None


class TestEveryCardLookupCarriesTheWait:
    """A lookup added to the fallback without the wait would bring the cost back."""

    def test_by_reading_the_source(self):
        source = (
            get_project_root() / "src/scraper/amazon/product_extractor.py"
        ).read_text()
        fn = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_card_facts_by_selectors"
        )
        missing = [
            node.lineno
            for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("select", "select_all")
            and not any(kw.arg == "wait" for kw in node.keywords)
        ]
        assert missing == [], f"card lookups without a wait argument at lines {missing}"
