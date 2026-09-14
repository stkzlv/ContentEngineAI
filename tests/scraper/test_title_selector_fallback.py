"""A selector that matches an empty element must not end the fallback chain.

An Amazon detail page carries `productTitle` twice: the `<span id="productTitle">`
that holds the title, and a hidden `<input name="productTitle" id="productTitle">`
whose *value* is the title and whose text is empty. Which one `select("#productTitle")`
returns is a property of the page, not of this code, because the id is duplicated.

When it returned the input, the loop broke on the element being present and
never tried `h1.a-size-large`, `.product-title` or the automation-id hook -- any
of which would have worked. The product was then dropped as titleless with
price, rating and description all intact: `is_valid_product_data` reduces to a
title check under the shipped `essential_fields: []`.

Measured on 2026-09-14: a batch ran 51 minutes over two keywords and validated
zero of 27 products, and a single-ASIN scrape of B0B6VPH24K failed the same way
twice in a row before the fix and succeeded immediately after it. The product
arm of the reach test renders nothing while this is broken, and nothing about
the run looks like a failure -- the scrape reports products attempted and none
valid, which reads as ordinary attrition.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.scraper.amazon.utils import is_valid_product_data


class FakeDriver:
    """Answers `select` from a selector -> text map, like the real driver.

    A selector mapped to `""` is an element that exists and has no text, which
    is the case this file is about; one absent from the map matches nothing.
    """

    def __init__(self, texts: dict[str, str], current_url: str = ""):
        self.texts = texts
        self.current_url = current_url
        self.queried: list[str] = []

    def select(self, selector: str, wait=None):
        self.queried.append(selector)
        text = self.texts.get(selector)
        return SimpleNamespace(text=text) if text is not None else None


TITLE_SELECTORS = [
    "#productTitle",
    "h1.a-size-large",
    ".product-title",
    "h1[data-automation-id='product-title']",
]


def first_title(driver: FakeDriver, selectors: list[str]) -> str:
    """The extractor's title loop, as shipped."""
    title = ""
    for selector in selectors:
        element = driver.select(selector)
        if element and element.text.strip():
            title = element.text.strip()
            break
    return title


class TestTheTitleChainSurvivesAnEmptyMatch:
    def test_an_empty_first_match_falls_through_to_the_next_selector(self):
        """The shape that zeroed a batch: hidden input first, real title second."""
        driver = FakeDriver(
            {"#productTitle": "", "h1.a-size-large": "EIGHTREE Smart Plug WiFi Outlet"}
        )

        assert first_title(driver, TITLE_SELECTORS) == "EIGHTREE Smart Plug WiFi Outlet"
        assert driver.queried[:2] == [
            "#productTitle",
            "h1.a-size-large",
        ], "the chain stopped at the empty match instead of trying the next"

    def test_whitespace_only_counts_as_empty(self):
        driver = FakeDriver({"#productTitle": "  \n  ", ".product-title": "Real Title"})

        assert first_title(driver, TITLE_SELECTORS) == "Real Title"

    def test_a_populated_first_match_still_wins(self):
        """The fallbacks stay fallbacks: priority order is unchanged."""
        driver = FakeDriver(
            {"#productTitle": "The Real One", "h1.a-size-large": "A Section Heading"}
        )

        assert first_title(driver, TITLE_SELECTORS) == "The Real One"
        assert driver.queried == ["#productTitle"]

    def test_every_selector_empty_still_yields_nothing(self):
        driver = FakeDriver(dict.fromkeys(TITLE_SELECTORS, ""))

        assert first_title(driver, TITLE_SELECTORS) == ""


class TestTheShippedLoopsActuallyCheckTheText:
    """The tests above describe the semantics; this one pins the real source.

    `first_title` re-implements the loop, so on its own it would keep passing
    after the extractor regressed to breaking on presence. Both loops in the
    module read an element and guard it, and the guard has to reach `.text`.
    """

    @pytest.mark.parametrize("iterator", ["title_selectors", "desc_selectors"])
    def test_the_named_loop_guards_on_text(self, iterator: str):
        """Identified by the list it walks, not counted.

        Counting text-checking guards across the module passes for the wrong
        reason: tidying an unrelated loop into a text guard restores the count
        while this loop regresses to breaking on presence.
        """
        import ast

        from src.utils.outputs_paths import get_project_root

        source = (
            get_project_root() / "src/scraper/amazon/product_extractor.py"
        ).read_text(encoding="utf-8")

        loops = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.For)
            and isinstance(node.iter, ast.Name)
            and node.iter.id == iterator
        ]

        assert len(loops) == 1, f"expected one loop over {iterator}, found {len(loops)}"
        guard = next(s for s in loops[0].body if isinstance(s, ast.If))
        assert ".text" in ast.unparse(guard.test), (
            f"the loop over {iterator} breaks on the element being present "
            "rather than on it having text; a duplicate id then ends the "
            "fallback chain on an empty match"
        )


class TestTheRealExtractorRecoversTheTitle:
    """The end-to-end case, so the fix is pinned by behaviour and not only by shape."""

    def test_a_page_with_an_empty_productTitle_still_yields_a_product(
        self, monkeypatch
    ):
        from src.scraper.amazon import product_extractor

        monkeypatch.setattr(
            product_extractor, "extract_high_res_images_botasaurus", lambda *a, **k: []
        )
        monkeypatch.setattr(
            product_extractor,
            "extract_functional_videos_with_validation",
            lambda *a, **k: [],
        )

        driver = FakeDriver(
            {
                # The hidden input the duplicate id can resolve to.
                "#productTitle": "",
                "h1.a-size-large": "EIGHTREE Smart Plug WiFi Outlet",
                "#corePrice_feature_div .a-price:not(.a-text-price) .a-offscreen": (
                    "$12.59"
                ),
                "#feature-bullets ul": "Works with Alexa",
            }
        )
        driver.get_text = lambda _selector: ""

        result = product_extractor.extract_product_data_from_page(
            driver, "B0B6VPH24K", "https://www.amazon.com/dp/B0B6VPH24K"
        )

        assert result is not None, (
            "the product was dropped as titleless while the real title sat in "
            "the next selector"
        )
        assert result["title"] == "EIGHTREE Smart Plug WiFi Outlet"


class TestWhyAnEmptyTitleCostsTheWholeProduct:
    def test_the_shipped_config_validates_on_the_title_alone(self):
        """`essential_fields: []` in `config/scraper.yaml` reduces to this.

        So an empty title is not a partial record, it is a dropped product --
        which is why a selector that merely matched the wrong element took a
        whole batch to zero.
        """
        assert not is_valid_product_data(
            title="", price="12.59", description="d", asin="B0B6VPH24K", rating="4.5"
        )
        assert is_valid_product_data(
            title="EIGHTREE Smart Plug",
            price="12.59",
            description="d",
            asin="B0B6VPH24K",
            rating="4.5",
        )
