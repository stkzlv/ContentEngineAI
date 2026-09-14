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

from src.scraper.amazon.utils import is_valid_product_data


class FakeDriver:
    """Answers `select` from a selector -> text map, like the real driver.

    A selector mapped to `""` is an element that exists and has no text, which
    is the case this file is about; one absent from the map matches nothing.
    """

    def __init__(self, texts: dict[str, str]):
        self.texts = texts
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

    def test_both_selector_loops_guard_on_text(self):
        import ast

        from src.utils.outputs_paths import get_project_root

        source = (
            get_project_root() / "src/scraper/amazon/product_extractor.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)

        guards_reaching_text = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.For):
                continue
            for statement in node.body:
                if not isinstance(statement, ast.If):
                    continue
                test_src = ast.unparse(statement.test)
                if "element" in test_src and ".text" in test_src:
                    guards_reaching_text += 1

        assert guards_reaching_text >= 2, (
            "a selector loop breaks on the element being present rather than on "
            "it having text; a duplicate id then ends the fallback chain on an "
            f"empty match (found {guards_reaching_text} text-checking guards)"
        )


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
