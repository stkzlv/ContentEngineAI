"""Page retry belongs to keyword searches, not to URL or ASIN inputs.

`_scrape_until_validated_count_reached` paginates when a page yields no
validated product. That is right for a keyword, where page 2 holds different
products. A URL or an ASIN names one product, so the next page re-resolves the
same one and the loop runs to `max_pages` on a listing that will never pass.

`scrape_products_unified` is called per input, so a media-poor entry cost all
seven of `max_pages` browser sessions re-fetching the same listing before the
loop gave up and returned empty. Nothing was dropped: the run recorded the
failure and moved on. The documented behaviour in `CLAUDE.md` already said page
retry applies to keyword searches only, and the global batch gates it that way;
this path did not.

The batch side of that gate is covered behaviourally by
`tests/pipeline/test_global_batch_integration.py`, in
`test_pipeline_skips_retry_for_asin_inputs`,
`test_pipeline_skips_retry_for_url_inputs` and
`test_pipeline_retries_next_page_on_validation_failure`. A test here asserting
that `global_batch` still mentions the two predicates stayed green with the
batch gate inverted, so it is not repeated. Dropping only the `_is_url` half
did defeat those tests until the URL case above was added, since the ASIN one
mocks `_is_url` to False.
"""

from __future__ import annotations

import pytest

from src.scraper.amazon.scraper import BotasaurusAmazonScraper


@pytest.fixture
def scraper(monkeypatch):
    """A real scraper with the two scrape paths replaced by recorders."""
    import src.scraper.amazon.scraper as scraper_module

    monkeypatch.setattr(
        scraper_module,
        "CONFIG",
        {"global_settings": {"count_products_with_media": True}},
    )
    s = BotasaurusAmazonScraper()
    s.amazon_config = {"max_products": 5}
    return s


def _record(scraper, monkeypatch):
    """Replace both scrape paths, returning the list of paths taken."""
    taken: list[str] = []

    def _loop(self, *a, **k):
        taken.append("paginated")
        return []

    def _single(self, *a, **k):
        taken.append("single_pass")
        return []

    monkeypatch.setattr(type(scraper), "_scrape_until_validated_count_reached", _loop)
    monkeypatch.setattr(type(scraper), "_scrape_single_pass", _single)
    return taken


@pytest.mark.parametrize(
    "value",
    [
        "B0B2Z8J1MJ",
        "https://www.amazon.com/dp/B0B2Z8J1MJ",
        "https://amzn.to/4pvapDs",
        "http://tr.ee/mUk1eH",
    ],
    ids=["asin", "full-url", "shortened-https", "shortened-http"],
)
def test_a_url_or_asin_never_paginates(scraper, monkeypatch, value):
    taken = _record(scraper, monkeypatch)

    scraper.scrape_products_unified(value)

    assert taken == ["single_pass"], (
        f"{value!r} entered the page-retry loop; it names one product, so "
        "every later page re-resolves the same listing"
    )


def test_a_keyword_still_paginates(scraper, monkeypatch):
    """The other half of the gate.

    Without this the conditional could be inverted, or widened to every
    input, and the URL test above would still pass while keyword searches
    lost the retry that finds products on later pages.
    """
    taken = _record(scraper, monkeypatch)

    scraper.scrape_products_unified("wireless earbuds")

    assert taken == ["paginated"]


def test_the_url_path_still_filters_on_media_validation(scraper, monkeypatch):
    """Skipping the loop must not also skip validation.

    The no-loop branch that already existed passes `filter_validated=False`,
    because it serves the `count_products_with_media: false` case. Reusing it
    unchanged would return a media-poor product as a success instead of
    dropping it, which is a worse bug than the one being fixed.
    """
    seen: dict[str, object] = {}

    def _single(self, keyword, search_params, products_limit, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(type(scraper), "_scrape_single_pass", _single)

    scraper.scrape_products_unified("https://www.amazon.com/dp/B0B2Z8J1MJ")

    assert seen.get("filter_validated") is True


def test_the_gate_is_off_when_media_counting_is_off(monkeypatch):
    """The gate lives inside the `count_products_with_media` branch.

    With counting off there is no loop to avoid, and the pre-existing
    unfiltered path must keep serving both keywords and URLs.
    """
    import src.scraper.amazon.scraper as scraper_module

    monkeypatch.setattr(
        scraper_module,
        "CONFIG",
        {"global_settings": {"count_products_with_media": False}},
    )
    s = BotasaurusAmazonScraper()
    s.amazon_config = {"max_products": 5}

    seen: dict[str, object] = {}

    def _single(self, keyword, search_params, products_limit, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(type(s), "_scrape_single_pass", _single)

    s.scrape_products_unified("https://www.amazon.com/dp/B0B2Z8J1MJ")

    assert seen.get("filter_validated") is False


def test_a_shortened_url_is_recognised_without_a_network_call():
    """`_is_url` decides before the URL resolves.

    The gate runs on the raw input, so a shortener that hides the ASIN must
    still be recognised. Matching on `/dp/` instead would send every
    shortened URL down the paginating path.
    """
    assert BotasaurusAmazonScraper._is_url("https://amzn.to/4pvapDs") is True
    assert BotasaurusAmazonScraper._is_url("wireless earbuds") is False
