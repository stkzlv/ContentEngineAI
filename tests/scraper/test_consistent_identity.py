"""The returning-browser identity behind `consistent_identity` (REQ-SCR-081).

With the switch on the browser keeps its own user agent, a window size drawn
once, and cookies across runs, and opens product pages from the results page.
With it off, nothing changes: a random user agent from the driver library and
a fresh window draw per run (REQ-SCR-039, REQ-SCR-041).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.scraper.amazon import browser_functions, identity
from src.scraper.base.throttle import (
    AmazonBlockError,
    AmazonErrorPageError,
    PageKind,
    ThrottleTracker,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUTS_DIR", str(tmp_path))
    monkeypatch.delenv("CONTENT_ENGINE_OUTPUT", raising=False)
    return tmp_path


def _switch(monkeypatch, on: bool) -> None:
    monkeypatch.setattr(browser_functions, "consistent_identity_on", lambda: on)


class _CookieDriver:
    def __init__(self, cookies=None):
        self.cookies = list(cookies or [])
        self.added: list = []

    def add_cookies(self, cookies):
        self.added.extend(cookies)

    def get_cookies(self):
        return self.cookies

    def run_js(self, _script):
        return {"user_agent": "UA/1", "hints": "Chromium/1"}


@pytest.mark.req("REQ-SCR-081")
class TestTheIdentityPersists:
    def test_the_window_is_drawn_once_and_kept(self, outputs):
        first = identity.window_size()

        assert all(identity.window_size() == first for _ in range(20))
        stored = json.loads((outputs / "state" / identity.IDENTITY_FILE).read_text())
        assert tuple(stored["window_size"]) == first

    def test_a_stored_narrow_window_is_redrawn_desktop_width(self, outputs):
        path = outputs / "state" / identity.IDENTITY_FILE
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"window_size": [800, 600]}))

        width, _ = identity.window_size()

        assert width >= 1280

    def test_cookies_carry_from_one_session_to_the_next(self, outputs):
        cookies = [{"name": "session-id", "value": "x", "domain": ".example.test"}]
        assert identity.save_cookies(_CookieDriver(cookies)) == 1

        later = _CookieDriver()

        assert identity.load_cookies(later) == 1
        assert later.added == cookies

    def test_the_browser_sends_its_own_user_agent(self, outputs):
        config = identity.apply_to_browser_config({"user_agent": "RANDOM"})

        assert config["user_agent"] == identity.REAL_USER_AGENT
        assert tuple(config["window_size"]) == identity.window_size()


@pytest.mark.req("REQ-SCR-081")
@pytest.mark.req("REQ-SCR-039")
class TestTheSwitch:
    def test_off_keeps_the_random_user_agent(self, monkeypatch, outputs):
        _switch(monkeypatch, False)

        config = browser_functions._build_browser_config()

        assert config.get("user_agent") != identity.REAL_USER_AGENT
        assert not (outputs / "state" / identity.IDENTITY_FILE).exists()

    def test_on_uses_the_real_user_agent_and_the_kept_window(
        self, monkeypatch, outputs
    ):
        _switch(monkeypatch, True)

        first = browser_functions._build_browser_config()
        second = browser_functions._build_browser_config()

        assert first["user_agent"] == identity.REAL_USER_AGENT
        assert first["window_size"] == second["window_size"]

    def test_on_carries_cookies_through_the_session(self, monkeypatch, outputs):
        _switch(monkeypatch, True)
        (outputs / "state").mkdir()
        (outputs / "state" / identity.COOKIES_FILE).write_text(json.dumps([{"n": 1}]))
        driver = _CookieDriver([{"n": 2}])

        with browser_functions.returning_identity(driver):
            assert driver.added == [{"n": 1}]

        assert json.loads((outputs / "state" / identity.COOKIES_FILE).read_text()) == [
            {"n": 2}
        ]

    def test_off_leaves_cookies_alone(self, monkeypatch, outputs):
        _switch(monkeypatch, False)
        driver = MagicMock()

        with browser_functions.returning_identity(driver):
            pass

        driver.add_cookies.assert_not_called()
        driver.get_cookies.assert_not_called()


@pytest.mark.req("REQ-SCR-081")
@pytest.mark.req("REQ-SCR-083")
class TestABlockedSessionsCookiesAreDropped:
    """The block travels with the kept cookies, so a block page drops them."""

    def _kept(self, outputs):
        path = outputs / "state" / identity.COOKIES_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([{"n": 1}]))
        return path

    @pytest.mark.parametrize("kind", [PageKind.BARE_TITLE, PageKind.CAPTCHA])
    def test_a_session_block_drops_them_and_saves_none(
        self, monkeypatch, outputs, kind
    ):
        _switch(monkeypatch, True)
        path = self._kept(outputs)
        driver = _CookieDriver([{"n": 2}])

        with (
            pytest.raises(AmazonBlockError),
            browser_functions.returning_identity(driver),
        ):
            raise AmazonBlockError(kind)

        assert not path.exists()

    def test_the_error_page_keeps_them(self, monkeypatch, outputs):
        """The error page is a rate limit, not the session."""
        _switch(monkeypatch, True)
        path = self._kept(outputs)

        with (
            pytest.raises(AmazonBlockError),
            browser_functions.returning_identity(_CookieDriver([{"n": 2}])),
        ):
            raise AmazonErrorPageError()

        assert json.loads(path.read_text()) == [{"n": 2}]

    def test_the_batch_session_loop_drops_them_too(self, monkeypatch, outputs):
        _switch(monkeypatch, True)
        path = self._kept(outputs)

        def impl(_driver, _item):
            raise AmazonBlockError(PageKind.BARE_TITLE)

        monkeypatch.setattr(
            browser_functions, "scrape_amazon_products_browser_impl", impl
        )
        driver = _CookieDriver([{"n": 2}])

        with browser_functions.returning_identity(driver):
            browser_functions.scrape_batch_items(
                driver, [{"keyword": "a"}], ThrottleTracker(), sleep=lambda _s: None
            )

        assert not path.exists()


@pytest.mark.req("REQ-SCR-081")
class TestProductPagesFromTheResults:
    def test_a_result_is_opened_by_clicking_it(self):
        link = MagicMock()
        card = MagicMock()
        card.select.return_value = link
        driver = MagicMock()
        driver.current_url = "https://www.amazon.com/dp/B0TEST0001?ref=sr"

        assert browser_functions._open_from_results(driver, card, "B0TEST0001")
        link.click.assert_called_once()

    @pytest.mark.parametrize(
        "card_link, url",
        [(None, "https://www.amazon.com/s?k=x"), (MagicMock(), "https://elsewhere/")],
    )
    def test_no_link_or_a_click_that_lands_elsewhere_falls_back(self, card_link, url):
        card = MagicMock()
        card.select.return_value = card_link
        driver = MagicMock()
        driver.current_url = url
        now = iter(range(100))

        assert not browser_functions._open_from_results(
            driver, card, "B0TEST0001", sleep=lambda _s: None, clock=lambda: next(now)
        )

    def test_a_slow_product_page_is_waited_for(self):
        link = MagicMock()
        card = MagicMock()
        card.select.return_value = link
        driver = MagicMock()
        urls = iter(["https://www.amazon.com/s?k=x"] * 3)
        type(driver).current_url = property(
            lambda _self: next(urls, "https://www.amazon.com/dp/B0TEST0001")
        )
        pauses: list[float] = []

        assert browser_functions._open_from_results(
            driver, card, "B0TEST0001", sleep=pauses.append, clock=lambda: 0.0
        )
        assert len(pauses) == 3

    @pytest.mark.parametrize("on", [True, False])
    def test_the_search_arm_navigates_only_when_off(self, monkeypatch, on):
        _switch(monkeypatch, on)
        monkeypatch.setattr(
            browser_functions,
            "raise_if_blocked",
            lambda _d: PageKind.RESULTS,
        )
        monkeypatch.setattr(
            browser_functions,
            "extract_serp_product_info",
            lambda card, _kw: SimpleNamespace(
                asin="B0TEST0001", url="https://www.amazon.com/dp/B0TEST0001"
            ),
        )
        monkeypatch.setattr(
            browser_functions,
            "extract_product_data_from_page",
            lambda *_a, **_k: {"asin": "B0TEST0001", "images": ["x"], "videos": []},
        )
        monkeypatch.setattr(browser_functions, "_open_from_results", lambda *_a: True)
        driver = MagicMock()
        driver.is_bot_detected.return_value = False
        driver.select_all.return_value = [MagicMock()]

        browser_functions.scrape_amazon_products_browser_impl(
            driver, {"keyword": "phone stand", "debug_mode": False, "max_products": 1}
        )

        product_visits = [
            c for c in driver.google_get.call_args_list if "/dp/" in str(c.args[0])
        ]
        assert bool(product_visits) is (not on)


def _two_card_search(monkeypatch, on: bool, product_effects):
    """Run the search arm over two cards; `product_effects` per product page."""
    _switch(monkeypatch, on)
    monkeypatch.setattr(
        browser_functions, "raise_if_blocked", lambda _d: PageKind.RESULTS
    )
    asins = iter(["B0TEST0001", "B0TEST0002"])
    monkeypatch.setattr(
        browser_functions,
        "extract_serp_product_info",
        lambda card, _kw: SimpleNamespace(
            asin=card._mock_name, url=f"https://www.amazon.com/dp/{card._mock_name}"
        ),
    )
    effects = iter(product_effects)

    def extract(*_a, **_k):
        effect = next(effects)
        if isinstance(effect, Exception):
            raise effect
        return effect

    monkeypatch.setattr(browser_functions, "extract_product_data_from_page", extract)
    monkeypatch.setattr(browser_functions, "_open_from_results", lambda *_a: True)
    driver = MagicMock()
    driver.is_bot_detected.return_value = False
    cards = [MagicMock(name=next(asins)), MagicMock(name="B0TEST0002")]
    driver.select_all.return_value = cards
    browser_functions.scrape_amazon_products_browser_impl(
        driver, {"keyword": "phone stand", "debug_mode": False, "max_products": 2}
    )
    return driver


@pytest.mark.req("REQ-SCR-081")
class TestTheWayBackToTheResults:
    PRODUCT = {"asin": "B0TEST0001", "images": ["x"], "videos": []}

    def test_on_steps_back_through_history(self, monkeypatch):
        driver = _two_card_search(monkeypatch, True, [self.PRODUCT, self.PRODUCT])

        driver.run_js.assert_any_call("history.back()")
        searches = [
            c for c in driver.google_get.call_args_list if "/s?" in str(c.args[0])
        ]
        assert len(searches) == 1, "only the first search; the way back is history"

    def test_off_navigates_to_the_search_again(self, monkeypatch):
        driver = _two_card_search(monkeypatch, False, [self.PRODUCT, self.PRODUCT])

        searches = [
            c for c in driver.google_get.call_args_list if "/s?" in str(c.args[0])
        ]
        assert len(searches) == 2


@pytest.mark.req("REQ-SCR-081")
def test_a_block_after_a_product_still_drops_the_cookies(monkeypatch, outputs):
    path = outputs / "state" / identity.COOKIES_FILE
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps([{"n": 1}]))
    product = {"asin": "B0TEST0001", "images": ["x"], "videos": []}

    with browser_functions.returning_identity(_CookieDriver([{"n": 2}])):
        _two_card_search(
            monkeypatch, True, [product, AmazonBlockError(PageKind.BARE_TITLE)]
        )

    assert not path.exists()
