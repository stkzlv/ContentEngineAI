"""Telling Amazon's rate limit apart from a query that never works.

Both produce the same error page, so the scraper logged both with the same
line and spent the same three short retries on each. Neither response fit: the
block was measured clearing after several minutes, so every retry landed
inside it, and no wait at all helps a query that is simply dead.

The evidence that separates them is what the run's *other* inputs did. These
tests drive that decision directly, and drive both scraping paths with a fake
driver and an injected sleep, so nothing here waits or opens a browser.
"""

from __future__ import annotations

import pytest

from src.scraper.base.throttle import (
    AMAZON_ERROR_PAGE_MESSAGE,
    AmazonErrorPageError,
    ThrottleSettings,
    ThrottleTracker,
    Verdict,
    is_error_page_failure,
)

pytestmark = pytest.mark.unit


def _tracker(**kwargs) -> ThrottleTracker:
    return ThrottleTracker(settings=ThrottleSettings(**kwargs))


class TestTheErrorPageIsRecognisedByOneSpelling:
    def test_the_sentinel_matches(self) -> None:
        assert is_error_page_failure(RuntimeError(AMAZON_ERROR_PAGE_MESSAGE))

    def test_added_context_does_not_break_the_match(self) -> None:
        """A caller may wrap the error; the classification must survive it."""
        wrapped = RuntimeError(f"page 3 of 'usb hub': {AMAZON_ERROR_PAGE_MESSAGE}")

        assert is_error_page_failure(wrapped)

    def test_an_unrelated_failure_is_not_an_error_page(self) -> None:
        assert not is_error_page_failure(RuntimeError("connection reset"))
        assert not is_error_page_failure(TimeoutError())


class TestAThrottleIsWaitedOut:
    def test_a_first_failure_with_nothing_else_succeeding_is_a_retry(self) -> None:
        tracker = _tracker()

        assert tracker.record_error_page("usb hub") is Verdict.RETRY

    def test_the_wait_reaches_minutes_rather_than_seconds(self) -> None:
        """The defect: the old policy capped at ten seconds.

        The block clears in minutes, so a schedule topping out in seconds
        retries inside it every time and loses the input.
        """
        tracker = _tracker(backoff_base_sec=60.0, backoff_max_sec=600.0)

        tracker.record_error_page("usb hub")
        first = tracker.backoff_sec("usb hub")
        tracker.record_error_page("usb hub")
        second = tracker.backoff_sec("usb hub")

        assert first == 60.0
        assert second == 120.0

    def test_the_wait_stops_at_the_ceiling(self) -> None:
        tracker = _tracker(
            backoff_base_sec=60.0, backoff_max_sec=200.0, max_attempts=10
        )

        for _ in range(6):
            tracker.record_error_page("usb hub")

        assert tracker.backoff_sec("usb hub") == 200.0

    def test_the_shipped_budget_can_reach_the_shipped_ceiling(self) -> None:
        """Otherwise the documented remedy for a block does nothing.

        Raising `throttle_backoff_max_sec` after a persistent block is what
        the troubleshooting section tells an operator to do. With too few
        attempts the schedule never gets near the ceiling and the change is
        byte-identical.
        """
        settings = ThrottleSettings()
        tracker = ThrottleTracker(settings=settings)

        waits = []
        while tracker.record_error_page("usb hub") is Verdict.RETRY:
            waits.append(tracker.backoff_sec("usb hub"))

        assert (
            max(waits) == settings.backoff_max_sec
        ), f"waits {waits} never reach the {settings.backoff_max_sec}s ceiling"

    def test_the_budget_runs_out(self) -> None:
        tracker = _tracker(max_attempts=3)

        verdicts = [tracker.record_error_page("usb hub") for _ in range(3)]

        assert verdicts == [Verdict.RETRY, Verdict.RETRY, Verdict.EXHAUSTED]

    def test_recovering_leaves_nothing_to_report(self) -> None:
        """A wait that worked is the mechanism doing its job, not a problem."""
        tracker = _tracker()

        tracker.record_error_page("usb hub")
        tracker.record_success("usb hub")

        assert tracker.throttled_inputs == []
        assert tracker.dead_queries == []
        assert tracker.summary_lines() == []


class TestADeadQueryIsSkipped:
    def test_a_later_success_settles_it(self) -> None:
        """The whole point: one keyword failing while its neighbours work."""
        tracker = _tracker(dead_query_after=2)

        assert tracker.record_error_page("wifi extender") is Verdict.RETRY
        tracker.record_success("usb hub")

        assert tracker.record_error_page("wifi extender") is Verdict.DEAD_QUERY
        assert tracker.dead_queries == ["wifi extender"]

    def test_an_earlier_success_settles_it_too(self) -> None:
        """Order is not the evidence, and requiring it misses the common case.

        A dead query is as likely to be the last input in a run as any other,
        and nothing succeeds after the last input by definition. Requiring the
        success to come afterwards left exactly that input un-rulable, so it
        spent the whole retry budget waiting for a block that was never there.
        """
        tracker = _tracker(dead_query_after=2)

        tracker.record_success("usb hub")
        assert tracker.record_error_page("wifi extender") is Verdict.RETRY
        assert tracker.record_error_page("wifi extender") is Verdict.DEAD_QUERY

    def test_an_input_alone_in_a_run_is_never_called_dead(self) -> None:
        """With nothing to compare against, a block is the better reading."""
        tracker = _tracker(dead_query_after=2, max_attempts=4)

        verdicts = [tracker.record_error_page("wifi extender") for _ in range(4)]

        assert Verdict.DEAD_QUERY not in verdicts
        assert tracker.dead_queries == []

    def test_a_success_by_the_same_input_is_not_evidence_against_it(self) -> None:
        """It has to be a *different* input.

        Asserted on the predicate rather than through `record_error_page`,
        which also resets the input's own count on success and would pass
        either way -- two properties, and only one of them is this one.
        """
        tracker = _tracker(dead_query_after=2, max_attempts=6)

        tracker.record_success("wifi extender")

        assert not tracker._another_input_got_through("wifi extender")
        assert tracker._another_input_got_through("usb hub")

    def test_one_failure_alongside_a_success_is_not_enough(self) -> None:
        """A single coincidence should not condemn a keyword."""
        tracker = _tracker(dead_query_after=2)

        tracker.record_error_page("wifi extender")
        tracker.record_success("usb hub")

        assert tracker.dead_queries == []

    def test_a_dead_query_is_ruled_before_the_budget_is_spent(self) -> None:
        """Waiting cannot help it, so it must not consume the retry budget."""
        tracker = _tracker(dead_query_after=2, max_attempts=6)

        tracker.record_error_page("wifi extender")
        tracker.record_success("usb hub")
        verdict = tracker.record_error_page("wifi extender")

        assert verdict is Verdict.DEAD_QUERY

    def test_no_input_appears_under_both_headings(self) -> None:
        """A dead input reached the throttled list through its own retries.

        Reporting it twice told the operator to replace a keyword and, in the
        next line, that the same keyword only needed a longer gap.
        """
        tracker = _tracker(dead_query_after=2, max_attempts=6)

        # Exhausted before anything succeeded: throttled, not dead.
        for _ in range(6):
            tracker.record_error_page("laptop stand")
        tracker.record_success("usb hub")
        # Now that something has got through, this one is ruled dead.
        tracker.record_error_page("wifi extender")
        tracker.record_error_page("wifi extender")

        dead, stuck = tracker.summary_lines()

        assert "wifi extender" in dead and "laptop stand" not in dead
        assert "laptop stand" in stuck and "wifi extender" not in stuck
        assert set(tracker.dead_queries) & set(tracker.throttled_inputs) == set()

    def test_an_input_that_later_got_through_is_not_reported_dead(self) -> None:
        """The batch reuses one tracker across pages of the same keyword.

        A keyword that delivered products on page one and met the error page
        on page four is not dead, whatever the later pages did.
        """
        tracker = _tracker(dead_query_after=2, max_attempts=6)

        tracker.record_success("usb hub")
        tracker.record_error_page("wifi extender")
        tracker.record_error_page("wifi extender")
        assert tracker.dead_queries == ["wifi extender"]

        tracker.record_success("wifi extender")

        assert tracker.dead_queries == []
        assert tracker.summary_lines() == []

    def test_a_recovery_does_not_condemn_the_next_failure(self) -> None:
        """`record_success` clearing the input's own count is load-bearing.

        Without it a keyword that recovered mid-run and met the error page
        once afterwards would be ruled dead on that single failure.
        """
        tracker = _tracker(dead_query_after=2, max_attempts=6)

        tracker.record_success("usb hub")
        tracker.record_error_page("wifi extender")
        tracker.record_success("wifi extender")

        assert tracker.record_error_page("wifi extender") is Verdict.RETRY

    def test_every_tail_input_is_named_once_a_block_starts(self) -> None:
        """The misread the docs admit to, pinned so the prose stays true.

        A rate limit beginning after something succeeded makes every later
        input reach the dead-query rule, not only the first.
        """
        tracker = _tracker(dead_query_after=2, max_attempts=6)

        tracker.record_success("usb hub")
        for label in ("wifi extender", "laptop stand", "hdmi cable"):
            tracker.record_error_page(label)
            tracker.record_error_page(label)

        assert tracker.dead_queries == ["wifi extender", "laptop stand", "hdmi cable"]


class TestSettingsComeFromConfig:
    def test_an_absent_block_uses_the_defaults(self) -> None:
        assert ThrottleSettings.from_config(None) == ThrottleSettings()
        assert ThrottleSettings.from_config({}) == ThrottleSettings()

    def test_configured_values_are_read(self) -> None:
        settings = ThrottleSettings.from_config(
            {
                "inter_input_delay_sec": [1.0, 2.0],
                "throttle_backoff_base_sec": 30,
                "throttle_backoff_max_sec": 300,
                "throttle_max_attempts": 6,
                "dead_query_after": 3,
            }
        )

        assert settings.inter_input_delay_sec == (1.0, 2.0)
        assert settings.backoff_base_sec == 30.0
        assert settings.max_attempts == 6
        assert settings.dead_query_after == 3

    @pytest.mark.parametrize(
        "bad",
        [
            {"throttle_backoff_base_sec": "soon"},
            {"inter_input_delay_sec": 10},
            {"inter_input_delay_sec": [1.0, 2.0, 3.0]},
            {"inter_input_delay_sec": "fast"},
        ],
    )
    def test_a_malformed_value_falls_back_to_the_whole_defaults(self, bad) -> None:
        """Half-applying is worse: the run would not say which half it kept.

        `inter_input_delay_sec: 10` is the likely mistake, since the three
        settings beside it are scalars. It used to fall back on its own while
        every other override stuck, silently.
        """
        settings = ThrottleSettings.from_config({"throttle_max_attempts": 9, **bad})

        assert settings == ThrottleSettings()

    def test_a_ceiling_below_the_base_is_refused(self) -> None:
        with pytest.raises(ValueError, match="backoff_max_sec"):
            ThrottleSettings(backoff_base_sec=60.0, backoff_max_sec=10.0)

    def test_the_shipped_config_parses(self) -> None:
        """The YAML and the model must agree, or the file is decoration."""
        from pathlib import Path

        import yaml

        repo = Path(__file__).resolve().parents[2]
        raw = yaml.safe_load((repo / "config" / "scraper.yaml").read_text())
        block = raw["global_settings"]["rate_limiting"]

        # Assert the keys are present first. Without this the comparison
        # below passes on an empty block, since the shipped values and the
        # model defaults agree by design -- a vacuous test that would not
        # notice the config section being deleted.
        for key in (
            "inter_input_delay_sec",
            "throttle_backoff_base_sec",
            "throttle_backoff_max_sec",
            "throttle_max_attempts",
            "throttle_max_total_wait_sec",
            "dead_query_after",
        ):
            assert key in block, f"{key} is missing from the shipped config"

        settings = ThrottleSettings.from_config(block)

        assert settings == ThrottleSettings(
            inter_input_delay_sec=tuple(block["inter_input_delay_sec"]),
            backoff_base_sec=block["throttle_backoff_base_sec"],
            backoff_max_sec=block["throttle_backoff_max_sec"],
            max_attempts=block["throttle_max_attempts"],
            max_total_wait_sec=block["throttle_max_total_wait_sec"],
            dead_query_after=block["dead_query_after"],
        )


class TestTheBatchLoopActsOnTheVerdict:
    """The batch's single-session loop, driven with a fake driver."""

    @staticmethod
    def _items(*labels: str) -> list[dict]:
        return [{"keyword": label} for label in labels]

    def test_a_throttled_input_is_retried_after_a_real_wait(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.scraper.amazon import browser_functions

        calls: list[str] = []
        waits: list[float] = []

        def impl(driver, item):
            calls.append(item["keyword"])
            if len(calls) == 1:
                raise RuntimeError(AMAZON_ERROR_PAGE_MESSAGE)
            return [{"asin": "B0TEST0001"}]

        monkeypatch.setattr(
            browser_functions, "scrape_amazon_products_browser_impl", impl
        )

        results = browser_functions.scrape_batch_items(
            driver=object(),
            items=self._items("usb hub"),
            tracker=_tracker(backoff_base_sec=60.0),
            sleep=waits.append,
        )

        assert calls == ["usb hub", "usb hub"]
        assert waits == [60.0]
        assert results[0]["products"] == [{"asin": "B0TEST0001"}]

    def test_a_dead_query_is_dropped_and_labelled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.scraper.amazon import browser_functions

        def impl(driver, item):
            if item["keyword"] == "wifi extender":
                raise RuntimeError(AMAZON_ERROR_PAGE_MESSAGE)
            return [{"asin": "B0TEST0001"}]

        monkeypatch.setattr(
            browser_functions, "scrape_amazon_products_browser_impl", impl
        )
        tracker = _tracker(dead_query_after=2, max_attempts=8)

        results = browser_functions.scrape_batch_items(
            driver=object(),
            items=self._items("usb hub", "wifi extender"),
            tracker=tracker,
            sleep=lambda _: None,
        )

        dead = next(r for r in results if r["input"] == "wifi extender")
        assert dead["failure_kind"] == "dead_query"
        assert tracker.dead_queries == ["wifi extender"]

    def test_a_dead_query_does_not_spend_the_whole_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Eight attempts of waiting on a query that never clears."""
        from src.scraper.amazon import browser_functions

        attempts: list[str] = []

        def impl(driver, item):
            attempts.append(item["keyword"])
            if item["keyword"] == "wifi extender":
                raise RuntimeError(AMAZON_ERROR_PAGE_MESSAGE)
            return [{"asin": "B0TEST0001"}]

        monkeypatch.setattr(
            browser_functions, "scrape_amazon_products_browser_impl", impl
        )

        browser_functions.scrape_batch_items(
            driver=object(),
            items=self._items("usb hub", "wifi extender"),
            tracker=_tracker(dead_query_after=2, max_attempts=8),
            sleep=lambda _: None,
        )

        assert attempts.count("wifi extender") == 2

    def test_inputs_are_paced_even_when_nothing_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Back-to-back searches are what draws the block in the first place."""
        from src.scraper.amazon import browser_functions

        waits: list[float] = []
        monkeypatch.setattr(
            browser_functions,
            "scrape_amazon_products_browser_impl",
            lambda driver, item: [{"asin": "B0TEST0001"}],
        )

        browser_functions.scrape_batch_items(
            driver=object(),
            items=self._items("a", "b", "c"),
            tracker=_tracker(inter_input_delay_sec=(3.0, 3.0)),
            sleep=waits.append,
        )

        assert waits == [3.0, 3.0], "one pause between inputs, none before the first"

    def test_an_unrelated_failure_is_not_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Waiting does not fix a broken selector, and the run has other work."""
        from src.scraper.amazon import browser_functions

        attempts: list[str] = []

        def impl(driver, item):
            attempts.append(item["keyword"])
            raise RuntimeError("no such element")

        monkeypatch.setattr(
            browser_functions, "scrape_amazon_products_browser_impl", impl
        )

        results = browser_functions.scrape_batch_items(
            driver=object(),
            items=self._items("usb hub"),
            tracker=_tracker(),
            sleep=lambda _: None,
        )

        assert attempts == ["usb hub"]
        assert results[0]["error"] == "no such element"
        assert "failure_kind" not in results[0]


class TestTheStandalonePathActsOnTheVerdict:
    """The other scraper entry point, which re-implements the same decision.

    It used to carry a fixed tenacity policy: three attempts at most ten
    seconds apart. Every attempt landed inside a block that clears in minutes,
    and the same three attempts were spent on a query no wait can fix.
    """

    @staticmethod
    def _scraper(**settings):
        from src.scraper.amazon.scraper import BotasaurusAmazonScraper

        scraper = BotasaurusAmazonScraper.__new__(BotasaurusAmazonScraper)
        scraper.debug_mode = False
        scraper.logger = __import__("logging").getLogger("test")
        scraper.throttle = _tracker(**settings)
        return scraper

    def test_a_throttled_input_waits_minutes_and_retries(self) -> None:
        waits: list[float] = []
        calls: list[int] = []

        def browser_func(data):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError(AMAZON_ERROR_PAGE_MESSAGE)
            return ["product"]

        scraper = self._scraper(backoff_base_sec=60.0)

        result = scraper._scrape_with_retry(
            browser_func, {"keyword": "usb hub"}, sleep=waits.append
        )

        assert result == ["product"]
        assert waits == [60.0], "the old policy capped this at ten seconds"

    def test_a_dead_query_raises_instead_of_waiting_again(self) -> None:
        waits: list[float] = []
        scraper = self._scraper(dead_query_after=2, max_attempts=8)
        scraper.throttle.record_success("usb hub")

        def browser_func(data):
            raise RuntimeError(AMAZON_ERROR_PAGE_MESSAGE)

        with pytest.raises(RuntimeError, match="Amazon error page"):
            scraper._scrape_with_retry(
                browser_func, {"keyword": "wifi extender"}, sleep=waits.append
            )

        assert scraper.throttle.dead_queries == ["wifi extender"]
        assert len(waits) == 1, "one backoff, then the verdict; not the whole budget"

    def test_an_unrelated_runtime_error_is_raised_untouched(self) -> None:
        scraper = self._scraper()

        def browser_func(data):
            raise RuntimeError("no such element")

        with pytest.raises(RuntimeError, match="no such element"):
            scraper._scrape_with_retry(
                browser_func, {"keyword": "usb hub"}, sleep=lambda _: None
            )

    def test_a_success_is_recorded_so_neighbours_can_be_ruled_on(self) -> None:
        """Both paths feed the same tracker; a success here is evidence there."""
        scraper = self._scraper()

        scraper._scrape_with_retry(
            lambda data: ["product"], {"keyword": "usb hub"}, sleep=lambda _: None
        )

        assert scraper.throttle._another_input_got_through("wifi extender")


class TestTheRunHasAWaitBudget:
    """Per-input budgets do not compose.

    Five inputs at fifteen minutes each is seventy-five minutes of an
    unattended run sleeping, and by the second exhaustion with nothing having
    succeeded the answer is already known.
    """

    def test_a_fully_blocked_run_stops_waiting(self) -> None:
        from src.scraper.amazon import browser_functions

        waits: list[float] = []

        def impl(driver, item):
            raise RuntimeError(AMAZON_ERROR_PAGE_MESSAGE)

        tracker = _tracker(max_total_wait_sec=1800.0)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(browser_functions, "scrape_amazon_products_browser_impl", impl)
            browser_functions.scrape_batch_items(
                driver=object(),
                items=[{"keyword": c} for c in "abcde"],
                tracker=tracker,
                sleep=waits.append,
            )

        # The cap governs backoff. The inter-input pacing is seconds and is
        # deliberately outside it, so this reads the tracker rather than the
        # sleep log, which holds both.
        assert (
            tracker.total_wait_sec <= 1800.0
        ), f"the run committed {tracker.total_wait_sec / 60:.0f} minutes"
        assert sum(waits) < 1800.0 + 5 * 5, "pacing should be seconds, not minutes"

    def test_the_budget_is_charged_by_the_tracker(self) -> None:
        """A caller that forgets to report its sleep cannot spend it unseen."""
        tracker = _tracker(backoff_base_sec=60.0, max_attempts=6)

        tracker.record_error_page("usb hub")
        tracker.backoff_sec("usb hub")
        tracker.record_error_page("usb hub")
        tracker.backoff_sec("usb hub")

        assert tracker.total_wait_sec == 180.0

    def test_an_input_ends_once_the_budget_is_gone(self) -> None:
        tracker = _tracker(
            backoff_base_sec=60.0, max_attempts=99, max_total_wait_sec=100.0
        )

        assert tracker.record_error_page("usb hub") is Verdict.RETRY
        tracker.backoff_sec("usb hub")

        # The next wait would be 120s against 40s of headroom, so the input
        # ends here rather than overshooting the cap and stopping afterwards.
        assert tracker.record_error_page("usb hub") is Verdict.EXHAUSTED
        assert tracker.total_wait_sec <= 100.0

    def test_the_cap_is_a_ceiling_not_a_tripwire(self) -> None:
        """Checked against what the retry would cost, not what was spent.

        Checking after the fact let an input already mid-schedule overshoot
        by its whole remaining schedule.
        """
        tracker = _tracker(
            backoff_base_sec=60.0, max_attempts=99, max_total_wait_sec=150.0
        )

        while tracker.record_error_page("usb hub") is Verdict.RETRY:
            tracker.backoff_sec("usb hub")

        assert tracker.total_wait_sec <= 150.0


class TestTheErrorPageIsCheckedOnEveryArmAndEveryMode:
    """The checks used to sit inside `if DEBUG_MODE:` blocks.

    The bundled config ships `debug_mode: false`, so on a normal run Amazon's
    error page produced an empty result and no exception at all -- which the
    batch loop then recorded as a success, turning a throttled input into
    evidence that the connection works. One of the two was also in the
    keyword-search branch only, so a scrape by ASIN or by URL never reached it
    even with `--debug`.
    """

    class _Driver:
        title = f"Amazon.com: {AMAZON_ERROR_PAGE_MESSAGE.split(': ')[1]}"
        current_url = "https://www.amazon.com/errors/validateCaptcha"

        def google_get(self, *args, **kwargs):
            return None

        def short_random_sleep(self):
            return None

        def run_js(self, *args, **kwargs):
            return None

    def test_the_helper_raises_on_the_error_page(self) -> None:
        from src.scraper.amazon.browser_functions import raise_if_error_page

        with pytest.raises(RuntimeError, match="Amazon error page"):
            raise_if_error_page(self._Driver())

    def test_a_normal_page_passes(self) -> None:
        from src.scraper.amazon.browser_functions import raise_if_error_page

        driver = self._Driver()
        driver.title = "Amazon.com: usb hub"

        raise_if_error_page(driver)

    def test_an_unreadable_title_is_not_an_error_page(self) -> None:
        """A driver that cannot be read is a different failure entirely."""
        from src.scraper.amazon.browser_functions import raise_if_error_page

        class _Broken:
            @property
            def title(self):
                raise RuntimeError("session closed")

        raise_if_error_page(_Broken())

    @pytest.mark.parametrize(
        ("arm", "item"),
        [
            ("keyword", {"keyword": "usb hub"}),
            ("asin", {"keyword": "B0TEST0001", "is_asin": True}),
            ("url", {"keyword": "https://a.co/d/xyz", "is_url": True}),
        ],
    )
    @pytest.mark.parametrize("debug", [False, True])
    def test_every_arm_raises_in_every_mode(self, arm, item, debug) -> None:
        from src.scraper.amazon import browser_functions

        # `debug_mode` on the item is the only switch; the impl reads it into
        # a local. That is what made the old checks unreachable by default.
        with pytest.raises(RuntimeError, match="Amazon error page"):
            browser_functions.scrape_amazon_products_browser_impl(
                self._Driver(), {**item, "debug_mode": debug}
            )


class _StubDriver:
    """Enough of a Botasaurus driver for the decorator's own bookkeeping."""

    config = None

    def close(self):
        return None

    def quit(self):
        return None


class TestTheStandaloneWrapperLetsTheErrorPageOut:
    """The swallow sat one level above the check.

    `create_dynamic_browser_function` wraps the impl in `except Exception:
    return []`, so the standalone path saw an empty page indistinguishable
    from a keyword with no matches -- and `_scrape_with_retry` then recorded
    the throttled input as a success, making it evidence that the connection
    works. None of the backoff, the verdicts or the pagination break could
    fire there.
    """

    @staticmethod
    def _inner(monkeypatch, impl):
        """The real decorator, with a stub driver instead of Chrome."""
        from src.scraper.amazon import browser_functions

        # The REAL decorator, with a stub driver factory. A pass-through
        # double is worthless here: Botasaurus catches everything the
        # decorated function raises, re-runs the whole task `max_retry` times
        # and returns None, so a double that propagates asserts the opposite
        # of what production does. The escape is `must_raise_exceptions`,
        # which only the real decorator honours.
        monkeypatch.setattr(
            browser_functions, "scrape_amazon_products_browser_impl", impl
        )
        config = browser_functions._build_browser_config(False)
        config["create_driver"] = lambda *a, **kw: _StubDriver()
        config["parallel"] = 1
        monkeypatch.setattr(
            browser_functions,
            "_build_browser_config",
            lambda debug_mode=False: config,
        )
        return browser_functions.create_dynamic_browser_function(False)

    def test_the_error_page_reaches_the_caller(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def impl(driver, data):
            raise AmazonErrorPageError(AMAZON_ERROR_PAGE_MESSAGE)

        func = self._inner(monkeypatch, impl)

        with pytest.raises(AmazonErrorPageError):
            func({"keyword": "usb hub"})

    def test_the_message_alone_is_not_enough(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The decorator matches by class, so the type is load-bearing.

        `is_error_page_failure` matches on the message, which is right for
        the caller. It is not what gets the exception past Botasaurus, so a
        site that raises a bare RuntimeError with the same text is swallowed
        and retried four times into a live block.
        """
        calls: list[int] = []

        def impl(driver, data):
            calls.append(1)
            raise RuntimeError(AMAZON_ERROR_PAGE_MESSAGE)

        func = self._inner(monkeypatch, impl)

        assert func({"keyword": "usb hub"}) is None
        assert len(calls) > 1, "swallowed and retried, as the type exists to avoid"
        assert issubclass(
            AmazonErrorPageError, RuntimeError
        ), "the caller catches RuntimeError; the type must stay a subclass"

    def test_every_other_failure_is_still_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One page failing must not lose the rest of the run."""

        def impl(driver, data):
            raise RuntimeError("no such element")

        func = self._inner(monkeypatch, impl)

        assert func({"keyword": "usb hub"}) == []

    def test_a_throttled_input_is_not_recorded_as_a_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The worst consequence: it became evidence the connection works."""
        from src.scraper.amazon.scraper import BotasaurusAmazonScraper

        def impl(driver, data):
            raise AmazonErrorPageError(AMAZON_ERROR_PAGE_MESSAGE)

        func = self._inner(monkeypatch, impl)

        scraper = BotasaurusAmazonScraper.__new__(BotasaurusAmazonScraper)
        scraper.debug_mode = False
        scraper.logger = __import__("logging").getLogger("test")
        scraper.throttle = _tracker(max_attempts=2)

        with pytest.raises(RuntimeError, match="Amazon error page"):
            scraper._scrape_with_retry(
                func, {"keyword": "wifi extender"}, sleep=lambda _: None
            )

        assert not scraper.throttle._another_input_got_through("usb hub")
        assert scraper.throttle.throttled_inputs == ["wifi extender"]


class TestAnInputThatSucceededThenStuckIsStillReported:
    """Both lists filtered on "ever succeeded", which erased it.

    The batch's page-retry loop reuses one tracker, so a keyword that
    delivered on page one and met the error page from page two onward waited
    out its whole budget and appeared in no summary line at all.
    """

    def test_it_appears_in_the_throttled_line(self) -> None:
        tracker = _tracker(max_attempts=3)

        tracker.record_success("usb hub")
        verdicts = [tracker.record_error_page("usb hub") for _ in range(3)]

        assert verdicts[-1] is Verdict.EXHAUSTED
        assert tracker.throttled_inputs == ["usb hub"]
        assert "usb hub" in "\n".join(tracker.summary_lines())

    def test_a_keyword_that_delivered_is_never_called_a_dead_query(self) -> None:
        """The batch's page-retry loop only runs after page one delivered.

        So reading the most recent outcome here named the keyword that had
        just produced products, and told the operator to replace it.
        """
        tracker = _tracker(dead_query_after=2, max_attempts=9)

        tracker.record_success("other keyword")
        tracker.record_success("usb hub")
        for _ in range(3):
            tracker.record_error_page("usb hub")

        assert tracker.dead_queries == []
        assert tracker.throttled_inputs == ["usb hub"]

    def test_a_recovery_after_the_failures_clears_it_again(self) -> None:
        tracker = _tracker(max_attempts=3)

        tracker.record_success("usb hub")
        for _ in range(3):
            tracker.record_error_page("usb hub")
        tracker.record_success("usb hub")

        assert tracker.summary_lines() == []

    def test_a_success_does_not_restore_the_run_budget(self) -> None:
        """Resetting it reads fairer and stops the cap capping anything.

        Measured on the shipped settings, a run where one input in five still
        got through spent eight hours asleep against a configured one. An
        operator who sets a bound wants a bound.
        """
        tracker = _tracker(backoff_base_sec=60.0, max_attempts=9)

        tracker.record_error_page("usb hub")
        tracker.backoff_sec("usb hub")

        tracker.record_success("usb hub")

        assert tracker.total_wait_sec == 60.0

    def test_a_mixed_run_stays_inside_the_budget(self) -> None:
        """The shape the reset made unbounded: blocks broken by successes."""
        tracker = _tracker(
            backoff_base_sec=60.0, max_attempts=9, max_total_wait_sec=300.0
        )

        for i in range(40):
            label = f"input-{i}"
            while tracker.record_error_page(label) is Verdict.RETRY:
                tracker.backoff_sec(label)
            if i % 5 == 0:
                tracker.record_success(f"other-{i}")

        assert tracker.total_wait_sec <= 300.0, (
            f"the run committed {tracker.total_wait_sec / 60:.0f} minutes "
            "against a five-minute cap"
        )


class TestAMalformedRateLimitingBlock:
    @pytest.mark.parametrize("raw", [[2.0, 5.0], "fast", 10])
    def test_a_non_mapping_block_does_not_raise(self, raw) -> None:
        """It used to kill the scraper's constructor with AttributeError."""
        assert ThrottleSettings.from_config(raw) == ThrottleSettings()


class TestTheStandalonePathIsPacedToo:
    """Module/Batch Alignment: this arm launches a fresh browser per input."""

    @staticmethod
    def _controller(monkeypatch, keywords):
        from unittest.mock import Mock

        from src.scraper.amazon.batch_controller import BatchController
        from src.scraper.amazon.models import BatchConfig, SearchParameters

        waits: list[float] = []
        monkeypatch.setattr(
            "src.scraper.amazon.batch_controller.time.sleep", waits.append
        )

        scraper = Mock()
        scraper.logger = __import__("logging").getLogger("test")
        scraper.throttle = _tracker(inter_input_delay_sec=(3.0, 3.0))
        scraper.scrape_products_unified.return_value = []
        scraper.pillar_for_keyword.return_value = None

        config = BatchConfig(
            product_ids=[],
            keywords=keywords,
            fail_fast=False,
            search_params=SearchParameters(),
            max_products=10,
            products_per_keyword=1,
        )
        return BatchController(scraper, config), waits

    def test_one_pause_between_keywords_and_none_before_the_first(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        controller, waits = self._controller(
            monkeypatch, ["usb hub", "wifi extender", "hdmi cable"]
        )

        controller._process_keywords()

        assert waits == [3.0, 3.0]
