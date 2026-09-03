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
        """It has to be a *different* input; otherwise a recovery condemns it."""
        tracker = _tracker(dead_query_after=2, max_attempts=6)

        tracker.record_error_page("wifi extender")
        tracker.record_success("wifi extender")

        assert tracker.record_error_page("wifi extender") is Verdict.RETRY
        assert tracker.dead_queries == []

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

    def test_the_two_are_named_separately_in_the_summary(self) -> None:
        tracker = _tracker(dead_query_after=2, max_attempts=2)

        tracker.record_error_page("wifi extender")
        tracker.record_success("usb hub")
        tracker.record_error_page("wifi extender")
        tracker.record_error_page("laptop stand")
        tracker.record_error_page("laptop stand")

        lines = "\n".join(tracker.summary_lines())

        assert "wifi extender" in lines
        assert "Dead queries" in lines
        assert "laptop stand" in lines
        assert "Rate-limited" in lines


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

    def test_a_malformed_value_falls_back_to_the_whole_defaults(self) -> None:
        """Half-applying is worse: the run would not say which half it kept."""
        settings = ThrottleSettings.from_config(
            {"throttle_backoff_base_sec": "soon", "throttle_max_attempts": 6}
        )

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
            "dead_query_after",
        ):
            assert key in block, f"{key} is missing from the shipped config"

        settings = ThrottleSettings.from_config(block)

        assert settings == ThrottleSettings(
            inter_input_delay_sec=tuple(block["inter_input_delay_sec"]),
            backoff_base_sec=block["throttle_backoff_base_sec"],
            backoff_max_sec=block["throttle_backoff_max_sec"],
            max_attempts=block["throttle_max_attempts"],
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
