"""A no-flag standalone run rotates the configured keyword pool.

`BatchController._process_keywords` walks `config.keywords` from index 0 and
breaks at `max_products`, so a run only ever reaches the first few entries.
With the pool taken in order, `make scrape-lowpri` with no arguments searched
the same head of the list every time and returned the same products.

#248 fixed this for the global batch. The rotation helper now lives beside
`read_keyword_pillars`, which both config loaders already go through, so the
two paths cannot drift apart again.

Only the no-flag path rotates. `--keywords` is what the operator typed, and a
date-dependent result there would be surprising in a tool used to reproduce a
problem. The distinction has to be made where the pool is read, because the
CLI assigns it to `args.keywords` before `load_batch_config` sees it, and by
then a typed keyword and a configured one are indistinguishable.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.scraper.base.keyword_pillars import rotate_keyword_pool

POOL = [f"kw{i:02d}" for i in range(54)]

# 700000 * 5 % 54 == 44, so the default start is not the identity. The
# harness pins the date: run against `date.today()`, the reorder test is the
# identity whenever `ordinal % 54 == 0`, which is 2026-10-19 and every 54th
# day after, and would turn CI red with no code change.
DAY = 700_000


def _frozen_at(ordinal):
    import datetime

    class _Date(datetime.date):
        @classmethod
        def today(cls):
            return datetime.date.fromordinal(ordinal)

    return _Date


class TestTheRotationItself:
    def test_the_pool_stays_whole(self):
        """A rotation, not a slice.

        Slicing to what the run consumes removes the fallback the keyword
        loop depends on: on the bundled config that width is 1, so a barren
        keyword failed the whole run instead of moving to the next.
        """
        got = rotate_keyword_pool(POOL, 5, day_ordinal=700_000)

        assert sorted(got) == sorted(POOL)
        assert len(got) == len(POOL)

    def test_consecutive_days_start_somewhere_else(self):
        a = rotate_keyword_pool(POOL, 5, day_ordinal=700_000)
        b = rotate_keyword_pool(POOL, 5, day_ordinal=700_001)

        assert a[0] != b[0]

    def test_what_a_run_reaches_does_not_repeat_next_day(self):
        """The stride is what one run consumes, so the reached heads differ.

        A stride of one would move the start by a single keyword and repeat
        all but one of yesterday's, which is the shape that made the batch's
        rotation nearly useless.
        """
        stride = 5
        a = rotate_keyword_pool(POOL, stride, day_ordinal=700_000)[:stride]
        b = rotate_keyword_pool(POOL, stride, day_ordinal=700_001)[:stride]

        assert set(a).isdisjoint(b)

    def test_the_order_within_a_day_is_stable(self):
        assert rotate_keyword_pool(POOL, 5, day_ordinal=700_000) == (
            rotate_keyword_pool(POOL, 5, day_ordinal=700_000)
        )

    def test_an_empty_pool_stays_empty(self):
        assert rotate_keyword_pool([], 5, day_ordinal=700_000) == []

    @pytest.mark.parametrize("stride", [0, -1])
    def test_a_non_positive_stride_leaves_the_order_alone(self, stride):
        assert rotate_keyword_pool(POOL, stride, day_ordinal=700_000) == POOL


class TestBothPathsShareOneImplementation:
    """Module/Batch Alignment: the helper has one home.

    It used to live in `src/pipeline/config.py`, which the standalone scraper
    does not import, so the standalone path could only have got a rotation by
    growing a second copy.
    """

    def test_the_batch_reads_it_from_the_shared_module(self):
        import inspect

        from src.pipeline import config as pipeline_config

        source = inspect.getsource(pipeline_config)

        assert "def keywords_for_run(" not in source, (
            "the batch grew its own copy again; the shared one is in "
            "src/scraper/base/keyword_pillars.py"
        )
        assert "keywords_for_run" in source

    def test_the_standalone_scraper_reads_the_same_one(self):
        import inspect

        from src.scraper.amazon import scraper as scraper_module

        source = inspect.getsource(scraper_module)

        assert "def rotate_keyword_pool(" not in source
        assert "rotate_keyword_pool" in source

    def test_both_read_their_rotation_from_keyword_pillars(self):
        from src.pipeline.config import keywords_for_run as batch_one
        from src.scraper.amazon.scraper import rotate_keyword_pool as standalone_one
        from src.scraper.base import keyword_pillars

        assert batch_one is keyword_pillars.keywords_for_run
        assert standalone_one is keyword_pillars.rotate_keyword_pool


class TestTheCliRotatesOnlyTheConfiguredPool:
    """Drives `main()`, not the helper.

    The helper tests above pass whether or not anything calls it. The
    decision that matters is made in the CLI's config-loading branch, and it
    has two halves that must behave differently.
    """

    @staticmethod
    def _run(
        argv,
        pool,
        max_products=10,
        products_per_keyword=2,
        day=DAY,
        route="batch",
    ):
        """Run `main()` and return what it handed to `load_batch_config`.

        `route` selects which config key carries the pool: `batch` for
        `batch.keywords`, `fallback` for `scrapers.amazon.keywords`, which the
        CLI reads only when the batch block is empty.
        """
        import contextlib
        import sys
        from unittest.mock import MagicMock

        from src.scraper.amazon import scraper as scraper_module

        seen: dict[str, object] = {}

        def _capture(**kwargs):
            seen.update(kwargs)
            raise SystemExit(0)

        amazon = {"max_products": max_products}
        batch = {"products_per_keyword": products_per_keyword}
        if route == "batch":
            batch["keywords"] = {"value": list(pool)}
        else:
            amazon["keywords"] = list(pool)
        yaml_config = {"batch": batch, "scrapers": {"amazon": amazon}}

        with (
            patch.object(sys, "argv", ["scraper", *argv]),
            patch.object(scraper_module, "BotasaurusAmazonScraper", MagicMock()),
            patch.object(scraper_module.yaml, "safe_load", return_value=yaml_config),
            patch("src.scraper.base.keyword_pillars.date", _frozen_at(day)),
            patch(
                "src.scraper.amazon.config.load_batch_config",
                side_effect=_capture,
            ),
            contextlib.suppress(SystemExit),
        ):
            scraper_module.main()
        return seen

    def _keywords_reaching_the_batch(self, *args, **kwargs):
        return self._run(*args, **kwargs).get("cli_keywords")

    def test_a_no_flag_run_reorders_the_pool_without_shortening_it(self):
        got = self._keywords_reaching_the_batch([], POOL)

        assert got is not None, "main() never reached load_batch_config"
        assert sorted(got) == sorted(POOL), "the pool lost or gained entries"
        assert got != POOL, "the pool reached the batch in config order"

    def test_the_starting_keyword_moves_with_the_date(self):
        """The assertion the first version of this test was missing.

        It only checked that the list was shorter than the pool, which a
        fixed head slice satisfies -- the exact behaviour #347 was filed
        about.
        """
        a = self._keywords_reaching_the_batch([], POOL, day=DAY)
        b = self._keywords_reaching_the_batch([], POOL, day=DAY + 1)

        assert a[0] != b[0], f"same starting keyword on both days: {a[0]}"

    def test_the_stride_is_what_a_run_consumes(self):
        """A stride of one satisfies the date test above; this pins the size.

        With 10/2 the run consumes five keywords, so the five reached on day
        N must be disjoint from the five reached on day N+1. A stride of one
        would overlap on four of them.
        """
        a = self._keywords_reaching_the_batch([], POOL, day=DAY)[:5]
        b = self._keywords_reaching_the_batch([], POOL, day=DAY + 1)[:5]

        assert set(a).isdisjoint(b), f"days overlap: {sorted(set(a) & set(b))}"

    def test_the_single_product_fallback_route_rotates_too(self):
        """`scrapers.amazon.keywords`, read only when the batch block is empty.

        The previous version rotated one of the two config-read routes; this
        is the other, and nothing exercised it.
        """
        a = self._keywords_reaching_the_batch([], POOL, day=DAY, route="fallback")
        b = self._keywords_reaching_the_batch([], POOL, day=DAY + 1, route="fallback")

        assert sorted(a) == sorted(POOL)
        assert a[0] != b[0]

    def test_the_bundled_config_still_takes_the_batch_arm(self):
        """max_products and products_per_keyword are both 1 in the shipped file.

        Slicing to what a run consumes gave one keyword there, which flips
        `is_batch_mode` false and routes the run to the single-keyword arm.
        That arm has no next keyword, so a barren search failed the whole
        run, and it does not honour --products-per-keyword either.
        """
        got = self._keywords_reaching_the_batch(
            [], POOL, max_products=1, products_per_keyword=1
        )

        assert got is not None, (
            "main() never reached load_batch_config: the run fell onto the "
            "single-keyword arm, losing the rest of the pool as fallback"
        )
        assert len(got) == len(POOL)

    def test_products_per_keyword_survives_a_no_flag_run(self):
        seen = self._run(
            ["--products-per-keyword", "3"],
            POOL,
            max_products=1,
            products_per_keyword=1,
        )

        assert seen.get("cli_keywords") is not None
        assert seen.get("cli_products_per_keyword") == 3

    def test_typed_keywords_are_passed_through_untouched(self):
        """`--keywords` is reproducible; only the configured pool rotates."""
        typed = ["wireless earbuds", "smart plug", "portable ssd"]

        got = self._keywords_reaching_the_batch(["--keywords", *typed], POOL)

        assert got == typed
