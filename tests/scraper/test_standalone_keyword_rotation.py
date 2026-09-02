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

import pytest

from src.scraper.base.keyword_pillars import keywords_for_run

POOL = [f"kw{i:02d}" for i in range(54)]


class TestTheRotationItself:
    def test_consecutive_days_share_nothing(self):
        """The stride is the slice width, not one.

        Stepping by one would repeat all but one of yesterday's keywords,
        which is the shape that made the batch's rotation nearly useless.
        """
        day = 700_000
        a = keywords_for_run(POOL, 10, day_ordinal=day)
        b = keywords_for_run(POOL, 10, day_ordinal=day + 1)

        assert set(a).isdisjoint(b)

    def test_a_full_width_slice_would_be_the_identity(self):
        """Why the width is what the run consumes, not the pool size.

        At `count == len(pool)` the start offset is a multiple of the length,
        so it is zero on every day. This is documented as a property rather
        than guarded against, because the caller choosing the width is what
        keeps it from happening.
        """
        for day in (700_000, 700_001, 700_002):
            assert keywords_for_run(POOL, len(POOL), day_ordinal=day) == POOL

    def test_the_pool_is_covered_before_it_repeats(self):
        day = 700_000
        seen: set[str] = set()
        for offset in range(6):
            seen.update(keywords_for_run(POOL, 10, day_ordinal=day + offset))

        assert seen == set(POOL)

    def test_it_never_returns_more_than_the_pool_holds(self):
        assert keywords_for_run(["a", "b"], 10, day_ordinal=700_000) == ["a", "b"]

    @pytest.mark.parametrize("count", [0, -1])
    def test_a_non_positive_width_selects_nothing(self, count):
        assert keywords_for_run(POOL, count, day_ordinal=700_000) == []


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

        assert "def keywords_for_run(" not in source
        assert "keywords_for_run" in source

    def test_both_import_it_from_keyword_pillars(self):
        from src.pipeline.config import keywords_for_run as batch_one
        from src.scraper.amazon.scraper import keywords_for_run as standalone_one

        assert batch_one is keywords_for_run
        assert standalone_one is keywords_for_run


class TestTheCliRotatesOnlyTheConfiguredPool:
    """Drives `main()`, not the helper.

    The helper tests above pass whether or not anything calls it. The
    decision that matters is made in the CLI's config-loading branch, and it
    has two halves that must behave differently.
    """

    @staticmethod
    def _keywords_reaching_the_batch(argv, pool):
        """Run `main()` and return the keyword list it handed downstream."""
        import contextlib
        import sys
        from unittest.mock import MagicMock, patch

        from src.scraper.amazon import scraper as scraper_module

        seen: dict[str, object] = {}

        def _capture(**kwargs):
            seen.update(kwargs)
            raise SystemExit(0)

        yaml_config = {
            "batch": {"keywords": {"value": list(pool)}, "products_per_keyword": 2},
            "scrapers": {"amazon": {"max_products": 10}},
        }

        with (
            patch.object(sys, "argv", ["scraper", *argv]),
            patch.object(scraper_module, "BotasaurusAmazonScraper", MagicMock()),
            patch.object(scraper_module.yaml, "safe_load", return_value=yaml_config),
            patch(
                "src.scraper.amazon.config.load_batch_config",
                side_effect=_capture,
            ),
            contextlib.suppress(SystemExit),
        ):
            scraper_module.main()
        return seen.get("cli_keywords")

    def test_a_no_flag_run_gets_a_slice_not_the_whole_pool(self):
        got = self._keywords_reaching_the_batch([], POOL)

        assert got is not None, "main() never reached load_batch_config"
        assert len(got) < len(POOL), (
            "the whole 54-keyword pool reached the batch; the run stops at "
            "max_products, so only the head would ever be searched"
        )
        assert set(got) <= set(POOL)

    def test_typed_keywords_are_passed_through_untouched(self):
        """`--keywords` is reproducible; only the pool rotates."""
        typed = ["wireless earbuds", "smart plug", "portable ssd"]

        got = self._keywords_reaching_the_batch(["--keywords", *typed], POOL)

        assert got == typed
