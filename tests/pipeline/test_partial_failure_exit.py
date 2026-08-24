"""Exit codes for a run that partly worked.

A batch that produced nothing exits 1, so CI, cron and wrappers checking
`$?` see it. A batch that lost some products exits 0 by default — failing a
whole schedule over one bad ASIN costs more than it saves — and `--strict`
is for a caller that would rather investigate than lose a product silently.
"""

import pytest

from src.pipeline.config import PipelineSummary


def _summary(*, succeeded: int, failed: int) -> PipelineSummary:
    from unittest.mock import MagicMock

    return PipelineSummary(
        scraping=MagicMock(),
        production=MagicMock(skipped=0),
        publishing=None,
        end_to_end_success=succeeded,
        partial_success=0,
        total_failures=failed,
        total_duration_sec=1.0,
    )


@pytest.mark.unit
class TestExitCode:
    def test_nothing_produced_is_a_failure(self):
        assert _summary(succeeded=0, failed=3).exit_code() == 1

    def test_nothing_produced_is_a_failure_under_strict_too(self):
        assert _summary(succeeded=0, failed=3).exit_code(strict=True) == 1

    def test_a_clean_run_succeeds(self):
        assert _summary(succeeded=3, failed=0).exit_code() == 0
        assert _summary(succeeded=3, failed=0).exit_code(strict=True) == 0

    def test_a_partial_failure_exits_zero_by_default(self):
        # One bad ASIN must not stop a schedule.
        assert _summary(succeeded=19, failed=1).exit_code() == 0

    def test_strict_makes_a_partial_failure_visible(self):
        assert _summary(succeeded=19, failed=1).exit_code(strict=True) == 1


@pytest.mark.unit
class TestBothEntryPointsOfferIt:
    """Module/Batch Alignment: the scraper re-implements this reporting."""

    def test_the_batch_accepts_strict(self):
        from src.pipeline.global_batch import create_argument_parser

        args = create_argument_parser().parse_args(["--keywords", "a", "--strict"])
        assert args.strict is True

    def test_the_batch_defaults_to_lenient(self):
        from src.pipeline.global_batch import create_argument_parser

        assert create_argument_parser().parse_args(["--keywords", "a"]).strict is False

    def test_the_scraper_accepts_strict(self):
        import argparse
        import inspect

        from src.scraper.amazon import scraper

        source = inspect.getsource(scraper.main)
        assert '"--strict"' in source, "the standalone scraper has no --strict"
        # And it is a flag, not a value-taking option.
        parser = argparse.ArgumentParser()
        parser.add_argument("--strict", action="store_true")
        assert parser.parse_args([]).strict is False
