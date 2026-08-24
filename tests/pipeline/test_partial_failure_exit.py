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

    def test_the_producer_accepts_strict(self):
        from src.video.producer.cli import create_argument_parser

        parser = create_argument_parser()
        assert parser.parse_args(["--batch", "--strict"]).strict is True
        assert parser.parse_args(["--batch"]).strict is False


@pytest.mark.unit
class TestTheScraperUnderStrict:
    """Driven through `main`, because the flag has to reach an exit code.

    A source grep passes while the flag is registered as a value-taking
    option, or read from a summary field the arm under test never fills.
    """

    @staticmethod
    def _run(argv, *, summary):
        import sys
        from unittest.mock import MagicMock, patch

        from src.scraper.amazon import scraper as scraper_module

        controller = MagicMock()
        controller.run_batch.return_value = summary

        with (
            patch.object(sys, "argv", ["scraper", *argv]),
            patch.object(scraper_module, "BotasaurusAmazonScraper", MagicMock()),
            patch(
                "src.scraper.amazon.batch_controller.BatchController",
                return_value=controller,
            ),
            patch("src.scraper.amazon.config.load_batch_config", MagicMock()),
        ):
            try:
                scraper_module.main()
            except SystemExit as exit_info:
                return exit_info.code
            return 0

    @staticmethod
    def _summary(*, successful, failed=0, failed_keywords=()):
        from src.scraper.amazon.models import BatchSummary

        return BatchSummary(
            total_attempted=successful + failed,
            product_ids_attempted=0,
            keywords_attempted=1,
            successful=successful,
            failed=failed,
            successful_products=["B0AAAAAAAA"] * successful,
            failed_products=["B0BBBBBBBB"] * failed,
            media_stats={},
            duration_sec=1.0,
            failed_keywords=list(failed_keywords),
        )

    def test_a_lost_keyword_trips_strict(self):
        # The keyword arm records no per-product failure, so a check reading
        # only `failed` would pass here while the keyword was lost.
        code = self._run(
            ["--keywords", "a", "b", "--strict"],
            summary=self._summary(successful=1, failed_keywords=["b"]),
        )
        assert code == 1

    def test_a_lost_keyword_is_tolerated_without_the_flag(self):
        code = self._run(
            ["--keywords", "a", "b"],
            summary=self._summary(successful=1, failed_keywords=["b"]),
        )
        assert code == 0

    def test_a_failed_product_trips_strict(self):
        code = self._run(
            ["--product-ids", "B0AAAAAAAA", "B0BBBBBBBB", "--strict"],
            summary=self._summary(successful=1, failed=1),
        )
        assert code == 1

    def test_a_clean_run_succeeds_under_strict(self):
        code = self._run(
            ["--keywords", "a", "b", "--strict"],
            summary=self._summary(successful=2),
        )
        assert code == 0
