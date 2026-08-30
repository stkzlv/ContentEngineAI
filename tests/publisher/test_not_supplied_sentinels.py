"""Two flags whose "off" value collided with "not supplied".

The same shape twice, and the same shape as the chunked-keywords defect: a
consumer resolves an option with `x if x is not None else <fallback>`, and the
caller hands it a value that is falsy but *supplied*. The fallback then never
runs, or runs when it should not.

- `--fail-fast` is `action="store_true"`, so an omitted flag arrived as
  `False`. `False is not None`, so the CLI default always won and
  `batch.fail_fast` in the YAML was unreachable.
- `--no-cleanup` was expressed by passing `cleanup_config=None`, which
  `auto_schedule` reads as "caller supplied nothing" and answers with a
  default `CleanupConfig()` whose `enabled` is True. The flag was a no-op.

Neither logged anything. A user setting `fail_fast: true` got a run that kept
going, and a user passing `--no-cleanup` got their directories deleted.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.publisher.models import CleanupConfig, RecurringSlot, ScheduleConfig


def _enabled_schedule() -> ScheduleConfig:
    """`auto_schedule` refuses to run at all without an enabled slot."""
    config = ScheduleConfig()
    config.enabled = True
    config.slots = [
        RecurringSlot(day_of_week="monday", time="09:00:00", timezone="UTC")
    ]
    return config


class TestFailFastReachesTheLoader:
    """`--fail-fast` omitted must let the configured value win."""

    def test_the_parser_default_is_the_not_supplied_sentinel(self):
        """`action="store_true"` defaults to False, which the loader cannot
        tell from a deliberate `--fail-fast` absent... or present.
        """
        from src.scraper.amazon.scraper import build_argument_parser

        args = build_argument_parser().parse_args([])

        assert args.fail_fast is None, (
            "an omitted --fail-fast arrives as False, which the loader reads "
            "as a supplied value and prefers over batch.fail_fast"
        )

    def test_the_flag_still_works_when_passed(self):
        from src.scraper.amazon.scraper import build_argument_parser

        args = build_argument_parser().parse_args(["--fail-fast"])

        assert args.fail_fast is True

    def test_the_flag_can_be_forced_off(self):
        """`store_true` with `default=None` produces True or unset, never
        False, which leaves a user who configured `fail_fast: true` no way to
        ask for continue-on-error on one run. The paired form restores it.
        """
        from src.scraper.amazon.scraper import build_argument_parser

        args = build_argument_parser().parse_args(["--no-fail-fast"])

        assert args.fail_fast is False

    def test_the_configured_value_wins_when_the_flag_is_omitted(self):
        """The behaviour the fix exists for, through the real loader."""
        from src.scraper.amazon import config as scraper_config

        with patch.dict(
            scraper_config.CONFIG,
            {"batch": {"fail_fast": True, "keywords": []}},
            clear=False,
        ):
            batch = scraper_config.load_batch_config(
                cli_product_ids=["B0AAAAAAAA"], cli_keywords=[], cli_fail_fast=None
            )

        assert batch.fail_fast is True

    def test_the_flag_overrides_the_configured_value(self):
        """Precedence is CLI over YAML, not the other way round."""
        from src.scraper.amazon import config as scraper_config

        with patch.dict(
            scraper_config.CONFIG,
            {"batch": {"fail_fast": True, "keywords": []}},
            clear=False,
        ):
            batch = scraper_config.load_batch_config(
                cli_product_ids=["B0AAAAAAAA"], cli_keywords=[], cli_fail_fast=False
            )

        assert batch.fail_fast is False


class TestNoCleanupIsNotNoConfig:
    """`--no-cleanup` must say "off", not "nothing supplied"."""

    @pytest.mark.asyncio
    async def test_a_none_config_still_means_cleanup_is_on(self, tmp_path):
        """Pinned, because it is why `None` cannot express the flag.

        Making `None` mean off would silently stop cleaning up for every caller
        that passes nothing, which is most of them -- so the premise has to be
        observed, not asserted about the function object.
        """
        from src.publisher.schedule import ScheduleManager

        built: list = []

        class _Manager:
            def __init__(self, *args, **kwargs):
                built.append(args)

        manager = ScheduleManager(config=_enabled_schedule())

        with patch("src.publisher.cleanup.CleanupManager", _Manager):
            await manager.auto_schedule(
                videos=[],
                platforms=[],
                publisher=AsyncMock(),
                cleanup_config=None,
                outputs_dir=tmp_path,
            )

        assert built, (
            "`None` no longer means 'not supplied, clean up by default'. If "
            "that is deliberate, `--no-cleanup` can go back to passing None "
            "-- but every caller that passes nothing has just stopped "
            "cleaning up too."
        )

    @pytest.mark.asyncio
    async def test_a_disabled_config_creates_no_cleanup_manager(self, tmp_path):
        """The observable difference, driven through `auto_schedule`.

        With nothing to schedule the method returns before publishing, but the
        cleanup manager is built first -- which is exactly the decision the
        flag is supposed to make.
        """
        from src.publisher.schedule import ScheduleManager

        built: list = []

        class _Manager:
            def __init__(self, *args, **kwargs):
                built.append(kwargs or args)

        manager = ScheduleManager(config=_enabled_schedule())

        with patch("src.publisher.cleanup.CleanupManager", _Manager):
            await manager.auto_schedule(
                videos=[],
                platforms=[],
                publisher=AsyncMock(),
                cleanup_config=replace(CleanupConfig(), enabled=False),
                outputs_dir=tmp_path,
            )

        assert built == [], "cleanup was set up on a run that asked for none"

    @pytest.mark.asyncio
    async def test_an_enabled_config_does_create_one(self, tmp_path):
        """The counterpart, so the test above cannot pass by never building."""
        from src.publisher.schedule import ScheduleManager

        built: list = []

        class _Manager:
            def __init__(self, *args, **kwargs):
                built.append(args)

        manager = ScheduleManager(config=_enabled_schedule())

        with patch("src.publisher.cleanup.CleanupManager", _Manager):
            await manager.auto_schedule(
                videos=[],
                platforms=[],
                publisher=AsyncMock(),
                cleanup_config=CleanupConfig(),
                outputs_dir=tmp_path,
            )

        assert built, "cleanup should be set up when the config enables it"

    def test_the_cli_expresses_the_flag_as_a_disabled_config(self):
        """The call site is the defect; the helper above is the mechanism.

        Read from source because `cmd_schedule_auto` needs a publisher, a scanner
        and a slot calendar before it reaches this line, none of which say
        anything about the decision being tested.
        """
        import inspect

        from src.publisher.late import cli

        source = inspect.getsource(cli.cmd_schedule_auto)

        assert "cleanup_config = None if args.no_cleanup" not in source, (
            "`--no-cleanup` is expressed as an absent config again, which "
            "`auto_schedule` reads as 'not supplied' and answers with a "
            "default whose `enabled` is True"
        )
        assert "enabled=False" in source


def test_no_other_store_true_flag_feeds_a_not_supplied_resolver():
    """The class of bug, swept rather than the two instances.

    `load_batch_config` resolves five CLI arguments with the same
    `is not None` sentinel. Any `store_true` flag among them has the defect by
    construction, whatever its name.
    """
    import inspect

    from src.scraper.amazon import config as scraper_config
    from src.scraper.amazon.scraper import build_argument_parser

    resolved = {
        name
        for name in inspect.signature(scraper_config.load_batch_config).parameters
        if name.startswith("cli_")
    }

    defaults = vars(build_argument_parser().parse_args([]))
    offenders = [
        name
        for name in sorted(resolved)
        if defaults.get(name.removeprefix("cli_")) is False
    ]

    assert not offenders, (
        f"{offenders} reach a loader that treats a supplied False as an "
        "override, so the configured value can never win"
    )
