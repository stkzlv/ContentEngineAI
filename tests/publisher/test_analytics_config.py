"""The sweep size is configured in one place, and the CLI can still override it.

A scheduled sweep passes no ``--limit``, deliberately, so that the size lives
in ``config/publisher.yaml`` and nowhere else. That only works if an omitted
flag is distinguishable from a passed one, which is why ``--limit`` defaults to
None rather than to a number.
"""

import argparse
import ast
from pathlib import Path
from unittest.mock import patch

import pytest

from src.publisher.config import load_publisher_config
from src.publisher.late.cli import _analytics_limit
from src.publisher.models import AnalyticsConfig, PublisherConfig

MINIMAL = "provider: late\napi_key: sk_live_key_12345\n"


def _write(tmp_path, body):
    path = tmp_path / "publisher.yaml"
    path.write_text(body)
    return path


class TestAnalyticsConfigSection:
    """The YAML section parses, and its absence is not a failure."""

    @patch.dict("os.environ", {}, clear=True)
    def test_defaults_when_section_missing(self, tmp_path):
        """No analytics section leaves the sweep size at the dataclass default.

        An install that never configured this still has to sweep, so a missing
        section is an unconfigured default rather than an error.
        """
        config = load_publisher_config(config_path=_write(tmp_path, MINIMAL))

        assert isinstance(config.analytics_config, AnalyticsConfig)
        assert config.analytics_config.limit == AnalyticsConfig().limit

    @patch.dict("os.environ", {}, clear=True)
    def test_defaults_when_section_empty(self, tmp_path):
        """An empty section parses to None and must be read as absent.

        Shares a path with the missing-section case: `analytics:` with no body
        yields None, which would be passed straight to the constructor by a
        loader that only checked for the key.
        """
        config = load_publisher_config(
            config_path=_write(tmp_path, MINIMAL + "analytics:\n")
        )

        assert config.analytics_config.limit == AnalyticsConfig().limit

    @patch.dict("os.environ", {}, clear=True)
    def test_limit_read_from_yaml(self, tmp_path):
        """A configured limit reaches the loaded config."""
        config = load_publisher_config(
            config_path=_write(tmp_path, MINIMAL + "analytics:\n  limit: 12\n")
        )

        assert config.analytics_config.limit == 12

    @patch.dict("os.environ", {}, clear=True)
    def test_invalid_limit_warns_and_falls_back(self, tmp_path, caplog):
        """A limit that would measure nothing falls back, and says so.

        The fallback matches every other section here, but silence would be
        wrong: this runs unattended on the timer, where the log line is the
        only evidence that the configured size was not the one used.
        """
        config = load_publisher_config(
            config_path=_write(tmp_path, MINIMAL + "analytics:\n  limit: 0\n")
        )

        assert config.analytics_config.limit == AnalyticsConfig().limit
        assert "analytics" in caplog.text.lower()

    @patch.dict("os.environ", {}, clear=True)
    def test_raw_section_never_reaches_the_constructor(self, tmp_path):
        """The raw YAML key is popped, leaving only the parsed object.

        Guards the pop: an unpopped section would be dropped by the
        unknown-key strip instead, which is silent, so the pop is what keeps
        the intent greppable.
        """
        config = load_publisher_config(
            config_path=_write(tmp_path, MINIMAL + "analytics:\n  limit: 7\n")
        )

        assert not hasattr(config, "analytics")
        assert config.analytics_config.limit == 7


class TestAnalyticsLimitPrecedence:
    """CLI flag beats config; config beats the dataclass default."""

    @staticmethod
    def _args(limit):
        return argparse.Namespace(limit=limit)

    @staticmethod
    def _config(limit):
        return PublisherConfig(
            provider="late",
            api_key="sk_live_key_12345",
            analytics_config=AnalyticsConfig(limit=limit),
        )

    def test_cli_flag_wins_over_config(self):
        assert _analytics_limit(self._args(7), self._config(50)) == 7

    def test_config_used_when_flag_omitted(self):
        assert _analytics_limit(self._args(None), self._config(12)) == 12

    def test_cli_flag_equal_to_the_shipped_default_still_wins(self):
        """Passing 50 explicitly must beat a configured 12.

        The resolver cannot tell these apart on its own; it relies on the
        parser handing it None for an omitted flag, which
        ``TestParserDefault`` is what actually guards.
        """
        assert _analytics_limit(self._args(50), self._config(12)) == 50

    @patch.dict("os.environ", {}, clear=True)
    def test_dataclass_default_reached_end_to_end(self, tmp_path):
        """With no flag and no section, nothing in between invents a number."""
        config = load_publisher_config(config_path=_write(tmp_path, MINIMAL))

        assert _analytics_limit(self._args(None), config) == AnalyticsConfig().limit


class TestParserDefault:
    """The analytics --limit flag must declare no numeric default.

    ``_analytics_limit`` can only prefer the configured value when an omitted
    flag arrives as None, so the parser declaration is the load-bearing half
    and the resolver tests cannot see it: they build a Namespace directly.
    The parser is built inline inside ``main()`` and cannot be imported, so
    this reads the declaration structurally instead of by string match.
    """

    @staticmethod
    def _limit_default_for(parser_var):
        source = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "publisher"
            / "late"
            / "cli.py"
        )
        for node in ast.walk(ast.parse(source.read_text())):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
                continue
            if not isinstance(func.value, ast.Name) or func.value.id != parser_var:
                continue
            if not node.args or getattr(node.args[0], "value", None) != "--limit":
                continue
            for kw in node.keywords:
                if kw.arg == "default":
                    return kw.value
            return "no default keyword"
        return "no such argument"

    def test_analytics_limit_defaults_to_none(self):
        default = self._limit_default_for("analytics_parser")

        assert isinstance(default, ast.Constant), f"unexpected default: {default}"
        assert default.value is None, (
            "analytics --limit declares a numeric default, so an omitted flag "
            "is indistinguishable from a passed one and analytics.limit in "
            "config/publisher.yaml can never take effect"
        )

    def test_a_sibling_command_is_unaffected(self):
        """verify-comments keeps its own numeric default.

        The three subcommands' --limit flags no longer behave alike, which is
        deliberate: only the analytics one is config-backed.
        """
        default = self._limit_default_for("verify_parser")

        assert isinstance(default, ast.Constant)
        assert default.value == 25


class TestMalformedSectionDoesNotBreakThePublisher:
    """A bad analytics section degrades; it must not abort the whole load.

    ``load_publisher_config`` backs every publisher subcommand, so an
    exception escaping the analytics parse stops publishing too, not just the
    sweep. Every sibling section degrades to its defaults on the same input.
    """

    @patch.dict("os.environ", {}, clear=True)
    @pytest.mark.parametrize(
        "section",
        [
            "analytics: 50",
            "analytics: fifty",
            "analytics:\n  - 50",
            "analytics:\n  limit: 12.5",
            "analytics:\n  limit: true",
            "analytics:\n  nope: 1",
        ],
        ids=["scalar-int", "scalar-str", "list", "float", "bool", "unknown-key"],
    )
    def test_bad_shape_falls_back_instead_of_raising(self, tmp_path, section):
        config = load_publisher_config(
            config_path=_write(tmp_path, MINIMAL + section + "\n")
        )

        assert config.analytics_config.limit == AnalyticsConfig().limit

    def test_a_float_limit_is_rejected_at_construction(self):
        """It would otherwise pass ``< 1`` and break ``posts[:limit]`` mid-sweep."""
        with pytest.raises(TypeError):
            AnalyticsConfig(limit=12.5)

    def test_a_bool_limit_is_rejected(self):
        """``bool`` is an int subclass, so ``posts[:True]`` measures one post."""
        with pytest.raises(TypeError):
            AnalyticsConfig(limit=True)


class TestTheSweepCoversTheShippedCadence:
    """The sweep size and the publishing cadence are coupled and nothing
    reads them together.

    A durability ratio needs a post older than ``DURABILITY_WINDOW_DAYS``
    while its timeline still reaches publication, and the provider retains
    about five weeks. So the sweep has to reach back at least that far, and
    how many posts that is depends entirely on how many slots a week the
    bundled schedule declares. Doubling the slots without raising the limit
    costs every post its ratio, permanently and silently -- a short sweep
    looks exactly like a complete one, and the rows are unrecoverable once
    they age past retention.

    This asserts the two shipped values against each other rather than
    pinning either one, so changing the cadence is free and changing it
    alone is not.
    """

    def test_the_shipped_limit_reaches_back_past_the_durability_window(
        self, monkeypatch
    ):
        from src.publisher.analytics import DURABILITY_WINDOW_DAYS

        # The bundled file carries no key; the loader refuses without one.
        monkeypatch.setenv("LATE_API_KEY", "sk_test_not_a_real_key")
        config = load_publisher_config(
            config_path=Path(__file__).resolve().parents[2]
            / "config"
            / "publisher.yaml"
        )
        slots_per_week = len(config.schedule_config.slots)
        assert slots_per_week, "the bundled schedule declares no slots"

        # Derived from the window rather than restated, or raising
        # DURABILITY_WINDOW_DAYS leaves this green while the sweep no longer
        # reaches the posts it governs. The five days are slack for a missed
        # sweep; the schedule runs daily.
        days = DURABILITY_WINDOW_DAYS + 5
        needed = round(days * slots_per_week / 7)

        assert config.analytics_config.limit >= needed, (
            f"{slots_per_week} slots a week fills {needed} posts into "
            f"{days} days, but the bundled sweep measures only "
            f"{config.analytics_config.limit}; a post ages out before day "
            f"{DURABILITY_WINDOW_DAYS} and its ratio is lost for good"
        )

        # The default carries the same requirement. A config with no
        # `analytics:` section falls back to it and logs nothing, so an
        # installation whose file predates that section is governed by this
        # number alone.
        assert AnalyticsConfig().limit >= needed, (
            f"the bundled schedule needs {needed}, but a config with no "
            f"analytics section silently gets {AnalyticsConfig().limit}"
        )
