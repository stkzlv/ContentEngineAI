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
