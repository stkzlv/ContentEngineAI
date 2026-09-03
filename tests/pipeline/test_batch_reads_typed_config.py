"""The batch reads its publisher settings from one typed object.

`global_batch.py` used to open `config/publisher.yaml` in four places with a
relative path and pull raw keys out of it. Two things followed, both silent.

A run from anywhere but the repository root took the `if config_path.exists()`
false branch and got hardcoded fallbacks, and one of those fallbacks was wrong:
`immediate_publish` defaulted to `True` here against a shipped `false`, so a
batch launched from another directory would have published immediately rather
than scheduling.

The second is the shape that already shipped a defect. Each section was
re-parsed at this call site, so a setting could reach `single` and not the
batch: `tiktok_settings` was absent from the batch's parse for several
releases, and the privacy level, comment permissions and AI label silently
differed between the two paths. Nothing logged the difference.

These tests therefore assert the property rather than the parse: no inline
read remains, and a value set in YAML arrives at the publish phase through the
typed config. Asserting that a shared loader exists would not have caught the
original defect, because the defect was a call site that called no loader.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BATCH = REPO / "src" / "pipeline" / "global_batch.py"


@pytest.mark.unit
class TestNoInlineConfigReadRemains:
    def test_the_batch_does_not_parse_yaml_itself(self) -> None:
        source = BATCH.read_text()

        assert "yaml.safe_load" not in source, (
            "the batch parses a config file itself again; load it through the "
            "typed loader so the path is absolute and the defaults are shared"
        )

    def test_the_batch_names_no_relative_config_path(self) -> None:
        source = BATCH.read_text()

        for literal in ('"config/publisher.yaml"', '"config/pipeline.yaml"'):
            assert literal not in source, (
                f"{literal} is relative to the working directory, so a run "
                "from elsewhere silently takes the fallbacks"
            )


@pytest.mark.unit
class TestABadConfigIsNotSwallowed:
    """The fallback read like caution and was the opposite.

    One unusable value discarded the entire file, and the run then published
    immediately to whichever platforms the dataclass defaults name, with the
    first comment, the disclosures, the stagger and the retention policy all
    silently reverted at the same time. Before this branch the same config
    aborted the publishing phase outright.
    """

    def test_an_unusable_value_reaches_the_caller(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.pipeline import global_batch
        from src.publisher import config as publisher_config

        path = tmp_path / "publisher.yaml"
        path.write_text(
            "late:\n  api_key_env_var: LATE_API_KEY\n"
            "default_platforms:\n  - instgram\n"
        )
        monkeypatch.setenv("LATE_API_KEY", "sk_live_" + "0" * 48)
        monkeypatch.setattr(publisher_config, "DEFAULT_PUBLISHER_CONFIG_PATH", path)
        global_batch._publisher_settings.cache_clear()

        with pytest.raises(ValueError, match="instgram"):
            global_batch._publisher_settings()

    def test_an_absent_credential_still_yields_the_configured_settings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The one case that is not about the file being unusable.

        The batch reads settings here and takes its key from the environment
        at publish time. Falling back to the dataclass defaults would discard
        the operator's schedule and platforms over a credential this code
        path never reads.
        """
        from src.pipeline import global_batch
        from src.publisher import config as publisher_config

        path = tmp_path / "publisher.yaml"
        path.write_text(
            "late:\n  api_key_env_var: LATE_API_KEY\n"
            "default_platforms:\n  - youtube\n"
            "immediate_publish: true\n"
        )
        monkeypatch.delenv("LATE_API_KEY", raising=False)
        monkeypatch.setattr(publisher_config, "DEFAULT_PUBLISHER_CONFIG_PATH", path)
        global_batch._publisher_settings.cache_clear()

        settings = global_batch._publisher_settings()

        assert [p.value for p in settings.default_platforms] == ["youtube"]
        assert settings.immediate_publish is True

    def test_every_publisher_takes_its_key_from_the_environment(self) -> None:
        """The placeholder exists so the dataclass will construct.

        Asserted at the call sites, not by slicing the source text. An earlier
        version split on the placeholder literal and inspected the one
        character that followed it, which no substring could ever match; it
        passed while `create_publisher` was handed the placeholder.
        """
        import ast

        tree = ast.parse(BATCH.read_text())
        keys = [
            keyword.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "create_publisher"
            for keyword in node.keywords
            if keyword.arg == "api_key"
        ]

        assert keys, "no create_publisher call passes an api_key"
        for value in keys:
            assert isinstance(value, ast.Name), (
                "api_key is built inline; it must be the environment variable "
                "the run already validated"
            )

    def test_a_malformed_slot_names_the_field(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """It used to escape the loader's own handler as a bare KeyError."""
        from src.publisher.config import load_publisher_config

        path = tmp_path / "publisher.yaml"
        path.write_text(
            "late:\n  api_key_env_var: LATE_API_KEY\n"
            "recurring_schedule:\n"
            "  enabled: true\n"
            "  slots:\n"
            "    - day: monday\n"
            "      time: '10:00:00'\n"
        )
        monkeypatch.setenv("LATE_API_KEY", "sk_live_" + "0" * 48)

        config = load_publisher_config(path)

        assert config.schedule_config.slots == [], (
            "a slot the loader cannot read is dropped with a warning, not "
            "raised as a KeyError out of the handler meant to catch it"
        )


@pytest.mark.unit
class TestAnUnreadableFileIsNotAnAbsentOne:
    """A parse failure used to become an empty mapping, then defaults.

    The defaults publish immediately, to the default platforms, with cleanup
    on. So one mis-indented line in `publisher.yaml` took a whole batch live
    on the spot and deleted the product directories afterwards.
    """

    def test_a_syntax_error_reaches_the_caller(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.pipeline import global_batch
        from src.publisher import config as publisher_config

        path = tmp_path / "publisher.yaml"
        path.write_text("late:\n  api_key_env_var: LATE_API_KEY\n bad: indent\n")
        monkeypatch.setenv("LATE_API_KEY", "sk_live_" + "0" * 48)
        monkeypatch.setattr(publisher_config, "DEFAULT_PUBLISHER_CONFIG_PATH", path)
        global_batch._publisher_settings.cache_clear()

        with pytest.raises(publisher_config.ConfigUnreadableError):
            global_batch._publisher_settings()

    def test_a_document_that_is_not_a_mapping_reaches_the_caller(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.publisher import config as publisher_config

        path = tmp_path / "publisher.yaml"
        path.write_text("- youtube\n- tiktok\n")
        monkeypatch.setenv("LATE_API_KEY", "sk_live_" + "0" * 48)

        with pytest.raises(publisher_config.ConfigUnreadableError):
            publisher_config.load_publisher_config(path)

    def test_an_absent_file_is_still_fine(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The supported state: everything from the environment."""
        from src.publisher.config import load_publisher_config

        monkeypatch.setenv("LATE_API_KEY", "sk_live_" + "0" * 48)

        config = load_publisher_config(tmp_path / "nothing-here.yaml")

        assert config.provider


@pytest.mark.unit
class TestOneBadSlotDoesNotDiscardTheSchedule:
    """An empty slot list with `enabled: true` publishes immediately.

    So a single typo in one slot used to turn a scheduled batch into a live
    one, while the warning said only that slots could not be parsed.
    """

    @staticmethod
    def _config(tmp_path: Path, slots: str) -> Path:
        path = tmp_path / "publisher.yaml"
        path.write_text(
            "late:\n  api_key_env_var: LATE_API_KEY\n"
            "recurring_schedule:\n  enabled: true\n  slots:\n" + slots
        )
        return path

    def test_the_good_slots_survive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.publisher.config import load_publisher_config

        monkeypatch.setenv("LATE_API_KEY", "sk_live_" + "0" * 48)
        path = self._config(
            tmp_path,
            "    - day_of_week: monday\n      time: '10:00:00'\n"
            "    - day: tuesday\n      time: '10:00:00'\n"
            "    - day_of_week: wednesday\n      time: '10:00:00'\n",
        )

        config = load_publisher_config(path)

        assert [s.day_of_week for s in config.schedule_config.slots] == [
            "monday",
            "wednesday",
        ]

    def test_a_slot_that_is_not_a_mapping_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`.get` on a string raises AttributeError, past the old handler."""
        from src.publisher.config import load_publisher_config

        monkeypatch.setenv("LATE_API_KEY", "sk_live_" + "0" * 48)
        path = self._config(
            tmp_path,
            "    - monday 10:00\n"
            "    - day_of_week: friday\n      time: '10:00:00'\n",
        )

        config = load_publisher_config(path)

        assert [s.day_of_week for s in config.schedule_config.slots] == ["friday"]


@pytest.mark.unit
class TestScheduleTimeIsNormalised:
    def test_an_unquoted_yaml_timestamp_becomes_a_string(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The consumer calls `.replace("Z", ...)` on it."""
        from src.publisher.config import load_publisher_config

        path = tmp_path / "publisher.yaml"
        path.write_text(
            "late:\n  api_key_env_var: LATE_API_KEY\n"
            "schedule_time: 2026-09-05 10:00:00\n"
        )
        monkeypatch.setenv("LATE_API_KEY", "sk_live_" + "0" * 48)

        config = load_publisher_config(path)

        assert isinstance(config.schedule_time, str)
        assert config.schedule_time.startswith("2026-09-05")


@pytest.mark.unit
class TestTheLoadersResolveFromAnyDirectory:
    """The defect one level down: both loaders defaulted to a relative path."""

    def test_the_publisher_default_is_absolute(self) -> None:
        from src.publisher.config import DEFAULT_PUBLISHER_CONFIG_PATH

        assert DEFAULT_PUBLISHER_CONFIG_PATH.is_absolute()
        assert DEFAULT_PUBLISHER_CONFIG_PATH.exists()

    def test_the_pipeline_default_is_absolute(self) -> None:
        from src.pipeline.config import DEFAULT_PIPELINE_CONFIG_PATH

        assert DEFAULT_PIPELINE_CONFIG_PATH.is_absolute()
        assert DEFAULT_PIPELINE_CONFIG_PATH.exists()

    def test_the_shipped_values_load_from_another_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The case the relative default got wrong."""
        from src.publisher.config import load_publisher_config

        monkeypatch.setenv("LATE_API_KEY", "sk_live_" + "0" * 48)
        monkeypatch.chdir(tmp_path)

        config = load_publisher_config()

        assert config.default_platforms, "fell back to the empty default"
        # The shipped file says false; the batch's old inline default said true.
        assert config.immediate_publish is False


@pytest.mark.unit
class TestDefaultPlatformsFlowFromYaml:
    """YAML -> typed config -> the value the publish phase would target."""

    @staticmethod
    def _write(tmp_path: Path, body: str) -> Path:
        path = tmp_path / "publisher.yaml"
        path.write_text("late:\n  api_key_env_var: LATE_API_KEY\n" + body)
        return path

    def test_a_changed_value_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.publisher.config import load_publisher_config

        monkeypatch.setenv("LATE_API_KEY", "sk_live_" + "0" * 48)
        path = self._write(tmp_path, "default_platforms:\n  - youtube\n")

        config = load_publisher_config(path)

        assert [p.value for p in config.default_platforms] == ["youtube"]

    def test_yaml_platforms_arrive_as_enums(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The annotation said Platform while YAML delivered strings.

        Nothing converted them, so `to_dict` and every caller reading `.value`
        raised on any config loaded from the shipped file, and mypy could not
        see it because it trusts the annotation.
        """
        from src.publisher.config import load_publisher_config
        from src.publisher.models import Platform

        monkeypatch.setenv("LATE_API_KEY", "sk_live_" + "0" * 48)
        path = self._write(
            tmp_path,
            "default_platforms:\n  - youtube\nprivacy_settings:\n  youtube: public\n",
        )

        config = load_publisher_config(path)

        assert config.default_platforms == [Platform.YOUTUBE]
        assert set(config.privacy_settings) == {Platform.YOUTUBE}
        assert config.to_dict()["default_platforms"] == ["youtube"]

    def test_an_unknown_platform_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dropping it would publish to fewer platforms than configured."""
        from src.publisher.config import load_publisher_config

        monkeypatch.setenv("LATE_API_KEY", "sk_live_" + "0" * 48)
        path = self._write(tmp_path, "default_platforms:\n  - facebok\n")

        with pytest.raises(ValueError, match="facebok"):
            load_publisher_config(path)

    def test_an_absent_key_falls_back_to_the_documented_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.publisher.config import load_publisher_config
        from src.publisher.models import PublisherConfig

        monkeypatch.setenv("LATE_API_KEY", "sk_live_" + "0" * 48)
        path = self._write(tmp_path, "")

        config = load_publisher_config(path)

        assert (
            config.default_platforms
            == PublisherConfig(provider="late", api_key="x").default_platforms
        )


@pytest.mark.unit
class TestThePublishPhaseUsesTheTypedObject:
    """A shared loader existing is not the guard; the call site is."""

    @staticmethod
    def _function(name: str) -> ast.AST:
        tree = ast.parse(BATCH.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef)
                and node.name == name
            ):
                return node
        raise AssertionError(f"{name} not found in {BATCH.name}")

    @pytest.mark.parametrize(
        "function",
        [
            "_execute_publishing_phase",
            "display_execution_plan",
            "_default_platforms",
            "_publisher_profiles",
        ],
    )
    def test_it_reads_the_typed_settings(self, function: str) -> None:
        node = self._function(function)

        called = {
            n.func.id
            for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        assert "_publisher_settings" in called, (
            f"{function} does not read the typed publisher config, so it can "
            "drift from the CLI path the way tiktok_settings did"
        )

    def test_the_phase_no_longer_re_parses_the_sections(self) -> None:
        """Each of these was its own parse, and one was simply missing."""
        source = ast.unparse(self._function("_execute_publishing_phase"))

        for section in (
            "first_comment",
            "tiktok_settings",
            "blob_retention",
            "delivery_sweep",
            "link_in_bio",
            "affiliate_disclosure",
        ):
            assert f"'{section}'" not in source, (
                f"{section} is parsed here again instead of read off the "
                "typed config, which is how a section goes missing"
            )


@pytest.mark.unit
class TestTheWebhookComesFromTheLoadedConfig:
    def test_main_does_not_reopen_the_pipeline_file(self) -> None:
        source = BATCH.read_text()

        assert (
            "webhook_yaml" in source
        ), "the webhook slice is not carried on the config"
        assert 'open("config/pipeline.yaml")' not in source

    def test_a_configured_url_survives_the_round_trip(self, tmp_path: Path) -> None:
        """YAML -> loader -> the object `main` hands to the notifier.

        Asserting only that the dataclass declares the field would pass with
        the loader never populating it, which is silent: webhooks would simply
        stop firing for every batch run and nothing would fail.
        """
        import argparse

        import yaml

        from src.pipeline.config import load_global_batch_config
        from src.pipeline.webhooks import load_webhook_config

        url = "https://hooks.example.test/abc"
        path = tmp_path / "pipeline.yaml"
        path.write_text(
            yaml.safe_dump({"global_batch": {"webhook": {"url": url, "enabled": True}}})
        )

        config = load_global_batch_config(argparse.Namespace(), config_path=path)

        assert load_webhook_config(config.webhook_yaml).url == url

    def test_an_absent_webhook_section_configures_nothing(self, tmp_path: Path) -> None:
        import argparse

        import yaml

        from src.pipeline.config import load_global_batch_config
        from src.pipeline.webhooks import load_webhook_config

        path = tmp_path / "pipeline.yaml"
        path.write_text(yaml.safe_dump({"global_batch": {}}))

        config = load_global_batch_config(argparse.Namespace(), config_path=path)

        assert not load_webhook_config(config.webhook_yaml).is_configured()
